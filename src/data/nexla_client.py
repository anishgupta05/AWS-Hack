"""
Nexla integration — schema normalization, incremental merging, and
feature transformation.

Architecture
------------
All public functions delegate to a module-level _backend singleton.
The active backend is chosen at import time from the environment:

    NEXLA_API_KEY set  →  _NexlaAPIBackend  (real Nexla jobs)
    not set            →  _LocalBackend     (pandas fallback, logs a warning)

To swap to the real Nexla API: set NEXLA_API_KEY (and optionally
NEXLA_WORKSPACE_ID) in the environment before importing this module.
Nothing else in the calling code changes — the public function
signatures are identical across both backends.

Public entry points
-------------------
merge_and_normalize(working_df, new_source_df, source_name) -> pd.DataFrame
    Primary entry point used by the agent loop on every incremental pull.
    Normalises new_source_df to the canonical schema then merges it into
    working_df. One call does both steps so there is no window where the
    working dataset is half-merged.

normalize(raw_df, source_name) -> pd.DataFrame
    Normalise a single raw source DataFrame to CANONICAL_COLUMNS + source.
    Called by merge_and_normalize; exposed separately for testing.

merge(working_df, new_normalized_df) -> pd.DataFrame
    Concatenate two already-normalised DataFrames. Called by
    merge_and_normalize; exposed separately for testing.

transform(working_df, spec) -> pd.DataFrame
    Apply a feature-engineering transformation described by spec.
    Called by the agent loop when the diagnosis verdict is NEED_TRANSFORM.

    Supported ops (spec dict):
        {"op": "log_scale",      "columns": [...]}
        {"op": "standardize",    "columns": [...]}
        {"op": "add_interaction","columns": [col_a, col_b], "name": str}
        {"op": "clip_outliers",  "columns": [...], "n_std": float}

    The diagnosis module produces these specs; this function executes them.

Job logging
-----------
Every backend call logs a "[nexla] job_id=<id> op=<op> ..." line.
Local job IDs look like  local-a3f2c1b8  (random hex).
Nexla API job IDs come from the API response.
Person B can capture these from the log or from the module-level job_log
list (list of dicts, one per completed operation).
"""

from __future__ import annotations

import logging
import os
import secrets
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target",
]

# Numeric feature columns (excludes target and source).
_NUMERIC_FEATURES: list[str] = [c for c in CANONICAL_COLUMNS if c != "target"]

# ---------------------------------------------------------------------------
# Job audit log — Person B reads this for the dashboard
# ---------------------------------------------------------------------------

job_log: list[dict[str, Any]] = []


def _record_job(job_id: str, backend: str, operation: str, **kwargs: Any) -> None:
    entry = {"job_id": job_id, "backend": backend, "operation": operation, **kwargs}
    job_log.append(entry)
    logger.info(
        "[nexla] job_id=%s backend=%s op=%s %s",
        job_id,
        backend,
        operation,
        " ".join(f"{k}={v}" for k, v in kwargs.items()),
    )


def _local_job_id() -> str:
    return f"local-{secrets.token_hex(4)}"


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class _Backend(ABC):
    @abstractmethod
    def normalize(self, raw_df: pd.DataFrame, source_name: str) -> pd.DataFrame: ...

    @abstractmethod
    def merge(self, working_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def transform(self, working_df: pd.DataFrame, spec: dict) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Local (pandas) backend — fully implemented, used when no API key is set
# ---------------------------------------------------------------------------

class _LocalBackend(_Backend):
    """Pure-pandas fallback. Behaviour is identical to the Nexla API backend;
    the only difference is that jobs run in-process instead of on Nexla's
    infra, and job IDs are synthetic."""

    def normalize(self, raw_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        job_id = _local_job_id()
        df = raw_df.copy()

        # Ensure every canonical column is present; add NaN column if absent.
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                logger.warning(
                    "[nexla] %s missing column '%s' — filling with NaN", source_name, col
                )
                df[col] = np.nan

        # Cast all numeric feature columns to float so dtypes are homogeneous
        # across hospital sources regardless of how UCI encoded them.
        for col in _NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        # Preserve 'source' provenance tag added by uci_client; keep only
        # canonical columns plus source.
        keep = CANONICAL_COLUMNS + (["source"] if "source" in df.columns else [])
        df = df[keep].reset_index(drop=True)

        _record_job(
            job_id, "local", "normalize",
            source=source_name, rows=len(df),
        )
        return df

    def merge(self, working_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        job_id = _local_job_id()

        # Align columns — new sources may have a subset if some canonical
        # columns were absent and filled with NaN during normalize().
        merged = pd.concat(
            [working_df, new_df],
            ignore_index=True,
            sort=False,
        )

        # Canonical column order, then source if present.
        col_order = CANONICAL_COLUMNS + (
            ["source"] if "source" in merged.columns else []
        )
        merged = merged[col_order].reset_index(drop=True)

        _record_job(
            job_id, "local", "merge",
            rows_before=len(working_df),
            rows_added=len(new_df),
            rows_after=len(merged),
        )
        return merged

    def transform(self, working_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
        job_id = _local_job_id()
        op = spec.get("op")
        df = working_df.copy()

        if op == "log_scale":
            cols = spec["columns"]
            for col in cols:
                df[col] = np.log1p(df[col].clip(lower=0))
            _record_job(job_id, "local", "transform:log_scale", columns=cols)

        elif op == "standardize":
            cols = spec["columns"]
            for col in cols:
                mean, std = df[col].mean(), df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
                else:
                    logger.warning(
                        "[nexla] standardize: '%s' has zero std — skipping", col
                    )
            _record_job(job_id, "local", "transform:standardize", columns=cols)

        elif op == "add_interaction":
            cols = spec["columns"]
            if len(cols) != 2:
                raise ValueError("add_interaction requires exactly 2 columns")
            col_a, col_b = cols
            name = spec.get("name", f"{col_a}_x_{col_b}")
            df[name] = df[col_a] * df[col_b]
            _record_job(
                job_id, "local", "transform:add_interaction",
                col_a=col_a, col_b=col_b, new_col=name,
            )

        elif op == "clip_outliers":
            cols = spec["columns"]
            n_std = float(spec.get("n_std", 3.0))
            for col in cols:
                mean, std = df[col].mean(), df[col].std()
                df[col] = df[col].clip(lower=mean - n_std * std, upper=mean + n_std * std)
            _record_job(
                job_id, "local", "transform:clip_outliers",
                columns=cols, n_std=n_std,
            )

        else:
            raise ValueError(
                f"Unknown transform op {op!r}. "
                "Valid ops: log_scale, standardize, add_interaction, clip_outliers"
            )

        return df


# ---------------------------------------------------------------------------
# Nexla API backend — stubbed; fill in HTTP calls when credentials arrive
# ---------------------------------------------------------------------------

class _NexlaAPIBackend(_Backend):
    """Real Nexla API backend.

    To complete this stub, replace each `raise NotImplementedError` block
    with the appropriate Nexla REST call. The expected flow per operation:
      1. POST to create/trigger a Nexla job  (returns job_id)
      2. Poll GET until job status == "completed"
      3. GET the output dataset and parse it back to a DataFrame
      4. Call _record_job(job_id, "nexla_api", op, ...) before returning

    Nexla API reference: https://developer.nexla.io/
    Auth header:   Authorization: Bearer <NEXLA_API_KEY>
    Workspace:     X-Nexla-Workspace: <NEXLA_WORKSPACE_ID>
    """

    def __init__(self, api_key: str, workspace_id: str) -> None:
        self._api_key = api_key
        self._workspace_id = workspace_id
        self._base = "https://api.nexla.com/v1"   # verify against current docs
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Nexla-Workspace": self._workspace_id,
            "Content-Type": "application/json",
        }

    def normalize(self, raw_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        # TODO: upload raw_df as a Nexla source dataset, apply the
        # heart-disease schema mapping flow, download normalised output.
        #
        # Rough skeleton:
        #   resp = requests.post(f"{self._base}/datasets", headers=self._headers,
        #                        json={"name": f"uci-{source_name}-raw",
        #                              "data": raw_df.to_dict(orient="records")})
        #   resp.raise_for_status()
        #   job_id = resp.json()["job_id"]
        #   _poll_until_done(self._base, self._headers, job_id)
        #   output = _download_output(self._base, self._headers, job_id)
        #   _record_job(job_id, "nexla_api", "normalize", source=source_name)
        #   return output
        raise NotImplementedError("Nexla API normalize — wire up HTTP calls here")

    def merge(self, working_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        # TODO: submit both datasets to a Nexla merge flow, poll, download.
        raise NotImplementedError("Nexla API merge — wire up HTTP calls here")

    def transform(self, working_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
        # TODO: submit working_df + spec to a Nexla transformation flow.
        raise NotImplementedError("Nexla API transform — wire up HTTP calls here")


# ---------------------------------------------------------------------------
# Active backend — swap point: one env-var change switches implementations
# ---------------------------------------------------------------------------

def _init_backend() -> _Backend:
    api_key = os.environ.get("NEXLA_API_KEY", "")
    workspace_id = os.environ.get("NEXLA_WORKSPACE_ID", "")
    if api_key:
        logger.info("[nexla] NEXLA_API_KEY found — using Nexla API backend")
        return _NexlaAPIBackend(api_key, workspace_id)
    logger.warning(
        "[nexla] NEXLA_API_KEY not set — using local pandas fallback. "
        "Set NEXLA_API_KEY to switch to real Nexla jobs."
    )
    return _LocalBackend()


_backend: _Backend = _init_backend()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_and_normalize(
    working_df: pd.DataFrame | None,
    new_source_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """Normalise new_source_df and merge it into working_df.

    Primary entry point for the agent loop on every incremental data pull.
    If working_df is None (first pull), the normalised source becomes the
    working dataset.

    Parameters
    ----------
    working_df:     existing working dataset, or None on the first pull.
    new_source_df:  raw DataFrame returned by uci_client.fetch_source().
    source_name:    name of the hospital source being added (for logging).

    Returns
    -------
    Merged DataFrame with CANONICAL_COLUMNS + source column.
    """
    normalised = _backend.normalize(new_source_df, source_name)
    if working_df is None or len(working_df) == 0:
        logger.info("[nexla] first source — working dataset initialised with %s", source_name)
        return normalised
    return _backend.merge(working_df, normalised)


def normalize(raw_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalise a single raw source DataFrame to the canonical schema."""
    return _backend.normalize(raw_df, source_name)


def merge(working_df: pd.DataFrame, new_normalized_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two already-normalised DataFrames."""
    return _backend.merge(working_df, new_normalized_df)


def transform(working_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Apply a feature-engineering transformation spec to the working dataset."""
    return _backend.transform(working_df, spec)
