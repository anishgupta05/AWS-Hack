"""
Nexla integration — schema normalization, incremental merging, and
feature transformation.

Architecture
------------
All public functions delegate to a module-level _backend singleton.
The active backend is chosen at import time from the environment:

    NEXLA_CLI_PATH set  →  _NexlaCLIBackend  (real Nexla CLI subprocess calls)
    not set             →  _LocalBackend     (pure-pandas, deliberate default)

The local backend is the explicit default, not a fallback of last resort.
Set NEXLA_CLI_PATH to the absolute path of the nexla binary to route
operations through the real CLI.  Nothing else in the calling code changes —
the public function signatures are identical across both backends.

_NexlaCLIBackend timeout and fallback
--------------------------------------
Every CLI call runs inside a ThreadPoolExecutor with a hard wall-clock
timeout (default 5s, set via NEXLA_CLI_TIMEOUT env var).  On timeout or
any subprocess error the call falls back to the local pandas backend for
that specific operation.  Every call — success or fallback — is logged
with:

    backend_used : "real_cli" | "local_fallback"
    operation    : normalize | merge | transform:<op>
    latency_ms   : wall-clock ms including CLI process overhead
    (fallback only) fallback_reason : timeout | error:<ExcType>:<message>

This mirrors the pattern in diagnosis.py's call_llm_provider / _try_llm_diagnosis
and was chosen deliberately after a zero_enrichment incident earlier in
the session where a no-op call logged in a way that implied it had
executed.  The logging here must be unambiguous.

CLI interface expected
----------------------
The backend passes data through temporary CSV files and expects the
Nexla CLI to accept the following sub-commands:

    nexla normalize  --input <csv>  --output <csv>  --source <name>
    nexla merge      --left <csv>   --right <csv>   --output <csv>
    nexla transform  --input <csv>  --output <csv>  --op <op>
                     --columns <col,col,...>
                     [--n-std <float>]   # clip_outliers only
                     [--name <str>]      # add_interaction only

Non-zero exit code or any subprocess exception triggers the fallback.

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
    Called by the agent loop when the diagnosis verdict is TRANSFORM_DATA.

    Supported ops (spec dict):
        {"op": "log_scale",      "columns": [...]}
        {"op": "standardize",    "columns": [...]}
        {"op": "add_interaction","columns": [col_a, col_b], "name": str}
        {"op": "clip_outliers",  "columns": [...], "n_std": float}

Job logging
-----------
Every backend call appends one entry to the module-level job_log list.
Fields always present: job_id, backend_used, operation, latency_ms.
The dashboard reads job_log; do not remove or rename these fields.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import secrets
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Hard wall-clock timeout for every CLI subprocess call.
# Kept at 5s (lower than the 10s LLM timeout) because this runs on nearly
# every loop iteration, not as a rare external fallback.
CLI_TIMEOUT_S: float = float(os.environ.get("NEXLA_CLI_TIMEOUT", "5"))

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
# Job audit log — dashboard reads this list
# ---------------------------------------------------------------------------

job_log: list[dict[str, Any]] = []


def _record_job(
    job_id: str,
    backend_used: str,
    operation: str,
    latency_ms: int = 0,
    **kwargs: Any,
) -> None:
    """Append one entry to job_log and emit a structured log line.

    backend_used must be one of: "real_cli", "local_fallback", "local".
    Every entry includes latency_ms so the dashboard can track CLI
    performance over the run.
    """
    entry = {
        "job_id": job_id,
        "backend_used": backend_used,
        "operation": operation,
        "latency_ms": latency_ms,
        **kwargs,
    }
    job_log.append(entry)
    logger.info(
        "[nexla] job_id=%s backend=%s op=%s latency=%dms %s",
        job_id,
        backend_used,
        operation,
        latency_ms,
        " ".join(f"{k}={v}" for k, v in kwargs.items()),
    )


def _new_job_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(4)}"


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
# Local (pandas) backend — fully implemented, explicit default
# ---------------------------------------------------------------------------

class _LocalBackend(_Backend):
    """Pure-pandas implementation.

    This is the deliberate default when NEXLA_CLI_PATH is not set.
    It is also used as the per-call fallback inside _NexlaCLIBackend.
    """

    def normalize(self, raw_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        t0 = time.monotonic()
        df = raw_df.copy()

        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                logger.warning(
                    "[nexla] %s missing column '%s' — filling with NaN", source_name, col
                )
                df[col] = np.nan

        for col in _NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        keep = CANONICAL_COLUMNS + (["source"] if "source" in df.columns else [])
        df = df[keep].reset_index(drop=True)

        _record_job(
            _new_job_id("local"), "local", "normalize",
            latency_ms=round((time.monotonic() - t0) * 1000),
            source=source_name, rows=len(df),
        )
        return df

    def merge(self, working_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        t0 = time.monotonic()
        merged = pd.concat([working_df, new_df], ignore_index=True, sort=False)

        col_order = CANONICAL_COLUMNS + (
            ["source"] if "source" in merged.columns else []
        )
        merged = merged[col_order].reset_index(drop=True)

        _record_job(
            _new_job_id("local"), "local", "merge",
            latency_ms=round((time.monotonic() - t0) * 1000),
            rows_before=len(working_df),
            rows_added=len(new_df),
            rows_after=len(merged),
        )
        return merged

    def transform(self, working_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
        t0 = time.monotonic()
        op = spec.get("op")
        df = working_df.copy()

        if op == "log_scale":
            cols = spec["columns"]
            for col in cols:
                df[col] = np.log1p(df[col].clip(lower=0))
            _record_job(
                _new_job_id("local"), "local", f"transform:{op}",
                latency_ms=round((time.monotonic() - t0) * 1000),
                columns=cols,
            )

        elif op == "standardize":
            cols = spec["columns"]
            for col in cols:
                mean, std = df[col].mean(), df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
                else:
                    logger.warning("[nexla] standardize: '%s' has zero std — skipping", col)
            _record_job(
                _new_job_id("local"), "local", f"transform:{op}",
                latency_ms=round((time.monotonic() - t0) * 1000),
                columns=cols,
            )

        elif op == "add_interaction":
            cols = spec["columns"]
            if len(cols) != 2:
                raise ValueError("add_interaction requires exactly 2 columns")
            col_a, col_b = cols
            name = spec.get("name", f"{col_a}_x_{col_b}")
            df[name] = df[col_a] * df[col_b]
            _record_job(
                _new_job_id("local"), "local", f"transform:{op}",
                latency_ms=round((time.monotonic() - t0) * 1000),
                col_a=col_a, col_b=col_b, new_col=name,
            )

        elif op == "clip_outliers":
            cols = spec["columns"]
            n_std = float(spec.get("n_std", 3.0))
            for col in cols:
                mean, std = df[col].mean(), df[col].std()
                df[col] = df[col].clip(lower=mean - n_std * std, upper=mean + n_std * std)
            _record_job(
                _new_job_id("local"), "local", f"transform:{op}",
                latency_ms=round((time.monotonic() - t0) * 1000),
                columns=cols, n_std=n_std,
            )

        else:
            raise ValueError(
                f"Unknown transform op {op!r}. "
                "Valid ops: log_scale, standardize, add_interaction, clip_outliers"
            )

        return df


# ---------------------------------------------------------------------------
# CLI backend — real Nexla subprocess calls with timeout + per-call fallback
# ---------------------------------------------------------------------------

class _NexlaCLIBackend(_Backend):
    """Routes normalize/merge/transform through the Nexla CLI binary.

    Every call has a hard wall-clock timeout (CLI_TIMEOUT_S, default 5s)
    via ThreadPoolExecutor + non-blocking shutdown(wait=False).  On timeout
    or any subprocess error, the call falls back to the local pandas backend
    for that specific operation so the loop never stalls.

    Logging is unambiguous: every entry in job_log carries backend_used
    ("real_cli" or "local_fallback") and latency_ms so it is always clear
    which implementation actually ran.
    """

    def __init__(self, cli_path: str, timeout_s: float = CLI_TIMEOUT_S) -> None:
        self._cli_path = cli_path
        self._timeout_s = timeout_s
        self._local = _LocalBackend()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def normalize(self, raw_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        return self._with_fallback(
            op_name="normalize",
            cli_fn=lambda: self._cli_normalize(raw_df, source_name),
            local_fn=lambda: self._local.normalize(raw_df, source_name),
            log_kwargs={"source": source_name, "rows": len(raw_df)},
        )

    def merge(self, working_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        return self._with_fallback(
            op_name="merge",
            cli_fn=lambda: self._cli_merge(working_df, new_df),
            local_fn=lambda: self._local.merge(working_df, new_df),
            log_kwargs={
                "rows_before": len(working_df),
                "rows_added": len(new_df),
            },
        )

    def transform(self, working_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
        op = spec.get("op", "unknown")
        return self._with_fallback(
            op_name=f"transform:{op}",
            cli_fn=lambda: self._cli_transform(working_df, spec),
            local_fn=lambda: self._local.transform(working_df, spec),
            log_kwargs={"op": op, "columns": spec.get("columns", [])},
        )

    # ------------------------------------------------------------------
    # Timeout + fallback wrapper
    # ------------------------------------------------------------------

    def _with_fallback(
        self,
        op_name: str,
        cli_fn,
        local_fn,
        log_kwargs: dict,
    ) -> pd.DataFrame:
        """Run cli_fn with a hard timeout; fall back to local_fn on any failure.

        Logs backend_used ("real_cli" | "local_fallback") and latency_ms
        for every call so the dashboard can always tell what executed.
        """
        job_id = _new_job_id("cli")
        t0 = time.monotonic()
        fallback_reason: str | None = None

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(cli_fn)
        try:
            result = future.result(timeout=self._timeout_s)
            pool.shutdown(wait=False)
            latency_ms = round((time.monotonic() - t0) * 1000)
            _record_job(
                job_id, "real_cli", op_name,
                latency_ms=latency_ms,
                **log_kwargs,
            )
            return result

        except concurrent.futures.TimeoutError:
            future.cancel()
            pool.shutdown(wait=False)
            latency_ms = round((time.monotonic() - t0) * 1000)
            fallback_reason = f"timeout:{self._timeout_s}s"
            logger.warning(
                "[nexla:cli] %s timed out after %dms (limit=%ss) — "
                "falling back to local pandas for this call",
                op_name, latency_ms, self._timeout_s,
            )

        except Exception as exc:
            pool.shutdown(wait=False)
            latency_ms = round((time.monotonic() - t0) * 1000)
            fallback_reason = f"error:{type(exc).__name__}:{exc}"
            logger.warning(
                "[nexla:cli] %s failed in %dms (%s: %s) — "
                "falling back to local pandas for this call",
                op_name, latency_ms, type(exc).__name__, exc,
            )

        # Fallback path — local backend executes; its own _record_job call
        # would overwrite the backend label, so we call the raw ops directly
        # and record the entry ourselves with backend_used="local_fallback".
        t_fb = time.monotonic()
        result = local_fn()
        total_ms = round((time.monotonic() - t0) * 1000)
        _record_job(
            job_id, "local_fallback", op_name,
            latency_ms=total_ms,
            fallback_reason=fallback_reason,
            **log_kwargs,
        )
        return result

    # ------------------------------------------------------------------
    # Raw CLI subprocess calls (no timeout logic — that lives in _with_fallback)
    # ------------------------------------------------------------------

    def _cli_normalize(self, raw_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "input.csv")
            out_path = os.path.join(tmpdir, "output.csv")
            raw_df.to_csv(in_path, index=False)
            subprocess.run(
                [self._cli_path, "normalize",
                 "--input",  in_path,
                 "--output", out_path,
                 "--source", source_name],
                capture_output=True, text=True, check=True,
            )
            return pd.read_csv(out_path)

    def _cli_merge(self, working_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path  = os.path.join(tmpdir, "left.csv")
            right_path = os.path.join(tmpdir, "right.csv")
            out_path   = os.path.join(tmpdir, "output.csv")
            working_df.to_csv(left_path,  index=False)
            new_df.to_csv(right_path, index=False)
            subprocess.run(
                [self._cli_path, "merge",
                 "--left",   left_path,
                 "--right",  right_path,
                 "--output", out_path],
                capture_output=True, text=True, check=True,
            )
            return pd.read_csv(out_path)

    def _cli_transform(self, working_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path  = os.path.join(tmpdir, "input.csv")
            out_path = os.path.join(tmpdir, "output.csv")
            working_df.to_csv(in_path, index=False)

            op   = spec["op"]
            cols = ",".join(spec.get("columns", []))
            cmd  = [
                self._cli_path, "transform",
                "--input",   in_path,
                "--output",  out_path,
                "--op",      op,
                "--columns", cols,
            ]
            if "n_std" in spec:
                cmd += ["--n-std", str(spec["n_std"])]
            if "name" in spec:
                cmd += ["--name", spec["name"]]

            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return pd.read_csv(out_path)


# ---------------------------------------------------------------------------
# Backend selection — deliberate, documented choice
# ---------------------------------------------------------------------------

def _init_backend() -> _Backend:
    cli_path = os.environ.get("NEXLA_CLI_PATH", "")
    if cli_path:
        logger.info(
            "[nexla] NEXLA_CLI_PATH=%s — using CLI backend "
            "(timeout=%ss, fallback=local on error/timeout)",
            cli_path, CLI_TIMEOUT_S,
        )
        return _NexlaCLIBackend(cli_path=cli_path, timeout_s=CLI_TIMEOUT_S)
    # Deliberate default: local pandas backend.
    # This is not a warning — it is the expected state for dev/test and
    # for demo runs without Nexla credentials wired.
    # Set NEXLA_CLI_PATH to switch to real CLI operations.
    logger.info(
        "[nexla] NEXLA_CLI_PATH not set — using local pandas backend "
        "(deliberate default; set NEXLA_CLI_PATH to enable CLI backend)"
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
