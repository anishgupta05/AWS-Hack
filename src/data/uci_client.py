"""
UCI Heart Disease data client — incremental, live pulls only.

The four hospital sources in pull order (smallest first so the first
iteration is guaranteed to be data-starved):
  1. Cleveland   (~303 records) — always the starting point
  2. Hungary     (~294 records)
  3. Switzerland (~123 records)
  4. Long Beach VA (~200 records)

Each raw file is comma-separated, 14 columns, "?" for missing values.
Column 14 ('target') contains raw UCI 'num' values 0–4; binarization
(1-4 → 1) is the harness's responsibility, not ours.

Public entry points
-------------------
fetch_source(source_name: str) -> pd.DataFrame
    Download and parse one named source. Returns a clean DataFrame with
    standardised column names, "?" replaced with NaN, and basic median
    imputation applied to feature columns. A 'source' column is added so
    the merged working dataset retains provenance.

fetch_next_source(already_pulled: list[str]) -> tuple[str, pd.DataFrame] | None
    Convenience wrapper: returns (name, df) for the first source in
    SOURCES not yet in already_pulled, or None when all are exhausted.
    Thin wrapper around fetch_source — all real logic lives there.

DataSourceTracker
    Stateful helper that owns pull-ordering and remembers what has been
    fetched this run. The loop uses this rather than managing the list
    itself.

SOURCES : list[str]
    Canonical pull order. Do not hardcode this sequence anywhere else.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Canonical pull order — smallest source first so iteration 1 is always
# data-starved, guaranteeing the correction loop fires in the demo.
SOURCES: list[str] = ["cleveland", "hungary", "switzerland", "va"]

COLUMNS: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target",
]

_BASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
)
_FILE_NAMES: dict[str, str] = {
    "cleveland":   "processed.cleveland.data",
    "hungary":     "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va":          "processed.va.data",
}

# Feature columns eligible for median imputation (all except the target).
_FEATURE_COLUMNS: list[str] = [c for c in COLUMNS if c != "target"]


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def fetch_source(source_name: str) -> pd.DataFrame:
    """Download one hospital source and return a clean, imputed DataFrame.

    Steps:
      1. HTTP GET the raw .data file from UCI archive.
      2. Parse as CSV with standardised column names; "?" → NaN.
      3. Tag each row with a 'source' column for provenance.
      4. Median-impute missing values in every feature column.
      5. Drop rows whose *target* is missing (can't supervise them).

    Raises
    ------
    ValueError          if source_name is not in SOURCES.
    requests.HTTPError  on a non-2xx response from UCI.
    """
    if source_name not in _FILE_NAMES:
        raise ValueError(
            f"Unknown source {source_name!r}. Valid choices: {SOURCES}"
        )

    url = _BASE_URL + _FILE_NAMES[source_name]
    logger.info("[uci] pulling %s from %s", source_name, url)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(
        io.StringIO(response.text),
        header=None,
        names=COLUMNS,
        na_values="?",
    )

    # Provenance tag — preserved through Nexla merge so the working
    # dataset always shows which hospital each row came from.
    df["source"] = source_name

    # Median imputation on feature columns only.
    # 'ca' and 'thal' are the most frequently missing across hospital
    # sources; median is conservative and fast enough for a demo.
    for col in _FEATURE_COLUMNS:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(
                "[uci] %s.%s: imputed %d missing values with median %.3f",
                source_name, col, n_missing, median_val,
            )

    # Drop rows with a missing target — we cannot supervise them and
    # they should be rare in the processed files.
    before = len(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    n_dropped = before - len(df)
    if n_dropped:
        logger.warning(
            "[uci] %s: dropped %d row(s) with missing target",
            source_name, n_dropped,
        )

    logger.info(
        "[uci] %s: %d records ready (%d columns including source)",
        source_name, len(df), len(df.columns),
    )
    return df


# ---------------------------------------------------------------------------
# Stateless convenience wrapper (used by agent when it holds its own list)
# ---------------------------------------------------------------------------

def fetch_next_source(
    already_pulled: list[str],
) -> tuple[str, pd.DataFrame] | None:
    """Return (source_name, df) for the next unpulled source, or None if exhausted.

    Iterates SOURCES in canonical order and returns the first name not in
    already_pulled. The caller is responsible for updating already_pulled
    after use. Prefer DataSourceTracker for stateful usage.
    """
    for source in SOURCES:
        if source not in already_pulled:
            logger.info("[uci] next source: %s (already pulled: %s)", source, already_pulled)
            return source, fetch_source(source)
    logger.info("[uci] all sources exhausted")
    return None


# ---------------------------------------------------------------------------
# Stateful tracker (used by the loop agent)
# ---------------------------------------------------------------------------

class DataSourceTracker:
    """Tracks which UCI hospital sources have been pulled this run.

    Owns pull ordering so the agent loop doesn't need to manage it.
    Starts with no sources pulled; call pull_next() to fetch sequentially.

    Example
    -------
    >>> tracker = DataSourceTracker()
    >>> name, df = tracker.pull_next()   # Cleveland
    >>> tracker.pulled                   # ['cleveland']
    >>> tracker.remaining()              # ['hungary', 'switzerland', 'va']
    >>> tracker.exhausted                # False
    """

    def __init__(self) -> None:
        self._pulled: list[str] = []

    @property
    def pulled(self) -> list[str]:
        """Sources pulled so far, in pull order."""
        return list(self._pulled)

    def remaining(self) -> list[str]:
        """Sources not yet pulled, in canonical SOURCES order."""
        return [s for s in SOURCES if s not in self._pulled]

    @property
    def exhausted(self) -> bool:
        """True when all four UCI sources have been fetched."""
        return len(self._pulled) == len(SOURCES)

    def pull_next(self) -> tuple[str, pd.DataFrame]:
        """Fetch the next available source and record it as pulled.

        Returns (source_name, df).

        Raises
        ------
        RuntimeError        if all sources are already exhausted.
        requests.HTTPError  on a network failure from UCI.
        """
        if self.exhausted:
            raise RuntimeError(
                "All UCI sources exhausted. "
                "Use Zero.xyz enrichment for additional data."
            )

        next_source = self.remaining()[0]
        df = fetch_source(next_source)
        self._pulled.append(next_source)
        return next_source, df

    def __repr__(self) -> str:
        return (
            f"DataSourceTracker("
            f"pulled={self._pulled}, "
            f"remaining={self.remaining()}, "
            f"exhausted={self.exhausted})"
        )
