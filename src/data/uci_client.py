"""
UCI Heart Disease data client — incremental, live pulls only.

The four hospital sources in pull order (smallest first so the first
iteration is guaranteed to be data-starved):
  1. Cleveland   (~303 records) — always the starting point
  2. Hungary     (~294 records)
  3. Switzerland (~123 records)
  4. Long Beach VA (~200 records)

Public entry points
-------------------
fetch_next_source(already_pulled: list[str]) -> tuple[str, pd.DataFrame] | None
    Query the UCI ML Repository API live for the next hospital source not
    yet pulled. Returns (source_name, raw_df), or None when all four
    sources are exhausted. This is the only function the loop calls
    directly; it decides which source comes next so the loop doesn't
    need to track ordering itself.

    Raises requests.HTTPError on a failed API call so the loop can
    decide whether to retry or abort.

fetch_source(source_name: str) -> pd.DataFrame
    Fetch a single named source from the UCI API. Called by
    fetch_next_source internally; exposed so tests can target a specific
    hospital without going through the ordering logic.

SOURCES : list[str]
    Ordered list of source names in pull order. Treat this as the
    canonical ordering — don't hardcode the sequence anywhere else.

Implementation notes
--------------------
- UCI ML Repository dataset ID for Heart Disease is 45. The API endpoint
  is https://archive.ics.uci.edu/api/datasets/45 (check live docs for
  the exact query parameter to filter by hospital/source).
- Raw column names differ across hospital sources — that's Nexla's job to
  reconcile. Return the raw DataFrame as-is; do not rename columns here.
- The target column in the raw data is 'num' (0 = no disease, 1-4 =
  disease present). Do not binarize here; leave that to the harness.
"""

import requests
import pandas as pd


SOURCES: list[str] = ["cleveland", "hungary", "switzerland", "va"]


def fetch_source(source_name: str) -> pd.DataFrame:
    """Fetch a single hospital source from the UCI API and return raw DataFrame."""
    raise NotImplementedError


def fetch_next_source(already_pulled: list[str]) -> tuple[str, pd.DataFrame] | None:
    """Return (source_name, raw_df) for the next unpulled source, or None if exhausted."""
    raise NotImplementedError
