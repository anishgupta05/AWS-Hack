"""
Nexla integration — schema normalization, feature transformation, and
incremental dataset merging.

Nexla sits between every UCI API pull and the training pipeline. It does
real work at two distinct points in the loop (which is what makes the
integration non-decorative):

  1. On every ingest: normalize the raw hospital-source DataFrame to a
     canonical schema so the training harness always sees the same columns
     regardless of how UCI encoded that hospital's data.

  2. On correction (option a): when the diagnosis is NEED_TRANSFORM,
     execute a feature-level transformation job — e.g. adding interaction
     terms, log-scaling skewed columns, encoding categoricals differently —
     as a Nexla job rather than an inline pandas call.

Public entry points
-------------------
normalize(raw_df: pd.DataFrame, source_name: str) -> pd.DataFrame
    Submit raw_df to Nexla as a normalization job and return the
    canonical DataFrame. The canonical schema is:
        age, sex, cp, trestbps, chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca, thal, target (0/1 binary)
    Missing columns from the raw source should come back as NaN so the
    harness can decide on imputation strategy.

merge(working_df: pd.DataFrame, new_normalized_df: pd.DataFrame) -> pd.DataFrame
    Ask Nexla to merge a newly normalized source into the existing working
    dataset. Returns the merged DataFrame. Keeps this as a Nexla job
    rather than a pd.concat so the merge is visible in Nexla's audit log
    (important for the demo: shows Nexla doing recurring work across
    multiple iterations, not just once at startup).

transform(working_df: pd.DataFrame, spec: dict) -> pd.DataFrame
    Execute a feature-transformation job on the working dataset. `spec`
    is a dict describing the transformation, e.g.:
        {"op": "log_scale", "columns": ["chol", "trestbps"]}
        {"op": "add_interaction", "columns": ["age", "thalach"]}
        {"op": "drop_high_nan", "threshold": 0.4}
    The diagnosis module produces these specs; this function submits them
    to Nexla and returns the transformed DataFrame.

Implementation notes
-------------------
- Nexla API credentials should come from environment variables
  NEXLA_API_KEY and NEXLA_WORKSPACE_ID — do not hardcode.
- All three functions should log the Nexla job ID they receive so Person B
  can surface these IDs in the demo dashboard.
- If the Nexla API is unavailable (e.g. running offline for dev), fall
  through to a local pandas equivalent so the core loop can still be
  developed and tested without a live Nexla connection. Log a warning
  when the fallback fires — it must never be silent.
"""

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target",
]


def normalize(raw_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize a raw hospital-source DataFrame to the canonical schema via Nexla."""
    raise NotImplementedError


def merge(working_df: pd.DataFrame, new_normalized_df: pd.DataFrame) -> pd.DataFrame:
    """Merge a newly normalized source into the working dataset via a Nexla job."""
    raise NotImplementedError


def transform(working_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Execute a feature-transformation spec on the working dataset via Nexla."""
    raise NotImplementedError
