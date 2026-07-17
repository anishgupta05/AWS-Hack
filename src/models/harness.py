"""
Training and evaluation harness.

Responsible for fitting a model, evaluating it on the held-out test set,
and returning a structured result the diagnosis module can reason about.
Also owns the train/test split so the split is fixed once and reused
across all iterations (the test set never changes mid-run).

Public entry points
-------------------
EvalResult : dataclass
    Structured evaluation output. Fields:
        accuracy: float
        f1: float                    # macro-averaged
        confusion_matrix: np.ndarray # shape (2, 2)
        per_class_recall: dict       # {0: float, 1: float}
        train_accuracy: float        # gap vs accuracy reveals overfitting
        n_train: int                 # records seen at training time
        n_test: int
        feature_importances: dict | None  # {feature_name: float}, None if unavailable

split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42)
        -> tuple[pd.DataFrame, pd.DataFrame]
    Perform a stratified train/test split on the full working dataset.
    Always call this once at the start of a run, then hold the test set
    fixed. Do not re-split between iterations — the test set must stay
    constant so accuracy comparisons across iterations are valid.

train(model, X_train: pd.DataFrame, y_train: pd.Series) -> fitted model
    Fit the model and return it. No preprocessing here — models that need
    scaling should have a Pipeline wrapper (see registry.py).

evaluate(model, X_train, y_train, X_test, y_test,
         feature_names: list[str]) -> EvalResult
    Evaluate a fitted model and return an EvalResult. Computes all fields
    needed by the diagnosis module in one pass. Also computes train
    accuracy so the diagnosis can detect the train/test gap.

Implementation notes
-------------------
- Target column name is 'target' (0 = no disease, 1 = disease present).
  The harness owns binarization: raw UCI 'num' values 1-4 all map to 1.
- feature_importances should be extracted from model.feature_importances_
  for tree-based models, from abs(model.coef_) for linear models, and
  set to None for KNN/SVM (these don't expose it cleanly).
- When the model is a Pipeline, unwrap the final step to get importances.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    accuracy: float
    f1: float
    confusion_matrix: np.ndarray
    per_class_recall: dict
    train_accuracy: float
    n_train: int
    n_test: int
    feature_importances: dict | None = field(default=None)


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) via stratified split. Call once; hold test set fixed."""
    raise NotImplementedError


def train(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit model on training data and return it."""
    raise NotImplementedError


def evaluate(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> EvalResult:
    """Evaluate a fitted model and return a fully populated EvalResult."""
    raise NotImplementedError
