"""
Training and evaluation harness.

Owns the train/test split, model fitting, and structured evaluation.
The test set is created once from the initial Cleveland pull and held
fixed for the entire run — as additional sources are merged into the
working dataset they go into the training portion only, so accuracy
numbers are comparable across iterations.

The harness also owns target binarization: raw UCI 'target' values
1–4 (disease present at varying severity) all map to 1. This happens
inside split() so every downstream function always sees a binary label.

Public entry points
-------------------
EvalResult          dataclass with all fields the diagnosis module needs
split(df)           stratified 80/20 split, binarizes target, call once
train(model, ...)   fit and return the model
evaluate(...)       compute all EvalResult fields in one pass
fit_and_eval(...)   convenience: train + evaluate from DataFrames
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Columns that are never used as features.
_NON_FEATURE_COLS = {"target", "source"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """All evaluation outputs needed by the diagnosis and state modules."""
    accuracy: float
    f1: float                        # macro-averaged over both classes
    confusion_matrix: np.ndarray     # shape (2, 2): [[TN, FP], [FN, TP]]
    per_class_recall: dict           # {0: recall_no_disease, 1: recall_disease}
    train_accuracy: float            # train/test gap = train_accuracy - accuracy
    n_train: int
    n_test: int
    feature_importances: dict | None = field(default=None)  # {col: importance}

    @property
    def overfit_gap(self) -> float:
        """Train accuracy minus test accuracy. Positive = overfitting."""
        return self.train_accuracy - self.accuracy

    @property
    def recall_imbalance(self) -> float:
        """Absolute difference between per-class recalls."""
        recalls = list(self.per_class_recall.values())
        return abs(recalls[0] - recalls[1]) if len(recalls) == 2 else 0.0

    def log_summary(self, model_name: str, sources: list[str]) -> None:
        """Emit a single human-readable iteration log line."""
        logger.info(
            "[eval] model=%-20s sources=%s  n_train=%d  "
            "acc=%.3f  train_acc=%.3f  gap=%.3f  f1=%.3f  "
            "recall={0:%.2f, 1:%.2f}",
            model_name,
            sources,
            self.n_train,
            self.accuracy,
            self.train_accuracy,
            self.overfit_gap,
            self.f1,
            self.per_class_recall.get(0, 0.0),
            self.per_class_recall.get(1, 0.0),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return column names to use as features (everything except target/source)."""
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def _extract_importances(model, feature_names: list[str]) -> dict | None:
    """Pull feature importances from the final estimator step.

    Returns a {feature: importance} dict for tree/linear models,
    None for KNN and SVM (which don't expose interpretable importances).
    """
    # Unwrap Pipeline to get the actual estimator.
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model

    if hasattr(estimator, "feature_importances_"):
        return dict(zip(feature_names, estimator.feature_importances_))

    if hasattr(estimator, "coef_"):
        # coef_ is shape (1, n_features) for binary LR.
        importances = np.abs(estimator.coef_[0])
        return dict(zip(feature_names, importances))

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split with target binarization.

    Binarizes 'target' in place (values 1–4 → 1) before splitting.
    Call this exactly once at run start on the initial Cleveland pull;
    hold test_df constant for all subsequent iterations.

    Returns (train_df, test_df), both with reset indices.
    """
    df = df.copy()
    df["target"] = (df["target"] > 0).astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"],
    )
    logger.info(
        "[harness] split: %d train / %d test  "
        "(disease prevalence train=%.1f%%  test=%.1f%%)",
        len(train_df),
        len(test_df),
        100 * train_df["target"].mean(),
        100 * test_df["target"].mean(),
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit the model and return it. Pipelines handle their own scaling."""
    model.fit(X_train, y_train)
    return model


def evaluate(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> EvalResult:
    """Evaluate a fitted model and return a fully populated EvalResult.

    Computes both train and test metrics so the diagnosis module can
    detect the train/test gap without a second call.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_test_pred))
    train_acc = float(accuracy_score(y_train, y_train_pred))
    f1 = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_test, y_test_pred)

    # recall_score(average=None) returns one value per class in label order.
    recalls = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    per_class_recall = {int(i): float(r) for i, r in enumerate(recalls)}

    fi = _extract_importances(model, feature_names)

    return EvalResult(
        accuracy=acc,
        f1=f1,
        confusion_matrix=cm,
        per_class_recall=per_class_recall,
        train_accuracy=train_acc,
        n_train=len(y_train),
        n_test=len(y_test),
        feature_importances=fi,
    )


def fit_and_eval(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[object, EvalResult]:
    """Train a model and evaluate it in one call.

    Extracts features and target from both DataFrames (target must already
    be binarized — call split() first). Returns (fitted_model, EvalResult).

    This is the function the agent loop calls each iteration:
        model = registry.get_model(name)
        fitted, result = harness.fit_and_eval(model, train_df, test_df)
    """
    feat_cols = _feature_cols(train_df)

    X_train = train_df[feat_cols]
    y_train = train_df["target"]
    X_test = test_df[feat_cols]
    y_test = test_df["target"]

    fitted = train(model, X_train, y_train)
    result = evaluate(fitted, X_train, y_train, X_test, y_test, feat_cols)
    return fitted, result
