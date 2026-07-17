"""
Diagnosis logic — decides what single thing to change after each evaluation.

This is the module that makes autonomy real. Every verdict must be
traceable to a specific numeric signal in the EvalResult; no branch
should fire based on iteration count or a fixed sequence.

Decision tree (evaluated top-to-bottom; first match wins)
----------------------------------------------------------
1. CONVERGED          accuracy >= target — stop the loop.

2. ENRICH_EXTERNALLY  all four UCI sources pulled AND at least one capable
                      model (not the weak baseline) has been tried — the
                      native dataset is genuinely exhausted; reach for
                      Zero.xyz. Checked before PULL_MORE_DATA because
                      there is literally no native data left to pull.

3. PULL_MORE_DATA     large train/test gap AND small training set AND native
                      sources remain. Evidence: train_acc ≫ test_acc means
                      the model CAN represent this function but doesn't have
                      enough examples to generalize. More records directly
                      address that. Requires both the gap AND size signal
                      so we don't fire on a model that simply can't fit the
                      data (a model switch fixes that, not more data).

4. TRANSFORM_DATA     recall badly skewed toward one class. Evidence: the
   (recall signal)    model consistently misses one class — most dangerously
                      the disease-positive class — which suggests the feature
                      space is distorted. High-magnitude features (chol ~200,
                      trestbps ~130) dominate Euclidean/linear distances while
                      binary flags (sex, fbs, exang) barely contribute.
                      Standardizing gives equal weight to all features.
                      Only fires if "standardize" hasn't been tried yet.

5. TRANSFORM_DATA     single feature absorbs > FEATURE_DOMINANCE_THRESHOLD of
   (dominance signal) total importance in a tree or linear model. That feature
                      is crowding out others; log-scaling reduces its leverage
                      without discarding the information.
                      Only fires if "log_scale" hasn't been tried yet.

6. SWITCH_MODEL       train_acc and test_acc are both low with a small gap —
   (underfitting)     the model cannot represent the decision boundary
                      regardless of data volume. More data or transforms won't
                      fix a model that genuinely underfits.

7. SWITCH_MODEL       catch-all: accuracy is still below target and no other
   (fallback)         specific signal was strong enough to act on.

Verdicts
--------
CONVERGED, PULL_MORE_DATA, TRANSFORM_DATA, SWITCH_MODEL, ENRICH_EXTERNALLY

The scaffold aliases NEED_MORE_DATA / NEED_TRANSFORM / NEED_MODEL_SWITCH
are kept as module-level aliases for backward compatibility with agent.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.models.harness import EvalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict constants  (use these in all comparisons, never raw strings)
# ---------------------------------------------------------------------------

CONVERGED          = "converged"
PULL_MORE_DATA     = "pull_more_data"
TRANSFORM_DATA     = "transform_data"
SWITCH_MODEL       = "switch_model"
ENRICH_EXTERNALLY  = "enrich_externally"

# Backward-compat aliases matching the original scaffold
NEED_MORE_DATA    = PULL_MORE_DATA
NEED_TRANSFORM    = TRANSFORM_DATA
NEED_MODEL_SWITCH = SWITCH_MODEL

# ---------------------------------------------------------------------------
# Thresholds — all in one place so judges can see them at a glance
# ---------------------------------------------------------------------------

# Branch 3 — PULL_MORE_DATA
# A gap above this suggests the model is overfitting on scarce data.
# 0.10 = 10 percentage-point spread between train and test accuracy.
OVERFIT_THRESHOLD = 0.10

# Below this training-set size, even a moderate gap warrants a data pull
# (the model hasn't seen enough examples to properly learn the boundary).
DATA_STARVATION_THRESHOLD = 400

# Branch 4 — TRANSFORM_DATA (recall signal)
# Absolute difference between class-0 and class-1 recall.  0.20 = 20 pp gap.
# Deliberately not set as low as 0.15 to avoid firing on minor imbalances.
RECALL_IMBALANCE_THRESHOLD = 0.20

# Branch 5 — TRANSFORM_DATA (dominance signal)
# Fraction of total normalized feature importance held by a single feature.
# Above this, that feature is crowding out all others.
FEATURE_DOMINANCE_THRESHOLD = 0.35

# Branch 2 — ENRICH_EXTERNALLY
# Total number of UCI hospital sources available natively.
TOTAL_UCI_SOURCES = 4

# ---------------------------------------------------------------------------
# Feature metadata used when building transform specs
# ---------------------------------------------------------------------------

# Numeric (continuous-valued) features that benefit from standardization.
# Excludes binary flags (sex, fbs, exang) — standardizing a 0/1 feature
# is harmless but misleading in the transform log.
_NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisResult:
    verdict: str
    reason: str                            # human-readable, shown on screen during demo
    transform_spec: dict | None = field(default=None)  # only set for TRANSFORM_DATA

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _top_feature(importances: dict) -> tuple[str, float]:
    """Return (name, fraction_of_total) for the most important feature."""
    total = sum(importances.values())
    if total == 0:
        return "", 0.0
    top_name = max(importances, key=importances.__getitem__)
    return top_name, importances[top_name] / total


def _dominant_features(importances: dict) -> list[str]:
    """Features that individually exceed FEATURE_DOMINANCE_THRESHOLD."""
    total = sum(importances.values())
    if total == 0:
        return []
    return [
        col for col, imp in importances.items()
        if imp / total > FEATURE_DOMINANCE_THRESHOLD
    ]

# ---------------------------------------------------------------------------
# Core diagnosis function
# ---------------------------------------------------------------------------

def diagnose(
    result: EvalResult,
    sources_pulled: list[str],
    models_tried: list[str],
    target_accuracy: float,
    transforms_applied: list[str] | None = None,
) -> DiagnosisResult:
    """Decide what single intervention will most likely improve accuracy.

    Parameters
    ----------
    result:              EvalResult from the most recent harness.fit_and_eval().
    sources_pulled:      hospital source names fetched so far (from DataSourceTracker).
    models_tried:        model names attempted so far (including current).
    target_accuracy:     the benchmark threshold — 0.833 to beat the SVM paper result.
    transforms_applied:  list of nexla transform ops already applied this run
                         (e.g. ["standardize", "log_scale"]). Pass [] initially;
                         append after each TRANSFORM_DATA action so we don't cycle.

    Returns
    -------
    DiagnosisResult with verdict, a display-ready reason string, and
    (for TRANSFORM_DATA) the transform spec to pass to nexla_client.transform().
    """
    if transforms_applied is None:
        transforms_applied = []

    gap             = result.overfit_gap       # positive = overfitting
    recall_imbal    = result.recall_imbalance  # |recall_0 - recall_1|
    sources_left    = TOTAL_UCI_SOURCES - len(sources_pulled)
    # "Capable" means anything beyond the deliberately-weak baseline.
    capable_tried   = [m for m in models_tried if m != "knn_weak"]

    # ------------------------------------------------------------------
    # Branch 1: CONVERGED
    # Signal: accuracy has cleared the benchmark target.
    # ------------------------------------------------------------------
    if result.accuracy >= target_accuracy:
        diagnosis = DiagnosisResult(
            verdict=CONVERGED,
            reason=(
                f"Accuracy {result.accuracy:.3f} meets target {target_accuracy:.3f} "
                f"after {len(models_tried)} model(s) and {len(sources_pulled)} "
                f"source(s). Loop complete."
            ),
        )
        _log(diagnosis)
        return diagnosis

    # ------------------------------------------------------------------
    # Branch 2: ENRICH_EXTERNALLY
    # Signal: all four UCI hospital sources have been pulled AND we have
    # already tried at least one capable model — the native dataset is
    # genuinely exhausted so external enrichment is the next logical step.
    #
    # "Capable model tried" guard prevents calling Zero.xyz when the only
    # attempted model is knn_weak; a model switch should come first in
    # that degenerate case (more data won't help a broken model anyway).
    # ------------------------------------------------------------------
    if sources_left == 0 and capable_tried:
        diagnosis = DiagnosisResult(
            verdict=ENRICH_EXTERNALLY,
            reason=(
                f"All {TOTAL_UCI_SOURCES} UCI hospital sources are exhausted "
                f"({result.n_train} training records) and accuracy {result.accuracy:.3f} "
                f"is still {target_accuracy - result.accuracy:.3f} below target "
                f"{target_accuracy:.3f}. Native data is genuinely exhausted; "
                f"reaching out to Zero.xyz marketplace for external enrichment."
            ),
        )
        _log(diagnosis)
        return diagnosis

    # ------------------------------------------------------------------
    # Branch 3: PULL_MORE_DATA
    # Signal: large train/test gap AND small training set AND native
    # sources remain.
    #
    # A big gap means the model CAN fit the data (high train accuracy)
    # but lacks enough examples to generalize (low test accuracy).
    # This is textbook overfitting-from-scarcity: more records directly
    # shrink the gap. We also require n_train < DATA_STARVATION_THRESHOLD
    # so we don't fire this on a 900-record dataset with a modest gap.
    #
    # Deliberately NOT fired on a small gap — a model that can't even
    # fit the training data (knn_weak with k=50) needs a model switch,
    # not more records to underfit on.
    # ------------------------------------------------------------------
    if (
        gap > OVERFIT_THRESHOLD
        and result.n_train < DATA_STARVATION_THRESHOLD
        and sources_left > 0
    ):
        diagnosis = DiagnosisResult(
            verdict=PULL_MORE_DATA,
            reason=(
                f"Train accuracy ({result.train_accuracy:.3f}) >> test accuracy "
                f"({result.accuracy:.3f}) — gap {gap:.3f} exceeds threshold "
                f"{OVERFIT_THRESHOLD} on only {result.n_train} training records. "
                f"Model can represent the function but doesn't have enough examples "
                f"to generalize; pulling the next hospital source."
            ),
        )
        _log(diagnosis)
        return diagnosis

    # ------------------------------------------------------------------
    # Branch 4: TRANSFORM_DATA — recall imbalance signal
    # Signal: one class's recall is much worse than the other.
    #
    # For a heart-disease classifier, the dangerous failure mode is
    # missing disease-positive patients (low class-1 recall).  A large
    # recall gap typically means the feature space is geometrically
    # distorted: high-magnitude features like chol (~200) and trestbps
    # (~130) dominate Euclidean distance and linear margins, making
    # low-magnitude binary flags (sex, fbs, exang) nearly invisible.
    # Standardizing puts all features on the same scale, giving the
    # model a fair view of the full feature set.
    #
    # Guard: only fires if "standardize" hasn't been applied yet — we
    # don't cycle on the same transform if it didn't close the gap.
    # ------------------------------------------------------------------
    if recall_imbal > RECALL_IMBALANCE_THRESHOLD and "standardize" not in transforms_applied:
        worse_cls   = min(result.per_class_recall, key=result.per_class_recall.__getitem__)
        better_cls  = 1 - worse_cls
        worse_rec   = result.per_class_recall[worse_cls]
        better_rec  = result.per_class_recall[better_cls]
        label       = "disease-positive (class 1)" if worse_cls == 1 else "disease-negative (class 0)"

        spec = {"op": "standardize", "columns": _NUMERIC_FEATURES}
        diagnosis = DiagnosisResult(
            verdict=TRANSFORM_DATA,
            reason=(
                f"Recall on {label} is {worse_rec:.3f} vs {better_rec:.3f} for the "
                f"other class — imbalance {recall_imbal:.3f} exceeds threshold "
                f"{RECALL_IMBALANCE_THRESHOLD}. High-magnitude features (chol ~200, "
                f"trestbps ~130) likely dominating the feature space. Requesting "
                f"Nexla standardization transform to equalize feature scales."
            ),
            transform_spec=spec,
        )
        _log(diagnosis)
        return diagnosis

    # ------------------------------------------------------------------
    # Branch 5: TRANSFORM_DATA — feature dominance signal
    # Signal: a single feature absorbs > FEATURE_DOMINANCE_THRESHOLD of
    # total normalized importance in the model (available for tree and
    # linear models; KNN and SVM don't expose importances).
    #
    # When one feature dominates, it crowds out the others — the model
    # is essentially a single-feature classifier. Log-scaling the
    # dominant feature compresses its range without discarding the signal,
    # making room for others to contribute.
    #
    # Guard: only fires if "log_scale" hasn't been applied yet.
    # ------------------------------------------------------------------
    if result.feature_importances and "log_scale" not in transforms_applied:
        dominant = _dominant_features(result.feature_importances)
        if dominant:
            top_name, top_frac = _top_feature(result.feature_importances)
            spec = {"op": "log_scale", "columns": dominant}
            diagnosis = DiagnosisResult(
                verdict=TRANSFORM_DATA,
                reason=(
                    f"Feature '{top_name}' carries {top_frac:.0%} of total model importance "
                    f"— dominating the model and suppressing other predictors. "
                    f"Requesting Nexla log-scale transform on: {dominant} to reduce "
                    f"its leverage and give other features room to contribute."
                ),
                transform_spec=spec,
            )
            _log(diagnosis)
            return diagnosis

    # ------------------------------------------------------------------
    # Branch 6: SWITCH_MODEL — underfitting
    # Signal: both train and test accuracy are low AND the gap is small.
    #
    # A small gap with low accuracy means the model isn't even fitting
    # the training data well — this is underfitting, not overfitting.
    # More data won't help (the model can't use what it already has).
    # Transforms won't help (the problem is model capacity, not data
    # shape). The only fix is a more expressive model class.
    #
    # knn_weak (k=50, unscaled) is the canonical example: accuracy ~62%
    # on train AND test, gap ~0.03. The model's inductive bias (majority
    # vote over 50 neighbors, distance warped by unscaled features) is
    # fundamentally incompatible with this decision boundary.
    # ------------------------------------------------------------------
    if gap < OVERFIT_THRESHOLD:
        diagnosis = DiagnosisResult(
            verdict=SWITCH_MODEL,
            reason=(
                f"Both train ({result.train_accuracy:.3f}) and test ({result.accuracy:.3f}) "
                f"accuracy are low with a small gap ({gap:.3f} < {OVERFIT_THRESHOLD}) — "
                f"the model is underfitting, not overfitting. More data or feature "
                f"transforms won't fix a model that can't represent this decision "
                f"boundary. Switching to a more expressive model class."
            ),
        )
        _log(diagnosis)
        return diagnosis

    # ------------------------------------------------------------------
    # Branch 7: SWITCH_MODEL — catch-all
    # Reaches here when: gap exists but sources are exhausted AND no
    # transform signal fired (or transforms already tried). The remaining
    # lever is a better model.
    # ------------------------------------------------------------------
    diagnosis = DiagnosisResult(
        verdict=SWITCH_MODEL,
        reason=(
            f"Accuracy {result.accuracy:.3f} is {target_accuracy - result.accuracy:.3f} "
            f"below target {target_accuracy:.3f}. No strong data-volume or "
            f"data-shape signal remains (transforms tried: {transforms_applied or 'none'}). "
            f"Switching model class."
        ),
    )
    _log(diagnosis)
    return diagnosis


def _log(d: DiagnosisResult) -> None:
    logger.info("[diagnosis] verdict=%-20s  %s", d.verdict, d.reason)
