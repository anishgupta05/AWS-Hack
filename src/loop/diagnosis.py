"""
Diagnosis logic — the most important module technically.

Given an EvalResult and the current loop state, decide what single thing
to change next. This is what makes "Autonomy" and "Technical
Implementation" scores real rather than theater — the decision must be
driven by the actual evaluation signal, not a fixed sequence.

The four possible diagnoses
---------------------------
CONVERGED
    Accuracy >= target. Stop the loop.

NEED_MORE_DATA
    Evidence: large train/test gap (overfitting on too little data) AND
    more UCI hospital sources are still available. Specifically:
      - train_accuracy - accuracy > OVERFIT_THRESHOLD (e.g. 0.10)
      - n_train < DATA_STARVATION_THRESHOLD (e.g. 400 records)
    Action the loop takes: call uci_client.fetch_next_source() and
    merge via Nexla.

NEED_TRANSFORM
    Evidence: per-class recall is highly imbalanced (one class recall
    >> the other) OR feature importances are dominated by a single
    feature (suggesting others are on wrong scales or need engineering).
    This pattern suggests the data shape is wrong, not the volume.
    Action the loop takes: call nexla_client.transform() with a spec
    derived from the specific signal (e.g. log-scale chol if chol
    dominates importance, or reweight if recall imbalance).

NEED_MODEL_SWITCH
    Evidence: train_accuracy and test_accuracy are both low and close
    (small gap) — the model genuinely can't fit this data, not just
    overfitting. Or: all UCI sources exhausted and still underperforming
    after trying transforms.
    Action the loop takes: call registry.next_model() and retrain.

    Note: NEED_MODEL_SWITCH is also the fallback when neither
    NEED_MORE_DATA nor NEED_TRANSFORM applies — but it should be reached
    by eliminating the other hypotheses, not as the default.

Public entry points
-------------------
DiagnosisResult : dataclass
    Fields:
        verdict: str          # one of the four constants above
        reason: str           # human-readable sentence for the demo log
        transform_spec: dict | None  # only set when verdict == NEED_TRANSFORM

diagnose(
    result: EvalResult,
    sources_pulled: list[str],
    models_tried: list[str],
    target_accuracy: float,
) -> DiagnosisResult
    Core function. Takes the evaluation result plus loop context and
    returns a DiagnosisResult. All logic lives here.

Constants
---------
CONVERGED = "converged"
NEED_MORE_DATA = "need_more_data"
NEED_TRANSFORM = "need_transform"
NEED_MODEL_SWITCH = "need_model_switch"

OVERFIT_THRESHOLD = 0.10        # train/test gap that signals data starvation
DATA_STARVATION_THRESHOLD = 400 # n_train below which gap → more data, not model switch
RECALL_IMBALANCE_THRESHOLD = 0.15  # per-class recall diff that signals transform

Implementation notes
-------------------
- reason must be a complete, human-readable sentence. This string is
  printed to the console during the demo to prove the decision is
  reasoned, not scripted. Examples:
    "Train accuracy (0.91) >> test accuracy (0.64) on only 303 records
     — dataset too small to generalize; pulling next hospital source."
    "Recall on disease-positive class (0.51) far below negative class
     (0.82) — class imbalance or feature scale issue; requesting
     Nexla transform."
- Do NOT hardcode "always try more data before model switch". The
  NEED_MORE_DATA diagnosis must require both the gap signal AND
  available sources — if sources are exhausted it cannot fire, even if
  the gap exists.
"""

from dataclasses import dataclass, field
from src.models.harness import EvalResult


CONVERGED = "converged"
NEED_MORE_DATA = "need_more_data"
NEED_TRANSFORM = "need_transform"
NEED_MODEL_SWITCH = "need_model_switch"

OVERFIT_THRESHOLD = 0.10
DATA_STARVATION_THRESHOLD = 400
RECALL_IMBALANCE_THRESHOLD = 0.15


@dataclass
class DiagnosisResult:
    verdict: str
    reason: str
    transform_spec: dict | None = field(default=None)


def diagnose(
    result: EvalResult,
    sources_pulled: list[str],
    models_tried: list[str],
    target_accuracy: float,
) -> DiagnosisResult:
    """Diagnose why the model is underperforming and return what to change next."""
    raise NotImplementedError
