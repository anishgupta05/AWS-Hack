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

5.5 LLM GRAY AREA    gap is small AND recall is balanced AND no feature
                      dominance signal fired — the threshold signals are
                      inconclusive. The full iteration context (history, sources,
                      models, transforms) is passed to call_llm_provider(), which
                      returns a branch decision and natural-language justification.
                      Falls back silently to Branch 6/7 on timeout or any error.
                      Hard timeout: LLM_TIMEOUT_S (default 10s, env-configurable).
                      Every call — success or failure — is appended to the JSONL
                      event log at DIAGNOSIS_LOG_PATH for the dashboard.

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

import concurrent.futures
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.models.harness import EvalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM fallback config  (all overridable via environment variables)
# ---------------------------------------------------------------------------

# Hard wall-clock timeout for call_llm_provider(). On expiry: log, return None,
# fall back to threshold-based Branch 6/7. Keeps the demo loop from hanging.
LLM_TIMEOUT_S: float = float(os.environ.get("DIAGNOSIS_LLM_TIMEOUT", "10"))

# Path of the newline-delimited JSON event log. Dashboard reads from here.
# Each line is one JSON object: LLM call inputs, outputs, latency, and verdict.
DIAGNOSIS_LOG_PATH: Path = Path(
    os.environ.get("DIAGNOSIS_LOG_PATH", "diagnosis_events.jsonl")
)

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
# Event log
# ---------------------------------------------------------------------------

def _log_diagnosis_event(event: dict) -> None:
    """Append one JSON event to DIAGNOSIS_LOG_PATH (dashboard reads this).

    Never raises — a log write failure must not crash the loop.
    """
    try:
        with DIAGNOSIS_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
    except OSError as exc:
        logger.warning("[diagnosis] event log write failed (%s): %s", DIAGNOSIS_LOG_PATH, exc)


# ---------------------------------------------------------------------------
# Swappable interface functions
# ---------------------------------------------------------------------------

def call_llm_provider(system_prompt: str, task_context: dict) -> dict:
    """Ask the active LLM for a branch decision on a gray-area iteration.

    Current implementation: Anthropic Claude Haiku (low latency).
    The function signature is the stable contract; the body can be
    replaced with a CLI-wrapped multi-provider router without changing
    any call sites or tests.

    Parameters
    ----------
    system_prompt : str
        Role + task description. Module-level constant ``_GRAY_AREA_SYSTEM_PROMPT``.
    task_context : dict
        Serialisable snapshot of the current iteration state (accuracy,
        gap, recall, history, sources, models, transforms, target).

    Returns
    -------
    dict with keys:
        "verdict"       : one of PULL_MORE_DATA / TRANSFORM_DATA /
                          SWITCH_MODEL / ENRICH_EXTERNALLY (uppercase strings)
        "justification" : one-sentence natural-language explanation
        "model"         : model identifier string (for the event log)

    Raises
    ------
    Any exception — caller wraps in a ThreadPoolExecutor with timeout and
    falls back to threshold-based diagnosis on any failure.
    """
    import anthropic  # local import — keeps module importable without the SDK

    client = anthropic.Anthropic()

    user_message = (
        "Current iteration state:\n"
        f"{json.dumps(task_context, indent=2)}\n\n"
        "Respond with valid JSON only (no markdown fences, no prose):\n"
        '{"verdict": "<PULL_MORE_DATA|TRANSFORM_DATA|SWITCH_MODEL|ENRICH_EXTERNALLY>", '
        '"justification": "<one sentence>"}'
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if the model adds them despite instructions.
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    parsed.setdefault("model", "claude-haiku-4-5-20251001")
    return parsed


def get_enrichment_data(context: dict) -> "pd.DataFrame | None":
    """Fetch external enrichment records for an ENRICH_EXTERNALLY verdict.

    Stub — always returns None. Replace this body with a Zero.xyz CLI
    subprocess call when the integration is ready. The loop handles None
    gracefully: it logs the best result achieved and exits cleanly rather
    than assuming enrichment always succeeds.

    Parameters
    ----------
    context : dict
        Serialisable loop state at the moment enrichment is needed.

    Returns
    -------
    pd.DataFrame with canonical UCI columns, or None on failure / stub.
    """
    return None


# ---------------------------------------------------------------------------
# Gray-area system prompt  (module-level — stable across calls, not per-iter)
# ---------------------------------------------------------------------------

_GRAY_AREA_SYSTEM_PROMPT = """\
You are a diagnostic agent embedded in an autonomous ML training loop on the UCI
Heart Disease dataset (Cleveland + Hungary + Switzerland + VA hospital sources).

You are called only for ambiguous iterations: the train/test gap is small
(neither clear overfitting nor clear underfitting), recall is balanced across
classes, and no single feature dominates. The threshold-based rules did not fire.

Your job: pick the single best next action given the full iteration context.

Available actions:
  PULL_MORE_DATA    — pull the next UCI hospital source (only if sources remain)
  TRANSFORM_DATA    — apply a feature transform via Nexla
  SWITCH_MODEL      — try the next model in the pool (LR → RF → SVM → GB)
  ENRICH_EXTERNALLY — all native sources exhausted; query Zero.xyz marketplace

Rules:
- Respond with valid JSON only. No markdown fences. No explanation outside JSON.
- Exactly one field "verdict" (from the list above) and one field "justification"
  (a single sentence explaining the reasoning traceable to the numbers).
- Do not repeat a verdict that appears in the last two history entries unless
  there is a strong numerical reason (e.g., accuracy clearly improved last time).
- Prefer SWITCH_MODEL when multiple models remain untried and the gap is stable.
- Prefer PULL_MORE_DATA only if sources remain AND the gap suggests data scarcity.

{"verdict": "<action>", "justification": "<one sentence>"}
"""


# ---------------------------------------------------------------------------
# Gray-area LLM wrapper  (timeout + fallback + event logging)
# ---------------------------------------------------------------------------

def _try_llm_diagnosis(
    result: EvalResult,
    sources_pulled: list[str],
    models_tried: list[str],
    target_accuracy: float,
    transforms_applied: list[str],
    history: list[tuple[str, float]],
    timeout_s: float = LLM_TIMEOUT_S,
) -> DiagnosisResult | None:
    """Attempt an LLM-based diagnosis for gray-area iterations.

    Wraps ``call_llm_provider`` with:
    - A hard wall-clock timeout (``timeout_s``, default ``LLM_TIMEOUT_S``).
    - Full input/output logging to ``DIAGNOSIS_LOG_PATH`` on every call.
    - Silent fallback: returns ``None`` on timeout or any error so the
      caller can fall through to threshold Branches 6/7.

    Returns
    -------
    DiagnosisResult on success, None on timeout / parse failure / API error.
    """
    task_context: dict = {
        "accuracy": round(result.accuracy, 4),
        "train_accuracy": round(result.train_accuracy, 4),
        "overfit_gap": round(result.overfit_gap, 4),
        "f1": round(result.f1, 4),
        "per_class_recall": {
            str(k): round(v, 4) for k, v in result.per_class_recall.items()
        },
        "n_train": result.n_train,
        "target_accuracy": target_accuracy,
        "gap_to_target": round(target_accuracy - result.accuracy, 4),
        "sources_pulled": sources_pulled,
        "sources_remaining": TOTAL_UCI_SOURCES - len(sources_pulled),
        "models_tried": models_tried,
        "transforms_applied": transforms_applied,
        # Last 5 iterations for trend context; oldest first.
        "history": [
            {"verdict": v, "accuracy": round(a, 4)} for v, a in history[-5:]
        ],
    }

    t0 = time.monotonic()
    llm_output: dict | None = None
    event_type = "llm_diagnosis"
    error_msg: str | None = None

    try:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(call_llm_provider, _GRAY_AREA_SYSTEM_PROMPT, task_context)
        try:
            llm_output = future.result(timeout=timeout_s)
            pool.shutdown(wait=False)
        except concurrent.futures.TimeoutError:
            future.cancel()
            pool.shutdown(wait=False)  # don't block on the in-flight HTTP request
            raise TimeoutError(
                f"call_llm_provider exceeded {timeout_s}s hard timeout"
            )

        raw_verdict = llm_output.get("verdict", "").strip().upper()
        justification = llm_output.get("justification", "LLM diagnosis.")
        model_id = llm_output.get("model", "unknown")

        _VERDICT_MAP = {
            "PULL_MORE_DATA":    PULL_MORE_DATA,
            "TRANSFORM_DATA":    TRANSFORM_DATA,
            "SWITCH_MODEL":      SWITCH_MODEL,
            "ENRICH_EXTERNALLY": ENRICH_EXTERNALLY,
        }
        verdict = _VERDICT_MAP.get(raw_verdict)
        if verdict is None:
            raise ValueError(
                f"LLM returned unrecognised verdict {raw_verdict!r}; "
                f"expected one of {list(_VERDICT_MAP)}"
            )

        latency_ms = round((time.monotonic() - t0) * 1000)

        _log_diagnosis_event({
            "event": event_type,
            "input": task_context,
            "output": llm_output,
            "verdict": verdict,
            "model": model_id,
            "latency_ms": latency_ms,
            "verdict_used": f"llm:{model_id}",
        })

        logger.info(
            "[diagnosis:llm] verdict=%-20s model=%s latency=%dms  %s",
            verdict, model_id, latency_ms, justification,
        )

        return DiagnosisResult(
            verdict=verdict,
            reason=f"[LLM/{model_id} +{latency_ms}ms] {justification}",
        )

    except Exception as exc:
        latency_ms = round((time.monotonic() - t0) * 1000)
        error_msg = str(exc)
        event_type = "llm_diagnosis_failed"

        _log_diagnosis_event({
            "event": event_type,
            "input": task_context,
            "output": llm_output,
            "error": error_msg,
            "latency_ms": latency_ms,
            "verdict_used": "threshold_fallback",
        })

        logger.warning(
            "[diagnosis:llm] FAILED in %.0fms — falling back to threshold logic. %s: %s",
            latency_ms,
            type(exc).__name__,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Core diagnosis function
# ---------------------------------------------------------------------------

def diagnose(
    result: EvalResult,
    sources_pulled: list[str],
    models_tried: list[str],
    target_accuracy: float,
    transforms_applied: list[str] | None = None,
    history: list[tuple[str, float]] | None = None,
    n_capable_models: int = 4,
) -> DiagnosisResult:
    """Decide what single intervention will most likely improve accuracy.

    Parameters
    ----------
    result:              EvalResult from the most recent harness.fit_and_eval().
    sources_pulled:      hospital source names fetched so far (from DataSourceTracker).
    models_tried:        model names attempted so far (including current).
    target_accuracy:     the benchmark threshold.
    transforms_applied:  list of nexla transform ops already applied this run
                         (e.g. ["standardize", "log_scale"]). Pass [] initially;
                         append after each TRANSFORM_DATA action so we don't cycle.
    history:             list of (verdict, accuracy) from prior iterations, oldest
                         first. Used to detect stalled actions and escalate.
    n_capable_models:    total count of non-weak-baseline models in the pool.
                         Used to detect when all capable models have been tried.

    Returns
    -------
    DiagnosisResult with verdict, a display-ready reason string, and
    (for TRANSFORM_DATA) the transform spec to pass to nexla_client.transform().
    """
    if transforms_applied is None:
        transforms_applied = []
    if history is None:
        history = []

    gap             = result.overfit_gap       # positive = overfitting
    recall_imbal    = result.recall_imbalance  # |recall_0 - recall_1|
    sources_left    = TOTAL_UCI_SOURCES - len(sources_pulled)
    # "Capable" means anything beyond the deliberately-weak baseline.
    capable_tried   = [m for m in models_tried if m != "knn_weak"]
    # True once every capable model in the pool has been evaluated.
    all_capable_exhausted = len(capable_tried) >= n_capable_models

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
    #
    # Fires under two distinct conditions (first match wins):
    #
    # 3a. Starvation/overfitting: large train/test gap AND small training
    #     set AND sources remain. A big gap on few records means the model
    #     can fit but can't generalize — more records directly fix that.
    #     Anti-repetition: if the last action was also PULL_MORE_DATA and
    #     accuracy barely moved (< 1pp), skip — the pull didn't help so
    #     escalate to transforms/model switch instead.
    #
    # 3b. Model exhaustion: all capable models evaluated on current data
    #     and still below target, but native sources remain. More data is
    #     the remaining lever before going external (Zero.xyz). No
    #     anti-repetition here — each new source adds genuinely new signal.
    # ------------------------------------------------------------------
    starvation_triggered = (
        gap > OVERFIT_THRESHOLD
        and result.n_train < DATA_STARVATION_THRESHOLD
    )
    last_pull_stalled = (
        starvation_triggered          # guard only applies to starvation case
        and len(history) >= 1
        and history[-1][0] == PULL_MORE_DATA
        and (result.accuracy - history[-1][1]) < 0.01
    )

    if sources_left > 0 and not last_pull_stalled and (starvation_triggered or all_capable_exhausted):
        if starvation_triggered:
            reason = (
                f"Train accuracy ({result.train_accuracy:.3f}) >> test accuracy "
                f"({result.accuracy:.3f}) — gap {gap:.3f} exceeds threshold "
                f"{OVERFIT_THRESHOLD} on only {result.n_train} training records. "
                f"Model can represent the function but doesn't have enough examples "
                f"to generalize; pulling the next hospital source."
            )
        else:
            reason = (
                f"All {len(capable_tried)} capable models tried on {result.n_train} "
                f"training records; best accuracy {result.accuracy:.3f} is still "
                f"{target_accuracy - result.accuracy:.3f} below target {target_accuracy:.3f}. "
                f"Native data sources remain — pulling the next hospital source "
                f"before falling back to external enrichment."
            )
        diagnosis = DiagnosisResult(verdict=PULL_MORE_DATA, reason=reason)
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
    # Branch 5.5: LLM GRAY AREA
    # Signal: gap is small AND recall is balanced — no threshold rule
    # fired cleanly. The situation is genuinely ambiguous: we're below
    # target but neither clearly overfitting, underfitting, data-starved,
    # nor feature-distorted.
    #
    # Pass the full iteration context to call_llm_provider(). If it
    # returns a verdict within LLM_TIMEOUT_S, use it. Otherwise fall
    # through to Branch 6/7 exactly as before — the loop never stalls.
    #
    # "Meaningful gap to target" guard (> 2pp) prevents firing this on
    # cases that are within noise of the benchmark, where Branch 6/7 is
    # perfectly adequate.
    # ------------------------------------------------------------------
    gray_area = (
        gap < OVERFIT_THRESHOLD
        and recall_imbal < RECALL_IMBALANCE_THRESHOLD
        and (target_accuracy - result.accuracy) > 0.02
    )
    if gray_area:
        llm_diag = _try_llm_diagnosis(
            result=result,
            sources_pulled=sources_pulled,
            models_tried=models_tried,
            target_accuracy=target_accuracy,
            transforms_applied=transforms_applied,
            history=history,
        )
        if llm_diag is not None:
            _log(llm_diag)
            return llm_diag
        # LLM unavailable or timed out — fall through to Branch 6/7.

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
