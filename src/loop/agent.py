"""
Main loop orchestrator — PLAN → ACT → OBSERVE → CORRECT.

This is the entry point for Person A's half. It wires together the UCI
client, Nexla client, model harness, and diagnosis module into the
closed correction loop described in CLAUDE.md.

It also exposes two integration seams for Person B:
  - on_action hook: called before every autonomous action with a real-world
    side effect; Person B wires the Pomerium gate here.
  - zero_enrichment_hook: called when all UCI sources are exhausted and
    the agent is still underperforming; Person B wires the Zero.xyz
    marketplace discovery + enrichment pull here.

Public entry points
-------------------
run(
    target_accuracy: float = 0.87,
    max_iterations: int = 12,
    on_action: Callable[[str, dict], None] | None = None,
    zero_enrichment_hook: Callable[[LoopState], pd.DataFrame | None] | None = None,
) -> LoopState
    Execute the full correction loop and return the final LoopState.

    Loop sequence per iteration:
      1. PLAN: decide next model (from registry) and whether to pull data.
      2. ACT: if new data needed, fetch via uci_client → normalize+merge
              via nexla_client → split train/test (test held fixed after
              first iteration). Then train the current model.
      3. OBSERVE: evaluate via harness, print iteration log.
      4. CORRECT: call diagnose(). Based on verdict:
            CONVERGED         → break
            PULL_MORE_DATA    → fetch next UCI source (fires on_action)
            TRANSFORM_DATA    → request Nexla transform (fires on_action)
            SWITCH_MODEL      → swap model (fires on_action)
            ENRICH_EXTERNALLY → call zero_enrichment_hook; if data returned,
                                merge it and continue loop for one more iter;
                                if None or no hook, stop.
      5. Append IterationRecord to state.

    Stops early if target_accuracy is reached or max_iterations hit.
    Prints a human-readable log line at each step (not just numbers) so
    the demo can run with the terminal visible.

on_action hook signature
    on_action(action_name: str, payload: dict) -> None
    Called synchronously before the action executes. action_name is one of:
        "fetch_uci_source"    payload: {"source": str}
        "nexla_transform"     payload: {"spec": dict}
        "switch_model"        payload: {"from": str, "to": str}
        "zero_enrichment"     payload: {"reason": str}
    Person B uses this to route through Pomerium's policy check.
    If not provided, actions proceed without gating (dev/test mode).

zero_enrichment_hook signature
    zero_enrichment_hook(state: LoopState) -> pd.DataFrame | None
    Called when all UCI sources are exhausted and accuracy is still below
    target. Person B implements this: searches Zero.xyz marketplace live,
    selects a service, pulls enrichment data, and returns it as a
    normalized DataFrame to merge into the working dataset, or None if
    enrichment failed / was declined by Pomerium.

Implementation notes
-------------------
- The frozen multi-site test set is built ONCE at startup by sampling
  proportionally from ALL FOUR UCI sources. This means the test set is
  representative of the full multi-site distribution from the start,
  so accuracy is comparable across iterations regardless of which
  training sources have been revealed.
- Training data is revealed incrementally: the loop starts with only
  Cleveland's reserve slice and unlocks additional source slices as the
  diagnosis calls for more data.
- Transforms are re-applied from raw each iteration (stateless pipeline).
- Cap at max_iterations regardless of accuracy — do not let the loop run
  indefinitely during a live demo.
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

import numpy as np

from src.data import nexla_client, uci_client
from src.loop.diagnosis import (
    CONVERGED,
    ENRICH_EXTERNALLY,
    PULL_MORE_DATA,
    SWITCH_MODEL,
    TRANSFORM_DATA,
    diagnose,
)
from src.loop.state import IterationRecord, LoopState
from src.models import harness, registry

logger = logging.getLogger(__name__)

# Models whose sklearn Pipeline already contains a StandardScaler.
# Applying a Nexla "standardize" to the raw DataFrame before feeding these
# would cause train/test distribution mismatch (test is kept raw and the
# Pipeline's internal scaler handles test normalization during predict).
_PIPELINE_SCALED_MODELS = frozenset({"logistic_regression", "svm_rbf"})


def _apply_transforms(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    specs: list[dict],
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply transform specs to train and test consistently.

    Rules:
    - "standardize": skip for Pipeline-scaled models (they handle it
      internally via StandardScaler); otherwise fit stats on train and
      apply the same stats to test to avoid distribution mismatch.
    - "log_scale": parameter-free; apply to both train and test.
    - Other ops: applied to train only (no scale-sensitive test impact).
    """
    train = raw_train.copy()
    test = raw_test.copy()

    for spec in specs:
        op = spec.get("op")

        if op == "standardize":
            if model_name in _PIPELINE_SCALED_MODELS:
                continue  # Pipeline's internal scaler handles this
            cols = [c for c in spec["columns"] if c in train.columns]
            for col in cols:
                mean = train[col].mean()
                std = train[col].std()
                if std > 0:
                    train[col] = (train[col] - mean) / std
                    test[col] = (test[col] - mean) / std

        elif op == "log_scale":
            cols = [c for c in spec["columns"] if c in train.columns]
            for col in cols:
                train[col] = np.log1p(train[col].clip(lower=0))
                test[col] = np.log1p(test[col].clip(lower=0))

        else:
            train = nexla_client.transform(train, spec)

    return train, test


def _p(msg: str) -> None:
    print(msg, flush=True)


def run(
    target_accuracy: float = 0.87,
    max_iterations: int = 12,
    on_action: Callable[[str, dict], None] | None = None,
    zero_enrichment_hook: Callable[[LoopState], pd.DataFrame | None] | None = None,
) -> LoopState:
    """Execute the PLAN→ACT→OBSERVE→CORRECT loop and return final LoopState."""

    state = LoopState()
    state.target_accuracy = target_accuracy

    # -----------------------------------------------------------------------
    # STARTUP: fetch all 4 sources and build the frozen multi-site test set.
    #
    # The test set is fixed for the entire run so accuracy numbers are
    # directly comparable across iterations. Training data is revealed
    # source-by-source as the diagnosis calls for more data.
    # -----------------------------------------------------------------------
    _p("[startup] Fetching all 4 UCI sources to build frozen multi-site test set ...")
    all_sources: dict[str, pd.DataFrame] = {}
    for source_name in uci_client.SOURCES:
        raw_df = uci_client.fetch_source(source_name)
        normalized_df = nexla_client.normalize(raw_df, source_name)
        all_sources[source_name] = normalized_df
        _p(f"[startup]   {source_name}: {len(normalized_df)} records after normalization")

    frozen = harness.build_frozen_test_set(all_sources)
    n_test = len(frozen.test_df)
    _p(
        f"[startup] Frozen test set: {n_test} rows from {len(uci_client.SOURCES)} sites  "
        f"(disease prevalence={100 * frozen.test_df['target'].mean():.1f}%)"
    )
    state.test_df = frozen.test_df

    # -----------------------------------------------------------------------
    # Loop state — mutable across iterations
    # -----------------------------------------------------------------------
    pulled_sources: list[str] = [uci_client.SOURCES[0]]  # Cleveland only to start
    current_model_name: str = registry.WEAK_BASELINE_NAME
    transforms_applied: list[str] = []    # op names: prevent re-applying same op
    transform_specs: list[dict] = []      # full specs — re-applied each iteration
    models_tried_local: list[str] = []    # models evaluated so far (incl. current)
    diagnosis_history: list[tuple[str, float]] = []  # (verdict, accuracy) per iter
    enrichment_df: pd.DataFrame | None = None  # Zero.xyz data if hook returned it
    enrichment_attempted: bool = False    # True once hook has been called; prevents re-firing

    # How many non-weak-baseline models exist in the pool.
    n_capable = len([m for m in registry.MODEL_POOL if m != registry.WEAK_BASELINE_NAME])

    _p(
        f"\n[loop] Starting: source={pulled_sources[0]}, "
        f"model={current_model_name}, target={target_accuracy:.3f}, "
        f"max_iterations={max_iterations}"
    )

    for iteration in range(1, max_iterations + 1):
        model_name_this_iter = current_model_name  # save before any switch
        _p(f"\n{'─' * 68}")
        _p(
            f"[iter {iteration}] PLAN  model={model_name_this_iter}  "
            f"sources={pulled_sources}  transforms={transforms_applied or ['none']}"
        )

        # -------------------------------------------------------------------
        # ACT: assemble training slice, re-apply transforms, fit model
        # -------------------------------------------------------------------
        raw_train = harness.assemble_train(pulled_sources, frozen.train_reserves)

        # Merge Zero.xyz enrichment data if the hook returned it last iteration.
        # enrichment_df is normalized by nexla_client at merge time so column
        # schema is guaranteed to match before transforms are applied.
        if enrichment_df is not None:
            raw_train = nexla_client.merge(raw_train, enrichment_df)

        # Re-apply transforms from raw each iteration (stateless).
        # Transforms are applied consistently to both train and test so
        # there is no distribution mismatch at prediction time.
        working_train, working_test = _apply_transforms(
            raw_train, frozen.test_df, transform_specs, model_name_this_iter
        )

        model = registry.get_model(model_name_this_iter)
        fitted, result = harness.fit_and_eval(model, working_train, working_test)
        state.working_df = working_train

        # -------------------------------------------------------------------
        # OBSERVE
        # -------------------------------------------------------------------
        _p(
            f"[iter {iteration}] OBSERVE  n_train={result.n_train}  "
            f"acc={result.accuracy:.3f}  train_acc={result.train_accuracy:.3f}  "
            f"gap={result.overfit_gap:.3f}  f1={result.f1:.3f}  "
            f"recall={{0:{result.per_class_recall.get(0, 0):.2f}, "
            f"1:{result.per_class_recall.get(1, 0):.2f}}}"
        )

        # -------------------------------------------------------------------
        # CORRECT: diagnose
        # -------------------------------------------------------------------
        if model_name_this_iter not in models_tried_local:
            models_tried_local.append(model_name_this_iter)

        diag = diagnose(
            result=result,
            sources_pulled=pulled_sources,
            models_tried=models_tried_local,
            target_accuracy=target_accuracy,
            transforms_applied=transforms_applied,
            history=diagnosis_history,
            n_capable_models=n_capable,
        )
        _p(f"[iter {iteration}] DIAGNOSE {diag.verdict}")
        _p(f"           reason: {diag.reason}")

        # -------------------------------------------------------------------
        # Determine action, mutate loop state
        # -------------------------------------------------------------------
        action_taken: str
        should_break = False

        if diag.verdict == CONVERGED:
            action_taken = "stop — target reached"
            should_break = True

        elif diag.verdict == PULL_MORE_DATA:
            remaining = [s for s in uci_client.SOURCES if s not in pulled_sources]
            next_src = remaining[0]
            if on_action:
                on_action("fetch_uci_source", {"source": next_src})
            action_taken = f"fetch_uci_source({next_src})"
            pulled_sources.append(next_src)

        elif diag.verdict == TRANSFORM_DATA:
            spec = diag.transform_spec
            if on_action:
                on_action("nexla_transform", {"spec": spec})
            action_taken = f"nexla_transform(op={spec['op']})"
            transforms_applied.append(spec["op"])
            transform_specs.append(spec)

        elif diag.verdict == SWITCH_MODEL:
            next_name = registry.next_model(models_tried_local)
            if next_name is None:
                # All models exhausted and ENRICH_EXTERNALLY hasn't fired
                # (sources remain). This is a stop condition.
                action_taken = "all models exhausted — stopping"
                should_break = True
            else:
                if on_action:
                    on_action("switch_model", {"from": model_name_this_iter, "to": next_name})
                action_taken = f"switch_model({model_name_this_iter} → {next_name})"
                current_model_name = next_name

        elif diag.verdict == ENRICH_EXTERNALLY:
            if enrichment_attempted:
                # Hook was already called last time we hit this branch.
                # Whether it returned data or None, we have nothing left
                # to try — stop cleanly.
                action_taken = "zero_enrichment_already_attempted — stopping"
                should_break = True
            elif zero_enrichment_hook:
                if on_action:
                    on_action("zero_enrichment", {"reason": diag.reason})
                fetched = zero_enrichment_hook(state)
                enrichment_attempted = True
                if fetched is not None:
                    # Persist for ACT in the next iteration; the loop continues
                    # so the enriched data actually gets trained on and evaluated.
                    enrichment_df = fetched
                    action_taken = (
                        f"zero_enrichment_called(records_merged={len(fetched)})"
                    )
                    _p(
                        f"[iter {iteration}] Zero.xyz enrichment: {len(fetched)} records "
                        f"will be included in training next iteration"
                    )
                    # Do NOT set should_break — let the loop continue.
                else:
                    action_taken = "zero_enrichment_called(hook_returned_none)"
                    should_break = True
            else:
                action_taken = "zero_enrichment_attempted(no_hook_configured)"
                should_break = True

        else:
            action_taken = f"unknown verdict {diag.verdict!r} — stopping"
            should_break = True

        _p(f"[iter {iteration}] ACTION  {action_taken}")

        # -------------------------------------------------------------------
        # Record iteration using the model that was evaluated this round
        # -------------------------------------------------------------------
        iter_record = IterationRecord(
            iteration=iteration,
            model_name=model_name_this_iter,
            sources_pulled=list(pulled_sources),
            n_records=result.n_train,
            accuracy=result.accuracy,
            f1=result.f1,
            train_accuracy=result.train_accuracy,
            diagnosis=diag.verdict,
            diagnosis_reason=diag.reason,
            action_taken=action_taken,
        )
        diagnosis_history.append((diag.verdict, result.accuracy))
        state.record(iter_record)
        state.current_model_name = current_model_name

        if should_break:
            break

    else:
        best_so_far = max((r.accuracy for r in state.iterations), default=0.0)
        last_diag = state.iterations[-1].diagnosis if state.iterations else "n/a"
        _p(
            f"\n[loop] Max iterations ({max_iterations}) reached without converging.\n"
            f"  Best accuracy : {best_so_far:.3f}  (target: {target_accuracy:.3f}, "
            f"gap: {target_accuracy - best_so_far:.3f})\n"
            f"  Last diagnosis: {last_diag}\n"
            f"  The gap was not closed by native UCI data or available enrichment.\n"
            f"  Returning best result achieved."
        )

    best_acc = max((r.accuracy for r in state.iterations), default=0.0)
    _p(
        f"\n[loop] Done. Best accuracy: {best_acc:.3f}  "
        f"(target: {target_accuracy:.3f}, "
        f"{'✓ reached' if best_acc >= target_accuracy else '✗ not reached'})"
    )
    return state
