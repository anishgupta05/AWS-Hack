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
    target_accuracy: float = 0.833,
    max_iterations: int = 6,
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
            CONVERGED      → break
            NEED_MORE_DATA → fetch next UCI source (fires on_action)
            NEED_TRANSFORM → request Nexla transform (fires on_action)
            NEED_MODEL_SWITCH → swap model (fires on_action); if all
                                UCI sources exhausted, first call
                                zero_enrichment_hook before switching
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
- The test set is split once from the first fetched data and never
  re-split. As new sources are merged into working_df, only the training
  portion grows — the test set stays fixed so accuracy is comparable
  across iterations.
- Log format per iteration (print to stdout):
    [iter 1] model=knn_weak  sources=[cleveland]  n=242  acc=0.613  f1=0.601
    [iter 1] diagnosis: NEED_MORE_DATA — Train accuracy (0.91) >> test (0.61) ...
    [iter 1] action: fetch_uci_source(hungary)
  This is what appears on screen during the demo.
- Cap at max_iterations regardless of accuracy — do not let the loop run
  indefinitely during a live demo.
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from src.loop.state import LoopState

logger = logging.getLogger(__name__)


def run(
    target_accuracy: float = 0.833,
    max_iterations: int = 6,
    on_action: Callable[[str, dict], None] | None = None,
    zero_enrichment_hook: Callable[[LoopState], pd.DataFrame | None] | None = None,
) -> LoopState:
    """Execute the PLAN→ACT→OBSERVE→CORRECT loop and return final LoopState."""
    raise NotImplementedError
