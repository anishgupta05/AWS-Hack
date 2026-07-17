"""
Integration test: ENRICH_EXTERNALLY continues the loop when the hook returns data.

Requires network access (fetches real UCI data). Runtime: ~60s.

What this confirms:
- When zero_enrichment_hook returns a non-None DataFrame, the loop does NOT
  stop at that iteration — it continues and trains/evaluates on the merged data.
- The iteration record immediately after the enrichment action shows n_records
  larger than the pre-enrichment training set, proving the data was included.
- When the hook returns None, the loop stops immediately at that iteration.
- The loop stops after a second ENRICH_EXTERNALLY verdict even if a hook is
  wired (prevents infinite re-enrichment cycles).
"""

import numpy as np
import pandas as pd
import pytest

from src.loop.agent import run
from src.loop.state import LoopState


# ---------------------------------------------------------------------------
# Synthetic enrichment DataFrame (canonical UCI schema, already binarized)
# ---------------------------------------------------------------------------

def _make_enrichment_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age":      rng.integers(40, 75, n).astype(float),
        "sex":      rng.integers(0, 2, n).astype(float),
        "cp":       rng.integers(0, 4, n).astype(float),
        "trestbps": rng.integers(100, 180, n).astype(float),
        "chol":     rng.integers(150, 350, n).astype(float),
        "fbs":      rng.integers(0, 2, n).astype(float),
        "restecg":  rng.integers(0, 3, n).astype(float),
        "thalach":  rng.integers(100, 180, n).astype(float),
        "exang":    rng.integers(0, 2, n).astype(float),
        "oldpeak":  rng.uniform(0, 4, n),
        "slope":    rng.integers(0, 3, n).astype(float),
        "ca":       rng.integers(0, 4, n).astype(float),
        "thal":     rng.integers(3, 8, n).astype(float),
        "target":   rng.integers(0, 2, n).astype(int),  # already binarized
        "source":   ["zero_xyz"] * n,
    })


# ---------------------------------------------------------------------------
# Helper: run to completion, return state
# ---------------------------------------------------------------------------

def _run(hook=None, target: float = 0.9999, max_iter: int = 15) -> LoopState:
    """Run with an unreachable target so ENRICH_EXTERNALLY always fires."""
    return run(
        target_accuracy=target,
        max_iterations=max_iter,
        zero_enrichment_hook=hook,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loop_continues_after_enrichment_with_data():
    """Hook returns data → loop runs at least one more iteration afterward."""
    enrichment_df = _make_enrichment_df(n=40)
    hook_call_count = {"n": 0}

    def hook(state: LoopState) -> pd.DataFrame:
        hook_call_count["n"] += 1
        return enrichment_df

    state = _run(hook=hook)

    # Hook was called exactly once (second ENRICH_EXTERNALLY triggers stop, not a re-call).
    assert hook_call_count["n"] == 1, (
        f"Expected hook called once, got {hook_call_count['n']}"
    )

    # Find the iteration that recorded the enrichment action.
    enrich_iters = [
        r for r in state.iterations
        if "records_merged" in r.action_taken
    ]
    assert enrich_iters, "No iteration recorded a successful enrichment"

    enrich_iter = enrich_iters[0]
    enrich_idx = enrich_iter.iteration  # 1-based

    # There must be at least one iteration AFTER the enrichment iteration.
    post_enrich = [r for r in state.iterations if r.iteration > enrich_idx]
    assert post_enrich, (
        f"Loop stopped at iter {enrich_idx} (enrichment iter) — "
        "expected at least one more iteration training on enriched data"
    )

    # The post-enrichment iteration's training set must be larger (enrichment included).
    n_before = enrich_iter.n_records
    n_after   = post_enrich[0].n_records
    assert n_after > n_before, (
        f"n_records did not grow after enrichment: before={n_before}, after={n_after}. "
        "Enrichment data was not included in training."
    )

    # Confirm evaluation happened (accuracy is a real number, not a sentinel).
    assert 0.0 < post_enrich[0].accuracy <= 1.0


def test_loop_stops_when_hook_returns_none():
    """Hook returns None → loop stops at that iteration, not after."""
    def hook(state: LoopState):
        return None

    state = _run(hook=hook)

    last = state.iterations[-1]
    assert last.action_taken == "zero_enrichment_called(hook_returned_none)", (
        f"Unexpected final action: {last.action_taken!r}"
    )

    # No post-enrichment iteration should exist.
    enrich_iters = [r for r in state.iterations if "records_merged" in r.action_taken]
    assert not enrich_iters, "Found unexpected records_merged action when hook returned None"


def test_loop_stops_without_hook():
    """No hook configured → loop stops with explicit label, no crash."""
    state = _run(hook=None)

    last = state.iterations[-1]
    assert last.action_taken == "zero_enrichment_attempted(no_hook_configured)", (
        f"Unexpected final action: {last.action_taken!r}"
    )


def test_enrichment_not_called_twice():
    """Second ENRICH_EXTERNALLY verdict (post-enrichment iter) stops without re-calling hook."""
    hook_calls = {"n": 0}

    def hook(state: LoopState) -> pd.DataFrame:
        hook_calls["n"] += 1
        return _make_enrichment_df(n=40)

    state = _run(hook=hook, max_iter=15)

    assert hook_calls["n"] == 1, (
        f"Hook called {hook_calls['n']} times — expected exactly 1"
    )

    # The final iteration should be the post-enrichment stop.
    last = state.iterations[-1]
    assert "already_attempted" in last.action_taken or "records_merged" in last.action_taken, (
        f"Unexpected final action: {last.action_taken!r}"
    )
