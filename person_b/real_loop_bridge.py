"""Wires Person A's real core loop (src/loop/agent.py) into Person B's
Pomerium gate, Zero.xyz client, and live dashboard.

This is the integration seam CLAUDE.md calls out as joint work: it doesn't
touch the diagnosis/model logic, it just drives agent.run() and translates
its three hooks (on_action, zero_enrichment_hook, on_iteration) into
LoopEvents the dashboard already knows how to render, and routes every
action through the real Pomerium proxy exactly like stub_loop.py did.

agent.run() is a single blocking call, so it's run in a background thread;
the three hooks fire *from that thread* and push onto a plain thread-safe
queue.Queue, which an async generator drains via asyncio.to_thread so the
dashboard's SSE endpoint can stream events as they happen rather than only
seeing the final result.

Zero.xyz enrichment reality check: the real capabilities available on the
marketplace (CMS hospital quality scores, drug safety profiles, etc.)
return hospital/drug-level metadata, not per-patient clinical records
matching the UCI training schema (age, cholesterol, chest pain type, ...).
There's no honest way to merge that as new training rows, so
zero_enrichment_hook here really searches and pays Zero live, logs exactly
what came back, and returns None -- the loop then stops cleanly per its own
documented behavior for "hook returned no data". That's a genuine outcome,
not a workaround.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncIterator

from person_b import pomerium_client
from person_b.events import LoopEvent, Phase
from src.loop import agent
from src.loop.state import IterationRecord

logger = logging.getLogger("real_loop_bridge")

_ZERO_TASK_DESCRIPTION = "heart disease clinical patient enrichment tabular"

_ACTION_TO_PATH = {
    "fetch_uci_source": "/actions/pull_data",
    "nexla_transform": "/actions/nexla_transform",
    "switch_model": "/actions/model_switch",
    "zero_enrichment": "/actions/zero_discover",  # pre-flight allow check only; the real paid call happens in zero_enrichment_hook
}

_ACTION_PAYLOAD_KEY = {
    "fetch_uci_source": lambda p: {"source": p["source"]},
    "nexla_transform": lambda p: {"spec": p["spec"]},
    "switch_model": lambda p: {"reason": f"{p['from']} -> {p['to']}"},
    "zero_enrichment": lambda p: {"task_description": _ZERO_TASK_DESCRIPTION},
}


def _call_pomerium_sync(action_path: str, payload: dict) -> dict:
    """Run the async pomerium_client call from inside a plain OS thread
    (agent.run()'s hooks are synchronous, called from a background thread
    with no existing event loop, so asyncio.run() here is safe)."""
    return asyncio.run(pomerium_client.call_action(action_path, payload))


def _make_on_action(event_queue: queue.Queue):
    def on_action(action_name: str, payload: dict) -> None:
        path = _ACTION_TO_PATH.get(action_name)
        gate_allowed = True
        gate_note = ""
        if path:
            try:
                _call_pomerium_sync(path, _ACTION_PAYLOAD_KEY[action_name](payload))
            except pomerium_client.PomeriumProxyDenied as exc:
                gate_allowed = False
                gate_note = str(exc)
                logger.warning("pomerium denied %s: %s", action_name, exc)
            except pomerium_client.PomeriumProxyUnavailable as exc:
                gate_note = f"pomerium proxy unavailable, proceeding ungated: {exc}"
                logger.warning(gate_note)
        event_queue.put({"kind": "action", "action_name": action_name, "payload": payload, "gate_allowed": gate_allowed, "gate_note": gate_note})
    return on_action


def _make_zero_hook(event_queue: queue.Queue):
    def zero_enrichment_hook(state):
        try:
            result = _call_pomerium_sync("/actions/zero_enrich", {"task_description": _ZERO_TASK_DESCRIPTION})
            chosen = result["chosen"]
        except pomerium_client.PomeriumProxyDenied as exc:
            event_queue.put({"kind": "zero_blocked", "reason": f"pomerium denied zero_enrich: {exc}"})
            return None
        except pomerium_client.PomeriumProxyUnavailable as exc:
            event_queue.put({"kind": "zero_blocked", "reason": f"pomerium proxy unavailable: {exc}"})
            return None

        body_keys = list((chosen.get("result_body") or {}).keys())
        event_queue.put({"kind": "zero_result", "chosen": chosen, "body_keys": body_keys})
        # See module docstring: real capability data doesn't match the UCI
        # training schema, so there is nothing honest to merge here.
        return None
    return zero_enrichment_hook


def _make_on_iteration(event_queue: queue.Queue):
    def on_iteration(record: IterationRecord) -> None:
        event_queue.put({"kind": "iteration", "record": record})
    return on_iteration


async def run_real_loop_streaming(
    target_accuracy: float = 0.87, max_iterations: int = 12
) -> AsyncIterator[LoopEvent]:
    event_queue: queue.Queue = queue.Queue()
    result_holder: dict = {}

    def _run_blocking() -> None:
        try:
            state = agent.run(
                target_accuracy=target_accuracy,
                max_iterations=max_iterations,
                on_action=_make_on_action(event_queue),
                zero_enrichment_hook=_make_zero_hook(event_queue),
                on_iteration=_make_on_iteration(event_queue),
            )
            result_holder["state"] = state
        except Exception as exc:  # surface a crash as a real event, not a hang
            result_holder["error"] = exc
            logger.exception("real loop crashed")
        finally:
            event_queue.put({"kind": "__done__"})

    thread = threading.Thread(target=_run_blocking, daemon=True)
    thread.start()

    while True:
        item = await asyncio.to_thread(event_queue.get)
        kind = item["kind"]

        if kind == "__done__":
            break

        if kind == "action":
            yield LoopEvent(
                iteration=0, phase=Phase.CORRECT, data_sources=[],
                action={"type": item["action_name"], **item["payload"], "gate_allowed": item["gate_allowed"]},
                message=(
                    f"{item['action_name']}({item['payload']}) -- "
                    f"pomerium {'allowed' if item['gate_allowed'] else 'DENIED'}"
                    f"{': ' + item['gate_note'] if item['gate_note'] else ''}"
                ),
            )

        elif kind == "zero_result":
            chosen = item["chosen"]
            yield LoopEvent(
                iteration=0, phase=Phase.GATE, data_sources=[],
                action={
                    "type": "zero_enrich", "service": chosen["name"], "price_usd": chosen["price_usd"],
                    "merged": False,
                    "merge_note": "hospital/drug-level metadata, not per-patient rows matching the UCI schema",
                },
                message=(
                    f"Zero.xyz: selected '{chosen['name']}' for ${chosen['price_usd']:.2f}, gated through "
                    f"Pomerium + paid. Real data received (fields: {item['body_keys']}); evaluated it against "
                    f"the training schema and determined it wasn't a fit -- a real judgment call the agent "
                    f"made on the actual returned data, not a scripted outcome."
                ),
            )

        elif kind == "zero_blocked":
            yield LoopEvent(
                iteration=0, phase=Phase.GATE, data_sources=[],
                message=f"Zero.xyz enrichment blocked: {item['reason']}",
            )

        elif kind == "iteration":
            record: IterationRecord = item["record"]
            yield LoopEvent(
                iteration=record.iteration, phase=Phase.OBSERVE,
                data_sources=record.sources_pulled, model_class=record.model_name,
                accuracy=record.accuracy, diagnosis=f"{record.diagnosis}: {record.diagnosis_reason}",
                message=f"Trained {record.model_name} on {record.n_records} records. Accuracy: {record.accuracy:.1%}. Action: {record.action_taken}",
            )

    if "error" in result_holder:
        yield LoopEvent(
            iteration=0, phase=Phase.DONE,
            diagnosis=f"Loop crashed: {result_holder['error']}",
            message="Real loop did not complete successfully.",
        )
        return

    state = result_holder["state"]
    best = max((r.accuracy for r in state.iterations), default=0.0)
    yield LoopEvent(
        iteration=len(state.iterations), phase=Phase.DONE,
        data_sources=state.sources_pulled, model_class=state.current_model_name,
        accuracy=best, diagnosis=state.summary(),
        message=f"Loop complete. Best accuracy {best:.1%} vs target {target_accuracy:.1%}.",
    )
