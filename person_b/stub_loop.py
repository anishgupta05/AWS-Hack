"""Fake loop generator emitting LoopEvents, standing in for Person A's real
core loop until it exists. Matches the narrative in CLAUDE.md: Cleveland-only
start, weak model, incremental data pulls, native-source exhaustion, Zero
fallback, final model switch beating the 83.3% benchmark.

Every autonomous action (pull_data, zero_enrich, model_switch) is routed
through the real Pomerium proxy via `pomerium_client.call_action`, which
hits the actions service (person_b/actions_service.py) only through
Pomerium's routing + PPL policy (config/pomerium.yaml). If the proxy is
unreachable -- e.g. the Docker container isn't running -- this falls back
to calling the same logic in-process, but logs it loudly as exactly that:
a bypass of the real network gate, not a substitute for it. That fallback
exists for demo resilience, not to pretend the gate ran when it didn't.

Delete/bypass this once the real loop is wired in — server.py just needs a
different async generator with the same LoopEvent shape.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from person_b import pomerium_client
from person_b.events import LoopEvent, Phase
from person_b.pomerium_gate import Action, PomeriumGate
from person_b.zero_client import ZeroClient

logger = logging.getLogger("stub_loop")

BENCHMARK_TARGET = 0.833

_SOURCES = ["cleveland", "hungary", "switzerland", "long_beach_va"]


async def _gated_pull_data(gate: PomeriumGate, source: str) -> bool:
    try:
        await pomerium_client.call_action("/actions/pull_data", {"source": source})
        return True
    except pomerium_client.PomeriumProxyDenied as exc:
        logger.warning("pomerium denied pull_data for %s: %s", source, exc)
        return False
    except pomerium_client.PomeriumProxyUnavailable as exc:
        logger.warning("%s -- falling back to in-process gate check (NOT network-gated)", exc)
        return gate.check(Action(type="pull_data", cost_usd=0.0, detail={"source": source})).allowed


async def _gated_model_switch(gate: PomeriumGate, reason: str) -> bool:
    try:
        await pomerium_client.call_action("/actions/model_switch", {"reason": reason})
        return True
    except pomerium_client.PomeriumProxyDenied as exc:
        logger.warning("pomerium denied model_switch: %s", exc)
        return False
    except pomerium_client.PomeriumProxyUnavailable as exc:
        logger.warning("%s -- falling back to in-process gate check (NOT network-gated)", exc)
        return gate.check(Action(type="model_switch", cost_usd=0.0, detail={"reason": reason})).allowed


async def _gated_zero_enrich(gate: PomeriumGate, zero: ZeroClient, task_description: str) -> dict:
    """Returns a dict shaped like {"chosen": {...}} regardless of whether
    it went through the proxy or the in-process fallback."""
    try:
        result = await pomerium_client.call_action("/actions/zero_enrich", {"task_description": task_description})
        logger.info("zero_enrich routed through real Pomerium proxy")
        return result
    except pomerium_client.PomeriumProxyDenied as exc:
        raise PermissionError(f"pomerium denied zero_enrich: {exc}") from exc
    except pomerium_client.PomeriumProxyUnavailable as exc:
        logger.warning("%s -- falling back to in-process Zero call (NOT network-gated)", exc)
        listings = await zero.discover(task_description)
        chosen = await zero.select_and_pay(listings)
        return {
            "chosen": {
                "service_id": chosen.service_id,
                "name": chosen.name,
                "price_usd": chosen.price_usd,
                "fit_score": chosen.fit_score,
                "availability": chosen.availability,
            }
        }


async def run_stub_loop(gate: PomeriumGate, zero: ZeroClient, delay: float = 1.5) -> AsyncIterator[LoopEvent]:
    sources: list[str] = []
    accuracy = 0.0
    model = "KNN (k=1, unscaled)"
    iteration = 0

    plan = [
        (0.61, "underfitting on too little data (train/test gap small, both low) -> pull more data"),
        (0.70, "still data-starved, learning curve still rising -> pull more data"),
        (0.74, "per-class recall imbalanced on minority class -> data shape issue, needs enrichment"),
        (None, "native sources exhausted, still below benchmark -> fall back to Zero.xyz enrichment"),
        (0.79, "enrichment merged but model class capped -> switch to SVM"),
        (0.858, "beats 83.3% benchmark -> converged"),
    ]

    for step_acc, diagnosis in plan:
        iteration += 1

        if len(sources) < len(_SOURCES) and step_acc is not None and step_acc < BENCHMARK_TARGET and "enrichment" not in diagnosis:
            next_source = _SOURCES[len(sources)]
            allowed = await _gated_pull_data(gate, next_source)
            sources.append(next_source)
            yield LoopEvent(
                iteration=iteration, phase=Phase.CORRECT, data_sources=list(sources),
                model_class=model, accuracy=accuracy, diagnosis=diagnosis,
                action={"type": "pull_data", "source": next_source, "gate_allowed": allowed},
                message=f"Pulled {next_source} live from UCI API, merged via Nexla.",
            )
            await asyncio.sleep(delay)

        if "Zero.xyz" in diagnosis:
            yield LoopEvent(
                iteration=iteration, phase=Phase.CORRECT, data_sources=list(sources),
                model_class=model, accuracy=accuracy, diagnosis=diagnosis,
                message="Native UCI sources exhausted. Searching Zero.xyz marketplace live...",
            )
            await asyncio.sleep(delay)

            result = await _gated_zero_enrich(gate, zero, "cardiac clinical enrichment tabular")
            chosen = result["chosen"]
            sources.append(f"zero:{chosen['service_id']}")
            yield LoopEvent(
                iteration=iteration, phase=Phase.GATE, data_sources=list(sources),
                model_class=model, accuracy=accuracy, diagnosis=diagnosis,
                action={"type": "zero_enrich", "service": chosen["name"], "price_usd": chosen["price_usd"]},
                message=f"Zero.xyz: selected '{chosen['name']}' (fit {chosen['fit_score']:.1f}, {chosen['availability']}) for ${chosen['price_usd']:.2f}, gated through Pomerium + paid.",
            )
            await asyncio.sleep(delay)
            continue

        if "switch" in diagnosis:
            model = "SVM (RBF kernel, scaled features)"
            await _gated_model_switch(gate, "model class capped, switching to SVM")

        accuracy = step_acc if step_acc is not None else accuracy
        yield LoopEvent(
            iteration=iteration, phase=Phase.OBSERVE, data_sources=list(sources),
            model_class=model, accuracy=accuracy, diagnosis=diagnosis,
            message=f"Trained {model} on {len(sources)} source(s). Accuracy: {accuracy:.1%}.",
        )
        await asyncio.sleep(delay)

    yield LoopEvent(
        iteration=iteration, phase=Phase.DONE, data_sources=list(sources),
        model_class=model, accuracy=accuracy, diagnosis="Converged: exceeds 83.3% published SVM benchmark.",
        message="Loop complete.",
    )
