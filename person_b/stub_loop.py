"""Fake loop generator emitting LoopEvents, standing in for Person A's real
core loop until it exists. Matches the narrative in CLAUDE.md: Cleveland-only
start, weak model, incremental data pulls, native-source exhaustion, Zero
fallback, final model switch beating the 83.3% benchmark.

Delete/bypass this once the real loop is wired in — server.py just needs a
different async generator with the same LoopEvent shape.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from person_b.events import LoopEvent, Phase
from person_b.pomerium_gate import Action, PomeriumGate
from person_b.zero_client import ZeroClient

BENCHMARK_TARGET = 0.833

_SOURCES = ["cleveland", "hungary", "switzerland", "long_beach_va"]


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
            action = Action(type="pull_data", cost_usd=0.0, detail={"source": next_source})
            decision = gate.check(action)
            sources.append(next_source)
            yield LoopEvent(
                iteration=iteration, phase=Phase.CORRECT, data_sources=list(sources),
                model_class=model, accuracy=accuracy, diagnosis=diagnosis,
                action={"type": "pull_data", "source": next_source, "gate_allowed": decision.allowed},
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

            listings = await zero.discover("cardiac clinical enrichment tabular")
            chosen = await zero.select_and_pay(listings)
            sources.append(f"zero:{chosen.service_id}")
            yield LoopEvent(
                iteration=iteration, phase=Phase.GATE, data_sources=list(sources),
                model_class=model, accuracy=accuracy, diagnosis=diagnosis,
                action={"type": "zero_enrich", "service": chosen.name, "price_usd": chosen.price_usd},
                message=f"Zero.xyz: selected '{chosen.name}' (fit {chosen.fit_score:.1f}, {chosen.availability}) for ${chosen.price_usd:.2f}, gated + paid.",
            )
            await asyncio.sleep(delay)
            continue

        if "switch" in diagnosis:
            model = "SVM (RBF kernel, scaled features)"
            gate.check(Action(type="model_switch", cost_usd=0.0))

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
