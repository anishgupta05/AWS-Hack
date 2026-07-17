"""Internal actions service -- the real boundary Pomerium protects.

This service actually executes every autonomous action (Zero discovery and
payment, UCI data pulls, model switches). It is NOT meant to be reachable
directly by the loop or dashboard -- only the Pomerium proxy running in
front of it should ever call it, which is why it binds to 127.0.0.1 only.

Spend-ceiling bookkeeping (a running total across calls) lives here rather
than in Pomerium's own policy, because Pomerium's policy language (PPL)
evaluates each request statelessly against route/identity/context criteria
-- it has no notion of "total spent so far". What Pomerium genuinely
enforces at the network layer is the *allowlist*: each action type is its
own route with its own `http_path` policy, and any action type that isn't
an allowlisted route simply has nowhere to go -- Pomerium returns a routing
error before this service ever sees the request. The spend ceiling is
still a real, enforced boundary; it just lives one layer further in,
inside the only thing Pomerium will let traffic reach.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from person_b.pomerium_gate import Action, PomeriumGate
from person_b.zero_client import ZeroClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("actions_service")

app = FastAPI(title="Autonomous Actions Service (behind Pomerium)")

_gate = PomeriumGate("config/policy.yaml")
_zero = ZeroClient(_gate)


class PullDataRequest(BaseModel):
    source: str


class ZeroTaskRequest(BaseModel):
    task_description: str


class ModelSwitchRequest(BaseModel):
    reason: str


@app.post("/actions/pull_data")
async def pull_data(req: PullDataRequest):
    decision = _gate.check(Action(type="pull_data", cost_usd=0.0, detail={"source": req.source}))
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)
    logger.info("pull_data executed: source=%s", req.source)
    return {"allowed": True, "source": req.source}


@app.post("/actions/zero_discover")
async def zero_discover(req: ZeroTaskRequest):
    listings = await _zero.discover(req.task_description)
    return {"listings": [_listing_dict(listing) for listing in listings]}


@app.post("/actions/zero_enrich")
async def zero_enrich(req: ZeroTaskRequest):
    listings = await _zero.discover(req.task_description)
    chosen = await _zero.select_and_pay(listings)
    return {"chosen": _listing_dict(chosen)}


@app.post("/actions/model_switch")
async def model_switch(req: ModelSwitchRequest):
    decision = _gate.check(Action(type="model_switch", cost_usd=0.0, detail={"reason": req.reason}))
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)
    logger.info("model_switch executed: reason=%s", req.reason)
    return {"allowed": True}


def _listing_dict(listing) -> dict:
    return {
        "service_id": listing.service_id,
        "name": listing.name,
        "price_usd": listing.price_usd,
        "availability": listing.availability,
        "fit_score": listing.fit_score,
        "rating_state": listing.rating_state,
        "capability_url": listing.capability_url,
        "result_body": listing.result_body,
    }
