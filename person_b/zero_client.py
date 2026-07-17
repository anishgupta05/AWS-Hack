"""Zero.xyz integration: live marketplace discovery + payment for
enrichment/scraping services, used only after native UCI hospital sources
are exhausted and the agent is still underperforming.

Discovery and selection happen live, at the point the correction loop
decides it needs enrichment — nothing here is pre-selected.

Zero has no public REST API docs; the `zero` CLI (~/.zero/runtime/bin/zero)
is the supported interface, already authenticated via ~/.zero/config.json.

Confirmed real shape of `zero search "<query>" --json` (run 2026-07-17):
{
  "capabilities": [
    {
      "token": "z_9xRsEW.1", "canonicalName": "...", "description": "...",
      "whatItDoes": "...", "method": "GET", "url": "https://...",
      "cost": {"amount": "0.12", "asset": "USDC"},
      "rating": {"score": "0.00", "reviews": 0, "state": "unrated"},
      "availabilityStatus": "unknown" | "healthy"
    }, ...
  ]
}
Search already returns `url`/`method`/`rating`/`availabilityStatus` directly
-- no separate `zero get` detail round-trip is needed to select and pay.

`discover()` shells out to `zero search --json`, `select_and_pay()` pays via
`zero fetch --capability ... --max-pay ... --json`. If the CLI is missing or
a call fails (e.g. offline dev), both fall back to the mock catalog below so
the dashboard/demo stay runnable -- disclosed via a warning log, not silent.

Ranking has no real usable ratings to lean on (every capability observed so
far reports `"state": "unrated"`, 0 reviews) so `select_and_pay` picks by
task-fit -- keyword overlap between the task description and each listing's
name/description/whatItDoes -- weighted by availability and price, not by
community rating as originally sketched in CLAUDE.md.

Capabilities can require capability-specific query/body params (e.g. the CMS
Hospital Care Compare capability needs `hospital_name`). The first fetch
attempt is always param-free. If it fails, the 400 body is inspected for
candidate field names -- both Zod "invalid_type" issues with a `path` array
naming the field directly, and free-text "custom" issues ("Provide at least
one of facilityId, state, city, or name.") that only name fields in the
message. Candidates are intersected against `_PARAM_DEFAULTS`; any hit
triggers one retry with those defaults filled in -- a small, honest
self-correction step (matching the project's own diagnose-and-correct
ethos) rather than a silent failure. Values are anchored to the actual
project: "Cleveland Clinic" for hospital/facility-name fields, since
Cleveland is the first UCI hospital source this agent starts with. No
recognizable field names in the error aborts the retry -- no guessing at
arbitrary schema fields. The retry is still a real, separate x402 payment,
logged like any other call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

from person_b.pomerium_gate import Action, PomeriumGate

logger = logging.getLogger("zero_client")

ZERO_BIN = shutil.which("zero") or str(Path.home() / ".zero/runtime/bin/zero")

# Mock marketplace catalog, shaped like real `zero search --json` output --
# used as a fallback if the real CLI is unavailable or a call fails.
_MOCK_CATALOG: list[dict] = [
    {
        "service_id": "svc_clinical_scrape_01",
        "name": "Open Clinical Cardiac Data Scraper",
        "description": "Scrapes public clinical trial and patient outcome datasets for cardiac risk factors.",
        "price_usd": 1.50,
        "availability": "healthy",
        "method": "GET",
        "url": "",
    },
    {
        "service_id": "svc_health_enrich_07",
        "name": "HealthRecord General Enrichment API",
        "description": "General-purpose EHR feature enrichment, broad coverage but not cardiac-specific.",
        "price_usd": 0.75,
        "availability": "healthy",
        "method": "GET",
        "url": "",
    },
    {
        "service_id": "svc_lowquality_03",
        "name": "QuickScrape Health Forum Bot",
        "description": "Fast but low-quality scraping of forum health data.",
        "price_usd": 0.25,
        "availability": "unknown",
        "method": "GET",
        "url": "",
    },
]

_AVAILABILITY_WEIGHT = {"healthy": 1.0, "unknown": 0.4, "down": 0.0}

# Sensible defaults for common required-param names seen across capabilities
# in this marketplace, anchored to the actual project (Cleveland = the first
# UCI hospital source). Any required field not listed here aborts the retry
# rather than guessing.
_PARAM_DEFAULTS: dict[str, str] = {
    "hospital_name": "Cleveland Clinic",
    "name": "Cleveland Clinic",
    "facility_name": "Cleveland Clinic",
    "facilityId": "Cleveland Clinic",
    "term": "heart disease",
    "query": "heart disease",
    "q": "heart disease",
    "condition": "heart disease",
    "diagnosis": "heart disease",
    "city": "Cleveland",
    "state": "OH",
}


@dataclass
class ServiceListing:
    service_id: str
    name: str
    price_usd: float
    availability: str = "unknown"
    description: str = ""
    method: str = "GET"
    capability_url: str = ""
    rating_state: str = ""
    fit_score: float = 0.0
    result_body: dict = field(default_factory=dict)


def _parse_cost(item: dict) -> float:
    cost = item.get("cost")
    if isinstance(cost, dict):
        try:
            return float(cost.get("amount", 0))
        except (TypeError, ValueError):
            return 0.0
    if isinstance(cost, (int, float)):
        return float(cost)
    match = re.search(r"[\d.]+", str(cost or ""))
    return float(match.group()) if match else 0.0


def _task_fit_score(text: str, availability: str, price_usd: float, task_description: str) -> float:
    """No usable rating data exists in practice (every observed capability
    is `"state": "unrated"`) -- score by how well the listing's own text
    matches the task, weighted by availability, with a mild preference for
    cheaper options among comparable fits."""
    task_words = set(re.findall(r"[a-z]+", task_description.lower()))
    text_words = set(re.findall(r"[a-z]+", text.lower()))
    overlap = len(task_words & text_words)
    availability_bonus = _AVAILABILITY_WEIGHT.get(availability, 0.2)
    price_bonus = 1.0 / (price_usd + 0.01)
    return overlap * 10 + availability_bonus * 2 + price_bonus


def _parse_listing(item: dict, task_description: str) -> ServiceListing:
    """Parse one entry from `zero search --json`'s `capabilities` list."""
    name = item.get("canonicalName") or item.get("name") or item.get("token", "unnamed")
    description = item.get("description", "")
    what_it_does = item.get("whatItDoes", "")
    price_usd = _parse_cost(item)
    availability = item.get("availabilityStatus", "unknown")
    fit_text = f"{name} {description} {what_it_does}"

    return ServiceListing(
        service_id=item.get("token") or item.get("id", "unknown"),
        name=name,
        description=description,
        price_usd=price_usd,
        availability=availability,
        method=item.get("method", "GET").upper(),
        capability_url=item.get("url", ""),
        rating_state=(item.get("rating") or {}).get("state", "unrated"),
        fit_score=_task_fit_score(fit_text, availability, price_usd, task_description),
    )


def _extract_error_text(result: dict) -> str:
    """Get a human-readable error string regardless of shape -- observed
    error bodies vary between {"error": {"message": "..."}} and a bare
    {"error": "some string"}."""
    body = result.get("body") or {}
    error = body.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(result.get("error") or "")


def _extract_field_candidates(result: dict) -> list[str]:
    """Pull candidate required-field names out of a 400/422 body. At least
    three distinct shapes observed from real providers in this marketplace:
    (1) Zod "invalid_type" issues with a `path` array naming the field
        directly: {"error": {"details": {"issues": [{"path": [...]}]}}}
    (2) "custom" issues with an empty `path`, field names only in free text:
        {"error": {"details": {"issues": [{"message": "Provide at least
        one of facilityId, state, city, or name."}]}}}
    (3) a flat shape with the field name and an example object at the
        body's top level, sibling to a bare string error code:
        {"error": "invalid_argument", "param": "hospital_name",
         "example": {"hospital_name": "example"}}
    None of these give a single clean field list, so this collects
    everything plausible (path segments, top-level "param", "example"
    keys, tokenized message text) and lets the caller intersect against
    known defaults -- a false positive token just fails the intersection,
    it doesn't add wrong params to the retry."""
    body = result.get("body") or {}
    candidates: list[str] = []

    param = body.get("param")
    if isinstance(param, str):
        candidates.append(param)
    example = body.get("example")
    if isinstance(example, dict):
        candidates += list(example.keys())

    error = body.get("error")
    if isinstance(error, str):
        candidates += re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", error)
    elif isinstance(error, dict):
        details = error.get("details")
        issues = details.get("issues") if isinstance(details, dict) else None
        for issue in issues or []:
            if not isinstance(issue, dict):
                candidates += re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", str(issue))
                continue
            path = issue.get("path") or []
            if path:
                candidates.append(".".join(str(p) for p in path))
            else:
                candidates += re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", issue.get("message", ""))
        message = error.get("message")
        if isinstance(message, str):
            candidates += re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", message)

    top_message = body.get("message")
    if isinstance(top_message, str):
        candidates += re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", top_message)

    return candidates


def _build_retry_params(candidate_fields: list[str]) -> dict[str, str] | None:
    """Intersect candidate field names against known defaults. Returns None
    if nothing recognizable was found -- caller should not retry in that
    case rather than guessing at arbitrary schema fields."""
    params = {name: _PARAM_DEFAULTS[name] for name in candidate_fields if name in _PARAM_DEFAULTS}
    return params or None


async def _run_zero(*args: str) -> dict:
    """Run the zero CLI with --json and parse the output. Raises on any
    failure (missing binary, unparseable output) so callers can decide
    whether to fall back. Deliberately does NOT raise on a non-zero exit
    code alone: `zero fetch` exits 2 on an upstream 4xx (e.g. a capability
    rejecting missing params) while still printing valid JSON on stdout --
    and, per x402, may have already charged the wallet before validating
    the request. The JSON body (with its own "ok" field) is the source of
    truth, not the process exit code."""
    try:
        proc = await asyncio.create_subprocess_exec(
            ZERO_BIN, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"zero CLI not found at {ZERO_BIN}") from exc

    stdout, stderr = await proc.communicate()
    try:
        return json.loads(stdout.decode())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"zero {args[0]} exited {proc.returncode} with unparseable output: {stderr.decode().strip()}"
        ) from exc


class ZeroClient:
    def __init__(self, gate: PomeriumGate):
        self._gate = gate

    async def _fetch_once(
        self, listing: ServiceListing, params: dict[str, str] | None, method_override: str | None = None
    ) -> dict:
        method = method_override or listing.method
        url = listing.capability_url
        fetch_args = ["fetch", url, "--capability", listing.service_id, "--max-pay", str(listing.price_usd), "--json"]
        if params:
            if method == "GET":
                url = f"{url}{'&' if '?' in url else '?'}{urlencode(params)}"
                fetch_args[1] = url
            else:
                fetch_args += ["-d", json.dumps(params)]
        elif method != "GET":
            fetch_args += ["-d", "{}"]
        return await _run_zero(*fetch_args)

    @staticmethod
    def _log_fetch_result(listing: ServiceListing, result: dict) -> None:
        payment = result.get("payment") or {}
        paid_amount = payment.get("amount")
        paid_asset = payment.get("asset") or payment.get("currency")

        if result.get("ok"):
            listing.result_body = result.get("body") or {}
            logger.info("zero fetch (live): paid %s %s, latency=%sms", paid_amount, paid_asset, result.get("latencyMs"))
        else:
            # x402 can charge before validating the request -- if a payment
            # amount is present, the wallet was actually debited even
            # though the call failed. Log both facts; don't pretend
            # nothing happened.
            error_msg = _extract_error_text(result)
            if paid_amount:
                logger.warning(
                    "zero fetch call failed (%s) but %s %s was still charged via x402",
                    error_msg, paid_amount, paid_asset,
                )
            else:
                logger.warning("zero fetch call failed, no payment charged: %s", error_msg)

    async def discover(self, task_description: str, steps: list[str] | None = None) -> list[ServiceListing]:
        """Search Zero's marketplace live for services matching the task.
        This is a second-order fallback -- only call after native UCI
        sources are exhausted. If `steps` is provided, human-readable
        descriptions of each real thing that happens are appended to it --
        used by actions_service.py to surface the full live sequence to
        the dashboard, not just the terminal outcome."""
        decision = self._gate.check(Action(type="zero_discover", cost_usd=0.0))
        if not decision.allowed:
            raise PermissionError(f"zero_discover denied: {decision.reason}")

        try:
            raw = await _run_zero("search", task_description, "--json")
            items = raw.get("capabilities", []) if isinstance(raw, dict) else raw
            results = [_parse_listing(item, task_description) for item in items]
            if not results:
                raise RuntimeError("zero search returned no results")
            logger.info("zero discover (live): %d listings for %r", len(results), task_description)
            if steps is not None:
                steps.append(f"Searched Zero.xyz marketplace live for \"{task_description}\" — found {len(results)} listings.")
        except Exception as exc:
            logger.warning("zero search failed (%s) — falling back to mock catalog", exc)
            if steps is not None:
                steps.append(f"Live marketplace search failed ({exc}) — fell back to a local mock catalog.")
            results = [
                ServiceListing(
                    service_id=listing["service_id"],
                    name=listing["name"],
                    description=listing["description"],
                    price_usd=listing["price_usd"],
                    availability=listing["availability"],
                    method=listing["method"],
                    capability_url=listing["url"],
                    fit_score=_task_fit_score(
                        f"{listing['name']} {listing['description']}",
                        listing["availability"], listing["price_usd"], task_description,
                    ),
                )
                for listing in _MOCK_CATALOG
            ]

        return results

    async def select_and_pay(self, listings: list[ServiceListing], steps: list[str] | None = None) -> ServiceListing:
        """Pick the best task-fit listing (real marketplace has no usable
        ratings -- see module docstring), then pay for it (gated) via
        `zero fetch`. This is a real decision, not the first item in the
        list -- selection varies with the input catalog and task
        description. See `discover()` for what `steps` is for."""
        if not listings:
            raise ValueError("no listings to select from")

        best = max(listings, key=lambda s: s.fit_score)
        if steps is not None:
            steps.append(f"Ranked {len(listings)} listings by task fit — selected \"{best.name}\" (fit score {best.fit_score:.1f}, ${best.price_usd:.2f}).")

        decision = self._gate.check(Action(type="zero_enrich", cost_usd=best.price_usd))
        if not decision.allowed:
            raise PermissionError(f"zero_enrich denied: {decision.reason}")

        if not best.capability_url:
            logger.warning("zero select_and_pay: no capability_url on %s, skipping live fetch (mock mode)", best.service_id)
            if steps is not None:
                steps.append("No live capability URL on the selected listing — skipped the paid call (mock mode).")
            return best

        def _record_charge(result: dict, attempt_label: str) -> None:
            """x402 can charge before validating the request -- every
            attempt, successful or not, may have moved real money. Surface
            each one, not just the final attempt's."""
            if steps is None:
                return
            payment = result.get("payment") or {}
            amount, asset = payment.get("amount"), (payment.get("asset") or payment.get("currency"))
            if not amount:
                return
            outcome = "succeeded" if result.get("ok") else "failed but was still charged"
            steps.append(f"{attempt_label}: {outcome} — {amount} {asset or 'USDC'} moved via x402.")

        try:
            result = await self._fetch_once(best, params=None)
            _record_charge(result, "Attempt 1 (as advertised)")
            method_override = None

            if not result.get("ok"):
                error_text = _extract_error_text(result)
                # some providers advertise "GET" in search metadata but
                # only accept POST for the paid/execute call (GET returns
                # free tool metadata) -- self-heal that mismatch first.
                if best.method != "POST" and "post" in error_text.lower():
                    method_override = "POST"
                    logger.info(
                        "zero fetch: %s rejected GET (%r) -- retrying with POST (self-correction, real second payment)",
                        best.name, error_text,
                    )
                    if steps is not None:
                        steps.append(f"Attempt 1 rejected (GET not accepted: \"{error_text}\") — retrying live with POST.")
                    result = await self._fetch_once(best, params=None, method_override=method_override)
                    _record_charge(result, "Attempt 2 (POST retry)")

            if not result.get("ok"):
                candidates = _extract_field_candidates(result)
                retry_params = _build_retry_params(candidates) if candidates else None
                if retry_params:
                    logger.info(
                        "zero fetch: %s rejected params -- retrying with %s (self-correction, real payment)",
                        best.name, retry_params,
                    )
                    if steps is not None:
                        steps.append(f"Call rejected missing params — retrying live with {retry_params}.")
                    result = await self._fetch_once(best, params=retry_params, method_override=method_override)
                    _record_charge(result, "Attempt 3 (params retry)")

            self._log_fetch_result(best, result)
            if steps is not None and not result.get("ok"):
                steps.append("Final attempt did not succeed — see above for what, if anything, was actually charged.")
        except Exception as exc:
            logger.warning("zero fetch failed (%s) — proceeding without live payment confirmation", exc)
            if steps is not None:
                steps.append(f"Live fetch failed entirely ({exc}) — no payment confirmation available.")

        logger.info(
            "zero select_and_pay: chose %s (fit=%.2f, price=$%.2f, availability=%s)",
            best.name,
            best.fit_score,
            best.price_usd,
            best.availability,
        )
        return best
