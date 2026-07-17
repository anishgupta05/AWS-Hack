"""Standalone check: does the Zero.xyz integration actually hit the real
`zero` CLI, or silently fall back to the mock catalog?

Run this yourself, in the terminal where `zero` is on PATH and authenticated
(this sandbox doesn't have it, so I can't run it for you):

    source .venv/bin/activate
    python -m person_b.verify_zero_live "<task description>"

Watch the log lines:
  - "zero discover (live): N listings for ..."  -> real zero search worked
  - "zero search failed (...) -- falling back"    -> CLI missing/errored
  - "zero fetch (live): paid ..."                 -> real payment happened
  - "zero get/fetch failed (...) -- proceeding"    -> real payment did NOT happen

If you see the "live" lines, the real integration works end to end. If you
see the fallback warnings, read the exception text in the log — it tells you
exactly what failed (missing binary, no capability url, non-GET body schema
mismatch, insufficient wallet funds, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import sys

from person_b.pomerium_gate import PomeriumGate
from person_b.zero_client import ZeroClient

CONFIG_PATH = "config/policy.yaml"


async def main(task_description: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    gate = PomeriumGate(CONFIG_PATH)
    zero = ZeroClient(gate)

    print(f"\n--- discover('{task_description}') ---")
    listings = await zero.discover(task_description)
    for listing in listings:
        print(f"  {listing.service_id:>20}  {listing.name:<45} fit={listing.fit_score:6.2f}  ${listing.price_usd:.2f}  {listing.availability}")

    print("\n--- select_and_pay(...) ---")
    chosen = await zero.select_and_pay(listings)
    print(f"\nChosen: {chosen.name} ({chosen.service_id})")
    print(f"  price: ${chosen.price_usd:.2f}")
    print(f"  capability_url: {chosen.capability_url or '(not resolved -- see warnings above)'}")
    print(f"  rating_state: {chosen.rating_state or '(not resolved -- see warnings above)'}")
    print(f"\nRemaining Pomerium budget: ${gate.remaining_budget_usd:.2f}")


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "cardiac clinical enrichment tabular"
    asyncio.run(main(task))
