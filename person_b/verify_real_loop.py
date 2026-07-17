"""Standalone check: run the real integrated loop (Person A's agent.run()
wired through Pomerium + Zero via real_loop_bridge) and print every event.

Requires the Pomerium container and actions_service both running:
    docker run -d --name pomerium-gate -p 9080:9080 -p 9081:9081 \
      -v $(pwd)/config/pomerium.yaml:/pomerium/config.yaml:ro \
      --add-host=host.docker.internal:host-gateway \
      pomerium/pomerium:latest --config /pomerium/config.yaml
    uvicorn person_b.actions_service:app --host 127.0.0.1 --port 9100

Run:
    python -m person_b.verify_real_loop
"""

from __future__ import annotations

import asyncio
import logging

from person_b.real_loop_bridge import run_real_loop_streaming


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    async for event in run_real_loop_streaming(target_accuracy=0.87, max_iterations=12):
        print(f"[{event.phase}] iter={event.iteration} acc={event.accuracy} {event.message}")


if __name__ == "__main__":
    asyncio.run(main())
