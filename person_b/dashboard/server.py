"""Live demo dashboard server: streams loop events over SSE and serves the
static single-page dashboard.

Two event sources are exposed:
  /events       -- the stub loop (person_b/stub_loop.py), a fabricated
                   narrative for standalone Person B development/demoing
                   without depending on Person A's loop or real network
                   calls.
  /events/real  -- the real integration (person_b/real_loop_bridge.py),
                   which actually drives Person A's src/loop/agent.run(),
                   routing every autonomous action through the real
                   Pomerium proxy and Zero.xyz, live.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from person_b.pomerium_gate import PomeriumGate
from person_b.real_loop_bridge import run_real_loop_streaming
from person_b.stub_loop import run_stub_loop
from person_b.zero_client import ZeroClient

APP_DIR = Path(__file__).parent
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "policy.yaml"

app = FastAPI(title="Loop Engineering Demo Dashboard")


def _stub_event_source():
    gate = PomeriumGate(CONFIG_PATH)
    zero = ZeroClient(gate)
    return run_stub_loop(gate, zero)


@app.get("/")
async def index():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/events")
async def events():
    async def stream():
        async for event in _stub_event_source():
            yield event.to_sse()

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/events/real")
async def events_real():
    async def stream():
        async for event in run_real_loop_streaming():
            yield event.to_sse()

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/start")
async def start():
    # /events and /events/real already (re)start a fresh run each time
    # they're opened via EventSource; this endpoint exists for explicit
    # "start the demo" UX.
    return {"status": "ok", "message": "connect to /events or /events/real to stream a fresh run"}
