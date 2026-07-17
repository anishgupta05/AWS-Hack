"""Live demo dashboard server: streams loop events over SSE and serves the
static single-page dashboard. Currently wired to the stub loop; swapping in
Person A's real loop is a one-line change to `_event_source()`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from person_b.pomerium_gate import PomeriumGate
from person_b.stub_loop import run_stub_loop
from person_b.zero_client import ZeroClient

APP_DIR = Path(__file__).parent
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "policy.yaml"

app = FastAPI(title="Loop Engineering Demo Dashboard")


def _event_source():
    """Swap this for Person A's real loop generator once it exists —
    must yield the same LoopEvent shape defined in person_b/events.py."""
    gate = PomeriumGate(CONFIG_PATH)
    zero = ZeroClient(gate)
    return run_stub_loop(gate, zero)


@app.get("/")
async def index():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/events")
async def events():
    async def stream():
        async for event in _event_source():
            yield event.to_sse()

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/start")
async def start():
    # /events already (re)starts a fresh run each time it's opened via
    # EventSource; this endpoint exists for explicit "start the demo" UX.
    return {"status": "ok", "message": "connect to /events to stream a fresh run"}
