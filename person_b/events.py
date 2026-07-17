"""Shared event schema — the integration seam between Person A's loop and
Person B's Pomerium/Zero/dashboard pieces.

Person A's real loop should eventually emit LoopEvent objects (or dicts with
this shape) at each phase transition. Until then, stub_loop.py fabricates
them so the dashboard and gates are independently demoable.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Phase(str, Enum):
    PLAN = "PLAN"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    CORRECT = "CORRECT"
    GATE = "GATE"  # a Pomerium allow/deny decision
    DONE = "DONE"


class LoopEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    iteration: int
    phase: Phase
    data_sources: list[str] = Field(default_factory=list)
    model_class: Optional[str] = None
    accuracy: Optional[float] = None
    diagnosis: Optional[str] = None
    action: Optional[dict[str, Any]] = None
    message: str = ""

    def to_sse(self) -> str:
        return f"data: {self.model_dump_json()}\n\n"
