"""Pomerium-gated access-control boundary for every autonomous action.

For the demo, `check()` is real local policy evaluation against
config/policy.yaml (allowlist + spend ceiling) — no live Pomerium call
required. Swapping in a live Pomerium policy call later is a body-only
change inside `check()`; the interface (Action in, GateDecision out) stays
the same.

Real Pomerium wiring note (when credentials are available): Pomerium
issues policy decisions via its Enforcer/authorize service. The local
allowlist + spend-ceiling check below is the same *shape* of decision —
replace the local YAML lookup with a call to the Pomerium policy endpoint
and keep GateDecision as the return type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("pomerium_gate")


@dataclass
class Action:
    type: str
    cost_usd: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateDecision:
    allowed: bool
    reason: str
    action: Action


class PomeriumGate:
    def __init__(self, policy_path: str | Path = "config/policy.yaml"):
        with open(policy_path) as f:
            policy = yaml.safe_load(f)
        self.allowlist: set[str] = set(policy["allowlist"])
        self.spend_ceiling_usd: float = float(policy["spend_ceiling_usd"])
        self._spent_usd: float = 0.0

    def check(self, action: Action) -> GateDecision:
        if action.type not in self.allowlist:
            decision = GateDecision(
                allowed=False,
                reason=f"'{action.type}' is not in the allowlist",
                action=action,
            )
        elif self._spent_usd + action.cost_usd > self.spend_ceiling_usd:
            decision = GateDecision(
                allowed=False,
                reason=(
                    f"action would spend ${action.cost_usd:.2f}, pushing total to "
                    f"${self._spent_usd + action.cost_usd:.2f} over the "
                    f"${self.spend_ceiling_usd:.2f} ceiling"
                ),
                action=action,
            )
        else:
            self._spent_usd += action.cost_usd
            decision = GateDecision(
                allowed=True,
                reason="within allowlist and spend ceiling",
                action=action,
            )

        logger.info(
            "gate decision: action=%s allowed=%s reason=%s",
            action.type,
            decision.allowed,
            decision.reason,
        )
        return decision

    @property
    def remaining_budget_usd(self) -> float:
        return self.spend_ceiling_usd - self._spent_usd
