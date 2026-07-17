"""HTTP client used by the loop/dashboard to route every autonomous action
through the real Pomerium proxy (config/pomerium.yaml, running via Docker)
in front of person_b/actions_service.py.

Pomerium routes by the `from:` host in its config (`gate.localhost.pomerium.io`),
matched via the HTTP Host header at the Envoy layer -- not by real DNS
resolution. So requests go to the proxy's actual listening address
(localhost:9080) with that Host header set explicitly.

If the proxy is unreachable (Docker not running, container down), this
raises `PomeriumProxyUnavailable` rather than silently succeeding -- the
caller decides whether to fall back, and any fallback must be logged as
exactly that: a bypass of the real gate, not a substitute for it.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("pomerium_client")

PROXY_URL = "http://localhost:9080"
PROXY_HOST_HEADER = "gate.localhost.pomerium.io"


class PomeriumProxyUnavailable(Exception):
    """Raised when the Pomerium proxy can't be reached at all -- distinct
    from a 403/404 response, which means the proxy IS running and denied
    the request."""


class PomeriumProxyDenied(Exception):
    """Raised when Pomerium's routing/policy actually rejected the
    request (403 from PPL policy, or 404/502 because no route matches --
    i.e. the action type isn't allowlisted at all)."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Pomerium rejected the request: {status_code} {body}")


async def call_action(action_path: str, payload: dict) -> dict:
    """POST to an action route through the real Pomerium proxy. action_path
    like "/actions/zero_enrich"."""
    url = f"{PROXY_URL}{action_path}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers={"Host": PROXY_HOST_HEADER})
    except httpx.ConnectError as exc:
        raise PomeriumProxyUnavailable(
            f"could not reach Pomerium proxy at {PROXY_URL} -- is the container running? "
            f"({exc})"
        ) from exc

    if response.status_code >= 400:
        raise PomeriumProxyDenied(response.status_code, response.text)

    logger.info("pomerium: %s allowed and executed (status=%s)", action_path, response.status_code)
    return response.json()
