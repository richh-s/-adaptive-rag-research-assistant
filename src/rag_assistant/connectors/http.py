"""HTTP for connectors: bounded retries that respect the source system's rate limits.

Every SaaS API a connector talks to rate-limits, and a sync of a large space is exactly the
burst that trips it. A 429 or a 5xx is retried a few times with backoff, honouring
`Retry-After` when the server sends one; anything else fails the request immediately. The
ceiling is small on purpose: a sync that retried indefinitely would hold its lock and look
hung, while one that gives up is reported, and the next scheduled run tries again.
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4
_MAX_BACKOFF_SECONDS = 30.0


def make_client(transport=None, **kwargs) -> httpx.Client:
    # Redirects are not followed: an API that redirects a credentialed request elsewhere is
    # either misconfigured or trying to send the credential somewhere it should not go.
    return httpx.Client(
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=False,
        transport=transport,
        **kwargs,
    )


def request(client: httpx.Client, method: str, url: str, **kwargs) -> httpx.Response:
    delay = 1.0
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = client.request(method, url, **kwargs)
        except httpx.TransportError:
            if attempt == _MAX_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 2, _MAX_BACKOFF_SECONDS)
            continue
        if response.status_code not in _RETRY_STATUSES or attempt == _MAX_ATTEMPTS:
            response.raise_for_status()
            return response
        retry_after = response.headers.get("retry-after")
        wait = delay
        if retry_after:
            try:
                wait = min(float(retry_after), _MAX_BACKOFF_SECONDS)
            except ValueError:
                pass
        logger.info("connector request got %d; retrying in %.1fs", response.status_code, wait)
        time.sleep(wait)
        delay = min(delay * 2, _MAX_BACKOFF_SECONDS)
    raise RuntimeError("unreachable")
