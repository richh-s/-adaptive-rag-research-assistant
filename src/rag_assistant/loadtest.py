"""A load generator for measuring this service under concurrency.

The metrics layer can report p95 latency, but only for traffic that actually happened — and
nothing has ever driven concurrent traffic at this service, so its behaviour under load is
genuinely unknown rather than merely unmeasured. This closes that: it drives a fixed number of
requests at a fixed concurrency and reports the distribution.

Two things it deliberately does *not* do.

It does not default to `/api/v1/research`. Every research request costs several LLM calls, so
a thousand-request run against it is a real bill; the default target is `/health`, which
measures the HTTP stack, the middleware chain and the event loop for free. Pointing it at the
research endpoint is an explicit choice, and the CLI says what it will cost first.

It does not report an average. An average latency hides exactly the tail that matters — a run
where nineteen requests take 100ms and one takes 30 seconds has a fine mean and a terrible
p95, and the p95 is the number a user experiences.
"""

import asyncio
import time
from dataclasses import dataclass, field

import httpx


def percentile(values: list[float], p: float) -> float:
    """Linearly interpolated percentile. Matches numpy's default without the dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (p / 100.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


@dataclass
class LoadTestResult:
    total_requests: int
    concurrency: int
    wall_seconds: float = 0.0
    latencies: list[float] = field(default_factory=list)
    status_counts: dict[int, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    # Connection failures are recorded as status 0. The lower bound matters: `status < 400`
    # alone would count every refused connection as a success, which is precisely backwards
    # for the one tool whose job is to find out whether the service stops accepting them.
    @property
    def successful(self) -> int:
        return sum(count for status, count in self.status_counts.items() if 200 <= status < 400)

    @property
    def throughput(self) -> float:
        return self.total_requests / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return 1.0 - (self.successful / self.total_requests) if self.total_requests else 0.0

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "concurrency": self.concurrency,
            "wall_seconds": round(self.wall_seconds, 2),
            "throughput_rps": round(self.throughput, 2),
            "error_rate": round(self.error_rate, 4),
            "p50_ms": round(percentile(self.latencies, 50) * 1000, 1),
            "p95_ms": round(percentile(self.latencies, 95) * 1000, 1),
            "p99_ms": round(percentile(self.latencies, 99) * 1000, 1),
            "max_ms": round(max(self.latencies) * 1000, 1) if self.latencies else 0.0,
            "status_counts": dict(sorted(self.status_counts.items())),
        }


async def run_load_test(
    base_url: str,
    path: str = "/health",
    method: str = "GET",
    total_requests: int = 100,
    concurrency: int = 10,
    payload: dict | None = None,
    headers: dict | None = None,
    timeout: float = 60.0,
    client: httpx.AsyncClient | None = None,
) -> LoadTestResult:
    """Drives `total_requests` at `concurrency`, returning the latency distribution.

    Concurrency is bounded by a semaphore rather than by batching. Batching would make every
    request wait for the slowest in its group, which flattens the tail this is trying to
    measure — the whole point is to see what happens to the slow requests, not to hide them
    behind a barrier.
    """
    result = LoadTestResult(total_requests=total_requests, concurrency=concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    owns_client = client is None
    client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def one_request() -> None:
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await client.request(method, path, json=payload, headers=headers)
                elapsed = time.perf_counter() - started
                result.latencies.append(elapsed)
                result.status_counts[response.status_code] = (
                    result.status_counts.get(response.status_code, 0) + 1
                )
            except Exception as exc:
                # A connection error is a result, not a crash: refusing connections under
                # load is precisely the behaviour a load test exists to discover.
                result.latencies.append(time.perf_counter() - started)
                result.status_counts[0] = result.status_counts.get(0, 0) + 1
                result.errors.append(f"{type(exc).__name__}: {exc}")

    try:
        started = time.perf_counter()
        await asyncio.gather(*(one_request() for _ in range(total_requests)))
        result.wall_seconds = time.perf_counter() - started
    finally:
        if owns_client:
            await client.aclose()
    return result
