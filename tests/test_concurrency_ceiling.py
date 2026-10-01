"""What bounds concurrency on `/api/v1/research`, now that the handler is async.

This used to measure a threadpool. `/api/v1/research` was a sync handler, so FastAPI ran it
on the AnyIO worker pool and each in-flight question held one thread for the whole graph --
seconds, not milliseconds -- which made `API_THREADPOOL_SIZE` the hard concurrency ceiling of
the service no matter how idle the CPU was. The tests below asserted that the ceiling was
enforced, that requests past it queued rather than failed, and that saturation showed up as
wall-clock latency.

The graph's latency is almost entirely provider latency, which is I/O. So the nodes whose work
is an LLM call are coroutines now and await it on the event loop, the handler awaits the graph,
and only the nodes doing blocking library work -- retrieval, web search -- take a worker thread,
for milliseconds each. The ceiling those old tests measured is gone, and a test suite that
still asserted it would be pinning the bug.

What is measured instead is the property that replaced it: concurrency is bounded by the event
loop rather than by the thread pool, so a small thread limiter no longer serialises questions.
The probe is an async stand-in that sleeps -- it models provider latency, which is what a real
graph run overwhelmingly is, and it deliberately does not model CPU contention, memory
pressure, or the provider's own rate limit. This bounds the *concurrency* behaviour, not
performance under real load.
"""

import asyncio
import time

import anyio
import httpx
import pytest

from rag_assistant import api, metrics
from rag_assistant.config import get_settings

GRAPH_SECONDS = 0.15


class ConcurrencyProbe:
    """Stands in for the graph, recording how many runs are ever in flight at once."""

    def __init__(self, duration: float = GRAPH_SECONDS):
        self.duration = duration
        self.active = 0
        self.peak = 0
        self.calls = 0

    async def ainvoke(self, state, config=None):
        self.active += 1
        self.calls += 1
        self.peak = max(self.peak, self.active)
        try:
            # `asyncio.sleep`, not `time.sleep`: this models a provider round trip, and a
            # blocking sleep inside a coroutine would stall the loop and measure the opposite
            # of what the test is for.
            await asyncio.sleep(self.duration)
            return {"research_report": "ok", "route": "vector", "confidence_score": 0.9}
        finally:
            self.active -= 1


@pytest.fixture
def unlimited_rate(monkeypatch):
    """The rate limiter would otherwise 429 long before concurrency became interesting --
    which is the exact reason this could not be measured against the live service."""
    monkeypatch.setenv("RATE_LIMIT_RPM", "100000")
    monkeypatch.setenv("RATE_LIMIT_RPM_GLOBAL", "100000")
    get_settings.cache_clear()


async def _drive(app, count: int) -> list[httpx.Response]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=60.0
    ) as client:
        return await asyncio.gather(
            *(
                client.post("/api/v1/research", json={"question": f"question {i} about models?"})
                for i in range(count)
            )
        )


def _run(monkeypatch, requests: int, thread_ceiling: int = 2):
    probe = ConcurrencyProbe()
    monkeypatch.setattr(api._graph, "ainvoke", probe.ainvoke)

    async def main():
        # Deliberately tiny, and deliberately *not* the bound any more. The same call the
        # lifespan makes (api.py's `_lifespan`); set here because the limiter is loop-scoped
        # and this test owns the loop.
        anyio.to_thread.current_default_thread_limiter().total_tokens = thread_ceiling
        started = time.perf_counter()
        responses = await _drive(api.app, requests)
        return responses, time.perf_counter() - started

    responses, elapsed = asyncio.run(main())
    return probe, responses, elapsed


def test_the_thread_pool_no_longer_bounds_concurrent_questions(unlimited_rate, monkeypatch):
    """The regression this whole change exists to prevent coming back.

    Twelve questions against a two-thread pool. While the handler was synchronous this peaked
    at two and took six sequential waves; awaiting the graph instead, all twelve are in flight
    at once. A future change that reverts a node to blocking work in the request path shows up
    here as a peak collapsing back toward the thread count.
    """
    probe, responses, _elapsed = _run(monkeypatch, requests=12, thread_ceiling=2)

    assert probe.peak == 12, f"only {probe.peak} questions were ever in flight together"
    assert all(r.status_code == 200 for r in responses)


def test_concurrent_questions_finish_in_about_one_graph_duration(unlimited_rate, monkeypatch):
    """The wall-clock consequence. Twelve requests that once took six waves now overlap, so
    the whole batch costs roughly what one question costs."""
    _probe, _responses, elapsed = _run(monkeypatch, requests=12, thread_ceiling=2)

    assert elapsed < 4 * GRAPH_SECONDS, f"took {elapsed:.2f}s -- requests were serialised"


def test_every_question_still_reaches_the_graph(unlimited_rate, monkeypatch):
    """Concurrency must not come at the cost of dropping work: more overlap is only an
    improvement if every request is still answered."""
    probe, responses, _elapsed = _run(monkeypatch, requests=8)

    assert [r.status_code for r in responses] == [200] * 8
    assert probe.calls == 8


def test_the_in_flight_gauge_tracks_actual_concurrency(unlimited_rate, monkeypatch):
    """`rag_research_in_flight` is what the saturation objective is measured by, so it has to
    move with real concurrency rather than merely exist."""
    observed: list[float] = []
    probe = ConcurrencyProbe()

    async def watching(state, config=None):
        observed.append(metrics.research_in_flight._value.get())
        return await probe.ainvoke(state, config)

    monkeypatch.setattr(api._graph, "ainvoke", watching)

    responses = asyncio.run(_drive(api.app, 8))

    assert all(r.status_code == 200 for r in responses)
    assert max(observed) > 1, "the gauge never showed concurrent work despite 8 parallel requests"


def test_the_service_recovers_to_idle(unlimited_rate, monkeypatch):
    """No stuck gauge: the failure this guards against is a saturation alert that never clears
    because the counter was decremented on the happy path only."""
    _run(monkeypatch, requests=9)

    assert metrics.research_in_flight._value.get() == 0


def test_a_failing_graph_still_releases_its_slot(unlimited_rate, monkeypatch):
    """The nastier leak: if the gauge were only released on success, a burst of provider
    errors would strand the service at a permanently saturated reading."""

    async def exploding(state, config=None):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(api._graph, "ainvoke", exploding)

    responses = asyncio.run(_drive(api.app, 6))

    assert all(r.status_code in (500, 503) for r in responses)
    assert metrics.research_in_flight._value.get() == 0
