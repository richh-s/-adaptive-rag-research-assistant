"""Tests for the load-test harness.

Driven against an httpx MockTransport rather than a live server, so the concurrency and
statistics logic is verified without a socket. A harness that miscounts or hides the tail
would produce confident, wrong performance numbers — worse than having none.
"""

import asyncio

import httpx
import pytest

from rag_assistant.loadtest import LoadTestResult, percentile, run_load_test


def client_returning(status: int = 200, delay: float = 0.0) -> httpx.AsyncClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        if delay:
            await asyncio.sleep(delay)
        return httpx.Response(status, json={"ok": True})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")


# ---- statistics ----


def test_percentiles_interpolate():
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == pytest.approx(2.5)


def test_percentile_of_one_value():
    assert percentile([7.0], 95) == 7.0


def test_percentile_of_nothing_is_zero():
    assert percentile([], 95) == 0.0


def test_the_summary_reports_the_tail_not_an_average():
    """An average hides the case this exists to find: nineteen fast requests and one very
    slow one has a fine mean and a terrible p95."""
    result = LoadTestResult(total_requests=20, concurrency=1, wall_seconds=1.0)
    result.latencies = [0.1] * 19 + [30.0]
    result.status_counts = {200: 20}

    summary = result.summary()

    assert "avg_ms" not in summary
    assert summary["p95_ms"] > 1000
    assert summary["max_ms"] == pytest.approx(30000.0)


def test_error_rate_counts_non_2xx_as_failures():
    result = LoadTestResult(total_requests=10, concurrency=1, wall_seconds=1.0)
    result.status_counts = {200: 8, 500: 1, 429: 1}

    assert result.successful == 8
    assert result.error_rate == pytest.approx(0.2)


def test_throughput_is_requests_over_wall_time():
    result = LoadTestResult(total_requests=100, concurrency=10, wall_seconds=4.0)

    assert result.throughput == pytest.approx(25.0)


def test_throughput_of_a_zero_length_run_is_zero_not_a_division_error():
    assert LoadTestResult(total_requests=0, concurrency=1, wall_seconds=0.0).throughput == 0.0


# ---- driving requests ----


def test_every_request_is_sent_and_recorded():
    result = asyncio.run(
        run_load_test("http://test", total_requests=25, concurrency=5, client=client_returning())
    )

    assert result.status_counts == {200: 25}
    assert len(result.latencies) == 25
    assert result.error_rate == 0.0


def test_concurrency_is_bounded_by_the_semaphore():
    """The bound has to be real: an unbounded gather would fire everything at once and
    measure something other than what was asked for."""
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    asyncio.run(run_load_test("http://test", total_requests=20, concurrency=4, client=client))

    assert peak <= 4


def test_a_connection_failure_is_recorded_rather_than_raised():
    """Refusing connections under load is exactly what a load test exists to discover, so it
    must be a result rather than a crash."""

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")

    result = asyncio.run(
        run_load_test("http://test", total_requests=5, concurrency=2, client=client)
    )

    assert result.status_counts == {0: 5}
    assert result.error_rate == 1.0
    assert len(result.errors) == 5


def test_server_errors_are_counted_separately_from_successes():
    result = asyncio.run(
        run_load_test(
            "http://test", total_requests=6, concurrency=3, client=client_returning(status=500)
        )
    )

    assert result.status_counts == {500: 6}
    assert result.successful == 0


def test_a_post_sends_the_payload():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        seen["method"] = request.method
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")

    asyncio.run(
        run_load_test(
            "http://test",
            path="/api/v1/research",
            method="POST",
            total_requests=1,
            concurrency=1,
            payload={"question": "hello"},
            client=client,
        )
    )

    assert seen["method"] == "POST"
    assert b"hello" in seen["body"]


def test_headers_are_sent(monkeypatch):
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")

    asyncio.run(
        run_load_test(
            "http://test",
            total_requests=1,
            concurrency=1,
            headers={"X-API-Key": "secret"},
            client=client,
        )
    )

    assert seen["key"] == "secret"
