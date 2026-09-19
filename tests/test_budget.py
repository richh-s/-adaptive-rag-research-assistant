"""Tests for per-tenant token budgets.

Rate limiting bounds how often a tenant asks; this bounds what they spend. The two are not
interchangeable, because the cost of a run in this pipeline varies by orders of magnitude
depending on the route the router picked and how many sub-queries decomposition produced.
"""

import pytest
from fastapi.testclient import TestClient

from rag_assistant import api, budget
from rag_assistant.config import get_settings


class _FakeRedis:
    """Enough of the Redis surface for the counter, including INCRBY's atomicity, which is
    the property the shared path depends on."""

    def __init__(self):
        self.store: dict[str, int] = {}
        self.expires: dict[str, int] = {}

    def get(self, key):
        return self.store.get(key)

    def incrby(self, key, amount):
        self.store[key] = self.store.get(key, 0) + amount
        return self.store[key]

    def expire(self, key, ttl):
        self.expires[key] = ttl


class _FailingRedis:
    def get(self, key):
        raise ConnectionError("simulated Redis outage")

    def incrby(self, key, amount):
        raise ConnectionError("simulated Redis outage")

    def expire(self, key, ttl):
        raise ConnectionError("simulated Redis outage")


@pytest.fixture(autouse=True)
def _clean_budget():
    budget.reset_budget_state()
    yield
    budget.reset_budget_state()


@pytest.fixture
def shared_redis(monkeypatch):
    client = _FakeRedis()
    monkeypatch.setattr(budget, "get_redis_client", lambda: client)
    return client


@pytest.fixture
def no_redis(monkeypatch):
    monkeypatch.setattr(budget, "get_redis_client", lambda: None)


# ---- accounting ----


def test_charges_accumulate_within_a_day(shared_redis):
    budget.charge("alice", 100)
    budget.charge("alice", 250)

    assert budget.used_tokens("alice") == 350


def test_tenants_are_counted_separately(shared_redis):
    budget.charge("alice", 100)
    budget.charge("bob", 40)

    assert budget.used_tokens("alice") == 100
    assert budget.used_tokens("bob") == 40


def test_the_counter_is_shared_across_replicas(shared_redis):
    """The reason Redis is preferred: two processes charging the same tenant must see one
    total, not two. Simulated by charging through the same client from what would be separate
    processes -- INCRBY, not read-modify-write, is what makes that safe."""
    budget.charge("alice", 100)
    budget.reset_budget_state()  # a second replica: no local state at all
    budget.charge("alice", 100)

    assert budget.used_tokens("alice") == 200


def test_a_zero_charge_does_not_create_a_counter(shared_redis):
    budget.charge("alice", 0)

    assert budget.used_tokens("alice") == 0


def test_charges_expire_so_the_budget_resets(shared_redis):
    budget.charge("alice", 10)

    assert all(ttl > 0 for ttl in shared_redis.expires.values())


# ---- degradation ----


def test_a_redis_outage_does_not_fail_the_request(monkeypatch):
    """A cost control must not become an availability dependency. Reporting 0 lets the request
    through; the alternative -- failing closed -- turns a Redis blip into a full outage."""
    monkeypatch.setattr(budget, "get_redis_client", lambda: _FailingRedis())

    assert budget.charge("alice", 100) == 0
    assert budget.used_tokens("alice") == 0


def test_without_redis_the_counter_still_works_per_process(no_redis):
    budget.charge("alice", 100)

    assert budget.used_tokens("alice") == 100


def test_the_per_process_fallback_warns_once(no_redis, caplog, monkeypatch):
    """Silent per-process counting would mean the effective budget is N times the configured
    one under N replicas -- a budget that is quietly wrong is worse than one that is off."""
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "1000")
    get_settings.cache_clear()
    with caplog.at_level("WARNING"):
        budget.charge("alice", 10)
        budget.charge("alice", 10)
        budget.used_tokens("alice")

    warnings = [r for r in caplog.records if "per process" in r.message]
    assert len(warnings) == 1


# ---- enforcement ----


def test_enforce_is_a_no_op_when_the_budget_is_disabled(shared_redis, monkeypatch):
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "0")
    get_settings.cache_clear()
    budget.charge("alice", 10_000_000)

    budget.enforce("alice")  # must not raise


def test_enforce_raises_once_the_allowance_is_spent(shared_redis, monkeypatch):
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "1000")
    get_settings.cache_clear()
    budget.charge("alice", 1000)

    with pytest.raises(budget.BudgetExceeded) as excinfo:
        budget.enforce("alice")

    assert excinfo.value.used == 1000
    assert excinfo.value.budget == 1000
    assert "resets at 00:00 UTC" in str(excinfo.value)


def test_a_tenant_under_budget_is_unaffected(shared_redis, monkeypatch):
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "1000")
    get_settings.cache_clear()
    budget.charge("alice", 999)

    budget.enforce("alice")


def test_one_tenant_exhausting_its_budget_does_not_affect_another(shared_redis, monkeypatch):
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "100")
    get_settings.cache_clear()
    budget.charge("alice", 500)

    with pytest.raises(budget.BudgetExceeded):
        budget.enforce("alice")
    budget.enforce("bob")


# ---- the accountant ----


def test_the_accountant_totals_usage_across_calls():
    from langchain_core.outputs import LLMResult

    accountant = budget.TokenAccountant()
    result = LLMResult(
        generations=[],
        llm_output={"token_usage": {"prompt_tokens": 30, "completion_tokens": 12}},
    )
    accountant.on_llm_end(result)
    accountant.on_llm_end(result)

    assert accountant.input_tokens == 60
    assert accountant.output_tokens == 24
    assert accountant.total_tokens == 84


def test_an_unparseable_response_does_not_break_accounting():
    """Accounting must never fail a served answer."""
    accountant = budget.TokenAccountant()

    accountant.on_llm_end(object())

    assert accountant.total_tokens == 0


# ---- the API surface ----


def test_an_exhausted_budget_returns_429_before_any_model_call(shared_redis, monkeypatch):
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "100")
    get_settings.cache_clear()
    budget.charge("public", 100)

    def _explode(*args, **kwargs):
        raise AssertionError("the graph must not run once the budget is exhausted")

    monkeypatch.setattr(api._graph, "invoke", _explode)
    client = TestClient(api.app)

    response = client.post("/api/v1/research", json={"question": "Who founded Anthropic?"})

    assert response.status_code == 429
    assert "budget" in response.json()["detail"].lower()


def test_the_streaming_endpoint_rejects_before_the_status_is_locked(shared_redis, monkeypatch):
    """Once an SSE generator yields, the status is 200 forever. An exhausted budget has to be
    an HTTP 429 the client can act on, not an error frame inside a successful response."""
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "100")
    get_settings.cache_clear()
    budget.charge("public", 100)
    client = TestClient(api.app)

    response = client.post("/api/v1/research/stream", json={"question": "Who founded Anthropic?"})

    assert response.status_code == 429


def test_a_successful_run_is_charged(shared_redis, monkeypatch):
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "1000000")
    get_settings.cache_clear()

    def _fake_invoke(state, config=None):
        # The accountant arrives through config callbacks, exactly as LangGraph delivers it.
        accountant = config["callbacks"][0]
        accountant.input_tokens = 70
        accountant.output_tokens = 30
        return {"route": "vector", "final_answer": "ok", "research_report": "ok", "citations": []}

    monkeypatch.setattr(api._graph, "invoke", _fake_invoke)
    client = TestClient(api.app)

    client.post("/api/v1/research", json={"question": "Who founded Anthropic?"})

    assert budget.used_tokens("public") == 100


def test_a_failed_run_is_still_charged(shared_redis, monkeypatch):
    """A run that errored after four LLM calls cost what it cost. Not charging failures makes
    failure the cheap way to burn a provider quota."""
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "1000000")
    get_settings.cache_clear()

    def _fake_invoke(state, config=None):
        config["callbacks"][0].input_tokens = 55
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(api._graph, "invoke", _fake_invoke)
    client = TestClient(api.app)

    response = client.post("/api/v1/research", json={"question": "Who founded Anthropic?"})

    assert response.status_code == 500
    assert budget.used_tokens("public") == 55


# ---- the ingest path ----


def test_ingest_is_charged_for_embeddings_and_vision(shared_redis, monkeypatch):
    """Ingest is the expensive path -- an embedding per chunk and a vision call per figure --
    so leaving it unbudgeted made the budget cover mostly the wrong thing."""
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "1000000")
    monkeypatch.setenv("SYNTHESIS_CHARS_PER_TOKEN", "4")
    monkeypatch.setenv("VISION_CALL_TOKEN_ESTIMATE", "1500")
    get_settings.cache_clear()

    budget.charge_ingest("alice", embedded_chars=4000, vision_calls=2)

    # 4000 chars / 4 = 1000 embedding tokens, plus 2 * 1500 vision.
    assert budget.used_tokens("alice") == 4000


def test_an_ingest_that_embedded_nothing_is_not_charged(shared_redis, monkeypatch):
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "1000000")
    get_settings.cache_clear()

    budget.charge_ingest("alice", embedded_chars=0, vision_calls=0)

    assert budget.used_tokens("alice") == 0


def test_upload_is_rejected_before_bytes_are_accepted_when_over_budget(
    shared_redis, monkeypatch, tmp_path
):
    """Streaming 25MB to disk only to reject it wastes the resource the budget protects."""
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "100")
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    (tmp_path / "corpus").mkdir(parents=True, exist_ok=True)
    get_settings.cache_clear()
    budget.charge("public", 100)
    client = TestClient(api.app)

    response = client.post(
        "/api/v1/ingest", files={"file": ("a.md", b"some content", "text/markdown")}
    )

    assert response.status_code == 429
    assert list((tmp_path / "corpus").rglob("*.md")) == []


def test_url_ingest_is_rejected_before_the_fetch_when_over_budget(shared_redis, monkeypatch):
    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "100")
    get_settings.cache_clear()
    budget.charge("public", 100)

    def _explode(url):
        raise AssertionError("the URL must not be fetched once the budget is exhausted")

    monkeypatch.setattr(api, "fetch_page", _explode)
    client = TestClient(api.app)

    response = client.post("/api/v1/ingest/url", json={"url": "https://example.com"})

    assert response.status_code == 429


def test_a_completed_ingest_charges_the_uploading_tenant(shared_redis, monkeypatch, tmp_path):
    from rag_assistant.ingestion.build_index import IndexResult

    monkeypatch.setenv("TENANT_DAILY_TOKEN_BUDGET", "1000000")
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "corpus"))
    (tmp_path / "corpus").mkdir(parents=True, exist_ok=True)
    get_settings.cache_clear()

    monkeypatch.setattr(
        api,
        "build_index",
        lambda *, on_stage=None, **kw: IndexResult(
            indexed_chunks=2,
            changed_files=1,
            skipped_files=0,
            removed_files=0,
            embedded_chars=800,
            vision_calls=1,
        ),
    )
    client = TestClient(api.app)

    client.post("/api/v1/ingest", files={"file": ("a.md", b"content", "text/markdown")})

    # 800/4 = 200 embedding tokens + 1500 vision.
    assert budget.used_tokens("public") == 1700
