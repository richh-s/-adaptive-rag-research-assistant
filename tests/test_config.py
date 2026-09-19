import pytest

from rag_assistant.config import get_settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    settings = get_settings()

    assert settings.google_api_key == "test-google-key"
    assert settings.confidence_threshold == 0.6


def test_settings_missing_keys_raise_friendly_error(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env file here

    with pytest.raises(RuntimeError, match="Copy .env.example to .env"):
        get_settings()


# ---- deployment profiles ----


def test_the_default_profile_changes_nothing(monkeypatch):
    """single-node is the default and must stay infrastructure-free."""
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.deployment_profile == "single-node"
    assert settings.vector_backend == "chroma"
    assert settings.conversations_backend == "sqlite"
    assert settings.task_backend == "memory"


def test_multi_replica_turns_on_the_shared_backends_together(monkeypatch):
    """They are not independent choices: a shared index with per-process ingest tasks serves
    404s from whichever replica did not accept the upload, and shared tasks with a local
    index gives two divergent corpora."""
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("DEPLOYMENT_PROFILE", "multi-replica")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/rag")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.vector_backend == "pgvector"
    assert settings.conversations_backend == "postgres"
    assert settings.task_backend == "redis"


def test_an_explicit_switch_beats_the_profile(monkeypatch):
    """A profile that overrode explicit configuration would make the individual switches
    lie."""
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("DEPLOYMENT_PROFILE", "multi-replica")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/rag")
    monkeypatch.setenv("TASK_BACKEND", "memory")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.task_backend == "memory"
    assert settings.vector_backend == "pgvector"


def test_multi_replica_without_a_database_url_fails_at_startup(monkeypatch):
    """Rather than per request, inside a background task, far from the configuration that
    caused it."""
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("DEPLOYMENT_PROFILE", "multi-replica")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()

    with pytest.raises(Exception) as excinfo:
        get_settings()

    assert "DATABASE_URL" in str(excinfo.value)
