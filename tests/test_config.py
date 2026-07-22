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
