import hashlib
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from rag_assistant.config import get_settings


@pytest.fixture(autouse=True)
def _default_test_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeHashingEmbeddings(Embeddings):
    """Deterministic, offline stand-in for a real embeddings model. Uses word-level feature
    hashing (bag-of-words into fixed buckets) so texts sharing vocabulary end up close in
    vector space -- enough to test retrieval plumbing without hitting a live API."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in text.lower().split():
            bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dim
            vector[bucket] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def fake_embeddings() -> FakeHashingEmbeddings:
    return FakeHashingEmbeddings()


@pytest.fixture
def sample_corpus_dir(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "anthropic.md").write_text(
        "Anthropic was founded by Dario Amodei and builds the Claude model family, "
        "focused on Constitutional AI and safety research."
    )
    (corpus_dir / "mistral.md").write_text(
        "Mistral AI is a French company founded in Paris that builds open-weight models "
        "like Mixtral, emphasizing European AI sovereignty."
    )
    return corpus_dir
