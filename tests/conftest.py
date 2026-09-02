import hashlib
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from rag_assistant import auth, cache
from rag_assistant.config import get_settings
from rag_assistant.conversations import store as conversations_store
from rag_assistant.retrieval import parent_store, reranker


@pytest.fixture(autouse=True)
def _default_test_env(request, monkeypatch, tmp_path):
    # `live` tests hit real APIs and need the real keys from .env/the environment,
    # so don't clobber them with fake values here.
    if request.node.get_closest_marker("live"):
        return
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    # Tests run fully offline by default -- no local Redis is assumed to be running, and
    # caching behavior itself is tested separately with an explicit fake client.
    monkeypatch.setenv("USE_CACHE", "false")
    # Every test gets its own conversation store so persistence tests can't see each
    # other's rows -- and no test ever writes into the developer's real conversations.db.
    monkeypatch.setenv("CONVERSATIONS_DB_PATH", str(tmp_path / "conversations.db"))
    # Same reasoning for the corpus, and it is not hypothetical: before this, every test that
    # exercised an upload endpoint without overriding CORPUS_DIR wrote a file into the real
    # data/corpus -- which then got indexed, described to the router, and scored by the eval
    # harness. The repo had accumulated `passwd_*.md` from the path-traversal test.
    # Distinctly named so they can't collide with the `tmp_path / "corpus"` and
    # `tmp_path / "chroma"` that individual tests create for themselves.
    default_corpus = tmp_path / "_isolated_corpus"
    default_corpus.mkdir(exist_ok=True)
    monkeypatch.setenv("CORPUS_DIR", str(default_corpus))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "_isolated_chroma"))
    # PDF vision ingestion would otherwise attempt real API calls whenever a test PDF has
    # an image-only page; tests that exercise the vision path mock describe_image directly.
    monkeypatch.setenv("PDF_VISION", "false")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    cache.reset_client_cache()
    auth.reset_api_key_cache()
    conversations_store.reset_store_cache()
    parent_store.reset_parent_store_cache()
    reranker.reset_reranker_cache()
    yield
    get_settings.cache_clear()
    cache.reset_client_cache()
    auth.reset_api_key_cache()
    conversations_store.reset_store_cache()
    parent_store.reset_parent_store_cache()
    reranker.reset_reranker_cache()


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
