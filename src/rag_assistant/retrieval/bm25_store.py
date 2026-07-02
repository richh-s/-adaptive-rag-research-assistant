import re
import threading
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag_assistant.config import get_settings
from rag_assistant.ingestion.loaders import load_documents
from rag_assistant.ingestion.splitter import split_documents
from rag_assistant.schemas.models import RetrievedDoc

# Deliberately simple: no stemming, no stopword removal. BM25 is sensitive to exact token
# overlap, so e.g. "founded" vs "founding" won't match -- an accepted simplification for a
# small, low-vocabulary-variance corpus, not a production-grade tokenizer.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# rank_bm25 has no persistence API and the corpus is a handful of small files, so the index
# is rebuilt in-memory once per process and cached -- same lazy-singleton shape as
# vector_store.py's Chroma cache, keyed by resolved corpus dir.
_index_cache: dict[str, tuple[BM25Okapi | None, list]] = {}
_index_lock = threading.Lock()


def _build_index(source_dir: Path) -> tuple[BM25Okapi | None, list]:
    # Same loader + splitter defaults as ingestion/build_index.py uses for Chroma, so BM25
    # chunks are byte-identical to what's embedded -- this is what lets RRF's
    # SHA256(content) dedup key recognize the same chunk across both retrieval paths.
    documents = load_documents(source_dir)
    chunks = split_documents(documents)
    if not chunks:
        return None, chunks
    tokenized_corpus = [_tokenize(chunk.page_content) for chunk in chunks]
    return BM25Okapi(tokenized_corpus), chunks


def get_bm25_index(source_dir: Path | None = None) -> tuple[BM25Okapi | None, list]:
    settings = get_settings()
    resolved = str(source_dir or settings.corpus_dir)
    if resolved not in _index_cache:
        with _index_lock:
            if resolved not in _index_cache:
                _index_cache[resolved] = _build_index(Path(resolved))
    return _index_cache[resolved]


def bm25_search(sub_query: str, k: int = 4, source_dir: Path | None = None) -> list[RetrievedDoc]:
    bm25, chunks = get_bm25_index(source_dir)
    if bm25 is None:
        return []

    scores = bm25.get_scores(_tokenize(sub_query))
    ranked_indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]

    return [
        RetrievedDoc(
            content=chunks[i].page_content,
            metadata=chunks[i].metadata,
            source_id=chunks[i].metadata.get("source", ""),
            score=float(scores[i]),
        )
        for i in ranked_indices
        if scores[i] > 0  # no keyword overlap at all -- don't pad results with noise
    ]
