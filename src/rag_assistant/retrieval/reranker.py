"""Cross-encoder reranking of fused documents.

RRF ranks by *consensus*: a document that several retrieval paths agree on rises. That is a
good signal and a cheap one, but it is structurally blind to the question — it never compares
a document against the query, only against how other retrievers ranked it. A passage that all
three paths return for lexical reasons outranks the one passage that actually answers the
question, because the first one is popular and the second one is not.

A cross-encoder scores (query, document) pairs jointly, so it sees the interaction that
bi-encoder embeddings and rank fusion both miss. It is the standard fix, and it is expensive:
one forward pass per candidate, which is why it reranks a shortlist rather than the whole
fused set.

Off by default. `RERANKER=cohere` needs an API key and a network call; `RERANKER=cross_encoder`
needs `sentence-transformers`, which pulls in torch and several hundred megabytes. Both are
declared as optional extras and imported lazily, so a default install carries neither, and a
misconfigured one degrades to no reranking rather than failing a request.
"""

import logging
from functools import lru_cache

from rag_assistant.config import get_settings
from rag_assistant.schemas.models import FusedDocument

logger = logging.getLogger(__name__)


class Reranker:
    """Scores (query, document) pairs. Implementations return one score per document, in the
    order the documents were given."""

    def score(self, query: str, documents: list[str]) -> list[float]:  # pragma: no cover
        raise NotImplementedError


class CohereReranker(Reranker):
    """Cohere's hosted rerank endpoint. No local model, one network call per query."""

    def __init__(self, model: str, api_key: str):
        import cohere  # imported lazily: an optional extra, absent from a default install

        self._client = cohere.ClientV2(api_key=api_key)
        self._model = model

    def score(self, query: str, documents: list[str]) -> list[float]:
        response = self._client.rerank(model=self._model, query=query, documents=documents)
        # The API returns only the ranked subset, in relevance order -- scatter the scores
        # back onto the original positions so callers can zip them with their own list.
        scores = [0.0] * len(documents)
        for result in response.results:
            scores[result.index] = float(result.relevance_score)
        return scores


class CrossEncoderReranker(Reranker):
    """A local sentence-transformers cross-encoder. No API key, no per-query cost, but it
    loads a model into memory and wants a warm process."""

    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder  # lazy: optional extra

        self._model = CrossEncoder(model)

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self._model.predict([(query, doc) for doc in documents])]


@lru_cache
def get_reranker() -> Reranker | None:
    """The configured reranker, or None when reranking is off or unavailable.

    Cached because a local cross-encoder loads a model on construction and rebuilding it per
    request would dominate latency. Construction failures -- a missing extra, a bad key --
    return None and log once, so a misconfiguration costs ranking quality rather than
    availability.
    """
    settings = get_settings()
    choice = settings.reranker
    if choice == "none":
        return None
    try:
        if choice == "cohere":
            if not settings.cohere_api_key:
                logger.warning("RERANKER=cohere but COHERE_API_KEY is unset; reranking disabled")
                return None
            return CohereReranker(settings.cohere_rerank_model, settings.cohere_api_key)
        if choice == "cross_encoder":
            return CrossEncoderReranker(settings.cross_encoder_rerank_model)
    except Exception:
        logger.warning(
            "Reranker %r could not be constructed; continuing without reranking. Install the "
            "matching optional extra (`uv sync --extra rerank-cohere` / `--extra rerank-local`).",
            choice,
            exc_info=True,
        )
    return None


def reset_reranker_cache() -> None:
    get_reranker.cache_clear()


def rerank_documents(
    query: str, documents: list[FusedDocument], top_n: int | None = None
) -> list[FusedDocument]:
    """Reorders the top `top_n` fused documents by cross-encoder relevance.

    Only the shortlist is scored, and the untouched tail keeps its RRF order behind it. That
    bound is the whole reason this is affordable: reranking is per-pair work, so scoring
    everything fusion returned would scale cost with retrieval breadth rather than with how
    many documents synthesis can actually use.

    The rerank score is written to `rrf_score` so every downstream consumer -- grading,
    citation ordering, the context budget -- reads one ranking field rather than each having
    to know whether reranking happened to be enabled.
    """
    reranker = get_reranker()
    if reranker is None or not documents:
        return documents

    top_n = top_n or get_settings().rerank_top_n
    shortlist, tail = documents[:top_n], documents[top_n:]
    try:
        scores = reranker.score(query, [d.content for d in shortlist])
    except Exception:
        logger.warning("Reranking failed; falling back to fusion order", exc_info=True)
        return documents

    if len(scores) != len(shortlist):
        logger.warning(
            "Reranker returned %d scores for %d documents; keeping fusion order",
            len(scores),
            len(shortlist),
        )
        return documents

    reranked = [
        doc.model_copy(update={"rrf_score": score}) for doc, score in zip(shortlist, scores)
    ]
    reranked.sort(key=lambda d: d.rrf_score, reverse=True)
    return reranked + tail
