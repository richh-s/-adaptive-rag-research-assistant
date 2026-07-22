import hashlib

from rag_assistant.schemas.models import FusedDocument, RetrievedDoc


def _dedup_key(doc: RetrievedDoc) -> str:
    return hashlib.sha256(doc.content.encode()).hexdigest()


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedDoc]], k: int = 60
) -> list[FusedDocument]:
    """Merges multiple independently-ranked retrieval lists (one per sub-query/source pair)
    into a single ranking. Each list votes for its documents by rank rather than by raw
    similarity score -- vector cosine distance and web-search relevance scores aren't comparable
    on the same scale, but rank position always is. `score = sum(1 / (k + rank))` across
    every list a document appears in, so a document ranked highly across several lists
    outranks one that's #1 in only a single list."""
    scores: dict[str, float] = {}
    docs_by_key: dict[str, RetrievedDoc] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = _dedup_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs_by_key.setdefault(key, doc)

    fused = [
        FusedDocument(
            content=docs_by_key[key].content,
            metadata=docs_by_key[key].metadata,
            source_id=docs_by_key[key].source_id,
            rrf_score=score,
        )
        for key, score in scores.items()
    ]
    fused.sort(key=lambda d: d.rrf_score, reverse=True)
    return fused
