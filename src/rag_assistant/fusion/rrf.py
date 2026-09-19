"""Reciprocal Rank Fusion, with near-duplicate collapsing.

Fusion merges several independently-ranked lists, and the same passage routinely appears in
more than one of them -- a document retrieved by both vector and BM25 search, or a page whose
text the corpus already holds because someone uploaded it. Collapsing those is what stops one
passage from occupying several slots of the synthesis context budget and earning several
citation markers that all point at the same words.

Exact content hashing collapsed only byte-identical text, which is the case that matters least.
The local copy of a page and the web copy of it differ by a trailing newline, a "Read more"
suffix, a smart quote, or one sentence of boilerplate -- and then both survive, both are sent
to synthesis, and the answer cites [2] and [5] for one passage. So matching happens in two
tiers: a normalized hash catches the cosmetic differences, and shingle overlap catches the
substantive ones.

The second tier uses the **containment** (overlap) coefficient, `|A n B| / min(|A|, |B|)`,
rather than Jaccard. Measured over realistic pairs of a three-sentence passage:

    case                        jaccard   containment
    + "Read more."                0.933         1.000
    + a line of boilerplate       0.848         1.000
    + three lines of boilerplate  0.757         1.000
    one sentence dropped          0.321         1.000
    one extra sentence            0.718         1.000
    a different chunk, same doc   0.132         0.333
    same topic, different words   0.022         0.053
    an unrelated document         0.000         0.000

Jaccard punishes length differences, which is exactly what distinguishes a web copy from a
local one: a truncated copy of the same passage scores 0.321 and survives as a separate
document, while no threshold that catches it stays clear of genuinely distinct text. Under
containment every true near-duplicate lands at 1.000 and the nearest false positive at 0.333,
so the threshold sits in a gap rather than on a slope.

Containment's own hazard is that a short document contained in a long one matches it, and
keeping the short one would silently truncate the evidence. So the representative kept is the
one with *more* content -- text, metadata and source id together, never mixed, because citing
one source for another's words is worse than either copy alone.

Pairwise rather than MinHash or LSH. Those exist to make this sublinear over corpora; fusion
sees the candidates from a handful of sub-queries -- tens of documents, not millions -- so the
exact comparison is both cheaper in practice and free of approximation error. The cost is
quadratic in something that does not grow.
"""

import hashlib
import re

from rag_assistant.config import get_settings
from rag_assistant.schemas.models import FusedDocument, RetrievedDoc

_NON_WORD_RE = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE_RE = re.compile(r"\s+")

# Word count per shingle. 5 is long enough that shared stock phrases ("according to the
# company") do not by themselves make two documents look alike, and short enough that a
# single edited word does not destroy every shingle around it.
_SHINGLE_SIZE = 5

# Below this many shingles, Jaccard is too coarse to trust: a two-shingle document shares
# either none, half or all of them with another, so the threshold becomes a coin toss. Short
# documents are collapsed only on the exact normalized hash.
_MIN_SHINGLES_FOR_SIMILARITY = 4


def _exact_key(doc: RetrievedDoc) -> str:
    """Kept as the fast path and as the fallback for text too short to shingle."""
    return hashlib.sha256(doc.content.encode()).hexdigest()


def _normalize(text: str) -> str:
    """Lowercased, punctuation-stripped, whitespace-collapsed.

    This alone closes most of the gap the exact hash left: HTML-extracted text and the same
    passage from a markdown file differ far more often in punctuation and spacing than in
    words.
    """
    return _WHITESPACE_RE.sub(" ", _NON_WORD_RE.sub(" ", text.lower())).strip()


def _normalized_key(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode()).hexdigest()


def _shingles(text: str) -> frozenset[str]:
    words = _normalize(text).split()
    if len(words) < _SHINGLE_SIZE:
        return frozenset()
    return frozenset(
        " ".join(words[i : i + _SHINGLE_SIZE]) for i in range(len(words) - _SHINGLE_SIZE + 1)
    )


def _containment(left: frozenset[str], right: frozenset[str]) -> float:
    """`|A n B| / min(|A|, |B|)` -- how much of the smaller document the larger one contains."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / min(len(left), len(right))


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedDoc]], k: int = 60
) -> list[FusedDocument]:
    """Merges multiple independently-ranked retrieval lists (one per sub-query/source pair)
    into a single ranking. Each list votes for its documents by rank rather than by raw
    similarity score -- vector cosine distance and web-search relevance scores aren't comparable
    on the same scale, but rank position always is. `score = sum(1 / (k + rank))` across
    every list a document appears in, so a document ranked highly across several lists
    outranks one that's #1 in only a single list.

    Near-duplicates fuse into one entry whose score is the sum of its members', which is the
    same arithmetic an exact duplicate already received -- a passage found by three routes
    should outrank one found by a single route whether or not the three copies were
    byte-identical.
    """
    threshold = get_settings().fusion_near_duplicate_threshold

    scores: dict[str, float] = {}
    docs_by_key: dict[str, RetrievedDoc] = {}
    # Parallel to docs_by_key, so a near-duplicate check never re-shingles text.
    shingles_by_key: dict[str, frozenset[str]] = {}
    normalized_to_key: dict[str, str] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = _match_existing(doc, docs_by_key, shingles_by_key, normalized_to_key, threshold)
            if key is None:
                key = _exact_key(doc)
                docs_by_key[key] = doc
                shingles_by_key[key] = _shingles(doc.content)
                normalized_to_key[_normalized_key(doc.content)] = key
            elif len(doc.content) > len(docs_by_key[key].content):
                # Containment matches a short document against a long one that contains it,
                # so the merge must keep the longer text or it silently truncates the
                # evidence. Swapped wholesale -- content, metadata and source id -- because a
                # citation naming one source for another's words is worse than either copy.
                docs_by_key[key] = doc
                shingles_by_key[key] = _shingles(doc.content)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

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


def _match_existing(
    doc: RetrievedDoc,
    docs_by_key: dict[str, RetrievedDoc],
    shingles_by_key: dict[str, frozenset[str]],
    normalized_to_key: dict[str, str],
    threshold: float,
) -> str | None:
    """The key of an already-seen document this one duplicates, or None.

    First match wins, so cluster membership follows rank order; which member is *kept* as the
    representative is decided by the caller on content length, not by arrival.
    """
    exact = _exact_key(doc)
    if exact in docs_by_key:
        return exact

    normalized = _normalized_key(doc.content)
    if normalized in normalized_to_key:
        return normalized_to_key[normalized]

    if threshold <= 0:
        return None
    candidate_shingles = _shingles(doc.content)
    if len(candidate_shingles) < _MIN_SHINGLES_FOR_SIMILARITY:
        return None
    for key, existing in shingles_by_key.items():
        if len(existing) < _MIN_SHINGLES_FOR_SIMILARITY:
            continue
        if _containment(candidate_shingles, existing) >= threshold:
            return key
    return None
