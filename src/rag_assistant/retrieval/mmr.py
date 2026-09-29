"""Maximal Marginal Relevance: trading a little relevance for coverage.

Dense retrieval ranks every candidate against the query independently, which means the top-k
is free to be k paraphrases of one passage. On this corpus that is not hypothetical -- the
documents are structurally similar profiles, and a question about one company's safety work
pulls back that company's safety section, the neighbouring chunk of the same section, and the
same claim restated in its overview. All three are genuinely the most similar text to the
query, and together they answer less of the question than three documents that disagreed
about what to say.

Fusion's near-duplicate collapsing (see fusion/rrf.py) does not help here. It merges passages
that are *the same text*; these are different texts making the same point, which is precisely
the case containment scores at 0.333 and correctly leaves alone.

MMR picks greedily. The first pick is the nearest document to the query; each later pick
maximises

    lambda * sim(query, d) - (1 - lambda) * max sim(d, already_picked)

so a candidate has to be both relevant and unlike what is already selected. `lambda_mult` is
the dial: 1.0 is plain similarity ranking, 0.0 ignores the query entirely and returns the
most mutually dissimilar set it can find. The useful range is narrow -- around 0.5 the
selection stays anchored to the question while refusing the third restatement.

One implementation, used by both vector backends. Chroma and pgvector each ship (or could
ship) their own, and this codebase already asserts the two backends rank identically on the
same corpus -- a property that would quietly stop holding the moment diversity selection came
from two different pieces of code with two different tie-breaking rules.

Cosine throughout, matching the space both backends store vectors in.
"""

import numpy as np


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between two stacks of row vectors.

    Zero-length vectors norm to 1 rather than 0 so the division is defined: an all-zero
    embedding has no direction, and scoring it 0.0 against everything (which is what a norm
    of 1 produces for it) is the honest answer. Real embedding models do not emit them, but a
    test fixture can, and a NaN here would propagate silently into the ranking.
    """
    left_norms = np.linalg.norm(left, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right, axis=1, keepdims=True)
    left_norms[left_norms == 0] = 1.0
    right_norms[right_norms == 0] = 1.0
    return (left / left_norms) @ (right / right_norms).T


def maximal_marginal_relevance(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    k: int,
    lambda_mult: float = 0.5,
) -> list[int]:
    """Indices of the `k` candidates MMR selects, in selection order.

    Returns indices rather than documents so the caller keeps whatever it was carrying
    alongside the vectors -- content, metadata, source id -- without this module needing to
    know the shape of any of it.

    Ties break on the earlier candidate, which is the more similar one because candidates
    arrive in similarity order. That makes the selection deterministic, which matters more
    than which of two equally-good documents wins: a retriever that returns a different set
    on identical input turns every downstream eval score into noise.
    """
    if not candidate_embeddings or k <= 0:
        return []
    candidates = np.array(candidate_embeddings, dtype=float)
    query = np.array([query_embedding], dtype=float)

    to_query = _cosine_matrix(query, candidates)[0]
    # Precomputed in full: the greedy loop otherwise recomputes a candidate's similarity to
    # every selected document on every round, which is the same O(n^2) work spread out and
    # repeated. `fetch_k` is tens of documents, so the matrix is small either way.
    between = _cosine_matrix(candidates, candidates)

    selected = [int(np.argmax(to_query))]
    # Running max similarity from each candidate to the selected set, updated per pick rather
    # than re-derived -- the only state the greedy loop actually needs.
    redundancy = between[selected[0]].copy()

    while len(selected) < min(k, len(candidates)):
        scores = lambda_mult * to_query - (1.0 - lambda_mult) * redundancy
        scores[selected] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            break
        selected.append(best)
        redundancy = np.maximum(redundancy, between[best])
    return selected
