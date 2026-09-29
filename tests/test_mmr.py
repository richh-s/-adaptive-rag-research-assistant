"""Diversity selection, and the retrieval settings that turn it on."""

import pytest

from rag_assistant.config import get_settings
from rag_assistant.retrieval.mmr import maximal_marginal_relevance

# A query, a document that answers it, a near-restatement of that document, and a document
# that is also relevant but covers different ground. The restatement is what MMR exists to
# skip: it is genuinely the second-most-similar text to the query and adds nothing.
QUERY = [1.0, 0.0, 0.0]
ANSWER = [1.0, 0.1, 0.0]
RESTATEMENT = [0.99, 0.11, 0.0]
DIFFERENT_GROUND = [0.8, 0.0, 0.6]
CANDIDATES = [ANSWER, RESTATEMENT, DIFFERENT_GROUND]


def test_lambda_one_is_plain_similarity_ranking():
    """The degenerate case has to stay degenerate: at 1.0 the redundancy term is weightless,
    so MMR must return exactly what similarity ordering would."""
    assert maximal_marginal_relevance(QUERY, CANDIDATES, k=2, lambda_mult=1.0) == [0, 1]


def test_diversity_skips_the_near_restatement():
    assert maximal_marginal_relevance(QUERY, CANDIDATES, k=2, lambda_mult=0.5) == [0, 2]


def test_the_first_pick_is_always_the_nearest_document():
    """Whatever lambda says, the first selection has nothing to be redundant with, so it is
    the most relevant candidate -- MMR trades relevance away at the margin, never at the top."""
    for lambda_mult in (0.0, 0.3, 0.5, 1.0):
        picked = maximal_marginal_relevance(QUERY, CANDIDATES, k=3, lambda_mult=lambda_mult)
        assert picked[0] == 0


def test_k_larger_than_the_corpus_returns_everything_once():
    picked = maximal_marginal_relevance(QUERY, CANDIDATES, k=10, lambda_mult=0.5)
    assert sorted(picked) == [0, 1, 2]


def test_degenerate_inputs_return_empty_rather_than_raising():
    assert maximal_marginal_relevance(QUERY, [], k=3) == []
    assert maximal_marginal_relevance(QUERY, CANDIDATES, k=0) == []


def test_a_zero_vector_scores_zero_instead_of_producing_nan():
    """A zero-length embedding has no direction. Normalising it by a zero norm would put NaN
    into the ranking, where it propagates silently rather than failing."""
    picked = maximal_marginal_relevance(QUERY, [[0.0, 0.0, 0.0], ANSWER], k=2, lambda_mult=0.5)
    assert picked[0] == 1


def test_selection_is_deterministic_under_ties():
    """Two identical candidates must always resolve the same way. A retriever that returns a
    different set on identical input makes every downstream eval score noise."""
    tied = [ANSWER, list(ANSWER), DIFFERENT_GROUND]
    runs = {tuple(maximal_marginal_relevance(QUERY, tied, k=2, lambda_mult=0.5)) for _ in range(5)}
    assert len(runs) == 1


@pytest.mark.parametrize(
    "env,expected",
    [({}, 4), ({"RETRIEVAL_K": "9"}, 9)],
)
def test_retrieval_k_is_configurable(monkeypatch, env, expected):
    """The number that most directly controls recall used to be a literal in the node."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    assert get_settings().retrieval_k == expected


def test_retrieval_nodes_read_k_from_settings(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_K", "7")
    get_settings.cache_clear()
    captured = {}

    class _Retriever:
        def invoke(self, query):
            return []

    def fake_get_retriever(k, owner, filters, principals=None):
        captured["k"] = k
        return _Retriever()

    monkeypatch.setattr("rag_assistant.graph.nodes.retrieve.get_retriever", fake_get_retriever)
    from rag_assistant.graph.nodes.retrieve import retrieve_vector

    retrieve_vector({"sub_query": "anything", "owner": "public"})
    assert captured["k"] == 7


def test_mmr_lambda_is_bounded():
    """Outside [0, 1] the formula stops being a trade-off and starts inverting one of its
    own terms, so the bound belongs in config rather than in a comment."""
    from pydantic import ValidationError

    from rag_assistant.config import Settings

    with pytest.raises(ValidationError):
        Settings(google_api_key="k", anthropic_api_key="", retrieval_mmr_lambda=1.5)
