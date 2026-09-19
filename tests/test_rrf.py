import pytest

from rag_assistant.fusion.rrf import reciprocal_rank_fusion
from rag_assistant.schemas.models import RetrievedDoc


def _doc(content: str, source_id: str = "") -> RetrievedDoc:
    return RetrievedDoc(content=content, source_id=source_id)


def test_document_ranked_in_multiple_lists_outranks_single_list_top_result():
    shared = _doc("shared content", "shared")
    solo = _doc("solo content", "solo")

    fused = reciprocal_rank_fusion([[shared, solo], [shared]])

    assert fused[0].content == "shared content"
    assert fused[0].rrf_score > fused[1].rrf_score


def test_deduplicates_identical_content_across_lists():
    doc = _doc("same text", "a")

    fused = reciprocal_rank_fusion([[doc], [doc], [doc]])

    assert len(fused) == 1
    assert fused[0].rrf_score == pytest.approx(3 * (1 / 61))


def test_empty_input_produces_no_fused_documents():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_score_formula_matches_reciprocal_rank_sum():
    fused = reciprocal_rank_fusion([[_doc("a"), _doc("b")]], k=60)

    scores = {d.content: d.rrf_score for d in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)


def test_results_sorted_descending_by_score():
    fused = reciprocal_rank_fusion([[_doc("first"), _doc("second"), _doc("third")]])

    assert [d.rrf_score for d in fused] == sorted([d.rrf_score for d in fused], reverse=True)


def test_document_ranked_in_three_lists_outranks_document_in_one():
    consensus = _doc("consensus content", "consensus")
    solo = _doc("solo content", "solo")

    fused = reciprocal_rank_fusion([[consensus], [consensus, solo], [consensus]])

    assert fused[0].content == "consensus content"
    assert fused[0].rrf_score == pytest.approx(3 * (1 / 61))
    assert fused[1].content == "solo content"
    assert fused[0].rrf_score > fused[1].rrf_score


# ---- near-duplicate collapsing ----


def _doc(content: str, source_id: str = "a.md"):
    from rag_assistant.schemas.models import RetrievedDoc

    return RetrievedDoc(content=content, metadata={"source": source_id}, source_id=source_id)


PASSAGE = (
    "Anthropic was founded in 2021 by Dario Amodei and six colleagues from OpenAI. "
    "The company publishes research on interpretability and Constitutional AI, and has "
    "raised several billion dollars across successive funding rounds."
)


def test_cosmetically_different_copies_collapse():
    """The case exact hashing missed: HTML-extracted text and the markdown copy of the same
    passage differ in punctuation and spacing far more often than in words."""
    from rag_assistant.fusion.rrf import reciprocal_rank_fusion

    web_copy = PASSAGE.replace("2021", "2021,").replace(". ", ".  ").upper()
    fused = reciprocal_rank_fusion([[_doc(PASSAGE, "local.md")], [_doc(web_copy, "example.com")]])

    assert len(fused) == 1
    # Scores sum, exactly as they did for an exact duplicate: a passage found by two routes
    # should outrank one found by a single route regardless of byte-level differences.
    assert fused[0].rrf_score == pytest.approx(2 / 61)


def test_a_copy_with_extra_boilerplate_collapses():
    from rag_assistant.fusion.rrf import reciprocal_rank_fusion

    web_copy = PASSAGE + " Read more. Share this article. Subscribe to our newsletter."
    fused = reciprocal_rank_fusion([[_doc(PASSAGE, "local.md")], [_doc(web_copy, "example.com")]])

    assert len(fused) == 1
    # The boilerplate copy is longer, so it is the one kept -- keeping the shorter would
    # discard text on the strength of a similarity judgement.
    assert fused[0].content == web_copy


def test_the_fullest_copy_is_the_one_kept():
    """Containment matches a short document against a long one containing it, so the merge
    has to keep the longer text or it silently truncates the evidence -- even when the
    shorter copy ranked higher."""
    from rag_assistant.fusion.rrf import reciprocal_rank_fusion

    truncated = PASSAGE.rsplit(". ", 1)[0] + "."
    fused = reciprocal_rank_fusion([[_doc(truncated, "local.md")], [_doc(PASSAGE, "example.com")]])

    assert len(fused) == 1
    assert fused[0].content == PASSAGE


def test_content_and_source_are_never_taken_from_different_copies():
    """A citation naming one source for another's words is worse than keeping both copies."""
    from rag_assistant.fusion.rrf import reciprocal_rank_fusion

    longer = PASSAGE + " It is headquartered in San Francisco."
    fused = reciprocal_rank_fusion([[_doc(PASSAGE, "local.md")], [_doc(longer, "example.com")]])

    assert len(fused) == 1
    assert fused[0].content == longer
    assert fused[0].source_id == "example.com"
    assert fused[0].metadata["source"] == "example.com"


def test_a_truncated_copy_collapses():
    """The case Jaccard could not reach: a copy missing a sentence scores 0.321 by Jaccard
    and 1.000 by containment (see fusion/rrf.py)."""
    from rag_assistant.fusion.rrf import reciprocal_rank_fusion

    truncated = PASSAGE.rsplit(". ", 1)[0] + "."
    fused = reciprocal_rank_fusion([[_doc(PASSAGE, "a.md")], [_doc(truncated, "b.md")]])

    assert len(fused) == 1


def test_a_different_chunk_of_the_same_document_is_not_collapsed():
    """The nearest real false positive, measured at 0.333 containment against a 0.9
    threshold. Collapsing it would silently delete evidence from the answer."""
    from rag_assistant.fusion.rrf import reciprocal_rank_fusion

    other_chunk = (
        "The company publishes research on interpretability and Constitutional AI. "
        "Separately, it operates a commercial API business serving enterprise customers "
        "across several regulated industries."
    )
    fused = reciprocal_rank_fusion([[_doc(PASSAGE, "a.md")], [_doc(other_chunk, "a.md")]])

    assert len(fused) == 2


def test_genuinely_different_documents_are_not_collapsed():
    """The failure that would matter more than the one being fixed: collapsing distinct
    passages silently deletes evidence from the answer."""
    from rag_assistant.fusion.rrf import reciprocal_rank_fusion

    other = (
        "Mistral AI is a French company founded in Paris in 2023. It publishes open-weight "
        "models including Mixtral, and emphasizes European AI sovereignty in its positioning."
    )
    fused = reciprocal_rank_fusion([[_doc(PASSAGE, "a.md")], [_doc(other, "b.md")]])

    assert len(fused) == 2


def test_documents_sharing_only_stock_phrasing_are_not_collapsed():
    from rag_assistant.fusion.rrf import reciprocal_rank_fusion

    first = "According to the company, the results were strong. Revenue grew in every region."
    second = "According to the company, the outlook is poor. Headcount fell in every region."
    fused = reciprocal_rank_fusion([[_doc(first, "a.md")], [_doc(second, "b.md")]])

    assert len(fused) == 2


def test_short_documents_fall_back_to_exact_matching(monkeypatch):
    """Below a handful of shingles Jaccard is a coin toss -- two three-word snippets share
    none, half or all of them -- so short text is collapsed only on the normalized hash."""
    from rag_assistant.fusion.rrf import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(
        [[_doc("Revenue grew.", "a.md")], [_doc("Revenue fell.", "b.md")]]
    )

    assert len(fused) == 2


def test_setting_the_threshold_to_zero_disables_similarity_matching(monkeypatch):
    from rag_assistant.config import get_settings
    from rag_assistant.fusion.rrf import reciprocal_rank_fusion

    monkeypatch.setenv("FUSION_NEAR_DUPLICATE_THRESHOLD", "0")
    get_settings.cache_clear()

    web_copy = PASSAGE + " Read more."
    fused = reciprocal_rank_fusion([[_doc(PASSAGE, "local.md")], [_doc(web_copy, "example.com")]])

    assert len(fused) == 2
    # Normalized hashing still applies -- it is the cheap tier, not the similarity one.
    same = reciprocal_rank_fusion([[_doc(PASSAGE, "a.md")], [_doc(PASSAGE.upper(), "b.md")]])
    assert len(same) == 1
