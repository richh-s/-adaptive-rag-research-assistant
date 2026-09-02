from rag_assistant.config import get_settings
from rag_assistant.grading.relevance_grader import grade_documents
from rag_assistant.graph.state import ResearchState

TOP_N_TO_GRADE = 6


def grade_and_score(state: ResearchState) -> dict:
    """Corrective-RAG grading: judges the top fused documents' relevance, then aggregates
    into a single confidence_score. Correction only triggers when the route was vector-only
    and hasn't already been attempted -- if we already tried the web (route "web"/"both"),
    there's no further fallback to reach for."""
    docs = state.get("fused_documents", [])[:TOP_N_TO_GRADE]
    grades = grade_documents(state["question"], docs)

    # Average only over docs graded relevant, not every graded doc. The corpus has several
    # similarly-structured company profiles, so fusion often pulls in a couple of off-topic
    # chunks (e.g. another company's "safety focus" section) alongside the right ones --
    # averaging those low scores in with a genuinely strong match drags confidence below
    # threshold and falsely triggers corrective_web_search even when the local docs already
    # answer the question. What matters is the quality of what's actually relevant.
    relevant_scores = [g.score for g in grades if g.relevant]
    confidence = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0
    needs_correction = (
        confidence < get_settings().confidence_threshold
        and state["route"] == "vector"
        and not state.get("correction_attempted", False)
    )
    result = {
        "doc_grades": grades,
        "confidence_score": confidence,
        "needs_correction": needs_correction,
    }

    # Grade-informed rerank/prune: the grades were paid for to compute confidence, so reuse
    # them to clean up the synthesis context for free -- graded-relevant docs move to the
    # front ordered by grade score (a semantic judgment, sharper than RRF's rank-consensus),
    # graded-irrelevant docs are dropped so they can't pollute the answer or earn a citation,
    # and ungraded tail docs keep their RRF order behind the graded ones. Skipped when
    # nothing was graded relevant (dropping everything would leave synthesis with an empty
    # context that reads as "retrieval found nothing") and when a corrective pass is about to
    # rerun fuse_results and overwrite fused_documents anyway.
    all_docs = state.get("fused_documents", [])
    if relevant_scores and not needs_correction:
        graded_pairs = [(doc, g) for doc, g in zip(all_docs[: len(grades)], grades) if g.relevant]
        graded_pairs.sort(key=lambda pair: pair[1].score, reverse=True)
        result["fused_documents"] = [doc for doc, _ in graded_pairs] + all_docs[len(grades) :]

    return result


def after_grade(state: ResearchState) -> str:
    """Conditional edge function: loop back for one corrective web search pass, or
    proceed to synthesis."""
    return "corrective_web_search" if state["needs_correction"] else "synthesize_answer"
