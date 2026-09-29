from rag_assistant.config import get_settings
from rag_assistant.grading.relevance_grader import grade_documents
from rag_assistant.graph.state import ResearchState

TOP_N_TO_GRADE = 6


async def grade_and_score(state: ResearchState) -> dict:
    """Corrective-RAG grading: judges the top fused documents' relevance, then aggregates
    into a single confidence_score. Correction only triggers when the route was vector-only
    and hasn't already been attempted -- if we already tried the web (route "web"/"both"),
    there's no further fallback to reach for."""
    docs = state.get("fused_documents", [])[:TOP_N_TO_GRADE]
    grades = await grade_documents(state["question"], docs)

    # Average only over docs graded relevant, not every graded doc. The corpus has several
    # similarly-structured company profiles, so fusion often pulls in a couple of off-topic
    # chunks (e.g. another company's "safety focus" section) alongside the right ones --
    # averaging those low scores in with a genuinely strong match drags confidence below
    # threshold and falsely triggers corrective_web_search even when the local docs already
    # answer the question. What matters is the quality of what's actually relevant.
    relevant_scores = [g.score for g in grades if g.relevant]
    confidence = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0
    action = _choose_correction(state, confidence)
    needs_correction = action is not None
    result = {
        "doc_grades": grades,
        "confidence_score": confidence,
        "needs_correction": needs_correction,
        "correction_action": action,
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


def _choose_correction(state: ResearchState, confidence: float) -> str | None:
    """Which escalation a low-confidence result earns, or None to answer as-is.

    Ordered by what is likeliest and cheapest. A low grade is evidence that retrieval did not
    find the answer; it is *not* evidence about why. The previous rule -- web search, vector
    route only -- silently assumed the corpus could not contain the answer, when the ordinary
    cause is a question phrased the way a person asks rather than the way a document writes.
    So the corpus gets re-asked first, and only then does the pipeline leave it.

    The `both` and `web` routes used to get no correction at all, on the reasoning that a run
    which already searched the web has nothing left to escalate to. That is true of the web
    and false of the corpus: a `both` run whose local half retrieved badly can still be
    re-asked locally. `web` alone still gets nothing, because there is no local half to
    re-ask.
    """
    if confidence >= get_settings().confidence_threshold:
        return None
    route = state.get("route")
    if route in ("vector", "both") and not state.get("refinement_attempted", False):
        return "refine"
    if route == "vector" and not state.get("correction_attempted", False):
        return "web"
    return None


_CORRECTION_NODES = {"refine": "refine_retrieval", "web": "corrective_web_search"}


def after_grade(state: ResearchState) -> str:
    """Conditional edge function: loop back for one refinement or one corrective web search
    pass, or proceed to synthesis.

    Reads `correction_action` when present and falls back to the older boolean, so a state
    assembled without the newer key still routes somewhere sensible rather than silently
    skipping a correction it asked for.
    """
    action = state.get("correction_action")
    if action:
        return _CORRECTION_NODES[action]
    return "corrective_web_search" if state.get("needs_correction") else "synthesize_answer"
