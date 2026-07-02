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

    confidence = sum(g.score for g in grades) / len(grades) if grades else 0.0
    needs_correction = (
        confidence < get_settings().confidence_threshold
        and state["route"] == "vector"
        and not state.get("correction_attempted", False)
    )
    return {
        "doc_grades": grades,
        "confidence_score": confidence,
        "needs_correction": needs_correction,
    }


def after_grade(state: ResearchState) -> str:
    """Conditional edge function: loop back for one corrective web search pass, or
    proceed to synthesis."""
    return "corrective_web_search" if state["needs_correction"] else "synthesize_answer"
