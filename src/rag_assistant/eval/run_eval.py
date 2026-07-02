from dataclasses import dataclass
from pathlib import Path

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.evaluation import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    ResponseRelevancy,
)

from rag_assistant.eval.golden_dataset import load_golden_dataset
from rag_assistant.graph.build_graph import build_graph
from rag_assistant.llm import get_chat_model, get_embeddings_model
from rag_assistant.schemas.models import GoldenQuestion

_RECURSION_LIMIT = 50


@dataclass
class QuestionResult:
    """One golden question's graph output plus the free, non-LLM route/source checks --
    computed independently of whatever RAGAS metrics run alongside it."""

    question: str
    expected_route: str
    actual_route: str | None
    route_match: bool
    expected_sources: list[str]
    actual_sources: list[str]
    source_overlap: bool
    response: str
    retrieved_contexts: list[str]
    reference_contexts: list[str]
    reference: str


def _run_question(graph, golden_question: GoldenQuestion) -> QuestionResult:
    result = graph.invoke(
        {"question": golden_question.question}, config={"recursion_limit": _RECURSION_LIMIT}
    )
    actual_sources = [d.source_id for d in result.get("fused_documents", [])]
    return QuestionResult(
        question=golden_question.question,
        expected_route=golden_question.expected_route,
        actual_route=result.get("route"),
        route_match=result.get("route") == golden_question.expected_route,
        expected_sources=golden_question.expected_sources,
        actual_sources=actual_sources,
        source_overlap=bool(set(golden_question.expected_sources) & set(actual_sources)),
        response=result.get("final_answer", ""),
        retrieved_contexts=[d.content for d in result.get("fused_documents", [])],
        reference_contexts=golden_question.reference_contexts,
        reference=golden_question.ground_truth,
    )


def run_eval(
    limit: int = 3, llm_judge: bool = False, dataset_path: Path | None = None
) -> tuple[list[QuestionResult], EvaluationResult]:
    """Runs the graph on up to `limit` golden questions, then scores the results with RAGAS.
    Graph execution (~4 Gemini calls/question) dominates cost -- `limit` is the primary quota
    lever, independent of `llm_judge` which only gates the *scoring* metrics' extra calls."""
    questions = load_golden_dataset(dataset_path)[:limit]
    graph = build_graph()
    results = [_run_question(graph, q) for q in questions]

    samples = [
        SingleTurnSample(
            user_input=r.question,
            response=r.response,
            retrieved_contexts=r.retrieved_contexts,
            reference_contexts=r.reference_contexts,
            reference=r.reference,
        )
        for r in results
    ]
    dataset = EvaluationDataset(samples=samples)

    metrics = [NonLLMContextPrecisionWithReference(), NonLLMContextRecall()]
    llm = None
    embeddings = None
    if llm_judge:
        llm = LangchainLLMWrapper(get_chat_model())
        embeddings = LangchainEmbeddingsWrapper(get_embeddings_model())
        metrics += [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=embeddings)]

    eval_result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    return results, eval_result
