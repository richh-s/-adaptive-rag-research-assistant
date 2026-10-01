"""The corrective loop's first escalation: try the corpus again, differently.

Corrective-RAG had exactly one move here -- run a web search -- and it was armed only for the
`vector` route. That encodes an assumption the confidence score cannot actually support: that
a low grade means *the corpus does not contain the answer*. The other possibility is that the
corpus contains it and the query missed it, which is the ordinary failure of dense retrieval
against a question phrased the way a person asks rather than the way a document writes. For
that case, leaving the corpus is the one thing that cannot help.

So the escalation is ordered by what is likeliest and cheapest:

1. **Refine and re-retrieve locally** (this node). Rewrite the queries into the vocabulary the
   documents would use and widen `k`, then search the same corpus again. Costs one structured
   call plus local retrieval -- no external dependency, no network egress, and it is the only
   step that can recover an answer the corpus actually holds.
2. **Corrective web search**, unchanged, for when the corpus genuinely cannot answer.
3. **Answer anyway**, with the synthesis prompt's abstention instructions doing their job.

The `both` and `web` routes previously got no correction at all -- `grade_and_score` armed the
fallback only for `vector`, on the reasoning that a run which already searched the web has
nothing left to escalate to. True of the web, false of the corpus: a `both` run whose local
half retrieved badly can still be re-asked locally, and now is.

Widening `k` alongside the rewrite is deliberate. The two failures are different -- the right
document ranked below the cutoff, versus the right document not matching the query at all --
and a second attempt is expensive enough that addressing only one of them would leave the
other to a third pass nobody is going to pay for.
"""

import asyncio
import logging

from rag_assistant.auth import PUBLIC_OWNER
from rag_assistant.config import get_settings
from rag_assistant.graph.state import ResearchState, access_principals
from rag_assistant.llm import get_structured_llm
from rag_assistant.prompts.refine_prompt import REFINE_PROMPT
from rag_assistant.retrieval.bm25_store import bm25_search
from rag_assistant.retrieval.vector_store import get_retriever
from rag_assistant.schemas.models import RefinedQueries, RetrievedDoc, SubQueryResult

logger = logging.getLogger(__name__)


async def _rewrite(question: str, sub_queries: list[str]) -> list[str]:
    """Reformulated queries, or the originals when rewriting fails.

    Degrading to the originals rather than to nothing keeps the widened-`k` half of the
    retry working: a provider hiccup should cost the refinement its rewrite, not the whole
    second attempt.
    """
    llm = get_structured_llm(RefinedQueries)
    try:
        refined: RefinedQueries = await llm.ainvoke(
            REFINE_PROMPT.format(
                question=question,
                sub_queries="\n".join(f"- {q}" for q in sub_queries),
            )
        )
    except Exception:
        logger.warning("Query refinement failed; retrying with the original queries", exc_info=True)
        return sub_queries
    rewritten = [q.strip() for q in refined.sub_queries if q and q.strip()]
    return rewritten or sub_queries


async def refine_retrieval(state: ResearchState) -> dict:
    """Re-retrieves locally with rewritten queries and a widened `k`.

    Writes into `vector_results`/`bm25_results`, which carry an `operator.add` reducer, so the
    second attempt's documents join the first attempt's rather than replacing them -- fusion
    then ranks across both, and a document both attempts found gets the rank consensus it
    earned. Discarding the first attempt would throw away the evidence that it was graded
    against.
    """
    settings = get_settings()
    question = state["question"]
    original = state.get("sub_queries") or [question]
    queries = await _rewrite(question, original)

    owner = state.get("owner") or PUBLIC_OWNER
    principals = access_principals(state)
    filters = state.get("filters")
    k = max(settings.retrieval_k * settings.refine_k_multiplier, settings.retrieval_k)

    # Retrieval is blocking library work -- Chroma or psycopg, and the in-memory keyword
    # index -- so it runs in a worker thread rather than on the event loop. The ordinary
    # retrieval nodes get this for free by staying synchronous and letting LangGraph schedule
    # them; this node cannot, because it also awaits an LLM call, so it has to hand the
    # blocking half off itself. Without that, one refinement would stall every other request
    # sharing the loop for the duration of two searches per rewritten query.
    def _retrieve_all() -> tuple[list[SubQueryResult], list[SubQueryResult]]:
        vectors: list[SubQueryResult] = []
        keywords: list[SubQueryResult] = []
        for query in queries:
            docs = get_retriever(k=k, owner=owner, filters=filters, principals=principals).invoke(
                query
            )
            vectors.append(
                SubQueryResult(
                    sub_query=query,
                    docs=[
                        RetrievedDoc(
                            content=doc.page_content,
                            metadata=doc.metadata,
                            source_id=doc.metadata.get("source", ""),
                        )
                        for doc in docs
                    ],
                )
            )
            keywords.append(
                SubQueryResult(
                    sub_query=query,
                    docs=bm25_search(
                        query, k=k, owner=owner, filters=filters, principals=principals
                    ),
                )
            )
        return vectors, keywords

    vector_results, bm25_results = await asyncio.to_thread(_retrieve_all)

    logger.info(
        "retrieval refined after low confidence",
        extra={"trace_id": state.get("trace_id"), "refined_queries": queries, "widened_k": k},
    )
    return {
        "vector_results": vector_results,
        "bm25_results": bm25_results,
        "refined_sub_queries": queries,
        "refinement_attempted": True,
    }
