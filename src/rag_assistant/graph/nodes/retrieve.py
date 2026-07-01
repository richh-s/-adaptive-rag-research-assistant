from rag_assistant.retrieval.vector_store import get_retriever
from rag_assistant.schemas.models import RetrievedDoc, SubQueryResult


def retrieve_vector(state: dict) -> dict:
    """Local vector-store retrieval for one sub-query. Invoked once per sub-query via
    `Send`, so `state` here is just `{"sub_query": str}`, not the full graph state."""
    sub_query = state["sub_query"]
    docs = get_retriever(k=4).invoke(sub_query)
    retrieved = [
        RetrievedDoc(
            content=doc.page_content,
            metadata=doc.metadata,
            source_id=doc.metadata.get("source", ""),
        )
        for doc in docs
    ]
    return {"vector_results": [SubQueryResult(sub_query=sub_query, docs=retrieved)]}
