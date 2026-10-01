from rag_assistant.auth import PUBLIC_OWNER
from rag_assistant.config import get_settings
from rag_assistant.graph.state import access_principals
from rag_assistant.retrieval.bm25_store import bm25_search
from rag_assistant.retrieval.vector_store import get_retriever
from rag_assistant.schemas.models import RetrievedDoc, SubQueryResult


def retrieve_vector(state: dict) -> dict:
    """Local vector-store retrieval for one sub-query. Invoked once per sub-query via
    `Send`, so `state` here is just `{"sub_query": str, "owner": str}`, not the full graph
    state -- see `dispatch_retrieval` for why the owner has to be copied into the payload."""
    sub_query = state["sub_query"]
    docs = get_retriever(
        k=get_settings().retrieval_k,
        owner=state.get("owner", PUBLIC_OWNER),
        filters=state.get("filters"),
        principals=access_principals(state),
    ).invoke(sub_query)
    retrieved = [
        RetrievedDoc(
            content=doc.page_content,
            metadata=doc.metadata,
            source_id=doc.metadata.get("source", ""),
        )
        for doc in docs
    ]
    return {"vector_results": [SubQueryResult(sub_query=sub_query, docs=retrieved)]}


def retrieve_bm25(state: dict) -> dict:
    """Local BM25 keyword retrieval for one sub-query, mirroring retrieve_vector's Send-based
    shape. Complements dense vector search with exact keyword matching (names, acronyms) that
    embedding similarity sometimes under-ranks."""
    sub_query = state["sub_query"]
    docs = bm25_search(
        sub_query,
        k=get_settings().retrieval_k,
        owner=state.get("owner", PUBLIC_OWNER),
        filters=state.get("filters"),
        principals=access_principals(state),
    )
    return {"bm25_results": [SubQueryResult(sub_query=sub_query, docs=docs)]}
