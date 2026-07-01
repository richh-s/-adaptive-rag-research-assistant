from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from rag_assistant.config import get_settings
from rag_assistant.llm import get_embeddings_model

COLLECTION_NAME = "research_corpus"


def get_vector_store(
    embeddings: Embeddings | None = None, persist_dir: Path | None = None
) -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings_model(),
        persist_directory=str(persist_dir or settings.chroma_persist_dir),
    )


def get_retriever(
    k: int = 4, embeddings: Embeddings | None = None, persist_dir: Path | None = None
) -> VectorStoreRetriever:
    store = get_vector_store(embeddings=embeddings, persist_dir=persist_dir)
    return store.as_retriever(search_kwargs={"k": k})
