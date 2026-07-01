from pydantic import BaseModel


class RetrievedDoc(BaseModel):
    """Normalized shape for a single retrieved piece of content, whether it came from the
    local vector store or a web search — lets downstream nodes (fusion, grading, synthesis)
    treat both sources uniformly."""

    content: str
    metadata: dict = {}
    source_id: str
    score: float | None = None
