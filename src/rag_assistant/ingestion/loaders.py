from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_documents(source_dir: Path) -> list[Document]:
    """Load every supported text file in source_dir into a Document, tagging each with its
    filename as metadata so citations can later point back to a source."""
    documents = []
    for path in sorted(source_dir.iterdir()):
        if path.suffix not in SUPPORTED_SUFFIXES:
            continue
        documents.append(
            Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        )
    return documents
