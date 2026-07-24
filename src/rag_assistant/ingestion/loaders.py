import logging
from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}

logger = logging.getLogger(__name__)


def _load_pdf(path: Path) -> list[Document]:
    """Extract text natively as Markdown (preserves headings/tables/lists far better than
    raw text extraction) with one Document per page, so citations can point at an exact
    page instead of just the file. A PDF therefore contributes *multiple* Documents sharing
    one `source` -- callers must group by `metadata["source"]` rather than assume one
    Document per file, which held for every format before PDFs were added."""
    try:
        pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    except Exception:
        # Corrupt/encrypted/unreadable PDF -- skip rather than crash the whole ingestion run.
        logger.warning("Skipping unreadable PDF: %s", path.name, exc_info=True)
        return []

    documents = []
    for page in pages:
        text = (page.get("text") or "").strip()
        if not text:
            continue
        page_number = page.get("metadata", {}).get("page_number")
        documents.append(
            Document(page_content=text, metadata={"source": path.name, "page": page_number})
        )

    if not documents:
        # Scanned/image-only PDF with no extractable text layer -- nothing to index.
        logger.warning("No extractable text in PDF: %s", path.name)
    return documents


def load_documents(source_dir: Path) -> list[Document]:
    """Load every supported file in source_dir into one or more Documents, tagging each with
    its filename (and, for PDFs, page number) as metadata so citations can point back to a
    source. Returns a flat list -- a multi-page PDF contributes multiple entries sharing the
    same `metadata["source"]`."""
    documents = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in SUPPORTED_SUFFIXES:
            logger.warning("Skipping unsupported file: %s", path.name)
            continue

        if path.suffix == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            documents.append(
                Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
            )
    return documents
