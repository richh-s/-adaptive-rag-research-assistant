import logging
from pathlib import Path

import pymupdf
import pymupdf4llm
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from rag_assistant.ingestion import vision

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx", ".html", ".htm"}

logger = logging.getLogger(__name__)

# 2x zoom ≈ 144 dpi -- enough for legible OCR-style transcription without huge payloads.
_SCAN_RENDER_ZOOM = 2.0


def _describe_page_figures(doc: "pymupdf.Document", page_index: int, budget: list[int]) -> str:
    """Vision pass over one page's embedded images: returns "[Figure on page N: ...]" blocks
    to append to the page text, so charts/diagrams/photos become searchable. `budget` is a
    single-element mutable counter shared across the whole PDF (MAX_IMAGES_PER_PDF)."""
    blocks = []
    try:
        images = doc[page_index].get_images(full=True)
    except Exception:
        logger.warning("Could not enumerate images on page %d", page_index + 1, exc_info=True)
        return ""

    for image_info in images:
        if budget[0] <= 0:
            logger.info("Figure-description budget exhausted; remaining images skipped")
            break
        try:
            extracted = doc.extract_image(image_info[0])
        except Exception:
            continue
        if (
            extracted.get("width", 0) < vision.MIN_IMAGE_DIMENSION_PX
            or extracted.get("height", 0) < vision.MIN_IMAGE_DIMENSION_PX
        ):
            continue
        budget[0] -= 1
        description = vision.describe_image(
            extracted["image"], f"image/{extracted.get('ext', 'png')}", vision.FIGURE_PROMPT
        )
        if description:
            blocks.append(f"[Figure on page {page_index + 1}: {description}]")
    return "\n\n".join(blocks)


def _transcribe_scanned_page(doc: "pymupdf.Document", page_index: int) -> str:
    """A page with no text layer is a scan/photo: render it to PNG and let the vision model
    transcribe it -- same mechanism as figure description, no OCR dependency."""
    try:
        pixmap = doc[page_index].get_pixmap(matrix=pymupdf.Matrix(_SCAN_RENDER_ZOOM, _SCAN_RENDER_ZOOM))
        png_bytes = pixmap.tobytes("png")
    except Exception:
        logger.warning("Could not render page %d for transcription", page_index + 1, exc_info=True)
        return ""
    return vision.describe_image(png_bytes, "image/png", vision.SCANNED_PAGE_PROMPT) or ""


def _load_docx(path: Path) -> list[Document]:
    """Extract paragraphs and tables from a .docx in document order. Tables are flattened to
    one pipe-joined line per row -- crude, but it keeps cell values adjacent to their row
    context, which is what retrieval actually needs from a table."""
    import docx  # deferred: python-docx import is slow enough to matter at API startup

    try:
        document = docx.Document(str(path))
    except Exception:
        logger.warning("Skipping unreadable DOCX: %s", path.name, exc_info=True)
        return []

    parts: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    text = "\n\n".join(parts)
    if not text.strip():
        logger.warning("No extractable text in DOCX: %s", path.name)
        return []
    return [Document(page_content=text, metadata={"source": path.name})]


def extract_html_text(html: str) -> tuple[str | None, str]:
    """Shared by the .html file loader and URL ingestion: returns (title, visible text).
    Boilerplate tags (nav/script/style/etc.) are dropped wholesale rather than running a
    full readability pass -- good enough for docs/articles, and dependency-free beyond bs4."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "nav", "footer", "header", "aside"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
    text = "\n".join(line for line in lines if line)
    return title, text


def _load_html(path: Path) -> list[Document]:
    try:
        title, text = extract_html_text(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        logger.warning("Skipping unreadable HTML: %s", path.name, exc_info=True)
        return []

    if not text.strip():
        logger.warning("No extractable text in HTML: %s", path.name)
        return []
    metadata: dict = {"source": path.name}
    if title:
        metadata["title"] = title
    return [Document(page_content=text, metadata=metadata)]


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

    # One fitz handle for the vision passes (figure extraction, scanned-page rendering).
    # Opened lazily and only when vision is on -- plain text-only ingestion never pays for it.
    vision_doc: pymupdf.Document | None = None
    if vision.vision_available():
        try:
            vision_doc = pymupdf.open(str(path))
        except Exception:
            logger.warning("Could not reopen %s for vision passes", path.name, exc_info=True)
    figure_budget = [vision.MAX_IMAGES_PER_PDF]

    documents = []
    for page_index, page in enumerate(pages):
        text = (page.get("text") or "").strip()
        page_number = page.get("metadata", {}).get("page_number")

        if vision_doc is not None:
            if not text:
                # No text layer: a scanned/photographed page -- transcribe it.
                text = _transcribe_scanned_page(vision_doc, page_index).strip()
            figures = _describe_page_figures(vision_doc, page_index, figure_budget)
            if figures:
                text = f"{text}\n\n{figures}".strip()

        if not text:
            continue
        documents.append(
            Document(page_content=text, metadata={"source": path.name, "page": page_number})
        )

    if vision_doc is not None:
        vision_doc.close()

    if not documents:
        # Image-only PDF and vision disabled/unavailable -- nothing to index.
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
        elif path.suffix == ".docx":
            documents.extend(_load_docx(path))
        elif path.suffix in (".html", ".htm"):
            documents.extend(_load_html(path))
        else:
            documents.append(
                Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
            )
    return documents
