"""Chunking.

Fixed-size splitting cuts wherever the character count runs out, which on real documents
lands mid-sentence and mid-table, and strips a chunk of the one thing that tells you what it
is about -- the heading above it. A chunk reading "It raised $450M in a Series C" is nearly
useless to both retrieval paths: the embedding has no company in it, and BM25 has no company
token to match.

So this splits on document structure first (markdown headings, which is what the .md corpus
and pymupdf4llm's PDF output both use), then fixed-size *within* each section, and prepends
the heading breadcrumb to every chunk. That last step is what makes the chunk self-contained:
the text a reader would need to interpret it is the text the embedder and BM25 see.

Both retrieval paths call this same function, which is load-bearing: RRF dedups by
SHA256(content), so vector chunks and BM25 chunks must be byte-identical or the same passage
retrieved by both paths shows up twice, splits its own rank votes, and gets cited twice.
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Bumped whenever the chunking strategy changes. The ingestion manifest records this next to
# each file's content hash, so a strategy change invalidates every entry and forces a
# re-index -- without it, `build_index` would compare unchanged file hashes, skip every file,
# and leave the collection full of chunks built by the previous strategy while the code
# assumes the new one. Silent, and only visible as quietly worse retrieval.
CHUNKING_VERSION = 2

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
# A "#" inside a fenced code block is a comment, not a heading. Tracking fences costs one
# boolean and avoids shredding code samples into bogus sections.
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Below this, a breadcrumb would crowd out the actual content, so the section is chunked
# without one. Reached only with absurd heading nesting on a small chunk_size.
_MIN_BODY_CHARS = 120


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Splits markdown into (breadcrumb, body) pairs following its heading hierarchy.

    The breadcrumb is the trail of enclosing headings ("Anthropic > Funding"), not just the
    nearest one, so a chunk under a generic "## Funding" still carries the company name that
    makes it findable. Text with no headings at all (.txt, a bare .docx) comes back as one
    section with an empty breadcrumb, which is exactly the old fixed-size behavior.
    """
    sections: list[tuple[str, str]] = []
    # Index i holds the heading text at level i+1; deeper levels are cleared when a shallower
    # heading appears, which is what keeps the trail correct across sibling sections.
    stack: list[str | None] = [None] * 6
    current: list[str] = []
    in_fence = False

    def breadcrumb() -> str:
        return " > ".join(part for part in stack if part)

    def flush() -> None:
        body = "\n".join(current).strip()
        if body:
            sections.append((breadcrumb(), body))
        current.clear()

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            current.append(line)
            continue

        match = None if in_fence else _HEADER_RE.match(line)
        if match is None:
            current.append(line)
            continue

        # A heading ends the previous section, and the flush must happen *before* the stack
        # moves so the body is filed under the heading it actually sat beneath.
        flush()
        level = len(match.group(1))
        stack[level - 1] = match.group(2).strip()
        for deeper in range(level, 6):
            stack[deeper] = None

    flush()

    # Content-preservation fallback. Heading text normally survives in its descendants'
    # breadcrumbs, but a document whose entire content is headings has no descendants to
    # carry it -- and that is not hypothetical: pymupdf4llm renders a short PDF page as a
    # single `# ...` line with no body, so a page-per-heading PDF would index as nothing at
    # all. Treating a document that yielded no bodies as one unstructured section keeps the
    # invariant that splitting never silently drops content.
    if not sections and text.strip():
        return [("", text.strip())]
    return sections


def split_documents(
    documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 100
) -> list[Document]:
    """Structure-aware chunking: split on headings, then fixed-size within each section, with
    the heading breadcrumb prepended to every chunk.

    The breadcrumb is charged against `chunk_size` rather than added on top of it, so the
    caller's budget stays a real bound on chunk length -- a downstream context budget that
    trusts `chunk_size` would otherwise be quietly wrong by the length of the headings.
    """
    chunks: list[Document] = []
    for document in documents:
        for section_breadcrumb, body in _split_into_sections(document.page_content):
            prefix = f"{section_breadcrumb}\n\n" if section_breadcrumb else ""
            available = chunk_size - len(prefix)
            if available < _MIN_BODY_CHARS:
                prefix, available = "", chunk_size

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=available,
                chunk_overlap=min(chunk_overlap, max(available - 1, 0)),
            )
            for piece in splitter.split_text(body):
                metadata = dict(document.metadata)
                if section_breadcrumb:
                    metadata["section"] = section_breadcrumb
                chunks.append(Document(page_content=prefix + piece, metadata=metadata))
    return chunks
