"""Tests for structure-aware chunking.

Two properties matter more than the chunk boundaries themselves: splitting must never drop
content, and it must be deterministic -- BM25 and Chroma both call this function and RRF
dedups by SHA256(content), so any divergence between the two would show the same passage
twice, split its own rank votes, and cite it twice.
"""

from langchain_core.documents import Document

from rag_assistant.ingestion.splitter import (
    CHUNKING_VERSION,
    _split_into_sections,
    split_documents,
)

MARKDOWN = """# Anthropic

Anthropic is an AI safety company.

## Funding

It has raised substantial funding from several investors.

## Safety focus

The company is known for Constitutional AI.
"""


def test_sections_follow_the_heading_hierarchy():
    sections = _split_into_sections(MARKDOWN)

    assert [breadcrumb for breadcrumb, _ in sections] == [
        "Anthropic",
        "Anthropic > Funding",
        "Anthropic > Safety focus",
    ]


def test_breadcrumb_carries_ancestors_not_just_the_nearest_heading():
    """A chunk under a generic '## Funding' is useless without the company name -- neither
    the embedding nor BM25 has anything to match the question against."""
    sections = _split_into_sections(MARKDOWN)
    funding = next(body for crumb, body in sections if crumb == "Anthropic > Funding")

    assert "raised substantial funding" in funding


def test_chunks_are_prefixed_with_their_breadcrumb():
    chunks = split_documents([Document(page_content=MARKDOWN, metadata={"source": "a.md"})])

    funding = next(c for c in chunks if "raised substantial funding" in c.page_content)
    assert funding.page_content.startswith("Anthropic > Funding")
    assert funding.metadata["section"] == "Anthropic > Funding"
    assert funding.metadata["source"] == "a.md"


def test_breadcrumb_is_charged_against_the_chunk_size():
    """The prefix is included in the budget, not added on top, so `chunk_size` stays a real
    bound -- the context budget downstream trusts it."""
    body = "word " * 400
    document = Document(page_content=f"# Heading\n\n## Subheading\n\n{body}", metadata={})

    chunks = split_documents([document], chunk_size=300, chunk_overlap=20)

    assert chunks
    assert all(len(c.page_content) <= 300 for c in chunks)


def test_text_without_headings_becomes_one_unstructured_section():
    """.txt files and bare .docx have no headings; behavior there must match the old
    fixed-size splitter, with no breadcrumb invented."""
    chunks = split_documents([Document(page_content="Just a plain sentence.", metadata={})])

    assert len(chunks) == 1
    assert chunks[0].page_content == "Just a plain sentence."
    assert "section" not in chunks[0].metadata


def test_heading_only_document_still_produces_a_chunk():
    """pymupdf4llm renders a short PDF page as a single '# ...' line with no body. Filing
    that under a breadcrumb and emitting no body would index the page as nothing at all."""
    chunks = split_documents(
        [Document(page_content="# Cohere page one about embeddings.\n\n", metadata={})]
    )

    assert len(chunks) == 1
    assert "Cohere page one about embeddings." in chunks[0].page_content


def test_no_content_is_dropped_across_a_mixed_document():
    document = Document(page_content=MARKDOWN, metadata={})

    chunks = split_documents([document])
    combined = " ".join(c.page_content for c in chunks)

    for sentence in (
        "Anthropic is an AI safety company",
        "raised substantial funding",
        "known for Constitutional AI",
    ):
        assert sentence in combined


def test_hash_comments_inside_code_fences_are_not_treated_as_headings():
    """A '#' in a shell block is a comment. Reading it as a heading shreds the code sample
    into bogus sections and files the surrounding prose under a nonsense breadcrumb."""
    text = "# Guide\n\nIntro text.\n\n```bash\n# install the thing\nnpm install\n```\n\nOutro text."

    sections = _split_into_sections(text)

    assert [crumb for crumb, _ in sections] == ["Guide"]
    assert "npm install" in sections[0][1]


def test_splitting_is_deterministic():
    """RRF's cross-source dedup depends on vector and BM25 chunks being byte-identical."""
    documents = [Document(page_content=MARKDOWN, metadata={"source": "a.md"})]

    first = [c.page_content for c in split_documents(documents)]
    second = [c.page_content for c in split_documents(documents)]

    assert first == second


def test_empty_document_produces_no_chunks():
    assert split_documents([Document(page_content="   \n\n  ", metadata={})]) == []


def test_chunking_version_is_an_integer_the_manifest_can_compare():
    assert isinstance(CHUNKING_VERSION, int)
