from rag_assistant.ingestion.loaders import load_documents
from rag_assistant.ingestion.splitter import split_documents


def _make_minimal_pdf(text_or_pages: str | list[str]) -> bytes:
    """Hand-build a minimal PDF (one or more pages) with real text content streams, so tests
    can exercise actual text extraction without pulling in a PDF-generation dependency.
    Object numbering: 1=Catalog, 2=Pages, 3=Font, then a (Page, Contents) object pair per
    page."""
    pages = [text_or_pages] if isinstance(text_or_pages, str) else text_or_pages
    font_obj_id = 3
    page_obj_ids = [font_obj_id + 1 + 2 * i for i in range(len(pages))]

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_obj_ids)}] "
            f"/Count {len(pages)} >>").encode("latin-1"),
        font_obj_id: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_obj_id, text in zip(page_obj_ids, pages):
        contents_obj_id = page_obj_id + 1
        objects[page_obj_id] = (
            f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {font_obj_id} 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {contents_obj_id} 0 R >>"
        ).encode("latin-1")
        content = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode("latin-1")
        objects[contents_obj_id] = (
            f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream"
        )

    ordered_ids = sorted(objects)
    pdf = b"%PDF-1.4\n"
    offsets = {}
    for obj_id in ordered_ids:
        offsets[obj_id] = len(pdf)
        pdf += f"{obj_id} 0 obj\n".encode("latin-1") + objects[obj_id] + b"\nendobj\n"
    xref_offset = len(pdf)
    max_id = ordered_ids[-1]
    pdf += f"xref\n0 {max_id + 1}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for obj_id in range(1, max_id + 1):
        pdf += f"{offsets[obj_id]:010d} 00000 n \n".encode("latin-1")
    pdf += f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode(
        "latin-1"
    )
    return pdf


def test_load_documents_reads_all_supported_files(sample_corpus_dir):
    docs = load_documents(sample_corpus_dir)

    assert len(docs) == 2
    sources = {d.metadata["source"] for d in docs}
    assert sources == {"anthropic.md", "mistral.md"}


def test_load_documents_ignores_unsupported_files(sample_corpus_dir):
    (sample_corpus_dir / "notes.json").write_text("{}")

    docs = load_documents(sample_corpus_dir)

    assert all(d.metadata["source"] != "notes.json" for d in docs)


def test_load_documents_extracts_text_from_pdf(sample_corpus_dir):
    (sample_corpus_dir / "cohere.pdf").write_bytes(_make_minimal_pdf("Cohere builds enterprise LLMs."))

    docs = load_documents(sample_corpus_dir)

    pdf_docs = [d for d in docs if d.metadata["source"] == "cohere.pdf"]
    assert len(pdf_docs) == 1
    assert "Cohere builds enterprise LLMs." in pdf_docs[0].page_content


def test_load_documents_skips_corrupt_pdf(sample_corpus_dir):
    (sample_corpus_dir / "broken.pdf").write_bytes(b"%PDF-1.4\nnot a real pdf")

    docs = load_documents(sample_corpus_dir)

    assert all(d.metadata["source"] != "broken.pdf" for d in docs)


def test_load_documents_skips_image_only_pdf(sample_corpus_dir):
    (sample_corpus_dir / "scanned.pdf").write_bytes(_make_minimal_pdf(""))

    docs = load_documents(sample_corpus_dir)

    assert all(d.metadata["source"] != "scanned.pdf" for d in docs)


def test_load_documents_tags_pdf_pages_with_page_number(sample_corpus_dir):
    (sample_corpus_dir / "cohere.pdf").write_bytes(_make_minimal_pdf("Cohere builds enterprise LLMs."))

    docs = load_documents(sample_corpus_dir)

    pdf_docs = [d for d in docs if d.metadata["source"] == "cohere.pdf"]
    assert pdf_docs[0].metadata["page"] == 1


def test_load_documents_yields_one_document_per_pdf_page(sample_corpus_dir):
    (sample_corpus_dir / "multi.pdf").write_bytes(
        _make_minimal_pdf(["Page one content.", "Page two content."])
    )

    docs = load_documents(sample_corpus_dir)

    pdf_docs = sorted(
        (d for d in docs if d.metadata["source"] == "multi.pdf"), key=lambda d: d.metadata["page"]
    )
    assert [d.metadata["page"] for d in pdf_docs] == [1, 2]
    assert "Page one content." in pdf_docs[0].page_content
    assert "Page two content." in pdf_docs[1].page_content


def test_split_documents_produces_nonempty_chunks(sample_corpus_dir):
    docs = load_documents(sample_corpus_dir)

    chunks = split_documents(docs, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > len(docs)
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(chunk.metadata["source"] in {"anthropic.md", "mistral.md"} for chunk in chunks)
