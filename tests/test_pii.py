"""Personal data in the corpus.

The two properties that matter are opposite failures: finding what is there, and not claiming
to find what is not. The second is the harder one on this corpus -- annual reports are full of
long digit runs, and a detector that reports every one of them as a card number is a detector
whose output nobody reads.
"""

import pytest

from rag_assistant import pii
from rag_assistant.config import get_settings
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.retrieval.vector_store import get_vector_store, reset_store_cache

# A valid Luhn number from the standard test-card range; not anyone's card.
TEST_CARD = "4111 1111 1111 1111"


@pytest.mark.parametrize(
    "text,category",
    [
        ("write to jane.doe@example.co.uk about it", "email"),
        ("call +44 20 7946 0958 in the morning", "phone"),
        (f"paid with {TEST_CARD} last week", "credit_card"),
        ("his SSN is 123-45-6789 on the form", "ssn"),
        ("transfer to GB82 WEST 1234 5698 7654 32 today", "iban"),
        ("key AKIAIOSFODNN7EXAMPLE was committed", "aws_access_key"),
    ],
)
def test_each_category_is_detected(text, category):
    assert pii.scan(text) == {category: 1}


@pytest.mark.parametrize(
    "text",
    [
        # The case this corpus actually produces: long numbers everywhere, none of them cards.
        "Revenue of 1234567890123 birr against 98765432101234 the prior year.",
        "Document reference 12345678901234567 filed 2021.",
        "Section 4.2.1 covers the 2019 2020 2021 reporting periods.",
        "No personal data in this sentence at all.",
    ],
)
def test_ordinary_financial_prose_is_not_flagged(text):
    """A detector that fires on every figure in an annual report is one nobody reads."""
    assert pii.scan(text) == {}


@pytest.mark.parametrize(
    "text",
    ["+44 20 7946 0958", "+1 415 555 0142", "(415) 555-0142", "415-555-0142", "415.555.0142"],
)
def test_the_common_phone_formats_are_detected(text):
    assert pii.scan(text) == {"phone": 1}


def test_a_phone_number_needs_an_explicit_signal_not_just_spaced_digits():
    """Accepting any run of space-separated digit groups matched "2019 2020 2021" in a
    reporting-period heading -- which is how a detector ends up firing on most of an annual
    report and being ignored."""
    assert pii.scan("covering the 2019 2020 2021 periods") == {}
    assert pii.scan("lot 4111 1111 1111 1112 shipped") == {}


def test_the_luhn_check_is_what_separates_a_card_from_a_long_number():
    """Same length, same grouping, one digit different -- only the valid one is a card."""
    assert pii.scan(f"paid {TEST_CARD}") == {"credit_card": 1}
    assert pii.scan("paid 4111 1111 1111 1112") == {}


def test_redaction_names_the_category_rather_than_blanking_the_span():
    """A hole in a document invites the model to fill it. A marker tells it what was removed,
    so the answer says the contact details are unavailable instead of inventing them."""
    redacted, counts = pii.redact("Reach us at ops@example.com any time.")

    assert redacted == "Reach us at [REDACTED:email] any time."
    assert counts == {"email": 1}


def test_a_span_is_claimed_by_one_category_only():
    """A grouped card number matches several loose phone patterns too. Reporting it as a
    phone number would understate what was found, and redacting twice would nest markers."""
    counts = pii.scan(f"card {TEST_CARD}")

    assert counts == {"credit_card": 1}


def test_redaction_leaves_surrounding_text_exactly_intact():
    original = "Before. Email a@b.co. Middle. Card " + TEST_CARD + ". After."
    redacted, _ = pii.redact(original)

    assert redacted.startswith("Before. Email ")
    assert redacted.endswith(". After.")
    assert "Middle." in redacted


@pytest.mark.parametrize("mode", ["off", "flag"])
def test_flag_and_off_modes_store_the_text_unchanged(mode, monkeypatch):
    monkeypatch.setenv("PII_MODE", mode)
    get_settings.cache_clear()

    assert pii.apply("mail me at a@b.com") == "mail me at a@b.com"


def test_redact_mode_changes_what_apply_returns(monkeypatch):
    monkeypatch.setenv("PII_MODE", "redact")
    get_settings.cache_clear()

    assert pii.apply("mail me at a@b.com") == "mail me at [REDACTED:email]"


def _index_one(tmp_path, fake_embeddings, text: str):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "contacts.md").write_text(text)
    persist = tmp_path / "chroma"
    reset_store_cache()
    build_index(source_dir=corpus, persist_dir=persist, embeddings=fake_embeddings)
    store = get_vector_store(embeddings=fake_embeddings, persist_dir=persist)
    return corpus, persist, "\n".join(store.get(include=["documents"])["documents"])


def test_redaction_happens_before_the_text_is_embedded(tmp_path, fake_embeddings, monkeypatch):
    """The point of redaction: the text is not in the vector store, so it cannot be retrieved,
    cannot reach a synthesis prompt and cannot be quoted back in an answer."""
    monkeypatch.setenv("PII_MODE", "redact")
    get_settings.cache_clear()

    _, _, stored = _index_one(
        tmp_path, fake_embeddings, "Payroll contact: hr@acme.example. Direct line +1 415 555 0142."
    )

    assert "hr@acme.example" not in stored
    assert "[REDACTED:email]" in stored


def test_redaction_never_touches_the_uploaded_file(tmp_path, fake_embeddings, monkeypatch):
    """The original stays available for a human to look at, which is what makes the decision
    reversible by re-ingesting with the mode off."""
    monkeypatch.setenv("PII_MODE", "redact")
    get_settings.cache_clear()

    corpus, _, _ = _index_one(tmp_path, fake_embeddings, "Payroll contact: hr@acme.example.")

    assert "hr@acme.example" in (corpus / "contacts.md").read_text()


def test_flag_mode_indexes_the_original_text(tmp_path, fake_embeddings, monkeypatch):
    monkeypatch.setenv("PII_MODE", "flag")
    get_settings.cache_clear()

    _, _, stored = _index_one(tmp_path, fake_embeddings, "Payroll contact: hr@acme.example.")

    assert "hr@acme.example" in stored


def test_parent_sections_are_redacted_too(tmp_path, fake_embeddings, monkeypatch):
    """With PARENT_CONTEXT on, the section replaces the chunk in the synthesis prompt -- a
    redacted chunk whose parent still carried the original would put it straight back."""
    monkeypatch.setenv("PII_MODE", "redact")
    get_settings.cache_clear()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "contacts.md").write_text("# Contacts\n\nPayroll: hr@acme.example.\n")
    persist = tmp_path / "chroma"
    reset_store_cache()
    build_index(source_dir=corpus, persist_dir=persist, embeddings=fake_embeddings)

    from rag_assistant.retrieval.parent_store import get_parents

    manifest = load_manifest(persist)
    assert manifest
    parents = get_parents(persist, ["contacts.md::0"])
    assert all("hr@acme.example" not in text for text in parents.values())
