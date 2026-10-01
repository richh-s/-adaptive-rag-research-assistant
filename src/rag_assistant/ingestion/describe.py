"""One-line descriptions of indexed documents, for the router.

The router is shown what the local corpus contains and routes on it. Built from filenames
alone, that list fails exactly where it matters: `Annual_Report_JUNE-2021.pdf` says nothing
about *whose* report it is, so on a real 30-report corpus every question naming the publisher
was sent to web search instead. One short chat call per file, at ingest, labels it from its
opening text -- charged to the tenant like the rest of ingest (see budget.charge_ingest).

Failure is never fatal: no description means the router falls back to the filename, which is
exactly what it had before this existed.
"""

import logging
import re
from pathlib import Path

from rag_assistant.content_trust import fence_block, new_nonce
from rag_assistant.llm import get_chat_model
from rag_assistant.prompts.describe_prompt import DESCRIBE_PROMPT

logger = logging.getLogger(__name__)

# Enough for a cover page, a title page and the start of an introduction -- where publisher,
# document type and period are stated -- without paying to send the whole document.
SAMPLE_CHARS = 4000
MAX_DESCRIPTION_CHARS = 160


def _response_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        # Content blocks, as some providers return them.
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def clean_description(raw: str) -> str | None:
    """One bounded line, safe to interpolate into the router prompt.

    The description is model output about untrusted text, so it is treated as untrusted too:
    it is fenced wherever it is shown, and here it is cut to one line, stripped of anything
    resembling a fence marker, and length-capped so a document cannot turn its label into a
    paragraph of instructions.
    """
    line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    line = re.sub(r"<{2,}|>{2,}", "", line)
    line = re.sub(r"\s+", " ", line).strip(" \"'`")
    if not line or line.lower().startswith("unknown -- unknown -- unknown"):
        return None
    return line[:MAX_DESCRIPTION_CHARS]


def document_context_line(filename: str, description: str | None) -> str:
    """The one line prepended to every chunk of a document before it is embedded.

    A chunk carries the heading breadcrumb it came from but nothing about the document it came
    from, and in a corpus of near-identical annual reports that is what decides a year-specific
    question. Measured on 30 such reports: the passage answering "Ethiopian Reinsurance's
    2020/21 profit before tax" reads "During the period under review, the Company has
    registered Birr 220 million profit before tax" -- naming neither the company nor the year
    -- and did not reach the top 20 for that question, though its own report took the first
    four places.

    Falls back to the filename: weak, but never worse than nothing.
    """
    return description or Path(filename).stem.replace("_", " ").replace("-", " ").strip()


def describe_document(filename: str, text: str) -> str | None:
    """A one-line label for one document, or None when there is nothing usable to label."""
    sample = text.strip()[:SAMPLE_CHARS]
    if not sample:
        return None
    prompt = DESCRIBE_PROMPT.format(
        filename=filename,
        content=fence_block(sample, nonce=new_nonce(), label="DOCUMENT TEXT"),
    )
    try:
        return clean_description(_response_text(get_chat_model().invoke(prompt)))
    except Exception:
        logger.warning("Could not describe %s; the router will see its filename", filename)
        return None
