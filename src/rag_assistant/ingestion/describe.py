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
