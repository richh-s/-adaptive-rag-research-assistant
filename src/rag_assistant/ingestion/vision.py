"""Vision-model helpers for PDF ingestion: describing embedded figures and transcribing
scanned (image-only) pages.

Uses the same Anthropic-primary/Gemini-fallback chat stack as the rest of the app (llm.py)
rather than a local OCR/vision dependency -- both providers read images natively, the
deployment stays GPU-free, and chart/diagram understanding is far beyond what OCR gives.
Every call here is best-effort: ingestion must never fail because a figure couldn't be
described, so failures degrade to "no description" with a warning.

Cost note: this runs once per image/scanned page at ingestion time, not per query.
"""

import base64
import logging

from rag_assistant.config import get_settings
from rag_assistant.llm import get_chat_model

logger = logging.getLogger(__name__)

# Images smaller than this are almost always logos, icons, bullets, or decorative rules --
# describing them wastes vision calls and pollutes retrieval with noise.
MIN_IMAGE_DIMENSION_PX = 100
# Providers cap request sizes well below this; anything bigger is skipped, not resized,
# to keep this module dependency-free.
MAX_IMAGE_BYTES = 4 * 1024 * 1024
# Backstop against pathological PDFs (e.g. one image per bullet point across 200 pages).
MAX_IMAGES_PER_PDF = 20

FIGURE_PROMPT = """This image is a figure from a document being indexed for search. Describe it \
in 2-4 sentences so someone searching the document can find it: state what kind of figure it is \
(chart, diagram, photo, screenshot...), what it shows, and -- most importantly -- any concrete \
data, labels, numbers, or names visible in it. No preamble; output only the description."""

SCANNED_PAGE_PROMPT = """This image is a scanned page from a document being indexed for search. \
Transcribe ALL visible text faithfully, preserving headings and reading order; render tables as \
markdown tables. If part of the page is illegible, note '[illegible]' at that spot rather than \
guessing. No preamble; output only the transcription."""


def vision_available() -> bool:
    """Vision needs at least one configured provider key, and can be disabled outright via
    PDF_VISION=false (tests do this so offline ingestion never attempts a network call)."""
    settings = get_settings()
    if not settings.pdf_vision:
        return False
    return bool(settings.anthropic_api_key or settings.google_api_key)


def describe_image(image_bytes: bytes, media_type: str, prompt: str) -> str | None:
    """One best-effort vision call. Returns the model's text, or None if vision is disabled,
    the image is unsuitable, or the call fails."""
    if not vision_available():
        return None
    if len(image_bytes) > MAX_IMAGE_BYTES:
        logger.info("Skipping vision call for oversized image (%d bytes)", len(image_bytes))
        return None

    encoded = base64.b64encode(image_bytes).decode()
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            # OpenAI-style data-URL block: both langchain-anthropic and langchain-google-genai
            # translate this to their native image format.
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}},
        ],
    }
    try:
        response = get_chat_model().invoke([message])
        text = (response.text or "").strip()
        return text or None
    except Exception:
        logger.warning("Vision call failed; continuing without a description", exc_info=True)
        return None
