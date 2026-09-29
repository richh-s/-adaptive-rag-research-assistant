"""Personal data in the corpus: finding it, and optionally not storing it.

The pipeline accepts uploads from any tenant with a write scope and indexes whatever it
parses. Nothing looked at that text before storing it, which means a payroll spreadsheet, an
exported mailbox or a scanned form went into the vector store, the keyword index, the parent
sections and every prompt built from them, with no record that it had happened. For a demo
corpus of company profiles that is invisible. For a deployment where people upload their own
documents it is the difference between a search index and a copy of someone's personal data
in four places nobody is tracking.

Three modes, because the right answer genuinely differs by deployment:

* **off** -- no scanning. The setting exists so the cost can be removed entirely, not as a
  recommendation.
* **flag** (the default) -- detect, count, log; store the text unchanged. This mirrors
  `content_trust.scan_for_injection`, and for the same reason: a regex is a smoke detector,
  not a classifier. Every pattern here matches things that are not personal data (an order
  number that passes Luhn, a support address in a footer), and quietly mangling a legitimate
  document is a worse failure than indexing one that needed review. What the operator gets is
  the thing they did not have: a number that says personal data is arriving.
* **redact** -- replace each match with a `[REDACTED:<category>]` marker *before* the text is
  embedded, keyword-indexed and stored as a parent section. The corpus file on disk is never
  touched, so the original remains available for a human to look at and the decision stays
  reversible by re-ingesting with the mode off.

What redaction actually buys, stated precisely: the text is not embedded, so it cannot be
retrieved, cannot reach a synthesis prompt and cannot be quoted back in an answer. What it
does not buy: the uploaded file is still on disk, and anything indexed *before* the mode was
turned on is still indexed -- a mode change applies to what is ingested after it, and needs
`ingest --full` to apply to what came before.

The honest boundary, and it is a wide one: regexes find *formats*, not personal data. A name,
an address, a date of birth, a medical detail or a national ID this module has no pattern for
all pass through untouched, and no amount of pattern-writing closes that gap -- the categories
below are the ones with enough structure to match without drowning the signal in false
positives. A deployment with a real regulatory obligation wants a purpose-built classifier and
a human review step; this is the floor, not the ceiling, and its value is that the floor is no
longer zero.
"""

import re

from rag_assistant import metrics
from rag_assistant.config import get_settings

# (category, pattern). Ordered: the first pattern to match a span wins, so the more specific
# ones lead. Credit cards before phone numbers in particular -- a 16-digit card written with
# spaces matches several loose phone patterns, and reporting it as a phone number would
# understate what was found.
_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        # 13-19 digits in the groupings cards are actually written in. Validated by Luhn
        # below, which is what keeps this from matching every long number in a financial
        # report -- and this corpus is full of long numbers in financial reports.
        "credit_card",
        re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    ),
    (
        # US SSN. Narrow on purpose: the area/group/serial rules exclude the ranges the SSA
        # never issued, which is most of what a random nine-digit number would be.
        "ssn",
        re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    ),
    (
        "iban",
        re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}[ ]?[A-Z0-9]{1,4}\b"),
    ),
    (
        # Long-lived cloud credentials. Not personal data, but it is the other thing that
        # should never reach an embedding, and it is found by exactly this machinery.
        "aws_access_key",
        re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA)[0-9A-Z]{16}\b"),
    ),
    (
        # E.164 and the common national formats. Every branch requires an explicit telephone
        # signal -- a `+` country code, a parenthesised area code, or `-`/`.` separators --
        # rather than accepting any run of space-separated digit groups. Without that, the
        # pattern matched "2019 2020 2021" in a reporting-period heading and the tail of a
        # 16-digit number that failed the Luhn check, which is how a detector ends up firing
        # on most of an annual report and being ignored.
        "phone",
        re.compile(
            r"(?<![\d.])(?:"
            # +44 20 7946 0958 / +1 (415) 555-0142
            r"\+\d{1,3}[ .-]?(?:\(\d{2,4}\)[ .-]?|\d{2,4}[ .-]?)\d{3,4}[ .-]?\d{3,4}"
            # (415) 555-0142
            r"|\(\d{2,4}\)[ .-]?\d{3,4}[ .-]?\d{3,4}"
            # 415-555-0142 / 415.555.0142
            r"|\d{3}[.-]\d{3}[.-]\d{4}"
            r")(?![\d])"
        ),
    ),
)


def _luhn_valid(digits: str) -> bool:
    """The check digit every real card number carries.

    Without it the card pattern matches any 13-19 digit run, and a corpus of annual reports
    is full of those -- share counts, revenue in minor units, document reference numbers. The
    check turns a pattern that fires constantly into one that almost never fires by accident.
    """
    digits = re.sub(r"\D", "", digits)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _accept(category: str, matched: str) -> bool:
    """Per-category validation beyond the pattern. Only cards have any today."""
    if category == "credit_card":
        return _luhn_valid(matched)
    return True


def _spans(text: str) -> list[tuple[int, int, str]]:
    """Non-overlapping (start, end, category) matches, earliest and most specific first.

    Overlaps are resolved by discarding the later match rather than by merging: a span already
    claimed by `credit_card` must not be reported a second time as `phone`, which would both
    double-count the metric and produce nested redaction markers.
    """
    claimed: list[tuple[int, int, str]] = []
    for category, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if not _accept(category, match.group()):
                continue
            start, end = match.span()
            if any(start < seen_end and seen_start < end for seen_start, seen_end, _ in claimed):
                continue
            claimed.append((start, end, category))
    return sorted(claimed)


def scan(text: str) -> dict[str, int]:
    """Counts per category found in `text`. Never raises; an unscannable input finds nothing."""
    counts: dict[str, int] = {}
    for _, _, category in _spans(text or ""):
        counts[category] = counts.get(category, 0) + 1
    return counts


def redact(text: str) -> tuple[str, dict[str, int]]:
    """`text` with every match replaced by its category marker, plus the counts.

    The marker names the category rather than blanking the span, so a retrieved chunk still
    reads as a document with a phone number removed instead of as a document with a hole in
    it -- which is the difference between a model correctly saying the contact details are not
    available and a model inventing them to fill the gap.
    """
    spans = _spans(text or "")
    if not spans:
        return text, {}
    out: list[str] = []
    cursor = 0
    counts: dict[str, int] = {}
    for start, end, category in spans:
        out.append(text[cursor:start])
        out.append(f"[REDACTED:{category}]")
        counts[category] = counts.get(category, 0) + 1
        cursor = end
    out.append(text[cursor:])
    return "".join(out), counts


def apply(text: str, *, source: str = "") -> str:
    """The single seam ingestion calls. Returns what should actually be stored.

    Mode is read here rather than by each caller so `off` costs one comparison and no scan,
    and so a future caller cannot forget to check it.
    """
    mode = get_settings().pii_mode
    if mode == "off" or not text:
        return text
    if mode == "redact":
        redacted, counts = redact(text)
        _record(counts, source=source, redacted=True)
        return redacted
    _record(scan(text), source=source, redacted=False)
    return text


def _record(counts: dict[str, int], *, source: str, redacted: bool) -> None:
    """Counts to Prometheus, one log line per document that had anything.

    The log names the source and the categories, never the matched text: writing the value
    into a log to prove it was found would put the personal data somewhere new, which is the
    opposite of the point.
    """
    if not counts:
        return
    for category, count in counts.items():
        metrics.record_pii(category, count, redacted=redacted)
    import logging

    logging.getLogger(__name__).warning(
        "personal data detected in ingested content",
        extra={
            "route": source,
            "node": ("redacted " if redacted else "flagged ")
            + ",".join(f"{c}={n}" for c, n in sorted(counts.items())),
        },
    )
