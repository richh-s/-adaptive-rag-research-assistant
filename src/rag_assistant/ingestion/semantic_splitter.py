"""Semantic chunking: split where the topic changes, not where the character count runs out.

Fixed-size splitting cuts at an arbitrary offset. Even with structure-aware sections
(splitter.py), a long section still gets sliced every N characters, which routinely severs a
claim from the sentence that qualifies it — and a chunk that ends mid-argument retrieves badly
and reads worse when it lands in a synthesis prompt.

This finds the seams instead. Sentences are embedded, consecutive sentences are compared, and
a break is placed where similarity drops sharply — a topic shift. The threshold is a
*percentile of the distances actually observed in this section*, not an absolute number,
because cosine distances are not comparable across embedding models or across prose styles;
what matters is which gaps are unusually large relative to their neighbours.

It costs one embedding call per section at ingest time, which is why it is opt-in
(`CHUNKING_STRATEGY=semantic`). Structural chunking stays the default: free, deterministic,
and good enough on well-headed documents.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Sentence segmentation without an NLP dependency: split after ., ! or ? when followed by
# whitespace and something that starts a new sentence. Deliberately conservative -- an
# over-eager split merely creates a smaller comparison unit, while an under-eager one only
# costs a slightly coarser breakpoint. Common abbreviations are held back explicitly because
# "e.g. the model" would otherwise become a sentence boundary in the middle of a clause.
_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "cf.",
    "al.",
    "Inc.",
    "Ltd.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "Fig.",
    "No.",
    "approx.",
)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    """Segments prose into sentences. Falls back to the whole text when it finds none, so a
    caller always gets at least one unit to work with."""
    pieces = _SENTENCE_END_RE.split(text.strip())
    sentences: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        # Re-join a fragment that was split immediately after a known abbreviation.
        if sentences and sentences[-1].endswith(_ABBREVIATIONS):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences or ([text.strip()] if text.strip() else [])


def _cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 1.0
    return 1.0 - dot / (left_norm * right_norm)


def _percentile(values: list[float], percentile: float) -> float:
    """Linearly interpolated percentile, matching numpy's default -- without the dependency.

    Interpolation is load-bearing rather than cosmetic here. With nearest-rank, a high
    percentile over a handful of sentences snaps onto the largest observed distance, so the
    `distance > threshold` test can never be true and the single obvious topic shift is the
    one break that is guaranteed to be missed. Interpolating puts the threshold *between* the
    outlier and the rest, which is where a threshold has to sit to separate them.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def find_breakpoints(distances: list[float], percentile: float) -> list[int]:
    """Indexes after which a chunk boundary should fall.

    A distance is a breakpoint when it exceeds the given percentile of all distances in this
    section *and* is above a small absolute floor. The floor matters: in a section that is
    genuinely about one thing, every consecutive-sentence distance is tiny and nearly equal,
    and a pure percentile rule would still split it at whichever noise happened to be
    largest. Uniform prose should produce one chunk, not an arbitrary cut.
    """
    if not distances:
        return []
    threshold = _percentile(distances, percentile)
    spread = max(distances) - min(distances)
    if spread < 0.01:
        return []
    return [index for index, distance in enumerate(distances) if distance > threshold]


def semantic_split(
    text: str,
    embeddings,
    percentile: float = 85.0,
    min_chunk_chars: int = 200,
    max_chunk_chars: int = 2000,
) -> list[str]:
    """Splits one section at topic shifts, honouring size bounds.

    `min_chunk_chars` prevents a burst of short sentences from producing chunks too small to
    carry meaning; `max_chunk_chars` is a hard backstop so a section with no detectable topic
    shift still gets divided rather than becoming one enormous chunk that blows the context
    budget on its own.

    Any failure in the embedding call degrades to a single chunk rather than raising: chunking
    runs inside ingestion, and losing semantic boundaries is a quality regression while
    failing the ingest is an outage.
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return [text.strip()] if text.strip() else []

    try:
        vectors = embeddings.embed_documents(sentences)
    except Exception:
        logger.warning(
            "Semantic chunking failed to embed; falling back to one chunk", exc_info=True
        )
        return [text.strip()]

    distances = [_cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    breakpoints = set(find_breakpoints(distances, percentile))

    chunks: list[str] = []
    current: list[str] = []
    for index, sentence in enumerate(sentences):
        current.append(sentence)
        current_len = sum(len(s) + 1 for s in current)
        at_breakpoint = index in breakpoints and current_len >= min_chunk_chars
        over_budget = current_len >= max_chunk_chars
        if at_breakpoint or over_budget:
            chunks.append(" ".join(current))
            current = []
    if current:
        # A trailing fragment too small to stand alone is merged backwards rather than
        # emitted -- a two-sentence orphan retrieves poorly and dilutes its own embedding.
        tail = " ".join(current)
        if chunks and len(tail) < min_chunk_chars:
            chunks[-1] = f"{chunks[-1]} {tail}"
        else:
            chunks.append(tail)
    return chunks
