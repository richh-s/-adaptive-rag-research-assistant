"""Index generations: several complete indexes side by side, one of them serving.

Changing the embedding model -- or anything else that invalidates every stored vector -- used
to mean `ingest --full`: reset the collection, then re-parse and re-embed the corpus while the
service answered from a half-empty index, or stop serving until it finished. Neither is
acceptable for something people depend on.

A *generation* is one complete index: vectors, manifest, parent sections and the metadata
recording which embedding model built it. Generations are built beside each other, and a
single pointer names the one that serves. Re-embedding becomes

    build a new generation  ->  catch it up with whatever changed meanwhile  ->  flip the pointer

and the old generation stays on disk until it is garbage-collected, so rolling back is
flipping the pointer again rather than rebuilding anything.

Where each backend keeps a generation:

    chroma    <CHROMA_PERSIST_DIR>/                       the legacy generation ("")
              <CHROMA_PERSIST_DIR>/generations/<id>/      every generation built since
              <CHROMA_PERSIST_DIR>/active_index.json      the pointer
    pgvector  schema `public`                             the legacy generation
              schema `rag_idx_<id>`                        every generation built since
              table  `public.rag_active_index`            the pointer

The legacy generation is the index as it existed before generations did, so an existing
deployment needs no migration: with no pointer, the legacy generation serves, exactly as
before. Everything downstream already takes a `persist_dir`; a generation *is* a persist
directory, so the rest of the codebase needed only its default changed from the configured
directory to `active_index_dir()`.

Every replica reads the pointer, not a local copy of it, and caches it for at most
`INDEX_POINTER_POLL_SECONDS` -- the same bounded-staleness poll the BM25 index uses to notice
another replica's ingest. After a flip, replicas converge within that interval; each one
serves a complete, self-consistent generation the whole time, because a generation also
records the model its queries must be embedded with (see `embeddings_for_index`).
"""

import json
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from rag_assistant.config import get_settings

logger = logging.getLogger(__name__)

LEGACY = ""
POINTER_FILENAME = "active_index.json"
GENERATIONS_DIRNAME = "generations"
_GENERATION_RE = re.compile(r"^g[0-9a-z]{6,40}$")

_lock = threading.Lock()
# (root, backend) -> (generation, read_at monotonic)
_pointer_cache: dict[tuple[str, str], tuple[str, float]] = {}


@dataclass(frozen=True)
class Pointer:
    generation: str
    previous: str | None = None
    switched_at: float = 0.0


def index_root() -> Path:
    return Path(get_settings().chroma_persist_dir)


def _shared_backend() -> bool:
    return get_settings().vector_backend == "pgvector"


def new_generation_id() -> str:
    """Sortable by creation time and unique across replicas. Lowercase alphanumerics only,
    because the id becomes a Chroma collection suffix and a Postgres schema name."""
    return "g" + time.strftime("%Y%m%d%H%M%S", time.gmtime()) + secrets.token_hex(3)


def validate_generation(generation: str) -> str:
    if generation != LEGACY and not _GENERATION_RE.match(generation):
        raise ValueError(f"Not a valid index generation id: {generation!r}")
    return generation


def generation_dir(generation: str, root: Path | None = None) -> Path:
    root = Path(root or index_root())
    if generation == LEGACY:
        return root
    return root / GENERATIONS_DIRNAME / validate_generation(generation)


def generation_of_dir(persist_dir: Path | str | None) -> str:
    """The inverse of `generation_dir`. Any directory that is not a generation directory is
    treated as the legacy generation, which is what every existing caller (and every test that
    passes its own `tmp_path / "chroma"`) has always meant by it."""
    if persist_dir is None:
        return active_generation()
    path = Path(persist_dir)
    if path.parent.name == GENERATIONS_DIRNAME and _GENERATION_RE.match(path.name):
        return path.name
    return LEGACY


def schema_for_generation(generation: str) -> str:
    """The Postgres schema holding a generation's tables."""
    return "public" if generation == LEGACY else f"rag_idx_{validate_generation(generation)}"


def collection_suffix(generation: str) -> str:
    """Appended to Chroma collection names. Only matters in server mode, where every
    generation shares one Chroma server; embedded Chroma already separates them by directory,
    and the suffix keeps the two modes naming collections the same way."""
    return "" if generation == LEGACY else f"__{generation}"


# ---- the pointer ----


def _pointer_file(root: Path) -> Path:
    return root / POINTER_FILENAME


def read_pointer() -> Pointer:
    """The pointer as stored, uncached. No pointer means the legacy generation serves."""
    if _shared_backend():
        from rag_assistant.retrieval.pgvector_store import load_active_pointer

        row = load_active_pointer()
        if row is None:
            return Pointer(generation=LEGACY)
        return Pointer(generation=row[0], previous=row[1], switched_at=row[2])
    path = _pointer_file(index_root())
    if not path.exists():
        return Pointer(generation=LEGACY)
    try:
        payload = json.loads(path.read_text())
        return Pointer(
            generation=validate_generation(payload.get("generation", LEGACY)),
            previous=payload.get("previous"),
            switched_at=float(payload.get("switched_at", 0.0)),
        )
    except Exception:
        # A pointer that cannot be read must not silently fall back to the legacy generation:
        # that index may be months stale or already garbage-collected. Refusing to serve is
        # loud; answering from the wrong index is not.
        logger.error("Unreadable index pointer at %s", path, exc_info=True)
        raise


def active_generation() -> str:
    """The generation that serves, cached for INDEX_POINTER_POLL_SECONDS."""
    settings = get_settings()
    key = (str(index_root()), settings.vector_backend)
    now = time.monotonic()
    cached = _pointer_cache.get(key)
    if cached is not None and now - cached[1] < settings.index_pointer_poll_seconds:
        return cached[0]
    try:
        generation = read_pointer().generation
    except Exception:
        # A database blip while re-reading must not flip a healthy replica onto another
        # index. Keep serving the last generation this process knew was active; with no
        # previous reading there is nothing safe to fall back to, so the error propagates.
        if cached is not None:
            logger.warning("Could not re-read the index pointer; keeping %r", cached[0])
            _pointer_cache[key] = (cached[0], now)
            return cached[0]
        raise
    if cached is not None and cached[0] != generation:
        logger.info("index generation switched from %r to %r", cached[0], generation)
    _pointer_cache[key] = (generation, now)
    return generation


def active_index_dir() -> Path:
    """The persist directory of the serving generation -- the default for every caller that
    used to default to CHROMA_PERSIST_DIR."""
    return generation_dir(active_generation())


def set_active_generation(generation: str) -> Pointer:
    """Flips the pointer. The previous generation is recorded so a rollback needs no memory
    of what was serving before."""
    validate_generation(generation)
    with _lock:
        current = read_pointer()
        pointer = Pointer(
            generation=generation, previous=current.generation, switched_at=time.time()
        )
        if _shared_backend():
            from rag_assistant.retrieval.pgvector_store import save_active_pointer

            save_active_pointer(pointer.generation, pointer.previous, pointer.switched_at)
        else:
            root = index_root()
            root.mkdir(parents=True, exist_ok=True)
            path = _pointer_file(root)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "generation": pointer.generation,
                        "previous": pointer.previous,
                        "switched_at": pointer.switched_at,
                    },
                    indent=2,
                )
            )
            temporary.replace(path)
        reset_pointer_cache()
    logger.info(
        "index generation activated",
        extra={"node": f"{pointer.previous!r} -> {pointer.generation!r}"},
    )
    return pointer


def reset_pointer_cache() -> None:
    _pointer_cache.clear()


def list_generations() -> list[str]:
    """Every generation that exists, legacy first, then oldest to newest."""
    if _shared_backend():
        from rag_assistant.retrieval.pgvector_store import list_generation_schemas

        return [LEGACY, *sorted(list_generation_schemas())]
    base = index_root() / GENERATIONS_DIRNAME
    found = (
        sorted(p.name for p in base.iterdir() if p.is_dir() and _GENERATION_RE.match(p.name))
        if base.exists()
        else []
    )
    return [LEGACY, *found]
