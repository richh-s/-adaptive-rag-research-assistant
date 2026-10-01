"""The sync engine: mirroring a source system into a tenant's corpus, deletions included.

A connector's documents are written as ordinary corpus files under the tenant's subtree,

    data/corpus/_t/acme/_sources/eng-wiki/onboarding-guide_1f3a9c2b.html
    data/corpus/_t/acme/_sources/eng-wiki/onboarding-guide_1f3a9c2b.html.acl.json
    data/corpus/_t/acme/_sources/eng-wiki/.sync_state.json

and then the tenant's corpus is re-indexed incrementally. That is the whole design, and the
reason it is small: change detection, deletion, permission updates, parent sections, PII
policy, keyword indexing, backups and index generations all already work on corpus files, so
a synced document gets every one of them without any of them knowing connectors exist.

What the engine adds is the part specific to mirroring something you do not control:

* **Fetch only what changed.** The source's own version marker is compared with the one
  recorded at the last sync, so an unchanged document costs a listing entry, not a download.
* **Deletion sync.** A document that has disappeared from the listing is deleted from the
  corpus, and the next ingest removes its chunks. A search index that keeps serving a page
  someone deleted -- often deleted *because* it should not be read -- is worse than one that
  never had it.
* **A guard on deletion.** An expired token, a revoked share or an API outage can make a
  source *list* as empty or nearly so, and faithful deletion sync would propagate that as "the
  whole space was deleted". A sync that would delete more than CONNECTOR_MAX_DELETE_FRACTION of
  what it previously held is refused -- nothing is deleted, the run is reported as refused,
  and an operator can re-run it with `--force` once they have confirmed the deletion is real.
* **A failed listing changes nothing.** Deletion is only ever inferred from a *complete*
  listing; one that raised halfway through is not evidence that the rest was deleted.
* **A failed fetch keeps the previous copy.** One unreachable attachment must not cost the
  corpus its last good version, and must not fail the other 999 documents.
"""

import contextlib
import fcntl
import hashlib
import json
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag_assistant import metrics
from rag_assistant.config import get_settings
from rag_assistant.connectors.base import (
    ConnectorConfig,
    RemoteDocument,
    build_connector,
    load_connector_configs,
    supported_suffix,
)
from rag_assistant.ingestion.acl import sidecar_path, write_acl
from rag_assistant.ingestion.ownership import owner_corpus_dir

logger = logging.getLogger(__name__)

SOURCES_DIR = "_sources"
STATE_FILENAME = ".sync_state.json"
_LOCK_FILENAME = ".sync.lock"
_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")
# Below this many deletions the fraction test does not apply (an empty listing still trips
# the guard): a source of a handful of documents legitimately loses most of them at once.
_GUARD_MIN_DELETIONS = 3


class SyncInProgress(RuntimeError):
    pass


@dataclass
class SyncResult:
    connector: str
    owner: str
    outcome: str = "ok"
    listed: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    permissions_changed: int = 0
    deleted: int = 0
    failed: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    indexed_chunks: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.deleted or self.permissions_changed)


def connector_dir(config: ConnectorConfig) -> Path:
    return owner_corpus_dir(get_settings().corpus_dir, config.owner) / SOURCES_DIR / config.name


def local_filename(document: RemoteDocument) -> str:
    """A stable, safe filename for a remote document: its title for readability, a hash of
    its id for uniqueness. The `_<8 hex>` suffix is the same shape uploads carry, so the
    router and citations strip it the same way. A rename upstream changes the filename, which
    the engine handles as delete-then-add."""
    slug = _SLUG_RE.sub("_", Path(document.title).stem).strip("_")[:80] or "document"
    digest = hashlib.sha256(document.id.encode()).hexdigest()[:8]
    return f"{slug}_{digest}{document.suffix.lower()}"


def load_state(directory: Path) -> dict:
    path = directory / STATE_FILENAME
    if not path.exists():
        return {"items": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(directory: Path, state: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / STATE_FILENAME
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _remove(path: Path) -> None:
    path.unlink(missing_ok=True)
    sidecar_path(path).unlink(missing_ok=True)


@contextlib.contextmanager
def _sync_lock(directory: Path) -> Iterator[None]:
    """One sync per connector at a time, across processes on this host. Non-blocking: a
    second run that finds the lock held reports that rather than queueing behind the first,
    because two syncs of the same source can only produce the same result twice."""
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / _LOCK_FILENAME).open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SyncInProgress("a sync of this connector is already running") from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def sync_connector(
    config: ConnectorConfig,
    force: bool = False,
    ingest: bool = True,
    transport=None,
) -> SyncResult:
    """Mirrors one connector's source into its tenant's corpus and re-indexes what changed."""
    settings = get_settings()
    directory = connector_dir(config)
    result = SyncResult(connector=config.name, owner=config.owner, started_at=time.time())
    with _sync_lock(directory):
        state = load_state(directory)
        items: dict[str, dict] = state.setdefault("items", {})
        try:
            connector = build_connector(config, transport=transport)
            listed = list(connector.list_documents())
        except Exception as exc:
            logger.exception("connector %s: listing failed; nothing was changed", config.name)
            result.outcome = "error"
            result.error = f"listing failed: {exc}"
            return _finish(directory, state, result)

        listed = [d for d in listed if supported_suffix(d.suffix)]
        result.listed = len(listed)
        listed_ids = {d.id for d in listed}
        vanished = [item_id for item_id in items if item_id not in listed_ids]

        # Checked before anything is written, so a refused run leaves the corpus exactly as
        # the last good sync left it. Two triggers: a listing that came back empty while
        # documents were synced before (the signature of an expired credential or a revoked
        # share), and a deletion above the configured fraction. The second needs a few
        # documents behind it, or a source of one or two documents could never lose one.
        if vanished and items and not force:
            fraction = len(vanished) / len(items)
            emptied = not listed
            too_many = (
                fraction > settings.connector_max_delete_fraction
                and len(vanished) >= _GUARD_MIN_DELETIONS
            )
            if emptied or too_many:
                result.outcome = "refused"
                result.error = (
                    f"would delete {len(vanished)} of {len(items)} documents ({fraction:.0%}"
                    + (
                        ", the source listed nothing"
                        if emptied
                        else ", above CONNECTOR_MAX_DELETE_FRACTION="
                        f"{settings.connector_max_delete_fraction:.0%}"
                    )
                    + "); nothing was changed. If the deletion is real, re-run with --force."
                )
                logger.error("connector %s: %s", config.name, result.error)
                return _finish(directory, state, result)

        for document in listed:
            _apply(connector, document, directory, items, result)

        for item_id in vanished:
            _remove(directory / items[item_id]["path"])
            del items[item_id]
            result.deleted += 1

        if result.failed:
            result.outcome = "partial"

        if ingest and result.changed:
            from rag_assistant.ingestion.build_index import build_index

            indexed = build_index(owner=config.owner)
            result.indexed_chunks = indexed.indexed_chunks
            # Charged to the tenant the documents were synced into, like an upload would be.
            from rag_assistant import budget

            budget.charge_ingest(
                config.owner,
                indexed.embedded_chars,
                indexed.vision_calls,
                indexed.description_calls,
            )
        return _finish(directory, state, result)


def _apply(connector, document: RemoteDocument, directory: Path, items: dict, result) -> None:
    filename = local_filename(document)
    path = directory / filename
    previous = items.get(document.id)
    acl_hash = document.acl.fingerprint()

    if previous and previous["path"] != filename:
        # Renamed or moved upstream: the old file goes, the new one is fetched below.
        _remove(directory / previous["path"])
        previous = None

    if previous and previous.get("version") == document.version and path.exists():
        if previous.get("acl_hash") != acl_hash:
            write_acl(path, document.acl)
            previous["acl_hash"] = acl_hash
            result.permissions_changed += 1
        else:
            result.unchanged += 1
        return

    try:
        data = connector.fetch(document)
    except Exception as exc:
        logger.warning(
            "connector %s: could not fetch %r; keeping any previous copy",
            result.connector,
            document.title,
            exc_info=True,
        )
        result.failed.append(f"{document.title}: {exc}")
        return
    if not data:
        result.failed.append(f"{document.title}: empty document")
        return

    # Permissions before content: the next ingest must never see the new bytes without the
    # sidecar that restricts them.
    write_acl(path, document.acl)
    _write_atomic(path, data)
    items[document.id] = {
        "path": filename,
        "version": document.version,
        "acl_hash": acl_hash,
        "title": document.title,
        "url": document.url,
    }
    if previous:
        result.updated += 1
    else:
        result.added += 1


def _finish(directory: Path, state: dict, result: SyncResult) -> SyncResult:
    result.finished_at = time.time()
    summary = asdict(result)
    summary["failed"] = result.failed[:20]
    state["last_run"] = summary
    if result.outcome in ("ok", "partial"):
        state["last_success_at"] = result.finished_at
    _save_state(directory, state)
    metrics.record_connector_sync(
        result.connector,
        result.outcome,
        {
            "added": result.added,
            "updated": result.updated,
            "deleted": result.deleted,
            "permissions": result.permissions_changed,
        },
    )
    logger.info(
        "connector sync finished",
        extra={
            "route": result.connector,
            "node": (
                f"outcome={result.outcome} added={result.added} updated={result.updated} "
                f"deleted={result.deleted} acl={result.permissions_changed} "
                f"failed={len(result.failed)}"
            ),
        },
    )
    return result


def connector_status(config: ConnectorConfig) -> dict:
    """What the last sync did and when the next is due, for the CLI and the API."""
    state = load_state(connector_dir(config))
    last_success = state.get("last_success_at")
    return {
        "name": config.name,
        "type": config.type,
        "owner": config.owner,
        "interval_minutes": config.interval_minutes,
        "documents": len(state.get("items", {})),
        "last_success_at": last_success,
        "last_run": state.get("last_run"),
        "due": last_success is None or time.time() - last_success >= config.interval_minutes * 60,
    }


def run_due_syncs() -> list[SyncResult]:
    """Syncs every connector whose interval has elapsed. What the scheduler thread calls, and
    what a cron job running `rag-assistant connectors sync --due` calls."""
    results = []
    for config in load_connector_configs():
        if not connector_status(config)["due"]:
            continue
        try:
            results.append(sync_connector(config))
        except SyncInProgress:
            logger.info("connector %s: sync already running elsewhere; skipped", config.name)
        except Exception:
            logger.exception("connector %s: sync crashed", config.name)
    return results
