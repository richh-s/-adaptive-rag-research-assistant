import hashlib
import json
from pathlib import Path

MANIFEST_FILENAME = "ingestion_manifest.json"


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def manifest_path(persist_dir: Path) -> Path:
    return Path(persist_dir) / MANIFEST_FILENAME


def load_manifest(persist_dir: Path) -> dict[str, dict]:
    """Maps each source filename to the hash of its last-indexed content and the Chroma
    chunk IDs produced from it, so build_index can tell what changed without re-embedding
    everything."""
    path = manifest_path(persist_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_manifest(persist_dir: Path, manifest: dict[str, dict]) -> None:
    path = manifest_path(persist_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
