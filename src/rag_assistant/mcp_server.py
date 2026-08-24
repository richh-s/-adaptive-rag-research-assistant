"""MCP (Model Context Protocol) server: exposes the research pipeline as tools that any
MCP client -- Claude Desktop, Claude Code, agent frameworks -- can call directly.

Runs over stdio (launched by the client, speaks the protocol on stdout), so two rules hold
throughout: never print to stdout (logging_conf's StreamHandler goes to stderr, which is
safe), and call the pipeline in-process rather than through the HTTP API -- the server runs
on the user's own machine under their own credentials, so the API layer's auth/rate-limit
concerns don't apply.

Tool results are plain markdown/text: MCP clients hand them to a model, so human-readable
beats JSON. The graph is built lazily on first use -- importing this module (e.g. by the CLI
for --help) must not require configured API keys.
"""

import json
import logging
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from rag_assistant.graph.research_summary import build_research_summary

logger = logging.getLogger(__name__)

server = MCPServer(
    "adaptive-rag",
    instructions=(
        "Research assistant over a local knowledge base plus live web search. Use "
        "research_question for anything the user's indexed documents might answer; use "
        "ingest_file/ingest_url to add new material to the knowledge base first if needed."
    ),
)

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from rag_assistant.graph.build_graph import build_graph

        _graph = build_graph()
    return _graph


@server.tool()
def research_question(question: str) -> str:
    """Answer a research question with citations. Automatically decides whether to search
    the local knowledge base (the user's indexed documents), the live web, or both; fuses
    and grades results; falls back to web search when local confidence is low. Returns a
    markdown report ending with its sources and a transparency summary."""
    result = _get_graph().invoke(
        {"question": question, "chat_history": [], "trace_id": "mcp"},
        config={"recursion_limit": 50},
    )
    summary = build_research_summary(result)
    footer = (
        f"\n\n---\nroute: {summary.route} | confidence: {summary.confidence_score} | "
        f"sources fused: {summary.fused_document_count} | "
        f"corrective search: {'yes' if summary.correction_attempted else 'no'}"
    )
    return result.get("research_report", "") + footer


@server.tool()
def ingest_file(path: str) -> str:
    """Add a local document (PDF, DOCX, HTML, Markdown, or plain text) to the knowledge
    base and index it. PDFs get vision treatment: embedded charts/figures are described and
    scanned pages transcribed, so their contents become searchable too. Pass an absolute
    file path."""
    from rag_assistant.config import get_settings
    from rag_assistant.ingestion.build_index import build_index
    from rag_assistant.ingestion.loaders import SUPPORTED_SUFFIXES

    source = Path(path).expanduser()
    if not source.is_file():
        return f"Error: {path!r} is not a file."
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        return f"Error: unsupported file type {source.suffix!r}. Supported: {sorted(SUPPORTED_SUFFIXES)}."

    corpus_dir = get_settings().corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    destination = corpus_dir / source.name
    destination.write_bytes(source.read_bytes())

    result = build_index()
    return (
        f"Ingested {source.name}: {result.indexed_chunks} chunk(s) indexed from "
        f"{result.changed_files} changed file(s). The content is now searchable via "
        f"research_question."
    )


@server.tool()
def ingest_url(url: str) -> str:
    """Fetch a public web page and add its content to the knowledge base. Only http(s)
    URLs to public hosts are allowed; JavaScript-only pages may yield little text."""
    from rag_assistant.config import get_settings
    from rag_assistant.ingestion.build_index import build_index
    from rag_assistant.ingestion.url_fetch import UrlIngestError, fetch_page, page_to_markdown

    if not url.lower().startswith(("http://", "https://")):
        return "Error: url must start with http:// or https://."
    try:
        page = fetch_page(url)
    except UrlIngestError as exc:
        return f"Error: {exc}"

    corpus_dir = get_settings().corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() else "_" for c in (page.title or "webpage"))[:60].strip("_")
    destination = corpus_dir / f"{stem or 'webpage'}.md"
    destination.write_text(page_to_markdown(page), encoding="utf-8")

    result = build_index()
    return (
        f"Ingested {page.title or url!r} ({result.indexed_chunks} chunk(s) from "
        f"{result.changed_files} changed file(s)). The page is now searchable via "
        f"research_question."
    )


@server.tool()
def list_documents() -> str:
    """List every document currently indexed in the knowledge base, with chunk counts --
    useful for checking what research_question can draw on before asking."""
    from rag_assistant.config import get_settings
    from rag_assistant.ingestion.manifest import load_manifest

    manifest = load_manifest(get_settings().chroma_persist_dir)
    if not manifest:
        return "The knowledge base is empty -- ingest_file or ingest_url can add documents."
    lines = [
        f"- {source} ({len(entry.get('chunk_ids', []))} chunk(s))"
        for source, entry in sorted(manifest.items())
    ]
    return f"{len(manifest)} indexed document(s):\n" + "\n".join(lines)


@server.tool()
def list_conversations() -> str:
    """List research conversations saved by the web UI/API (title, id, message count),
    most recently active first."""
    from rag_assistant.conversations import store

    rows = store.list_conversations(owner="public")
    if not rows:
        return "No saved conversations."
    return json.dumps(
        [
            {"id": r.id, "title": r.title, "messages": r.message_count, "updated_at": r.updated_at}
            for r in rows
        ],
        indent=2,
    )


def run() -> None:
    """Entry point for `rag-assistant mcp`: serve over stdio until the client disconnects."""
    server.run()
