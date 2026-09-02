import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from rag_assistant.eval.baseline import (
    DEFAULT_TOLERANCE,
    BaselineNotFound,
    compare,
    load_baseline,
    save_baseline,
)
from rag_assistant.eval.golden_dataset import load_golden_dataset
from rag_assistant.backup import (
    create_backup,
    prune_backups,
    read_backup_metadata,
    restore_backup,
)
from rag_assistant.config import get_settings
from rag_assistant.graph.build_graph import build_graph
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.llm import (
    get_chat_model,
    primary_chat_provider_name,
    responding_provider_name,
)
from rag_assistant.logging_conf import configure_logging
from rag_assistant.retrieval.vector_store import get_retriever
from rag_assistant.retrieval.web_search import WebSearchTool

# Graph execution alone costs ~4 Gemini calls/question (route, decompose, grade, synthesize),
# independent of whether --llm-judge adds further scoring calls -- this is the dominant,
# easy-to-underestimate cost against the 20-calls/day free-tier quota.
_GRAPH_CALLS_PER_QUESTION = 4
_LLM_JUDGE_CALLS_PER_QUESTION = 2  # Faithfulness + ResponseRelevancy, each one extra call

app = typer.Typer(help="Adaptive RAG Research Assistant")
console = Console()


@app.callback()
def callback() -> None:
    pass


@app.command()
def hello() -> None:
    """Prove end-to-end connectivity to the configured chat model (Anthropic if set, else Gemini)."""
    configure_logging()
    configured = primary_chat_provider_name()
    try:
        response = get_chat_model().invoke("Reply with a short one-sentence greeting.")
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    # Report who *answered*, not who was configured to be asked first. The two differ exactly
    # when the fallback chain fired, which is the case worth knowing about: a dead primary
    # credential otherwise hides behind a series of successful-looking commands.
    responder = responding_provider_name(response)
    console.print(f"[green]{responder or configured} says:[/green] {response.text}")
    if responder and configured.split()[0].lower() not in responder.lower():
        console.print(
            f"[yellow]Note:[/yellow] {configured} is configured as primary but did not answer -- "
            f"the fallback chain handled this request. Check that provider's credentials."
        )


@app.command()
def ingest(
    full: bool = typer.Option(
        False, "--full", help="Reset the collection and re-embed every file from scratch."
    ),
) -> None:
    """Load, chunk, embed, and index the corpus into Chroma. Incremental by default: only
    new or changed files are (re)embedded, and files removed from the corpus have their
    chunks removed too. Pass --full to force a clean rebuild."""
    configure_logging()
    try:
        result = build_index(incremental=not full)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Indexed[/green] {result.indexed_chunks} chunks across "
        f"{result.changed_files} changed file(s); {result.skipped_files} unchanged file(s) "
        f"skipped; {result.removed_files} removed file(s) cleaned up."
    )


@app.command()
def retrieve(question: str, k: int = 4) -> None:
    """Debug command: run a raw vector-store retrieval for a question."""
    configure_logging()
    try:
        docs = get_retriever(k=k).invoke(question)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not docs:
        console.print("[yellow]No results.[/yellow]")
        return
    for i, doc in enumerate(docs, start=1):
        console.print(f"[bold]{i}. {doc.metadata.get('source', 'unknown')}[/bold]")
        console.print(doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""))
        console.print()


@app.command()
def search(query: str, max_results: int = 5) -> None:
    """Debug command: run a raw DuckDuckGo web search for a query."""
    configure_logging()
    try:
        results = WebSearchTool().search(query, max_results=max_results)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not results:
        console.print("[yellow]No results.[/yellow]")
        return
    for i, doc in enumerate(results, start=1):
        title = doc.metadata.get("title", "unknown")
        url = doc.metadata.get("url", "")
        console.print(f"[bold]{i}. {title}[/bold] ({url})")
        console.print(doc.content[:200] + ("..." if len(doc.content) > 200 else ""))
        console.print()


@app.command(name="loadtest")
def loadtest_(
    url: str = typer.Option("http://127.0.0.1:8000", help="Base URL of a running server."),
    path: str = typer.Option("/health", help="Path to hit."),
    requests: int = typer.Option(200, help="Total requests to send."),
    concurrency: int = typer.Option(10, help="Requests in flight at once."),
    question: str | None = typer.Option(
        None, help="Send this question to /api/v1/research instead (costs LLM calls)."
    ),
    api_key: str | None = typer.Option(None, help="X-API-Key, if the server requires one."),
) -> None:
    """Measure latency and throughput under concurrency against a running server.

    Defaults to /health, which exercises the HTTP stack, middleware chain and event loop for
    free. Passing --question points it at the research endpoint instead, which costs several
    LLM calls per request -- the estimate is printed before anything is sent.
    """
    configure_logging()
    method, payload = "GET", None
    target = path
    if question:
        method, target = "POST", "/api/v1/research"
        payload = {"question": question, "save": False}
        console.print(
            f"[yellow]This sends {requests} research requests -- roughly "
            f"{requests * _GRAPH_CALLS_PER_QUESTION} model calls. Ctrl-C now if that isn't "
            f"what you want.[/yellow]"
        )

    headers = {"X-API-Key": api_key} if api_key else None
    console.print(f"[bold]{method} {url}{target}[/bold] x{requests} at concurrency {concurrency}")

    import asyncio

    from rag_assistant.loadtest import run_load_test

    result = asyncio.run(
        run_load_test(
            base_url=url,
            path=target,
            method=method,
            total_requests=requests,
            concurrency=concurrency,
            payload=payload,
            headers=headers,
        )
    )

    summary = result.summary()
    table = Table(title="Load test")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key in ("total_requests", "concurrency", "wall_seconds", "throughput_rps", "error_rate"):
        table.add_row(key, str(summary[key]))
    for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms"):
        table.add_row(key, str(summary[key]))
    table.add_row("status_counts", str(summary["status_counts"]))
    console.print(table)

    if result.errors:
        console.print(
            f"[red]{len(result.errors)} connection error(s)[/red]; first: {result.errors[0]}"
        )
    if summary["error_rate"] > 0:
        raise typer.Exit(code=1)


@app.command()
def backup(
    output: Path | None = typer.Option(None, help="Directory to write the archive into."),
    keep: int = typer.Option(0, help="Delete all but this many newest archives (0 = keep all)."),
) -> None:
    """Snapshot the Chroma index, the ingestion manifest, the conversation database, and the
    corpus into one timestamped archive. SQLite files are captured with SQLite's online backup
    API, so the archive is consistent even if the server is running."""
    configure_logging()
    try:
        archive = create_backup(output_dir=output)
    except Exception as exc:
        console.print(f"[red]Backup failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    metadata = read_backup_metadata(archive)
    size_mb = archive.stat().st_size / (1024 * 1024)
    console.print(f"[green]Wrote[/green] {archive} ({size_mb:.1f} MB)")
    console.print(
        f"  {metadata.indexed_sources} indexed source(s), {metadata.corpus_files} corpus file(s), "
        f"{metadata.conversations} conversation(s), embeddings={metadata.embedding_model}"
    )

    if keep:
        removed = prune_backups(archive.parent, keep=keep)
        if removed:
            console.print(f"[yellow]Pruned {len(removed)} older archive(s).[/yellow]")


@app.command()
def restore(
    archive: Path = typer.Argument(..., help="Backup archive produced by `backup`."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
    discard_previous: bool = typer.Option(
        False, "--discard-previous", help="Delete the current data instead of moving it aside."
    ),
) -> None:
    """Replace the index and corpus with a backup's contents.

    The current data is moved aside (not deleted) unless --discard-previous, and the swap
    happens only after the archive extracts cleanly -- a corrupt archive fails with the live
    data untouched.
    """
    configure_logging()
    if not archive.exists():
        console.print(f"[red]No such archive: {archive}[/red]")
        raise typer.Exit(code=1)

    try:
        metadata = read_backup_metadata(archive)
    except Exception as exc:
        console.print(f"[red]Not a readable rag-assistant backup: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]Archive:[/bold] {archive}")
    console.print(f"  created {metadata.created_at}")
    console.print(
        f"  {metadata.indexed_sources} indexed source(s), {metadata.corpus_files} corpus file(s), "
        f"{metadata.conversations} conversation(s)"
    )
    console.print(f"  embeddings: {metadata.embedding_model}")

    if not yes:
        settings = get_settings()
        console.print(
            f"\n[yellow]This replaces {settings.chroma_persist_dir} and {settings.corpus_dir}."
            "[/yellow]"
        )
        typer.confirm("Continue?", abort=True)

    try:
        result = restore_backup(archive, keep_previous=not discard_previous)
    except Exception as exc:
        console.print(f"[red]Restore failed (live data left in place): {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Restored[/green] index={result.restored_chroma} corpus={result.restored_corpus}"
    )
    if result.previous_kept_at:
        console.print(f"  previous data kept alongside it in {result.previous_kept_at}")
    if result.embedding_model_changed:
        # Restoring an index built with a different embedding model is the silent-corruption
        # case index_metadata.py exists to catch -- say so now rather than let /ready say it.
        console.print(
            f"[red]Warning:[/red] this backup was built with embeddings "
            f"{metadata.embedding_model!r} but {get_settings().gemini_embedding_model!r} is "
            f"configured. /ready will report unavailable until you re-index with "
            f"`rag-assistant ingest --full` or restore the previous model setting."
        )
    console.print(
        "[yellow]Restart the server[/yellow] -- the index, BM25 and conversation "
        "caches are process-local and still hold pre-restore state."
    )


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the FastAPI server exposing POST /research."""
    import uvicorn

    configure_logging()
    uvicorn.run("rag_assistant.api:app", host=host, port=port, reload=reload)


@app.command()
def mcp() -> None:
    """Run the MCP server over stdio, exposing the research pipeline as tools for MCP
    clients (Claude Desktop, Claude Code, agent frameworks). Configure the client to launch
    this command; logs go to stderr so the protocol stream on stdout stays clean."""
    configure_logging()
    from rag_assistant.mcp_server import run

    run()


@app.command()
def ask(question: str) -> None:
    """Run the full adaptive research graph on a question."""
    configure_logging()
    try:
        result = build_graph().invoke({"question": question}, config={"recursion_limit": 50})
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(Markdown(result["research_report"]))


@app.command(name="eval")
def eval_(
    llm_judge: bool = False,
    limit: int = 3,
    output: Path | None = None,
    check: bool = typer.Option(
        False,
        "--check",
        help="Compare against the recorded baseline and exit non-zero on regression.",
    ),
    record_baseline: bool = typer.Option(
        False, "--record-baseline", help="Overwrite the baseline with this run's scores."
    ),
    tolerance: float = typer.Option(
        DEFAULT_TOLERANCE, help="How far a metric may fall below baseline before --check fails."
    ),
) -> None:
    """Run the RAGAS eval harness against the golden dataset. `limit` defaults to 3 (not the
    full dataset) because graph execution alone costs ~4 chat-model calls/question -- the
    primary quota lever when running on the Gemini free tier, independent of --llm-judge
    which only adds further scoring calls."""
    configure_logging()

    provider = primary_chat_provider_name()
    graph_calls = limit * _GRAPH_CALLS_PER_QUESTION
    judge_calls = limit * _LLM_JUDGE_CALLS_PER_QUESTION if llm_judge else 0
    total = graph_calls + judge_calls
    quota_note = (
        " against the 20/day free-tier quota"
        if provider == "Gemini"
        else " (Gemini free-tier"
        " quota no longer applies since Anthropic is primary; embeddings still call Gemini"
        " separately)"
    )
    console.print(
        f"[yellow]Estimated {provider} calls: ~{graph_calls} for graph execution"
        + (f" + ~{judge_calls} for LLM-judge scoring" if llm_judge else "")
        + f" = ~{total} total{quota_note}.[/yellow]"
    )

    from rag_assistant.eval.run_eval import compute_metrics, run_eval

    try:
        results, eval_result = run_eval(limit=limit, llm_judge=llm_judge)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Golden question checks")
    table.add_column("Question")
    table.add_column("Category")
    table.add_column("Route")
    table.add_column("Sources")
    for r in results:
        route_cell = (
            f"{r.actual_route} ({'✓' if r.route_match else '✗ expected ' + r.expected_route})"
        )
        if r.expected_sources:
            sources_cell = "✓" if r.source_overlap else f"✗ expected {r.expected_sources}"
        else:
            # Nothing to retrieve: the check is whether it correctly declined to cite.
            sources_cell = "✓ abstained" if r.citation_count == 0 else f"✗ cited {r.citation_count}"
        table.add_row(r.question, r.category, route_cell, sources_cell)
    console.print(table)

    deterministic = compute_metrics(results)
    scores = deterministic.gated_scores()
    console.print("[bold]Retrieval metrics (deterministic):[/bold]")
    for name, value in scores.items():
        console.print(f"  {name}: {value:.3f}")

    ragas_metrics = eval_result.to_pandas().mean(numeric_only=True).to_dict()
    console.print("[bold]RAGAS metrics:[/bold]", ragas_metrics)

    if output:
        output.write_text(
            json.dumps(
                {
                    "results": [r.__dict__ for r in results],
                    "retrieval_metrics": scores,
                    "ragas_metrics": ragas_metrics,
                },
                indent=2,
            )
        )
        console.print(f"[green]Wrote results to {output}[/green]")

    if record_baseline:
        written = save_baseline(deterministic)
        console.print(f"[green]Recorded baseline to {written}[/green]")

    if check:
        # A partial run can't be compared against a whole-dataset baseline: --limit 3 scores
        # three questions, and whether those three are the easy ones is luck, not quality.
        dataset_size = len(load_golden_dataset())
        if limit < dataset_size:
            console.print(
                f"[red]--check needs the full dataset ({dataset_size} questions); "
                f"--limit is {limit}. Re-run with --limit {dataset_size}.[/red]"
            )
            raise typer.Exit(code=2)

        try:
            baseline = load_baseline()
        except BaselineNotFound as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc

        comparison = compare(deterministic, baseline, tolerance=tolerance)
        gate = Table(title=f"Baseline comparison (tolerance {tolerance:.2f})")
        gate.add_column("Metric")
        gate.add_column("Baseline", justify="right")
        gate.add_column("Current", justify="right")
        gate.add_column("Delta", justify="right")
        for c in comparison.comparisons:
            marker = "[red]REGRESSED[/red]" if c.regressed else "[green]ok[/green]"
            gate.add_row(
                c.name, f"{c.baseline:.3f}", f"{c.current:.3f}", f"{c.delta:+.3f} {marker}"
            )
        console.print(gate)

        if not comparison.passed:
            names = ", ".join(c.name for c in comparison.regressions)
            console.print(f"[red]Eval gate failed -- regressed: {names}[/red]")
            raise typer.Exit(code=1)
        console.print("[green]Eval gate passed.[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
