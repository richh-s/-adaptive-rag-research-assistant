import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from rag_assistant.graph.build_graph import build_graph
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.llm import get_chat_model, primary_chat_provider_name
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
    provider = primary_chat_provider_name()
    try:
        response = get_chat_model().invoke("Reply with a short one-sentence greeting.")
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]{provider} says:[/green] {response.content}")


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


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the FastAPI server exposing POST /research."""
    import uvicorn

    configure_logging()
    uvicorn.run("rag_assistant.api:app", host=host, port=port, reload=reload)


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
        " against the 20/day free-tier quota" if provider == "Gemini" else " (Gemini free-tier"
        " quota no longer applies since Anthropic is primary; embeddings still call Gemini"
        " separately)"
    )
    console.print(
        f"[yellow]Estimated {provider} calls: ~{graph_calls} for graph execution"
        + (f" + ~{judge_calls} for LLM-judge scoring" if llm_judge else "")
        + f" = ~{total} total{quota_note}.[/yellow]"
    )

    from rag_assistant.eval.run_eval import run_eval

    try:
        results, eval_result = run_eval(limit=limit, llm_judge=llm_judge)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Golden question checks")
    table.add_column("Question")
    table.add_column("Route")
    table.add_column("Sources")
    for r in results:
        route_cell = f"{r.actual_route} ({'✓' if r.route_match else '✗ expected ' + r.expected_route})"
        sources_cell = "✓" if r.source_overlap else f"✗ expected {r.expected_sources}"
        table.add_row(r.question, route_cell, sources_cell)
    console.print(table)

    metrics = eval_result.to_pandas().mean(numeric_only=True).to_dict()
    console.print("[bold]RAGAS metrics:[/bold]", metrics)

    if output:
        output.write_text(
            json.dumps({"results": [r.__dict__ for r in results], "metrics": metrics}, indent=2)
        )
        console.print(f"[green]Wrote results to {output}[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
