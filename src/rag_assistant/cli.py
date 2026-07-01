import typer
from rich.console import Console

from rag_assistant.graph.build_graph import build_graph
from rag_assistant.ingestion.build_index import build_index
from rag_assistant.llm import get_chat_model
from rag_assistant.logging_conf import configure_logging
from rag_assistant.retrieval.vector_store import get_retriever
from rag_assistant.retrieval.web_search import WebSearchTool

app = typer.Typer(help="Adaptive RAG Research Assistant")
console = Console()


@app.callback()
def callback() -> None:
    pass


@app.command()
def hello() -> None:
    """Prove end-to-end connectivity to Gemini."""
    configure_logging()
    try:
        response = get_chat_model().invoke("Reply with a short one-sentence greeting.")
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Gemini says:[/green] {response.content}")


@app.command()
def ingest() -> None:
    """Load, chunk, embed, and index the sample corpus into Chroma."""
    configure_logging()
    try:
        num_chunks = build_index()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Indexed[/green] {num_chunks} chunks into Chroma.")


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
    """Debug command: run a raw Tavily web search for a query."""
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
def ask(question: str) -> None:
    """Run the full adaptive research graph on a question."""
    configure_logging()
    try:
        result = build_graph().invoke({"question": question}, config={"recursion_limit": 50})
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[dim]route: {result['route']} -- {result['route_reasoning']}[/dim]")
    sub_queries = result.get("sub_queries", [])
    if len(sub_queries) > 1:
        console.print("[dim]sub-queries:[/dim]")
        for sub_query in sub_queries:
            console.print(f"  [dim]- {sub_query}[/dim]")
    console.print()
    console.print(result["final_answer"])
    if result["citations"]:
        console.print()
        console.print("[bold]Sources:[/bold]")
        for citation in result["citations"]:
            console.print(f"  {citation['marker']} {citation['source_id']}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
