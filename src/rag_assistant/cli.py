import typer
from rich.console import Console

from rag_assistant.ingestion.build_index import build_index
from rag_assistant.llm import get_chat_model
from rag_assistant.logging_conf import configure_logging
from rag_assistant.retrieval.vector_store import get_retriever

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
