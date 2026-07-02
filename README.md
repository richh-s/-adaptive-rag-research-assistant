# Adaptive RAG Research Assistant

Ask a research question. The system autonomously decides whether to retrieve from a local
document store, search the web, or both, decomposes compound questions into sub-queries, fuses
results across retrieval paths, checks its own confidence, and falls back to web search when the
local knowledge base comes up short — then synthesizes a cited, transparency-reported answer.

Built with LangGraph, Google Gemini (free tier), Chroma, and Tavily — no paid services required.

## Concepts demonstrated

- **Agentic / Self-RAG routing** — an LLM router decides per-query whether to hit the local
  vector store, the web, both, or neither, before any retrieval happens.
- **Query decomposition** — compound questions are broken into focused, self-contained
  sub-queries that are retrieved independently and fused back together.
- **RAG Fusion (Reciprocal Rank Fusion)** — results from every sub-query and every retrieval path
  are merged and reranked by RRF score, not concatenated or naively deduplicated.
- **Confidence scoring / Corrective-RAG** — retrieved documents are graded for relevance; when
  confidence on a vector-only route falls below threshold, the system automatically falls back to
  a web search before answering.
- **Citation-mapped synthesis** — citation markers are assigned deterministically from fused rank
  order in code, not left to the LLM to invent.
- **LangGraph orchestration** — the whole pipeline is a `StateGraph` with conditional edges and
  `Send`-based fan-out for parallel sub-query retrieval, not a linear chain.

## Architecture

```
route_query --[none]--> synthesize_answer
    |
   [vector | web | both]
    v
decompose_query --Send--> retrieve_vector / web_search  (one per sub-query, per route)
    |
    v
fuse_results (Reciprocal Rank Fusion) <---------------------------+
    |                                                             |
    v                                                             |
grade_and_score --[low confidence, vector-only, not yet tried]----+
    |                                                  (corrective_web_search)
   [confident enough]
    v
synthesize_answer --> format_report --> END
```

`corrective_web_search` loops back into `fuse_results` at most once per question (guarded by
`correction_attempted` in state, backstopped by a `recursion_limit`).

## Setup

```bash
uv sync
cp .env.example .env
# fill in GOOGLE_API_KEY (https://aistudio.google.com/apikey) and
# TAVILY_API_KEY (https://app.tavily.com) in .env

uv run rag-assistant hello    # confirms Gemini connectivity
uv run rag-assistant ingest   # embeds the sample corpus (data/corpus/) into Chroma
```

## Usage

### CLI

```bash
uv run rag-assistant ask "Who founded Anthropic and what is their safety research called?"
uv run rag-assistant ask "What is the most recent Claude model release?"
uv run rag-assistant ask "Compare Anthropic and Mistral AI's founding stories and safety focus."
```

Debug commands for individual pieces of the pipeline:

```bash
uv run rag-assistant retrieve "anthropic founders" --k 4   # raw vector-store retrieval
uv run rag-assistant search "claude model releases 2026"   # raw Tavily web search
```

### API

```bash
uv run rag-assistant serve   # starts FastAPI on http://127.0.0.1:8000
```

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "Who founded Anthropic and what is their safety research called?"}'
```

Interactive API docs at `http://127.0.0.1:8000/docs`.

### Web UI

A React + Vite single-page app in `frontend/` calls the same API (`/health`, `/research`) and
renders the markdown report with routing/confidence metadata and one-click example questions.

```bash
uv run rag-assistant serve       # terminal 1 -- backend on http://127.0.0.1:8000

cd frontend
npm install
npm run dev                      # terminal 2 -- UI on http://localhost:5173
```

The backend allows CORS from `http://localhost:5173` by default. If the backend runs elsewhere,
copy `frontend/.env.example` to `frontend/.env` and set `VITE_API_BASE_URL`.

## Example questions per concept

| Concept | Example question |
| --- | --- |
| Vector routing | "Who founded Anthropic and what is their safety research called?" |
| Web routing | "What is the most recent Claude model release?" |
| No retrieval (general knowledge) | "What is the capital of France?" |
| Query decomposition | "Compare Anthropic and Mistral AI's founding stories and safety focus." |
| Corrective fallback | "What safety research did Anthropic publish this week?" (recent/narrow enough that the local corpus alone often scores low, triggering a web search fallback) |

## Testing

```bash
uv run pytest          # offline unit + node + e2e tests (no external API calls)
uv run pytest -m live  # also exercises real Gemini/Tavily calls; requires .env and a run of `ingest` first
uv run ruff check .
```

## Project layout

```
src/rag_assistant/
├── config.py, llm.py, logging_conf.py   # settings, model factories, logging
├── ingestion/                            # load -> split -> embed -> index the sample corpus
├── retrieval/                            # Chroma vector store + Tavily web search wrapper
├── fusion/rrf.py                         # Reciprocal Rank Fusion (pure function)
├── grading/relevance_grader.py           # batched LLM relevance grading
├── graph/                                # ResearchState, one node module per concept, build_graph()
├── prompts/                              # prompt templates per LLM-backed node
├── schemas/models.py                     # internal domain / structured-output schemas
├── schemas/api.py                        # external API request/response contracts
├── cli.py                                # Typer app: hello / ingest / retrieve / search / ask / serve
└── api.py                                # FastAPI: GET /health, POST /research

frontend/src/
├── api/client.ts                         # fetch wrapper for the backend API
├── hooks/useHealthStatus.ts              # polls GET /health on mount
├── constants/exampleQuestions.ts         # example-question chip data
├── components/                           # Header, AskCard, ResultCard, ErrorBanner, ErrorBoundary
├── App.tsx                               # composition root
└── index.css                             # shared theme (light/dark)
```
