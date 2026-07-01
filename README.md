# Adaptive RAG Research Assistant

Ask a research question. The system autonomously decides whether to retrieve from a local
document store, search the web, or both, decomposes compound questions into sub-queries, fuses
results across retrieval paths, checks its own confidence, and falls back to web search when the
local knowledge base comes up short — then synthesizes a cited report.

Built with LangGraph, Google Gemini (free tier), Chroma, and Tavily.

## Concepts demonstrated

- **Agentic / Self-RAG** — an LLM router decides retrieval strategy per query.
- **Query decomposition** — compound questions are broken into focused sub-queries.
- **RAG Fusion** — Reciprocal Rank Fusion merges/reranks results across sub-queries and sources.
- **Confidence scoring / Corrective-RAG** — low-confidence retrieval triggers a web search fallback.
- **LangGraph orchestration** — the whole pipeline is a `StateGraph` with conditional edges, not a
  linear chain.

## Setup

```bash
uv sync
cp .env.example .env
# fill in GOOGLE_API_KEY (https://aistudio.google.com/apikey) and
# TAVILY_API_KEY (https://app.tavily.com) in .env
uv run rag-assistant hello   # confirms Gemini connectivity
```

## Status

This project is being built incrementally, phase by phase. See `hello` above for Phase 0
(scaffolding + config). Later phases add document ingestion, web search, and the full LangGraph
research agent.
