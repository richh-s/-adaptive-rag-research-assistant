# Adaptive RAG Research Assistant

Ask a research question — then keep the conversation going with follow-ups. The system
resolves follow-up references against the chat history, autonomously decides whether to
retrieve from a local document store, search the web, or both, decomposes compound questions
into sub-queries, retrieves with both dense (vector) and sparse (BM25) search, fuses results
across every retrieval path, checks its own confidence, and falls back to web search when the
local knowledge base comes up short — then synthesizes a cited, transparency-reported answer,
streamed live to the browser as each step of the pipeline runs. The knowledge base ingests
PDF, Word, HTML, Markdown, and text files, or any public web page by URL.

Built with LangGraph, Chroma, and DuckDuckGo web search. Chat/reasoning defaults to Anthropic's Claude when an
`ANTHROPIC_API_KEY` is set, with automatic fallback to Google Gemini (free tier) on error.
Embeddings are a separate choice (`EMBEDDING_PROVIDER`): Gemini by default, OpenAI, or any
OpenAI-compatible server you host yourself. No paid services are required — leave
`ANTHROPIC_API_KEY` blank to run entirely on Gemini's free tier.

<!--
  TODO(portfolio polish): drop a screenshot or short GIF of the web UI here, e.g.
  ![Research summary panel](docs/screenshot-summary-panel.png)
  A ~3-5 min demo video link (YouTube/Loom) can go right below it.
-->

## Concepts demonstrated

- **Conversational memory / follow-up condensation** — a `condense_question` node rewrites
  follow-ups ("what about their pricing?") into standalone questions before routing, preserving
  what the user literally typed for the transparency panel ("interpreted as: ..."). Synthesis
  also sees recent turns, so answers read as a continuation instead of restarting the topic.
- **Persistent conversations** — every exchange is stored server-side (SQLite, WAL) under a
  conversation id: the server owns the transcript (clients can't forge or truncate history),
  conversations survive restarts, and `GET/DELETE /api/v1/conversations[/{id}]` powers a
  history sidebar in the UI where any conversation can be reopened and continued. Stateless
  callers can pass `save: false` (with an optional inline `history`) to opt out entirely.
- **Agentic / Self-RAG routing** — an LLM router decides per-query whether to hit the local
  knowledge base, the web, both, or neither, before any retrieval happens.
- **Query decomposition** — compound questions are broken into focused, self-contained
  sub-queries that are retrieved independently and fused back together.
- **Hybrid retrieval (dense + sparse)** — every sub-query is retrieved via both a Chroma vector
  store (semantic similarity) and BM25 keyword search over the same corpus, so exact
  names/acronyms that embeddings under-rank still surface.
- **RAG Fusion (Reciprocal Rank Fusion)** — results from every sub-query and every retrieval path
  (vector, BM25, web) are merged and reranked by RRF score, not concatenated or naively
  deduplicated.
- **Confidence scoring / Corrective-RAG** — retrieved documents are graded for relevance, and a
  low grade escalates in order of what is likeliest and cheapest rather than straight out of the
  corpus. First the corpus is re-asked: an LLM rewrites the sub-queries into the vocabulary a
  document would use (formal terms over casual ones, abbreviations expanded the other way) and
  re-retrieves with a widened `k`. Only if that still grades badly does a web search run. The
  reason is that a low grade says retrieval failed and says nothing about *why* — escalating
  straight to the web assumed the corpus could not contain the answer, when the ordinary cause
  is a question phrased the way a person asks rather than the way a document writes, which is
  the one case where leaving the corpus cannot help. The `both` route, which previously got no
  correction at all, now gets the local retry.
- **Answer groundedness verification** — everything else in the graph grades *retrieval*; this
  checks the answer. After synthesis, one structured call decomposes the answer into claims and
  asks whether the numbered context actually supports each one. The score, the unsupported
  claims and a `checked` flag reach the transparency panel and Prometheus, and the report
  carries a visible caveat below threshold. It reports rather than intervenes: regenerating
  doubles cost on exactly the questions the corpus is thinnest on, and stripping sentences
  edits prose on the word of a check that is itself a model call. The distinction it keeps is
  between "verified and clean" and "not verified" — a failed or disabled check is never
  reported as a clean bill of health.
- **Personal data detection and redaction** — ingested text is scanned for emails, phone
  numbers, Luhn-valid card numbers, SSNs, IBANs and cloud access keys. `PII_MODE=flag` (the
  default) counts and logs; `redact` replaces each match with a category marker *before* the
  text is embedded, keyword-indexed or stored as a parent section, leaving the uploaded file on
  disk untouched so the decision stays reversible. Honest boundary: regexes find formats, not
  people — a name, an address or a date of birth passes straight through.
- **Grade-informed reranking** — the relevance grades bought for confidence scoring are reused
  (zero extra LLM calls) to rerank the synthesis context: graded-relevant documents move to the
  front ordered by semantic relevance, graded-irrelevant ones are pruned so they can't pollute
  the answer or earn a citation.
- **Citation-mapped synthesis** — citation markers are assigned deterministically from fused rank
  order in code, not left to the LLM to invent.
- **LangGraph orchestration** — the whole pipeline is a `StateGraph` with conditional edges and
  `Send`-based fan-out for parallel sub-query retrieval, not a linear chain.
- **Streaming + explainability** — the API streams per-node progress over SSE, and every answer
  ships with a structured "Research Summary": route, sub-queries, per-source retrieval counts,
  fused document count, confidence, whether a corrective search fired, and a per-node latency
  breakdown — the same facts the graph already computes, surfaced instead of hidden.
- **RAGAS evaluation harness** — a golden-question dataset scored with non-LLM context
  precision/recall (string/set overlap against reference contexts, not semantic judgment)
  and, optionally, LLM-judged faithfulness/answer relevancy, run explicitly via
  `rag-assistant eval` rather than left unmeasured. The dataset spans every route (`vector`,
  `web`, `both`, `none`), including a case designed to exercise the corrective-fallback loop.
  It's a small, hand-curated set with no adversarial cases and no naive-RAG
  baseline to compare against — useful as a regression smoke test, not as proof the adaptive
  pipeline outperforms a simpler one. A second set (`conversational.jsonl`) carries prior turns
  on every row, because condensation is the first node in the graph and rewrites the question
  every later node reads: without it, a regression that broke follow-up resolution moved no
  measured number at all.
- **Provider fallback** — a three-tier chain, tried in priority order and skipping any tier
  that isn't configured: a self-hosted local model (when `LOCAL_LLM_BASE_URL` is set), then
  Anthropic Claude (when `ANTHROPIC_API_KEY` is set), then Gemini, wired with
  `.with_fallbacks()` so a rate limit, an outage, or an unreachable GPU box degrades to the
  next tier instead of failing the request. Embeddings deliberately have no such chain — see
  **Embedding providers** below.
- **Embedding providers** — `EMBEDDING_PROVIDER` selects Gemini (default), OpenAI, or a
  self-hosted OpenAI-compatible `/v1/embeddings` server (Ollama, vLLM, TEI) for the one model
  that builds *and* queries the index. Unlike chat there is no fallback, because only the model
  that built a collection can query it: the active model is recorded in the index metadata,
  `/ready` refuses to serve a collection built by another one, and a self-hosted server is
  probed on every readiness poll so an unreachable box pulls the replica from the load balancer
  instead of failing every question.
- **Per-document router labels** — one short chat call per file at ingest records
  `publisher -- document type -- period`, which the router sees next to the filename. Built
  from filenames alone the corpus list fails where it matters: on a 30-report corpus,
  `Annual_Report_JUNE-2021.pdf` told the router nothing about *whose* report it was, so every
  question naming the publisher was routed to web search. Files indexed before the labels
  existed are backfilled from their stored text on the next ingest, without re-parsing.
- **Self-hosted inference** — the primary tier can be your own hardware on any
  OpenAI-compatible `/v1` endpoint (Ollama, vLLM, LM Studio, llama.cpp), so every graph node
  runs at $0 while the box is reachable and silently falls back to Claude when it isn't. See
  [Running on self-hosted models](#running-on-self-hosted-models).
- **Incremental indexing** — `rag-assistant ingest` hashes file contents against a manifest and
  only re-embeds changed or new files, removing chunks for deleted files, instead of rebuilding
  the whole collection every run (`--full` forces a clean rebuild).
- **Multi-format + URL ingestion** — the corpus accepts PDF (page-aware, markdown-preserving),
  Word (.docx, paragraphs and tables), HTML, Markdown, and plain text, via CLI, drag-and-drop
  upload, or `POST /api/v1/ingest/url`, which fetches any public web page server-side with an
  SSRF guard (every hostname — including each redirect hop's — is resolved and refused if it
  lands on a private/loopback/link-local address) and a streamed size cap.
- **MCP server** — the pipeline doubles as a Model Context Protocol server
  (`rag-assistant mcp`), so Claude Desktop/Code and other MCP clients can research against
  the knowledge base, ingest files/URLs, and browse saved conversations as native tools.
- **Multimodal PDF ingestion (vision)** — charts, diagrams, and photos embedded in PDFs are
  described by the chat provider's vision capability and indexed as `[Figure on page N: ...]`
  blocks beside the page text, so data that exists only as pixels ("EMEA revenue $2.1M" in a
  bar chart) becomes retrievable and citable; pages with no text layer at all (scans) are
  rendered and transcribed by the same mechanism — OCR without an OCR dependency. One vision
  call per figure/scanned page at ingest time, never at query time; size/count budgets cap
  cost, and `PDF_VISION=false` disables it.
- **Live graph execution visualization** — the web UI renders the LangGraph pipeline as a stepper
  that highlights each node as it runs, sourced from the same per-node SSE progress events the
  streaming endpoint already emits.

## Architecture

```mermaid
flowchart TD
    START([question + chat history]) --> condense[condense_question]
    condense --> route[route_query]
    route -- none --> synth[synthesize_answer]
    route -- vector / web / both --> decompose[decompose_query]

    decompose == Send, one per sub-query ==> retrieveVector[retrieve_vector]
    decompose == Send, one per sub-query ==> retrieveBM25[retrieve_bm25]
    decompose == Send, one per sub-query ==> webSearch[web_search]

    retrieveVector --> fuse[fuse_results\nReciprocal Rank Fusion]
    retrieveBM25 --> fuse
    webSearch --> fuse

    fuse --> grade[grade_and_score]
    grade -- low confidence, corpus not yet re-asked --> refine[refine_retrieval\nrewrite queries, widen k]
    refine --> fuse
    grade -- still low, vector-only, web not yet tried --> corrective[corrective_web_search]
    corrective --> fuse
    grade -- confident enough --> synth

    synth --> verify[verify_groundedness\nclaims vs. retrieved context]
    verify --> format[format_report]
    format --> DONE([done])
```

Nodes drawn in the escalation path run at most once each: `refine_retrieval` re-asks the corpus
with rewritten queries before the pipeline is willing to leave it, and `corrective_web_search`
runs only if that still grades badly. Both rejoin at fusion rather than at synthesis, so a
second attempt's documents are ranked *against* the first attempt's instead of replacing them.

The nodes whose work is an LLM call — condensation, routing, decomposition, grading, synthesis,
refinement, verification — are coroutines and await it on the event loop; the ones doing
blocking library work — retrieval, web search, fusion, formatting — stay synchronous and run in
a worker thread. See [Concurrency](#concurrency) for why.

- `retrieve_vector` / `retrieve_bm25` / `web_search` fan out via `Send` — one invocation per
  sub-query, per applicable route — and join back at `fuse_results`.
- `corrective_web_search` loops back into `fuse_results` at most once per question (guarded by
  `correction_attempted` in state, backstopped by a `recursion_limit`).
- Every node is wrapped at registration time to record its own wall-clock latency into
  `node_timings` (an `operator.add`-reduced state field), which is what powers the latency
  breakdown in the Research Summary panel below — instrumentation with no changes to any node's
  own logic.

## Explainability: the Research Summary panel

Every answer — from both `POST /research` and the streaming `POST /research/stream` — carries a
structured summary alongside the prose report:

```json
{
  "route": "vector",
  "condensed_question": "What safety research does Anthropic do?",
  "sub_queries": ["...", "..."],
  "retrieval_counts": { "vector": 16, "bm25": 16, "web": 0 },
  "fused_document_count": 6,
  "confidence_score": 0.62,
  "correction_attempted": false,
  "refinement_attempted": true,
  "refined_sub_queries": ["annual revenue figure", "reported turnover"],
  "groundedness_checked": true,
  "groundedness_score": 0.83,
  "unsupported_claim_count": 1,
  "node_latencies_ms": [{ "node": "route_query", "latency_ms": 1523.7 }, "..."],
  "total_latency_ms": 21026.6
}
```

The web UI renders this as a panel: route, a sub-query checklist, per-source retrieval counts,
the post-fusion unique document count, confidence, whether either escalation fired, and a
latency table grouped by pipeline stage. It exists so the assistant doesn't just produce an
answer — it shows its work, which matters both for debugging and for demoing an agentic system as
something more than "a single LLM call with extra steps."

Two pairs of fields are easy to conflate and are deliberately distinct:

- `confidence_score` grades **retrieval** — how relevant the documents are to the question.
  `groundedness_score` grades the **answer** — the fraction of its factual claims the retrieved
  context actually supports. They routinely disagree, and the gap between them is the failure
  RAG exists to prevent: retrieval can be perfect and the write-up can still assert a figure no
  document contains. A UI that prints a confidence number next to a paragraph of prose is
  showing the first while the reader is asking about the second.
- `groundedness_checked` is not the same as a `groundedness_score` of `null`. A *checked*
  abstention scores `null` too, because an answer that makes no factual claims has nothing to
  ground — so `checked: false` means no check ran, and the absence of unsupported claims says
  nothing at all. Reporting the second as the first would turn a provider outage or a disabled
  setting into a clean bill of health.

## Setup

```bash
uv sync
cp .env.example .env
# fill in GOOGLE_API_KEY (https://aistudio.google.com/apikey) in .env
# web search (DuckDuckGo via `ddgs`) needs no key, signup, or billing account -- nothing to
# configure there
# optionally also fill in ANTHROPIC_API_KEY (https://console.anthropic.com/settings/keys) to
# use Claude as the primary chat model, with Gemini as automatic fallback

uv run rag-assistant hello    # confirms chat model connectivity (Anthropic if set, else Gemini)
uv run rag-assistant ingest   # embeds the sample corpus (data/corpus/) into Chroma, incrementally
```

To index your own documents instead of the sample corpus, point `CORPUS_DIR` at a directory
git ignores (`data/private_corpus/` is ignored for this) rather than adding files to
`data/corpus/`, which is what CI and the demo image index.

Or run the API + Redis via Docker Compose instead:

```bash
docker compose up --build   # api on http://localhost:8000, redis alongside it
```

> **Free-tier quota note:** if `ANTHROPIC_API_KEY` is unset, chat calls fall back to Gemini,
> whose free tier caps at ~20 requests/day; one research question costs ~4 calls (route,
> decompose, grade, synthesize) plus embedding calls (embeddings always go through Gemini
> regardless of the chat provider). Budget accordingly when running `ask`, `serve`, or `eval`
> repeatedly in a single day.

## Usage

### CLI

```bash
uv run rag-assistant ask "Who founded Anthropic and what is their safety research called?"
uv run rag-assistant ask "What is the most recent Claude model release?"
uv run rag-assistant ask "Compare Anthropic and Mistral AI's founding stories and safety focus."
```

`ingest` is incremental by default: it hashes each file in `data/corpus/` against a manifest and
only re-embeds new or changed files, removing chunks for any file that's been deleted since the
last run. Pass `--full` to reset the collection and re-embed everything from scratch:

```bash
uv run rag-assistant ingest --full
```

Debug commands for individual pieces of the pipeline:

```bash
uv run rag-assistant retrieve "anthropic founders" --k 4   # raw vector-store retrieval
uv run rag-assistant search "claude model releases 2026"   # raw DuckDuckGo web search
```

Operator commands:

```bash
uv run rag-assistant backup --keep 7          # snapshot index + corpus + conversations
uv run rag-assistant restore <archive.tar.gz> # roll back to a snapshot
uv run rag-assistant loadtest --requests 500 --concurrency 25
uv run rag-assistant labels                   # the router label each document carries
uv run rag-assistant labels --relabel report.pdf                  # ask the model again
uv run rag-assistant labels --relabel report.pdf --set "Ethio Re -- annual report -- 2020/21"
uv run rag-assistant delete report.pdf        # remove one document from the index and corpus
uv run rag-assistant feedback --export        # queue downvoted questions as eval candidates
```

Labels are worth a look after a first ingest. They are model output about a document's opening
pages, they prefix every chunk of that document, and they are what the router reads — so a
wrong one is not cosmetic. Re-labelling reads the text back from the index rather than
re-parsing the file, so correcting one never re-pays its vision calls; the router sees the new
label immediately, and retrieval sees it when that file is next re-indexed.

**Upgrading an existing index.** `CHUNKING_VERSION` and `LOADER_VERSION` are recorded per file
in the manifest, so a change to either makes the next `ingest` re-index the files it affects —
by design, since the alternative is a collection full of chunks the current code no longer
produces. For a PDF corpus that means re-paying vision calls, so run it deliberately rather
than discovering it during a deploy. Both versions moved in this release (scanned pages are
read once, and every chunk carries its document label), so an existing PDF index re-ingests in
full on its next run.

### API

```bash
uv run rag-assistant serve   # starts FastAPI on http://127.0.0.1:8000
```

```bash
curl -X POST http://127.0.0.1:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"question": "Who founded Anthropic and what is their safety research called?"}'
```

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/research/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Who founded Anthropic and what is their safety research called?"}'
```

Removing one indexed document — the unit that erasure was missing, since it previously existed
only per tenant:

```bash
curl -X DELETE http://127.0.0.1:8000/api/v1/sources/anthropic.md
curl -X DELETE http://127.0.0.1:8000/api/v1/sources/_t/alice/report.md   # tenant-owned
```

It removes the chunks, the parent sections, the manifest entry and the uploaded file, and
invalidates the keyword index so the text stops being searchable immediately. Scoped to the
caller's own sources; someone else's is reported as absent rather than forbidden, because
"forbidden" would confirm a filename exists to anyone willing to guess. Needs the `write`
scope, as `DELETE /api/v1/tenant/data` does.

`/api/v1/research/stream` emits Server-Sent Events — one `"progress"` frame per graph node as it
completes, then a final `"done"` frame carrying the report and the Research Summary above (or a
`"error"` frame on failure, since the HTTP status is already 200 by the time streaming starts).

Conversations are persisted server-side: the first request returns a `conversation_id`, and
follow-ups just send it back — the server loads its own transcript, condenses the follow-up
against it, and appends the new exchange:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"question": "what about their safety research?", "conversation_id": "<id from the first response>"}'
```

`GET /api/v1/conversations` lists them, `GET /api/v1/conversations/{id}` returns the full
transcript (including each answer's report and research summary), and `DELETE` removes one.
For fully stateless use, pass `"save": false` and (optionally) an inline `history` of
`{"role", "content"}` turns instead.

Ingest a public web page into the knowledge base by URL:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"}'
```

Every endpoint is under `/api/v1/`. The original unversioned `/research` and
`/research/stream` remain registered as deprecated aliases so clients written before
versioning keep working; they are hidden from the OpenAPI schema and carry the same rate
limits as the versioned paths.

Interactive API docs at `http://127.0.0.1:8000/docs`.

### Observability

```bash
curl http://127.0.0.1:8000/metrics        # Prometheus exposition
curl http://127.0.0.1:8000/ready          # dependency-aware readiness (503 when Chroma or web search is down)
```

Useful queries once a Prometheus is scraping it:

```promql
# p95 latency of a research call
histogram_quantile(0.95, sum by (le) (rate(rag_http_request_duration_seconds_bucket{route="/api/v1/research"}[5m])))

# is the primary LLM provider failing over?
sum by (provider, outcome) (rate(rag_llm_calls_total[5m]))

# output tokens per hour, by model -- the cost signal
sum by (model) (rate(rag_llm_tokens_total{kind="output"}[1h])) * 3600

# cache hit rate
sum(rate(rag_cache_operations_total{result="hit"}[5m])) / sum(rate(rag_cache_operations_total{result=~"hit|miss"}[5m]))
```

With `API_KEYS` set, `/metrics` requires a key like any other protected route — point the
scraper at it with an `X-API-Key` header, or set `METRICS_ENABLED=false` to remove the route.

Alert rules and a Grafana dashboard ship in `ops/`:

```yaml
rule_files:
  - ops/prometheus/alerts.yml   # then import ops/grafana/dashboard.json
```

`tests/test_ops_artifacts.py` asserts every metric they reference actually exists. Monitoring
config rots silently, and an alert that can never fire is worse than no alert because its
presence is taken as coverage. Two things are deliberately not alerted on: cache hit rate (a
cache outage is slower and costlier, never wrong) and absolute token spend (what matters is a
change in the *rate*, which the burn-rate rule catches without anyone guessing a budget).

### Web UI

The UI carries two controls tied to the features above: a **source filter** above the input,
which lists what `GET /api/v1/sources` reports this caller can retrieve from and narrows local
retrieval to the files you tick (web search is unaffected), and **thumbs up/down under each
answer**, which posts to `/api/v1/feedback`. A downvote reveals an optional note field — the
rating is recorded immediately either way, because demanding a note before accepting the rating
would cost most of the ratings.

A React + Vite single-page app in `frontend/` streams `/api/v1/research/stream` live as a
conversation: each turn shows the question, the streamed report, and its own collapsible
Research Summary, and follow-ups automatically carry the transcript. A graph visualization
stepper highlights each LangGraph node as it runs (grouping the fanned-out `retrieve_vector`
/ `retrieve_bm25` / `web_search` nodes into one "Retrieve" stage with per-source counts, and
marking `corrective_web_search` as skipped when the confidence gate doesn't trigger it). The
corpus drawer accepts drag-and-drop uploads (PDF/DOCX/HTML/MD/TXT) and web page URLs.

```bash
uv run rag-assistant serve       # terminal 1 -- backend on http://127.0.0.1:8000

cd frontend
npm install
npm run dev                      # terminal 2 -- UI on http://localhost:5173
```

The backend allows CORS from `http://localhost:5173` by default. If the backend runs elsewhere,
copy `frontend/.env.example` to `frontend/.env` and set `VITE_API_BASE_URL`.

### MCP server (use it from Claude Desktop / Claude Code)

The whole pipeline is also exposed as an [MCP](https://modelcontextprotocol.io) server, so any
MCP client can use the knowledge base as a tool — ask Claude Desktop a question and it calls
`research_question` behind the scenes, cited answer and all:

```bash
uv run rag-assistant mcp   # stdio transport; normally launched by the client, not by hand
```

Tools exposed: `research_question`, `ingest_file`, `ingest_url`, `list_documents`,
`list_conversations`. Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "adaptive-rag": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/this/repo", "rag-assistant", "mcp"]
    }
  }
}
```

The server runs in-process on your machine (no HTTP hop, logs on stderr so the stdio protocol
stream stays clean) under your own `.env` credentials.

### Deploying a live demo

The Docker image builds the frontend and serves it from FastAPI itself (`STATIC_DIR`), so one
container is the whole app. `render.yaml` is a ready-made [Render](https://render.com)
blueprint: connect the repo ("New" → "Blueprint"), set `GOOGLE_API_KEY` (and optionally
`ANTHROPIC_API_KEY`) when prompted, and the free instance serves the full demo — the baked-in
sample corpus is indexed at container startup. Any other Docker host works the same way:

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 --env-file .env rag-assistant   # full app on http://localhost:8000
```

#### Continuous deployment

`.github/workflows/deploy.yml` ships `main` automatically once CI is green. It is off until
configured, and says so rather than failing:

| Setting | Kind | What it is |
| --- | --- | --- |
| `RENDER_API_KEY` | secret | Render API key with deploy permission |
| `RENDER_SERVICE_ID` | secret | The `srv-...` id of the service |
| `DEPLOY_HEALTHCHECK_URL` | variable | Public base URL, e.g. `https://your-app.onrender.com` |

The trigger is a **successful CI run**, not a push. A `push` trigger races the test suite and
will deploy a commit whose tests are still running — and it deploys the SHA that CI actually
tested, because by the time a deploy starts another push may have landed and shipping that one
means shipping something no green run ever covered.

A deploy is not finished when the platform accepts it. The workflow waits for the deploy to
report live, then polls `/health` until the instance answers (a free instance cold-starts in
about a minute) and finally checks `/ready`, which pings Chroma and the embedding provider. A
container that boots perfectly against a mismatched index fails there rather than in front of
a user.

#### Rollback

Every successful deploy writes its SHA into the run summary, so the last known-good revision
is findable without opening the host's dashboard.

> **Actions → Deploy → Run workflow → `commit` = the last known-good SHA**

That redeploys the old revision through the same verified path — platform-live, then
`/health`, then `/ready` — so a rollback is checked exactly as carefully as a deploy. It is
deliberately a manual decision: sometimes the right response to a bad deploy is to roll
forward, and no workflow can tell which situation it is in.

What this does **not** cover is state. A rollback returns the *code*; it does not un-apply a
database migration, and the migrations here are forward-only (`_MIGRATIONS` in
`conversations/postgres_store.py` and `retrieval/pgvector_store.py`). A revision that adds a
column is safe to roll back from; one that drops or rewrites data is not, and needs a
deliberate plan before it ships. Nothing in the pipeline enforces that distinction.

### Evaluation

Two layers, because they answer different questions and only one of them can gate a build.

**Deterministic retrieval metrics** — computed from graph output, no judge, no extra model
calls, identical numbers on identical output:

| Metric | What regressing means |
| --- | --- |
| `route_accuracy` | The router started sending questions down the wrong retrieval path |
| `source_recall` | The right documents stopped being retrieved at all |
| `mean_reciprocal_rank` | The right documents are still found, but ranked lower |
| `abstention_accuracy` | The system started answering things it can't support — or refusing things it can |

**RAGAS** (`--llm-judge`) adds faithfulness and answer relevancy. Kept out of the gate on
purpose: it costs model calls per row and its scores drift slightly between runs on identical
output, so gating on it would fail builds for reasons unrelated to the change.

```bash
uv run rag-assistant eval --limit 50                    # score the full dataset
uv run rag-assistant eval --limit 50 --llm-judge        # ...plus RAGAS faithfulness/relevancy
uv run rag-assistant eval --limit 50 --record-baseline  # record baseline from a known-good build
uv run rag-assistant eval --limit 50 --check            # fail on regression vs. that baseline
```

A baseline recorded on Claude Sonnet against the sample corpus is committed at
`data/golden_eval/baseline.json`: `source_recall` and `mean_reciprocal_rank` at 1.000 (every
expected document retrieved, and ranked first), `abstention_accuracy` 0.778, `route_accuracy`
0.786 — see Known limitations for why the last two are lower, which is partly the dataset
rather than the system.

**Multi-turn coverage.** `data/golden_eval/conversational.jsonl` carries prior turns on every
row. It exists because condensation is the first node in the graph and rewrites the question
every later node reads — routing, decomposition, both retrieval paths and synthesis all see its
output rather than what the user typed — and no single-turn row exercises it. A regression that
broke follow-up resolution, or that dropped the fencing around the conversation (an untrusted
surface, since an assistant turn carries whatever the web path retrieved), moved no measured
number at all. Twelve rows cover pronoun resolution, bare ellipsis ("What about funding?"), a
referent two turns back with an intervening subject, a topic switch where the history is a
distractor, an already-self-contained follow-up that condensation must *not* damage, an
unanswerable follow-up, and a conversation whose own history contains injection-shaped text.

It is a separate dataset with its own baseline rather than rows added to `dataset.jsonl`,
following the same rule the private corpus does: scores are only comparable against a baseline
recorded on the same questions.

```bash
uv run rag-assistant eval \
  --dataset data/golden_eval/conversational.jsonl \
  --baseline data/golden_eval/conversational-baseline.json \
  --limit 12 --record-baseline
```

No baseline is committed for it, because recording one requires running the graph against real
models and this repo cannot do that in CI. Until someone records it, condensation is covered by
structural checks (`tests/test_conversational_eval.py`) and unit tests but is not *gated* — see
Known limitations.

`--check` compares against `data/golden_eval/baseline.json` with a tolerance (default 0.05),
rather than against absolute thresholds. Absolute numbers get set to whatever today's run
produced and then either block unrelated work or get quietly lowered until they block nothing;
a baseline asks the question that matters — *did this change make retrieval worse than it was?*
The tolerance absorbs a single borderline routing flip, since LLM routing isn't deterministic
even at temperature 0, and a gate that fails on noise is a gate people learn to ignore.

**Record your own baseline before the gate does anything.** No baseline ships with the repo:
the numbers depend on which chat provider you configured, so a committed one would be a
fiction on anyone else's setup. Run `--record-baseline` once against a build you trust and
commit `data/golden_eval/baseline.json` — that is what switches the CI gate on. Until then
CI reports the gate as skipped with a warning rather than failing, since failing red on a
fresh clone just teaches everyone to ignore the job. Re-record deliberately when a change is
a genuine improvement, never to make a failing gate pass.

The dataset (`data/golden_eval/dataset.jsonl`) is 28 questions across five categories —
`factual`, `multi_hop`, `unanswerable`, `current`, `no_retrieval`. The `unanswerable` rows
carry the most weight: a dataset of only answerable questions cannot catch the failure that
matters most in RAG, which is answering confidently from documents that don't contain the
answer. It is still small and hand-authored, with no baseline system to compare against.

**Evaluating against your own corpus.** A golden set is only meaningful against the corpus it
was written from, so `--dataset` and `--baseline` keep separate suites separate — scores are
comparable only against a baseline recorded on the same questions *and* the same documents.
Keep a private set beside a private corpus (both git-ignored; `tests/test_private_eval_dataset.py`
applies the same structural checks to it whenever it exists, and skips in CI):

```bash
uv run rag-assistant eval --dataset data/golden_eval/private/dataset.jsonl \
    --baseline data/golden_eval/private/baseline.json --limit 50 --record-baseline
uv run rag-assistant eval --dataset data/golden_eval/private/dataset.jsonl \
    --baseline data/golden_eval/private/baseline.json --limit 50 --check
```

Worth stating plainly, because it is the argument for doing this at all: run against 30 real
scanned annual reports, this harness found five defects the unit suite could not (see
[Self-audit](#self-audit-findings--fixes)) and measured three changes unit tests can only
assert *happened*, on the same 50 questions:

| Change | Source recall | MRR | Context precision | Context recall |
| --- | --- | --- | --- | --- |
| Baseline (all-minilm, no labels) | 0.868 | 0.882 | 0.283 | 0.456 |
| A stronger embedding model | 1.000 | 0.961 | 0.413 | 0.561 |
| Per-chunk document context lines | 0.985 | **1.000** | **0.642** | **0.926** |

The last row is the one that would have been hardest to guess: prefixing each chunk with its
document's label costs one line per chunk and moved context recall further than changing the
embedding model did.

**Against naive RAG.** Same index, same questions, same synthesis prompt — but retrieval as a
first implementation would write it: embed the question as typed, take the top 6 chunks,
answer. No routing, decomposition, BM25, fusion, grading or corrective search. Scored over the
34 rows that name a source document, since naive RAG has no path to the web or abstention rows:

| | Naive RAG (top-6) | This pipeline |
| --- | --- | --- |
| Source recall | 0.897 | **0.985** |
| MRR | 0.919 | **1.000** |
| Abstention accuracy | 0.882 | **1.000** |

Both ran against the *improved* index, so this isolates the pipeline from retrieval quality:
the labels and context lines help naive RAG too, and the remaining gap is what routing, hybrid
retrieval, fusion and grading add on top.

**Judged metrics.** `--llm-judge` adds RAGAS faithfulness and answer relevancy, scored by the
chat model rather than by string overlap. On 8 rows: **faithfulness 0.970, answer relevancy
0.939**. Worth reading with the deterministic metrics rather than instead of them — they cost
several model calls per row and move a little between runs, which is why the gate uses the
free ones.

### Retrieval tuning

Three knobs, all off by default, because each trades cost or a dependency for quality. Turning
any of them on is a configuration change — none requires a re-index except where noted.

| Setting | What it changes | What it costs |
| --- | --- | --- |
| `CHUNKING_STRATEGY=semantic` | Splits a section where consecutive sentences stop being similar, instead of every N characters. Fixed-size splitting routinely severs a claim from the sentence that qualifies it | One embedding call per section at ingest. Re-indexes automatically (`CHUNKING_VERSION`) |
| `PARENT_CONTEXT=true` | Small-to-big: retrieve on precise chunks, then hand synthesis the whole section each winner came from. Retrieval wants small chunks for precision, synthesis wants large ones for context — this refuses the trade | More of the context budget per document. No re-index: sections are always recorded |
| `RERANKER=cohere` / `cross_encoder` | Scores (query, document) pairs jointly. RRF ranks by retriever *consensus* and never compares a document against the question, so a passage every path returns for lexical reasons outranks the one that answers it | An API key, or `sentence-transformers` (torch). Both are optional extras |
| `RETRIEVAL_MMR=true` | Maximal Marginal Relevance over local retrieval. Dense search ranks candidates against the query independently, so the top-k may be k restatements of one passage — likely on a corpus of structurally similar documents, and *not* something fusion's near-duplicate collapsing catches, since those are different texts making the same point rather than the same text | Fetches `RETRIEVAL_FETCH_K` candidates instead of `RETRIEVAL_K`. No extra model call. `RETRIEVAL_MMR_LAMBDA` is the dial: 1.0 is plain similarity, 0.5 refuses the third restatement |

`RETRIEVAL_K` (default 4) is how many documents each path returns per sub-query — the number
that most directly controls recall, and until recently the only retrieval knob that was a
literal in the node rather than a setting. Raising it costs context budget and grading tokens,
not extra round trips.

Both vector backends run the *same* MMR implementation (`retrieval/mmr.py`) rather than each
using its own. Chroma ships one and pgvector has none, so the obvious split — native on one
side, hand-written on the other — would make the two backends agree on plain retrieval and
disagree the moment diversity was switched on. `tests/test_pgvector_store.py` asserts they
select identically, against a real Postgres in CI.

```bash
uv sync --extra rerank-cohere   # RERANKER=cohere, needs COHERE_API_KEY
uv sync --extra rerank-local    # RERANKER=cross_encoder, pulls in torch
```

Requests can also narrow local retrieval by metadata. Filters are pushed into the query rather
than applied to the results — post-filtering silently shrinks `k`, and the graph reads a short
result as "the corpus has nothing" and falls back to web search:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What is their safety approach?",
       "filters": {"sources": ["anthropic.md"], "ingested_after": "2026-01-01T00:00:00Z"}}'
```

### Feedback

`POST /api/v1/feedback` records a thumbs up/down against an answer; the web UI shows the
buttons under each one. `GET /api/v1/feedback/summary` returns the counts and — the useful part
— the recently downvoted questions.

Those questions are the material a golden eval dataset goes stale for lack of. The gate below
catches regressions against a *fixed* set of questions; it cannot tell you the set stopped
resembling what people actually ask. This is the only signal here sourced from a human rather
than from the system's own behaviour, which is why it is also the only alert of its kind.

`rag-assistant feedback --export` is what consumes it: downvoted questions become rows in
`data/golden_eval/candidates.jsonl`, deduplicated against the questions already in the dataset
so repeated exports converge instead of accumulating.

```bash
uv run rag-assistant feedback            # counts, plus the downvoted questions
uv run rag-assistant feedback --export   # queue them as golden-dataset candidates
```

The candidate rows carry the question and the route the system actually chose, and leave
`ground_truth`, `reference_contexts` and `expected_sources` **blank**. That is the point, not
an omission: those three fields are the assertions every metric is computed against, and
auto-filling them from the answer a user had just rejected would encode the failure as the
expected behaviour — after which the gate would defend the bug. A person completes each row
and moves it into the dataset, and the baseline is re-recorded.

### Backup and restore

```bash
uv run rag-assistant backup --output backups/ --keep 7
uv run rag-assistant restore backups/rag-assistant-backup-<timestamp>.tar.gz
```

One archive holds the vector index, the ingestion manifest, the conversation database and the
corpus — everything that isn't in git. Two design points worth stating, because both are places
a naive implementation is quietly wrong:

- **SQLite goes through SQLite's online backup API, not `cp`.** In WAL mode the committed data
  lives across `.db`, `.db-wal` and `.db-shm`, and copying the three catches them at different
  instants — producing an archive that restores, opens without complaint, and is missing recent
  writes. The `-wal`/`-shm` sidecars are deliberately *not* copied, because the snapshot has
  already folded them in.
- **Restore stages the whole archive before swapping anything**, and moves the existing data
  aside rather than deleting it. A corrupt or truncated archive fails with the live deployment
  untouched, instead of halfway replaced. Archives are also checked for path-traversal members,
  since a restore runs wherever the operator happens to be.

Caches are process-local, so restart the server afterwards — the CLI says so.

### Scaling out

The default is one container with no infrastructure, and that is a deliberate constraint rather
than a limitation nobody noticed: embedded Chroma locks its SQLite file to one process, the
ingest task registry is in-memory, and conversations are SQLite. The image pins `--workers 1`
for exactly that reason.

Each of those ceilings is now a setting rather than a rewrite:

| Setting | Removes |
| --- | --- |
| `CHROMA_SERVER_HOST` | The vector index's file lock — replicas share a Chroma server. Verified against a real Chroma server in CI (`tests/test_chroma_server.py`), not just at the construction boundary |
| `TASK_BACKEND=redis` | Per-process ingest tasks. Without it a client polling a load-balanced deployment gets "unknown ingest task" from every replica that didn't accept the upload |
| `CONVERSATIONS_BACKEND=postgres` + `DATABASE_URL` | SQLite's single-writer lock, the main obstacle to a second replica. Needs `uv sync --extra postgres`; migrations are advisory-locked so replicas can start simultaneously |
| `VECTOR_BACKEND=pgvector` + `DATABASE_URL` | The same file lock `CHROMA_SERVER_HOST` removes, but without operating a second service — the index becomes a table in a database that is already backed up, replicated and monitored. Needs the `vector` extension in that database |
| `KEYWORD_BACKEND=postgres` | The in-memory keyword index. BM25 is built by reading *every* chunk in the collection into the process and keeping it there, once per replica, so build time and resident memory both scale with the whole corpus and a restart pays for it again before the first keyword query is served. This moves keyword search to a full-text index in the same table the vectors live in. The trade is ranking, not correctness — Postgres ranks with `ts_rank_cd` rather than BM25, which fusion tolerates because it combines paths by *rank position* rather than by score |
| `RATE_LIMIT_STORAGE_URI` | Per-process rate-limit counters. Without it every replica enforces its own private copy of `RATE_LIMIT_RPM_GLOBAL`, so the "global" cap is really N times the configured number — the one shared-state ceiling that fails silently rather than loudly, since nothing errors and the service simply absorbs more load than it was told to |

`DEPLOYMENT_PROFILE=multi-replica` sets all of these together (and requires `DATABASE_URL`),
because they are not independent choices: every half-shared combination breaks as intermittent
flakiness rather than as an obvious misconfiguration. Anything you set explicitly still wins.

Both Postgres-backed paths are verified against a real Postgres — `tests/test_postgres_store.py`
for conversations and `tests/test_pgvector_store.py` for the index — and both skip unless
`RAG_TEST_DATABASE_URL` points at one:

```bash
initdb -D /tmp/pg/data -U postgres --auth=trust
pg_ctl -D /tmp/pg/data -o "-p 55432 -k /tmp/pg" -l /tmp/pg/log start
RAG_TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/postgres \
    uv run pytest tests/test_postgres_store.py tests/test_pgvector_store.py
```

The pgvector suite asserts the same behaviours the Chroma tests do — retrieval, idempotent
re-ingest, tenant isolation, metadata filtering — because two backends are interchangeable only
if they actually behave the same, and one that stored vectors correctly while dropping the
tenant predicate would be a data-isolation bug introduced by flipping a config value. Two tests
carry the load beyond that:

- **`test_ranking_matches_the_chroma_backend_on_the_same_corpus`** indexes one corpus into both
  backends with the same deterministic embeddings and asserts the same documents come back in
  the same *order*. Retrieving the right set in the wrong order changes which document reaches
  the synthesis prompt first, and nothing else would catch it.
- **`test_ranking_is_cosine_not_euclidean`** pins the distance metric. The parity test above
  cannot: the shared fake embeddings return unit vectors, and on unit vectors cosine and
  Euclidean rank identically — mutating `<=>` to `<->` leaves it green. This one uses
  deliberately unnormalized embeddings, where the two metrics disagree, and fails under that
  mutation. An HNSW index built with the wrong operator class never errors either; the `<=>`
  query silently stops using it and falls back to a sequential scan.

### When the embedding server is unreachable

The one dependency with no fallback, so it gets a runbook rather than a paragraph. Chat degrades
between providers; embeddings cannot, because only the model that built the index can query it.

**Symptom.** `/ready` returns 503 with `embeddings.error` naming the server, and every question
fails. `/health` stays 200: the process is fine, the dependency is not.

```bash
curl -s localhost:8000/ready | jq .embeddings      # what readiness saw
curl -sS $LOCAL_EMBEDDING_BASE_URL/models          # is the server itself up?
```

**If the server is coming back:** nothing to do. Requests fail fast (short connect timeout) and
recover the moment it answers; the index is untouched and no re-ingest is needed.

**If it is not coming back**, switch to a hosted provider — which means re-embedding the corpus,
because the stored vectors belong to the old model:

```bash
EMBEDDING_PROVIDER=gemini   # or openai
uv run rag-assistant ingest --full     # re-parses and re-embeds; PDFs pay vision calls again
```

Avoiding that outage entirely is a deployment choice, not a code one: run the embedding model
in the same failure domain as the app, or embed with a hosted provider and accept the bill.

### Concurrency

`POST /api/v1/research` is an async handler and the graph is driven through `ainvoke`. That
is a deliberate structural choice rather than a style one, and it is the difference between
the two paragraphs below.

**What it used to be.** The handler was a synchronous `def`, so FastAPI ran it on the AnyIO
worker threadpool and each in-flight question held one thread for the whole graph run --
seconds, not milliseconds. A single worker's ceiling was therefore `API_THREADPOOL_SIZE`
(default 40) no matter how idle the CPU was, because the work being waited on was not CPU at
all. It was almost entirely provider latency, which is to say it was I/O being done with
threads.

**What it is now.** The graph is mixed on purpose. Nodes whose work is an LLM call --
condensation, routing, decomposition, grading, synthesis, refinement, groundedness -- are
coroutines and await it on the event loop. Nodes whose work is a blocking library call --
Chroma, psycopg, the web-search client -- stay synchronous, and LangGraph runs them in a
worker thread for the milliseconds they take. So the threadpool no longer bounds how many
questions can be in flight; `tests/test_concurrency_ceiling.py` pins that by running twelve
concurrent questions against a **two**-thread pool and asserting all twelve overlap, which is
the assertion that fails if a future change puts blocking work back in the request path.

`API_THREADPOOL_SIZE` still bounds the blocking half, and `rag_research_in_flight` is still
the gauge to alert on -- it now tracks concurrent questions rather than occupied threads. The
binding constraint for most deployments is the provider's own rate limit, which this change
makes it much easier to actually reach.

Two caveats, stated because the change is easy to over-claim. The graph is the part that was
made async; the conversation store and the token budget are still synchronous calls inside
the handler (SQLite and Redis, milliseconds each) and have not been moved off the loop. And
the probe in that test sleeps rather than computing, so it models provider latency -- which is
what a real graph run overwhelmingly is -- and not CPU contention or memory pressure. It
bounds the concurrency behaviour, not performance under real load.

### Load testing

```bash
uv run rag-assistant loadtest --requests 500 --concurrency 25          # /health, free
uv run rag-assistant loadtest --question "Who founded Anthropic?"      # real pipeline, costs LLM calls
```

Defaults to `/health` on purpose: that exercises the HTTP stack, middleware chain and event
loop for nothing, while pointing it at the research endpoint is a real bill and the CLI prints
the estimate first. It reports p50/p95/p99 and never a mean — an average hides exactly the tail
that matters.

Measured on a laptop, single worker, concurrency 25: **376 rps on `/health`** and **407 rps on a
SQLite-backed endpoint**, p95 172ms and 109ms, no errors.

The pipeline itself, measured once against the 30-report corpus (12 requests at concurrency 4,
a `vector`-routed question, Claude for chat and a tailnet-hosted embedding model): **p50 10.3s,
p95 11.0s, p99 11.1s**, 0.4 rps. Two of the twelve came back 429 — the per-IP limiter
(`RATE_LIMIT_RPM`, default 10/min) doing its job, which is worth knowing before reading a load
test's error rate as failure. p95 sits inside the 12s objective below, and the flat spread
between p50 and p99 says the time is provider latency rather than queueing at this concurrency.

### Running on self-hosted models

Any OpenAI-compatible `/v1` endpoint works — Ollama (`:11434/v1`), vLLM (`:8000/v1`), LM Studio,
llama.cpp. Setting `LOCAL_LLM_BASE_URL` makes it the primary chat/reasoning provider; Anthropic
and Gemini stay configured behind it as automatic fallbacks.

Get the real model list from the box rather than guessing at names:

```bash
curl http://<host>:11434/api/tags | jq -r '.models[].name'   # Ollama
curl http://<host>:8000/v1/models                            # vLLM
```

Then:

```bash
LOCAL_LLM_BASE_URL=http://<host>:11434/v1
LOCAL_LLM_CHAT_MODEL=<one of the names above>
```

```bash
rag-assistant hello     # prints "Local (<model>) says: ..." when it's actually being used
curl localhost:8000/ready | jq .local_llm
```

Two things worth knowing:

- **Reachability is a property of the server, not your laptop.** If the box is on a tailnet or a
  VPN, use the MagicDNS/hostname rather than the raw `100.x` IP — the IP is stable per node but
  the name survives a node being removed and re-added. And a Render/Fly/Vercel deploy is not on
  your tailnet: leave `LOCAL_LLM_BASE_URL` blank in those environments, or accept that every
  call pays a 2s connect timeout before falling through to Claude.
- **`/ready` reports the local endpoint but doesn't fail on it.** An unreachable box is a cost
  and latency regression, not an outage — the graph still answers on Claude — so it shouldn't
  pull a healthy replica out of a load balancer. It's surfaced because "the bill moved because a
  route dropped" is exactly the failure you want to be able to see.

## Example questions per concept

| Concept | Example question |
| --- | --- |
| Vector routing | "Who founded Anthropic and what is their safety research called?" |
| Web routing | "What is the most recent Claude model release?" |
| No retrieval (general knowledge) | "What is the capital of France?" |
| Query decomposition | "Compare Anthropic and Mistral AI's founding stories and safety focus." |
| Corrective fallback | "What safety research did Anthropic publish this week?" (recent/narrow enough that the local corpus alone often scores low, triggering a web search fallback) |

## Design decisions

**Why is the local model the *primary* provider rather than a fallback behind Claude?**
Because the point is cost, and a fallback never gets called on the happy path. The order is
local → Anthropic → Gemini, and every tier that isn't configured is skipped without being
constructed (the Gemini client validates its key in `__init__`, so merely *building* it as an
unused fallback crashes an Anthropic-only deployment). The degradation this buys is the useful
kind: a laptop on the tailnet answers for free, and the same image deployed to a host with no
route to the box answers on Claude without a config change.

**Why a short connect timeout but a long read timeout on the local provider?** They're solving
opposite problems. Local generation is genuinely slow — a 26B model on one GPU takes 20-25s for
a real answer — so the read timeout is 180s. But off the tailnet there is no route to the box
at all, and a connect that hangs would burn the whole `GRAPH_TIMEOUT_SECONDS` budget before
Anthropic ever saw the call. A 2s connect timeout is what turns "unreachable box" into a fast
failover instead of a dead request. One retry, not zero: single-model Ollama returns a transient
500 while swapping models, and that one is worth absorbing.

**Why can't `with_structured_output(method="json_schema")` be used on a local server?**
`langchain_openai` routes that method through OpenAI's Structured Outputs parser, which reads a
`parsed` field only hosted OpenAI populates. Against Ollama or vLLM it raises
`"response does not have a 'parsed' field"` on *every* call — including calls where the server
returned perfectly valid JSON — so every structured node in the graph would fail through to the
paid fallback while looking like a local-model quality problem. The schema is bound as
`response_format` by hand (the half that matters: the server constrains decoding) and parsed
back off `content` like every other provider. See `_local_structured_runnable` in `llm.py`.

**Why does the graph tolerate a local model returning an empty answer?** It doesn't — that's the
bug it's built to avoid. Reasoning models (Qwen3-class) sometimes leave `content` empty and put
the whole answer in a `reasoning`/`reasoning_content` field, which `langchain_openai` discards
because it isn't in the OpenAI schema. Downstream that reads as "the model said nothing": an
empty synthesis, or a structured parse failure. `_LocalChatOpenAI` recovers it from the raw
payload on both the blocking and the streaming path (the synthesis node streams, so handling
only one of them would still relay a stream of empty SSE tokens). Truncation at `max_tokens`
with nothing in `content` is treated differently — that's a config problem, not an answer, so it
raises and lets the fallback chain answer while the error stays visible in the logs.

**Why can chat fall back between providers but embeddings cannot?** Chat providers are
interchangeable mid-flight; embeddings are not. The collection is built at one provider's vector
dimension, and pointing queries at a different embedding model doesn't error — it silently
returns nonsense neighbours. So `EMBEDDING_PROVIDER` is a deliberate one-way choice (switching
means `rag-assistant ingest --full`) rather than something the graph can fall back into at
runtime, and the consequence is accepted openly: with a self-hosted embedding server, an
unreachable box is an outage rather than a cost regression, which is why readiness probes it.

**Why hybrid (vector + BM25) retrieval, not vector-only?** Dense embeddings are strong on
semantic/paraphrased queries but can under-rank exact keyword matches — model names, acronyms,
proper nouns — that a small corpus makes easy to miss entirely if the wording doesn't line up.
BM25 costs nothing extra to add (`rank_bm25`, no external service, rebuilt in-memory from the
same chunks that get embedded) and only ever adds candidates into fusion; it never replaces the
vector path.

**Why Reciprocal Rank Fusion over an LLM-based reranker?** RRF is a pure, deterministic function
of rank position across ranked lists — no additional model call, no added latency, no added
quota cost — and is a well-established way to combine heterogeneous retrieval paths (vector,
BM25, web) without having to calibrate their scores onto a common scale.

**Why Corrective-RAG (confidence-gated web fallback) instead of always searching the web?**
Always searching the web on every question would add latency even when the
local corpus already answers confidently. Gating the fallback on a relevance-graded confidence
score means the web search only fires when the vector-only route is actually falling short —
demonstrating self-assessment rather than blind escalation.

**Why SSE streaming instead of a single blocking response?** The graph can take 10-20+ seconds
end-to-end (multiple sequential LLM calls plus fanned-out retrieval). A blocking response gives
no feedback during that window; `stream_mode="updates"` gives per-node progress essentially for
free, since LangGraph already emits these events — the only added work is reshaping them into SSE
frames.

**Why RAGAS for evaluation instead of eyeballing answers?** Manually judging "is this answer
good" doesn't scale and isn't repeatable across changes to prompts or retrieval. RAGAS's
non-LLM metrics (`NonLLMContextPrecisionWithReference`, `NonLLMContextRecall`) score retrieval
quality against a golden dataset with zero additional LLM calls, so regressions in retrieval can
be caught without spending quota — LLM-judged metrics (faithfulness, relevancy) are opt-in for
when that extra cost is worth it.

## Authentication & multi-tenancy

Set `API_KEYS` (comma-separated `label:key` entries) and every data/LLM endpoint requires
`X-API-Key: <key>` (or `Authorization: Bearer <key>`); the web UI shows an access-key gate.
Each key's label is a tenant: conversations are stored, listed, and deletable only within it
(a foreign conversation id 404s identically to a nonexistent one), and rate limits are keyed
per tenant rather than per IP. Leave `API_KEYS` blank to run fully open (local development /
public demo mode) — everything then belongs to the shared `public` tenant. `/health`,
`/ready`, the docs, and the static frontend stay open either way. Behind a load balancer the
Docker CMD passes `--proxy-headers --forwarded-allow-ips '*'` so anonymous rate limiting keys
on the real client IP, not the LB's. Set `SENTRY_DSN` to capture unhandled exceptions.

For anything beyond a demo, `API_KEYS_FILE` points at a JSON file that expresses what a
comma-separated env var cannot — a key that only reads, one that stops working in March, one
allowed more requests than the rest:

```json
{"keys": [
  {"key": "sk-live-a1b2", "owner": "alice", "scopes": ["read", "write"],
   "expires_at": "2026-12-31T23:59:59Z", "rate_limit_rpm": 120},
  {"key": "sk-ro-c3d4", "owner": "reporting", "scopes": ["read"]}
]}
```

Writes (`POST /api/v1/ingest*`, `DELETE /api/v1/conversations/*`, `DELETE`/`PUT
/api/v1/sources/*`, `POST /api/v1/connectors/*`) need the `write` scope; everything under
`/api/v1/admin` needs `admin`, which a key file entry must name explicitly and which no SSO
token carries; everything else needs `read`. A valid key without the scope gets **403, not 401** — 401 would
tell a read-only client to re-authenticate, which presenting the same key again cannot fix.
Every decision is audited with the key's fingerprint, never the key: an audit trail that
records secrets is a secret store nobody is guarding. The key cache is keyed on the file's
mtime, so editing it revokes or rotates a credential on the next request rather than the next
restart — the difference between revocation being an operation and being an outage.

**The knowledge base is tenant-scoped too**, not just conversations. Ownership lives in the
corpus layout rather than a sidecar table, because the layout is the thing that survives — a
manifest can be deleted, reset by a fresh deploy, or drift from the files on disk, and every
one of those failures defaults documents to visible-to-everyone, which is the wrong direction
to fail in:

```
data/corpus/anthropic.md           # public — the shared baseline corpus, everyone sees it
data/corpus/_t/alice/report.md     # private to tenant "alice"
```

Uploads land in the uploading tenant's subtree; flat files stay public, so an open demo's
on-disk layout is exactly what it was before tenancy existed and the baked-in corpus needs no
migration. Both retrieval paths filter on ownership — Chroma via a `$in` filter applied
*during* search (post-filtering would silently shrink `k`, so a tenant whose top hits belong
to someone else would get fewer documents with no indication why), BM25 by narrowing
candidates before the top-k cut. The router's corpus description is scoped as well: listing
another tenant's filenames would leak them through the prompt even though retrieval filters
them out.

Ingestion is scoped the same way. An upload re-indexes only the uploading tenant's scope, and
within it only files whose bytes actually changed — decided from a raw-byte fingerprint, so
the check never runs the parse it exists to avoid. Removal detection is scoped to the same
slice, since comparing one tenant's scan against the whole manifest would read every other
tenant's documents as deleted and drop their chunks. A full rebuild (`ingest --full`) refuses
to be scoped to one owner at all, because resetting the collection would delete everyone else's.

The measured effect on a 8-file corpus, one tenant uploading one file: **16 file parses → 1.**
Most of that came from a second, less obvious source — the BM25 index used to rebuild by
re-reading and re-splitting the entire corpus from disk after every ingest. It now builds from
the chunks already stored in Chroma, which removes the second parse pass and, more usefully,
makes the two retrieval paths index the identical chunk set *by construction* rather than by
the convention that both happened to call the same splitter. The tradeoff is that keyword
search reflects what has been indexed rather than what is on disk — which is the honest
behaviour, since an un-ingested file was always invisible to vector search.

### Single sign-on

API keys identify a *tenant*; a company needs to know which *person* is asking, from the
directory it already runs. Set `OIDC_ISSUER` and `OIDC_AUDIENCE` and the API accepts access
tokens from that identity provider — Okta, Entra ID, Auth0, Google, Keycloak — as
`Authorization: Bearer <jwt>`, alongside any API keys (keys are compared first, so turning SSO
on breaks no existing integration). Set `OIDC_CLIENT_ID` too and the web UI's access gate
offers **Sign in with SSO**: an authorization-code + PKCE flow with no client secret and no
library, the access token held in `sessionStorage` so it dies with the tab. Register the UI's
URL as the client's redirect URI; on Entra ID, add the API's scope to `OIDC_SCOPES` (e.g.
`openid profile email api://rag-assistant/access`) so the token issued is one for this API,
and configure the app to emit group names rather than object ids if ACLs use names.

Tokens are verified locally against the issuer's published keys (`oidc.py`), and strictly,
because every relaxation is a known way JWT validation fails in practice: only the configured
asymmetric algorithms (never `none`, never HS256 keyed with the public key — both have tests),
exact issuer, required audience (the config refuses to start with an issuer and no audience,
since without it a token minted for any other app at the same IdP is honoured here), required
expiry. The tenant comes from `OIDC_TENANT_CLAIM` and a token *missing* that claim is rejected
rather than defaulted, since defaulting would show the user the default tenant's documents.
Signing keys are cached; an unknown `kid` triggers at most one refetch a minute, so tokens
with random key ids cannot turn every request into outbound traffic.

Groups map to capabilities: `OIDC_WRITE_GROUPS` may upload and delete, `OIDC_ADMIN_GROUPS`
additionally bypass document permissions inside their tenant. Neither grants the deployment
`admin` scope — a tenant's admins are not the operator. Machine clients using client
credentials get `write` from the `rag.write` OAuth scope instead of a group. Conversations
and feedback become per *user* (`<tenant>::<user>`) once a caller has a user identity: with
per-document permissions a transcript is as sensitive as the documents it quotes. Erasing a
tenant still reaches every user's rows.

### Document-level permissions

Tenancy says which organisation owns a document; it cannot say "only finance may read the
board pack". A document now either has no ACL — visible to its whole tenant, exactly as
before — or a list of users, groups and email domains that may read it (`ingestion/acl.py`).
The caller's principals come from their identity: user id, email, email domain, SSO groups,
or the `user`/`groups` an API key file entry declares. A key with neither is tenant-wide.

```bash
curl -X POST .../api/v1/ingest -F file=@board.pdf -F allowed_groups=finance,leadership
curl -X PUT  .../api/v1/sources/_t/acme/board_1f3a9c2b.pdf/acl \
     -H 'Content-Type: application/json' -d '{"groups": ["finance"]}'
```

The ACL lives in a sidecar file next to the document (`board.pdf.acl.json`) for the reason
ownership lives in the path: it survives manifest resets, backups, restores and re-indexes,
where a permission stored only in the manifest would silently revert to "visible to
everyone". A sidecar that exists but cannot be parsed makes the document readable by nobody.

At index time the ACL is stamped on every chunk and enforced **inside** the vector search —
Chroma `$or`/`$contains`, a Postgres `?|` predicate, the same predicate in the in-memory
keyword filter — so a restricted document never shrinks `k` for someone who cannot see it.
Listings, the router's corpus description, deletion and ACL edits all apply the same check: a
filename is information, so a document you cannot read is one you cannot see listed, delete,
or re-permission either, and an ACL change that would lock its author out is refused. A
permission change does not re-embed anything: the manifest records an ACL fingerprint, and
when only that differs the chunks' metadata is rewritten in place — measured in the tests as
zero embedding calls. That matters because permission changes are the most frequent change a
synced source produces.

With auth disabled nobody holds any principals, so an open demo serves every unrestricted
document and no restricted one. Permissions are only as good as the identity behind them.

### Strict tenant isolation

`TENANT_ISOLATION=filter` (the default) keeps one shared collection with a tenant predicate on
every query: correct, tested, and one bug away from a cross-tenant leak, because the predicate
is the only thing between tenants. `strict` adds a boundary that does not depend on every
query remembering it:

- **Chroma** gets one collection per tenant. A tenant's search goes to their collection and
  the public one and merges by distance (comparable, because one generation is one embedding
  space). Another tenant's vectors are not filtered out of the search; they are not in it — a
  test replaces the `where` clause with one that matches everything and confirms a tenant
  still cannot reach another's documents. Deletes route by the owner encoded in the chunk id,
  and everything that reads "the whole index" (BM25 build, readiness, re-embedding) unions
  the collections.
- **pgvector** always carries a row-level-security policy on the chunks table, keyed on a
  per-connection setting that every connection the store hands out sets. It fails closed — a
  connection that never set it sees no rows — and `FORCE` keeps it binding on the table's
  owner. Postgres exempts superusers and `BYPASSRLS` roles from every policy, so the app must
  connect as an ordinary role for this layer to exist; `/ready` reports `row_security` so a
  deployment believing it has two layers and having one is told. The tests create such a
  role, because the CI container's superuser would pass them while proving nothing.

An index keeps the layout it was written with, so flipping the setting never hides existing
documents. Moving to `strict` is a re-index into a new generation and a pointer flip — see
below — with no downtime.

## Source connectors

Uploading files by hand keeps a corpus current for about a week. `CONNECTORS_FILE` names a
JSON file of sources to mirror into tenants' corpora — Confluence spaces, Google Drive
folders (My Drive or shared drives, walked recursively; Docs exported as HTML) and mounted
file shares:

```json
{"connectors": [
  {"name": "eng-wiki", "type": "confluence", "owner": "acme", "interval_minutes": 60,
   "base_url": "https://acme.atlassian.net/wiki", "space_key": "ENG",
   "email_env": "CONFLUENCE_EMAIL", "token_env": "CONFLUENCE_API_TOKEN",
   "default_acl": {"groups": ["engineering"]}},
  {"name": "policies", "type": "google_drive", "owner": "acme",
   "folder_id": "1AbC...", "credentials_file_env": "DRIVE_SERVICE_ACCOUNT_FILE"}
]}
```

```bash
rag-assistant connectors list
rag-assistant connectors sync [name] [--due] [--force]   # cron: `connectors sync --due`
```

or `CONNECTOR_SCHEDULER=true` to run due syncs inside the API, or
`POST /api/v1/connectors/{name}/sync` for one tenant's connector on demand. Secrets are never
in the file — each connector names the *environment variables* holding them.

A connector's documents are written as ordinary corpus files (plus ACL sidecars) under the
tenant's `_sources/<connector>/` directory, then the tenant is re-indexed incrementally. That
is why the design is small: change detection, deletion, permission updates, PII policy,
backups and index generations already work on corpus files, so a synced document gets all of
them without any of them knowing connectors exist. What the sync engine adds is what mirroring
a system you do not control needs:

- **Fetch only what changed**, by the source's own version marker — a sync of an unchanged
  space is a listing, not a download of every page.
- **Deletion sync.** A page deleted upstream — often deleted *because* it should not be
  read — is deleted from the corpus and its chunks from the index.
- **A deletion guard.** An expired token or revoked share makes a source *list* as empty,
  and faithful deletion sync would erase the corpus. A sync whose listing came back empty, or
  that would delete more than `CONNECTOR_MAX_DELETE_FRACTION` (50%) of what it holds and at
  least three documents, is refused, nothing changes, and an alert fires; `--force` applies it
  once someone has confirmed it is real.
- **Failures change nothing.** A listing that raised is not evidence the rest was deleted; a
  document that fails to fetch keeps its previous copy. An unmounted share raises rather than
  listing empty.
- **Permissions are translated conservatively.** Drive exposes each file's full permission
  list, so users, groups, domains and "anyone" map directly, and a file whose permissions
  cannot be read is indexed as readable by nobody. Confluence exposes page restrictions
  (read here for the page *and* every ancestor, since a parent's restriction binds its
  children) but not space permissions, so `default_acl` is required. A page restricted at
  several levels needs a reader to pass all of them, which an "any of these principals" ACL
  cannot express; it is flattened to the principals named at every level — someone
  Confluence would admit through two different groups may be refused, nobody it would refuse
  is admitted. Misconfiguration that would publish documents to a whole tenant, or send a
  credential over plain HTTP, fails at load, in front of the operator.

`rag_connector_last_success_timestamp_seconds` drives a staleness alert: a connector that has
quietly stopped syncing leaves deleted and re-permissioned documents searchable, and the
service keeps answering perfectly well from the stale copy, so nothing else would say so.

## Re-indexing without downtime

Changing the embedding model used to mean `ingest --full`: reset the collection, then serve
from a half-empty index (or not at all) while every file was re-parsed and re-embedded. Now
the index has **generations** (`ingestion/generations.py`): complete indexes built side by
side — vectors, manifest, parent sections and the metadata naming the model that built them
— with one pointer saying which serves.

```bash
rag-assistant reindex build --embedding-model openai/text-embedding-3-large
rag-assistant reindex activate g20260929120000a1b2c3
rag-assistant reindex rollback     # flip back; the previous generation is kept
rag-assistant reindex gc           # delete everything but the serving and rollback generations
rag-assistant reindex status
```

(`POST /api/v1/admin/index/reindex` does the same inside the API process, which embedded
Chroma requires since the server holds the index open.)

- **Re-embed, don't re-parse.** By default a build copies chunk text, parent sections,
  document labels and ACLs from the serving generation and recomputes only the vectors. A
  model change is a change of vector space, not of text; re-parsing would repeat every vision
  and description call the corpus ever cost (on the private 30-report corpus, ~370 vision
  transcriptions) for byte-identical text. `--from-corpus` re-parses when the text itself
  must change.
- **Catch-up.** Ingestion keeps writing to the serving generation during a build, so the
  build ends — and activation begins — with an incremental pass against the corpus on disk,
  bringing across every upload, deletion and permission change made meanwhile. The final pass
  and the pointer flip happen under one hold of the ingest lock.
- **Queries follow the index, not the config.** Every reader embeds with the model the
  serving generation recorded. Changing `EMBEDDING_PROVIDER` therefore no longer silently
  breaks retrieval; it chooses the model the *next* generation is built with, and `/ready`
  reports the difference as a pending migration rather than an outage. Readiness fails only
  when the recorded model cannot be used at all (its credentials or server are gone).
- **Every replica converges within `INDEX_POINTER_POLL_SECONDS`**, each serving a complete
  generation throughout. On pgvector a generation is a schema (`rag_idx_<id>`) and the
  pointer a row every replica reads; on Chroma it is a directory, and in server mode a
  collection-name suffix. Backups archive the serving generation.

## Production readiness

Beyond the core RAG pipeline, the API is hardened for running as an actual service rather than a
local demo script:

| Area | What's there |
| --- | --- |
| Containerization | Multi-stage `Dockerfile` (non-root user), `docker-compose.yml` wiring `api` + `redis` with a named volume for the Chroma persist directory. Embedded Chroma's SQLite backing locks the file to one process, which is why the image pins `--workers 1`; `CHROMA_SERVER_HOST` switches to server mode when that ceiling matters (see [Scaling out](#scaling-out)) |
| Health & readiness | `GET /health` is a pure liveness check; `GET /ready` actually pings Chroma (`_collection.count()`) and DuckDuckGo (`HEAD` request) and returns 503 if either dependency is down, so an orchestrator can distinguish "process is up" from "can actually serve a request" |
| Input validation | `question` is required, capped at 2000 chars, HTML-tag-stripped, and rejected as gibberish if under 10% alphanumeric — all in a pydantic `field_validator`, so bad input 422s before it ever reaches the graph |
| Rate limiting | `slowapi`-based, both per-caller (`RATE_LIMIT_RPM`, default 10/min) and a global cap (`RATE_LIMIT_RPM_GLOBAL`, default 30/min) across `/research` and `/research/stream`. Counters live wherever `RATE_LIMIT_STORAGE_URI` points — in-process by default, which is correct for one container and wrong for two, so `multi-replica` fills it in from `REDIS_URL`. Without that the "global" cap is really N times the configured number, once per replica. An unreachable store degrades to per-process counting rather than failing requests, and `/ready` reports it |
| Timeouts | The web search client is capped at `WEB_SEARCH_TIMEOUT_SECONDS` (default 10s); the whole graph execution behind `/research/stream` is bounded by `GRAPH_TIMEOUT_SECONDS` (default 45s) via a monotonic-clock deadline around `astream()`, emitting an `"error"` SSE frame and closing the connection instead of hanging indefinitely |
| Graceful shutdown | SIGTERM is caught via `loop.add_signal_handler` inside the FastAPI lifespan; active SSE connections (tracked in a `weakref.WeakSet`) are sent a `"close"` frame before the process exits, instead of being cut off mid-stream |
| Structured logging | JSON logs (`python-json-logger`) with a UUID4 `trace_id` generated per request by an ASGI middleware, propagated through `contextvars` *and* threaded explicitly into the LangGraph state (belt-and-suspenders, since LangGraph's internal scheduling isn't guaranteed to preserve context automatically) — every log line, including each node's completion log, carries `trace_id`/`node`/`route`/`latency_ms`, and the response carries the same trace ID in an `X-Trace-Id` header |
| Caching | Redis-backed, best-effort (`USE_CACHE=false` or any Redis error both degrade silently to "no cache" — a cache outage is never worse than having no cache): router decisions keyed by question (`CACHE_TTL_ROUTER`, 5min), web search results keyed by query (`CACHE_TTL_WEB_SEARCH`, 10min), synthesized answers keyed by question + route + fused source IDs (`CACHE_TTL_SYNTHESIS`, 30min) — all under a `v1:` key prefix so a payload-shape change can be rolled out by bumping the prefix rather than migrating existing entries |
| Configuration | All of the above is `pydantic-settings`-driven (`config.py`), reading exclusively from environment variables with fail-fast validation at startup instead of scattered `os.environ.get()` calls with silent defaults |
| Metrics | Prometheus exposition at `GET /metrics` (`metrics.py`): request rate/latency by *templated* route, LLM calls and **token usage** by provider/model/outcome, cache hit-rate by namespace, per-node graph latency, graph outcomes split `ok`/`error`/`timeout`, and a live SSE-connection gauge. Every label is drawn from a bounded set — the route label is Starlette's path template, never the UUID-bearing real path, and unmatched requests collapse to one sentinel series, so a scanner hitting random URLs can't blow up the registry. The token counter is the one that maps onto money: it's how a provider fallback shows up as a cost change rather than a surprise at the end of the month |
| Schema migrations | The conversation store applies an ordered, append-only migration list stamped via `PRAGMA user_version` (`conversations/store.py`), each in its own transaction so a failure mid-chain resumes rather than replays. The baseline migration is written to converge all three databases that predate versioning — brand new, pre-`owner`-column, and already-ALTERed — onto one shape |
| Data retention | Conversations are bounded by both an age cutoff (`CONVERSATION_RETENTION_DAYS`, 90d) and a per-tenant count cap (`CONVERSATION_MAX_PER_OWNER`, 500), pruned inline after each write and scoped to the tenant that wrote, so no cron is needed and the sweep never scans other tenants. Messages follow via `ON DELETE CASCADE` |
| CORS | Origins come from `CORS_ALLOW_ORIGINS` rather than being hardcoded, so a split deploy (UI on Vercel, API on Render) is configuration rather than a code change; the single-container deploy is same-origin and needs none. `X-Trace-Id` is in `expose_headers` so a cross-origin frontend can actually read the trace ID it's meant to report |
| API versioning | Everything is under `/api/v1/`; the pre-versioning `/research` and `/research/stream` stay registered as deprecated, schema-hidden aliases carrying the same rate limits, so older clients keep working without the docs showing two ways to do one thing |
| Supply chain | CI audits Python dependencies (`pip-audit` over the exported lockfile) and npm dependencies, and scans the built image with Trivy. Reported rather than blocking, since a fresh advisory against a transitive dependency shouldn't block unrelated work behind a fix nobody has published yet |
| Deploy verification | The docker CI job doesn't stop at "the image builds" — it runs the real image and waits on `/health`, so a container that builds and then crashes on boot fails CI instead of failing the deploy. Both the image and compose file carry healthchecks (compose uses `/ready`, which actually pings Chroma and web search) |
| Single-worker constraint | `--workers 1` is stated explicitly in the Dockerfile CMD with the reasoning, rather than left to uvicorn's default: embedded Chroma locks its SQLite file to one process, and the ingest task registry and conversation write lock are per-process. Made explicit so nobody "optimizes" it into `--workers $WEB_CONCURRENCY` and gets intermittent 404s and database-locked errors |
| Retrieval quality gate | `rag-assistant eval --check` scores the golden dataset on deterministic, judge-free metrics (route accuracy, source recall, MRR, abstention accuracy) and fails against a recorded baseline. CI runs it on every push where API keys are available. This is the gate for the failure mode nothing else catches: a prompt or chunking change that raises no exception and fails no unit test, because the system keeps returning confident prose about the wrong documents |
| Adversarial eval coverage | The dataset carries `unanswerable`, `multi_hop`, `no_retrieval` and `current` rows alongside the happy path, so abstention is scored as a first-class metric — a dataset of only answerable questions cannot catch confidently answering something the corpus doesn't contain |
| Context budget | Synthesis is capped at `SYNTHESIS_CONTEXT_BUDGET_TOKENS` (see `graph/context_budget.py`). Fusion's output scales with (sub-queries x retrieval paths), not with the question, so an uncapped prompt grows with retrieval breadth until it overflows the context window — at the very end of the pipeline, after every retrieval and grading call has been paid for. Documents arrive ranked, so the cap drops what the pipeline already judged least useful, truncates rather than drops the top document, and surfaces the count in the research summary |
| Structure-aware chunking | Splitting follows markdown headings first and fixed-size only within a section, prepending the heading breadcrumb to every chunk (`ingestion/splitter.py`). A chunk reading "It raised $450M in a Series C" is nearly useless to both retrieval paths — no company in the embedding, no company token for BM25. The breadcrumb is charged against `chunk_size`, so it stays a real bound, and `CHUNKING_VERSION` in the manifest makes a strategy change re-index itself instead of silently serving chunks built by the previous splitter |
| Corpus tenant isolation | Ownership is encoded in the corpus layout (`ingestion/ownership.py`): flat files are the shared public corpus, `_t/<owner>/` is private to that tenant. Both retrieval paths filter on it — Chroma via a `$in` filter applied *during* search rather than after (post-filtering silently shrinks k), BM25 by narrowing candidates before the top-k cut. The router's corpus description is scoped too, since listing another tenant's filenames leaks them through the prompt even when retrieval filters them out |
| Incremental ingestion cost | Re-indexing is decided from a raw-byte fingerprint plus `CHUNKING_VERSION`/`LOADER_VERSION`, so unchanged files are never parsed — not merely never re-embedded. That distinction is the whole cost: a parse runs pymupdf4llm and, with `PDF_VISION` on, a vision API call per figure and per scanned page. Ingestion is also scoped to the uploading tenant. One upload into an 8-file corpus went from 16 file parses to 1 |
| BM25 / vector chunk parity | The keyword index is built from the chunks stored in Chroma rather than by re-reading the corpus. Beyond removing a second full parse pass per ingest, it makes RRF's `SHA256(content)` cross-source dedup correct by construction — previously the two paths produced identical text only by the convention that both called the same splitter, and any drift would have silently double-counted and double-cited the same passage |
| Backup & restore | One archive holds the index, manifest, conversations and corpus — including when those live in Postgres rather than on disk. `VECTOR_BACKEND=pgvector` moves the vectors, manifest and parent sections out of the persist directory, and archiving only the directories in that configuration produced something worse than a failure: a well-formed archive whose metadata correctly reported eight indexed sources (the count reads through the live manifest, which *does* reach Postgres) and which restored an empty index, with no error anywhere. The Postgres-backed tables are now dumped under `postgres/` inside the archive, read in a single `REPEATABLE READ` transaction so a manifest can never name chunk ids the chunk dump lacks, and a restore **refuses** an archive holding tables the target is not configured to read rather than loading an index into a backend nothing queries. SQLite goes through SQLite's online backup API rather than a file copy — in WAL mode a `cp` of `.db`/`-wal`/`-shm` catches them at different instants and restores into a database that opens cleanly and is missing recent writes. Restore stages the whole archive before swapping and moves the existing data aside rather than deleting it, so a corrupt archive fails with the deployment untouched |
| Embedding-model drift | The model the index was built with is recorded and checked by `/ready`. This is the one dependency whose failure is *silent*: a changed model with the same dimension embeds queries into a space the stored vectors don't occupy and returns plausible nonsense with no error anywhere. Readiness failing pulls the replica from the load balancer instead |
| Key management | Scopes (`read`/`write`, 403 not 401), expiry, per-key rate limits, and an audit trail recording key fingerprints and never keys. The key cache is keyed on the key file's mtime, so revocation takes effect on the next request rather than the next restart |
| Horizontal scaling | Every single-process ceiling is a setting rather than a rewrite: `CHROMA_SERVER_HOST` or `VECTOR_BACKEND=pgvector` (vector index file lock), `TASK_BACKEND=redis` (per-process ingest tasks), `CONVERSATIONS_BACKEND=postgres` (SQLite's single-writer lock). Defaults keep a single container infrastructure-free; both Postgres-backed paths are verified against a real Postgres **in CI**, with advisory-locked migrations so replicas can start at once |
| Interchangeable vector backends | `VECTOR_BACKEND=pgvector` moves the index into Postgres (`retrieval/pgvector_store.py`), keeping the tenant predicate in SQL — applied *during* the search, not after, for the same reason as the Chroma path: post-filtering silently shrinks k, and the graph reads a short result set as "the corpus has nothing" and falls back to web search. The embedding column sizes itself on the first write and builds its HNSW index then, so there is no dimension setting to get wrong, and pgvector afterwards rejects a vector of the wrong width — an embedding-model swap fails at insert instead of silently returning neighbours from a space the stored vectors don't occupy. Backend parity is asserted on retrieval *order*, and the distance metric is pinned by a test that fails when `<=>` is mutated to `<->` |
| Shared index state | `VECTOR_BACKEND=pgvector` moves the *whole* index, not just the vectors: the ingestion manifest and the parent-section store go to Postgres with them. Vectors alone are not the index — a replica that reads a manifest which never saw another replica's ingest decides what to re-index from a record of someone else's collection, and serves chunks whose parent sections it cannot resolve. A manifest that disagrees with the vectors it describes is worse than either being missing |
| Cross-replica index invalidation | The in-memory BM25 index has a second way to go stale that no local call can catch: another replica ingesting. Each replica records the shared index version it built from and re-checks it at most every `BM25_VERSION_POLL_SECONDS`, rebuilding on a mismatch. A poll rather than a subscription on purpose — no broker, no delivery guarantee, no reconnect logic, and it converges from any state including a replica that was down for three ingests. Exposure is bounded stale *keyword ranking*, never stale answers, since vector retrieval reads through to the shared store every query |
| Untrusted content boundary | Applied at **both** LLM surfaces that see retrieved text — synthesis and relevance grading. Grading is the earlier and arguably more consequential one: its grades set the confidence score and decide whether corrective web search runs, so a document that talks its way to a high grade also suppresses the search that might have found something better. Retrieved documents are attacker-influenceable — uploaded by any tenant with a write scope, or fetched from whatever the web search returned — and reach the same prompt as the system's own instructions. Each is fenced with a **per-request random nonce** (`content_trust.py`), which is what defeats the obvious attack on any fencing scheme: a document that closes its own fence escapes into instruction position, and that requires a marker the attacker has never seen. The instruction hierarchy is stated *before* the content, since instructions after untrusted text occupy the position an injection is trying to claim. Detection is advisory and counted (`rag_prompt_injection_signals_total`), never enforced — silently dropping text that matched a regex would corrupt legitimate documents and replace a visible risk with an invisible one |
| Per-tenant cost control | `TENANT_DAILY_TOKEN_BUDGET` caps tokens per tenant per UTC day (`budget.py`), across **both** the research and ingest paths. Ingest is the expensive one — an embedding per chunk and a vision call per figure or scanned page — and neither reports usage the way a chat completion does, so both are estimated deliberately high: an estimate that under-counts lets a tenant exceed the cap the budget exists to enforce, while over-counting only makes it conservative. Enforced before the upload is accepted, since streaming 25MB to disk only to reject it wastes the resource being protected. Rate limiting bounds request *count*, which says nothing about cost when one question routes to `none` and the next decomposes into four sub-queries with corrective search. Usage was already metered; a Prometheus counter cannot be consulted to decide whether to serve a request. Checked before, charged after — including on failures, since a run that died after four LLM calls cost what it cost, and not charging failures makes failure the cheap way to burn a provider quota |
| Ingest idempotency | Uploading identical bytes twice returns the original task instead of parsing and embedding the file again — the retry case after a timeout, a dropped connection or a double-clicked button. Keyed on content and scoped per tenant: the same filename with new bytes is exactly when the work *is* needed, and collapsing two tenants' identical uploads would put one tenant's document in the other's corpus |
| Restart reconciliation | An ingest runs as a background task inside the process that accepted the upload, so a deploy or a crash stops the work but not the record — the task sits at `parsing` forever while a client polls a job nobody is doing. Startup fails those with a message saying what happened and what to do. There is no resumption to offer (the work was in memory), and a terminal failure is more honest than a status indistinguishable from slow progress |
| Explicit concurrency ceiling | `POST /api/v1/research` is a sync handler, so it runs on the worker threadpool and holds one thread for the whole multi-second graph run. `API_THREADPOOL_SIZE` (default 40) is therefore the real per-worker concurrency limit — the 41st concurrent research request queues rather than starts, however idle the CPU. Stated as a setting rather than inherited from AnyIO's default so the number is visible, with `rag_research_in_flight` to watch against it |
| Distributed tracing | Optional OpenTelemetry (`uv sync --extra otel`, `OTEL_EXPORTER_OTLP_ENDPOINT`). The existing per-request `trace_id` correlates log lines but cannot show where the time went; spans add the parent/child structure. One span per node, emitted from the shared `_timed` wrapper rather than from eleven nodes that would each carry their own copy and drift. Entirely inert when unconfigured — the disabled path is a working context manager, not a guard at every call site |
| Outage vs. defect | A self-hosted embedding server going away returns **503** with the server's address and a pointer to `/ready`, not a generic 500: embeddings have no fallback, so it is a dependency outage a client should retry and an orchestrator should route around. Every other failure stays a 500, because dressing a bug as an outage hides it |
| Per-tenant footprint | `GET /api/v1/tenant/usage` reports sources, chunks, corpus bytes and today's token spend for the calling tenant — the index-size half of "what is this tenant costing me", which used to mean reading the manifest by hand while spend was already metered |
| Right to erasure | `DELETE /api/v1/tenant/data` removes everything belonging to the calling tenant across all five stores that hold any: corpus files, embeddings, parent sections, manifest entries, and conversations plus feedback. Ordered so that files go last and each manifest entry is cleared only after its chunks and parents are — an interrupted purge leaves the remaining work still described, so a re-run finishes it. Scoped to the caller rather than taking an owner parameter, because erasing someone else's data should not share an endpoint with erasing your own |
| One coherent deployment switch | `DEPLOYMENT_PROFILE=multi-replica` turns on pgvector, Postgres conversations and Redis tasks together, and fails at startup without `DATABASE_URL`. They are not independent choices: a shared index with per-process ingest tasks serves "unknown ingest task" 404s from whichever replica did not accept the upload, and shared tasks with a local index give two divergent corpora. Every half-shared combination breaks as flakiness rather than as misconfiguration. Explicitly-set switches still win — a profile that overrode them would make the individual settings lie |
| Resumable ingestion | An ingest interrupted by a deploy or a crash is **resumed**, not failed. The uploaded file was written into the corpus before the task existed and `build_index` decides what to do from a fingerprint, so re-running costs only what was unfinished and costs nothing at all after a completed run. Transient failures retry in place up to `INGEST_MAX_ATTEMPTS`; the attempt counter lives on the task record, so a crash mid-retry resumes at the right attempt rather than granting a fresh budget every restart. A task that exhausts its attempts becomes terminally `failed` carrying the count — the record is the dead-letter queue, already queryable through the status endpoint |
| Near-duplicate fusion | Fusion collapses passages by **shingle containment**, not byte equality. A local copy and a web copy of one page previously both reached synthesis and earned separate citation markers pointing at the same words. Containment rather than Jaccard because Jaccard punishes length differences — a truncated copy of the same passage scores 0.321 by Jaccard and 1.000 by containment, and no Jaccard threshold that catches it stays clear of genuinely distinct text. Measured, every true near-duplicate lands at 1.000 and the nearest false positive (a different chunk of the same document) at 0.333, so the threshold sits in a gap rather than on a slope. The fuller copy is kept — text, metadata and source id together — because a merge must not truncate evidence, and citing one source for another's words is worse than keeping both |
| Defensible routing metric | `route_accuracy` scores against `acceptable_routes` rather than one asserted answer. Routing is genuinely ambiguous for a real share of questions — a company's private GPU count can defensibly go to `web` or to `both` — so a single-answer dataset measured its own labelling as much as the router. 17 of 50 rows now list more than one defensible route, and a test fails the build if the dataset is ever widened until every route is acceptable everywhere |
| Baselines that refuse to lie | A baseline recorded before a change that invalidates it is marked `stale`, and the gate **refuses to run** against it rather than comparing. An invalid baseline reports a pass or a failure with equal confidence and neither means anything. `--check` exits 2 for "the gate could not run" and 1 for "the gate ran and failed", and CI treats them differently — a shared exit code would make a misconfiguration look like a quality regression |
| Alerting | Prometheus rules and a Grafana dashboard in `ops/`, with tests asserting every metric they reference exists. Thresholds are stated with their reasoning — an alert whose number nobody can justify is one that gets silenced the first time it fires at 3am |
| Load testing | `rag-assistant loadtest` reports p50/p95/p99 and never a mean. Measured single-worker at concurrency 25: 376 rps on `/health`, 407 rps on a SQLite-backed endpoint, p95 172ms/109ms, no errors |
| Quality signal | Thumbs up/down per answer, surfacing recently downvoted questions. The eval gate catches regressions against a fixed dataset; only this can tell you the dataset stopped resembling what people ask |
| Continuous deployment | `.github/workflows/deploy.yml` deploys on a **successful** CI run for `main`, not on push — a push trigger races the suite and ships commits whose tests are still running, or have already failed. It deploys the SHA CI tested rather than the branch tip, waits for the platform to report the deploy live, then polls the public `/health` and `/ready`; `/ready` is the one that matters, since it pings Chroma and the embedding provider, so an image that boots cleanly against a broken index fails here instead of in front of a user. Skips with a notice when unconfigured, so a fork carries no red badge for a deployment it was never meant to do |
| Rollback | `workflow_dispatch` with a `commit` input redeploys any SHA, and each successful deploy writes its own SHA into the run summary — so the last known-good revision is findable without reading the host's dashboard. Deliberately manual: the right answer to a bad deploy is sometimes to roll forward, and a workflow cannot tell which. Deploys never run concurrently and never cancel each other, because interrupting a rollout leaves the service half-updated |
| Static analysis | `bandit` over `src` at medium severity and above, **blocking** rather than warning — unlike the dependency audit, a finding here is code in this repo that someone can act on today. The baseline is zero: the ten findings it started with were reviewed individually and carry an inline `# nosec` naming the reason (every one was a module-constant table name in an otherwise fully parameterised query), so a new finding is genuinely new. The one real finding it surfaced — `extractall` on a restore archive — is fixed |
| Secret scanning | `gitleaks` over the **full history**, not the tip. Scanning only the current tree misses the case that actually matters: a credential committed once and "removed" in a later commit is still in the history, and still compromised |
| Threat model | `SECURITY.md` states the disclosure path and names, per threat, what the mitigation does *not* cover — a control nobody can name the attacker for is a control nobody can evaluate. It also says plainly that no independent review or penetration test has been done |

## Self-audit: findings & fixes

A structured pass through routing, retrieval, corrective RAG, citations, evaluation, and
streaming — the kind of review that unit tests alone don't catch — surfaced real gaps beyond
happy-path correctness. Fixed:

| Area | Finding | Fix |
| --- | --- | --- |
| Vector store | Chroma had no explicit distance metric, silently defaulting to L2 while Gemini embeddings are meant to be compared via cosine similarity | Set `hnsw:space: cosine` explicitly and rebuilt the index (`ingest --full`) |
| Web search resilience | A web-search outage/rate-limit raised unhandled and crashed the graph node | `WebSearchTool.search` now catches the failure and degrades to `[]` |
| Answer synthesis | An empty `fused_documents` was treated as one case ("no retrieval needed"), but it also happens when retrieval is attempted and comes back empty — same prompt, very different risk of confident hallucination | Split into `NO_CONTEXT_PROMPT` (route == `none`) vs. `EMPTY_RETRIEVAL_PROMPT` (retrieval ran, found nothing), which forces the model to state upfront that no sources were found |
| Non-streaming API | `/research` only caught `RuntimeError`; any other exception fell through to a bare, contentless 500 | Broadened to `except Exception`, still raised as a proper `HTTPException` with `detail` |
| Documentation | README implied RAGAS's semantic, LLM-judged `context_precision`/`context_recall`, when the harness actually runs the non-LLM overlap variants | Relabeled accurately, and noted the eval set is small and non-adversarial with no baseline comparison |

A second pass, against a 30-document corpus of real scanned annual reports rather than the
sample corpus, surfaced five more — every one of them invisible to the unit suite, and four
found by running the eval rather than by reading code:

| Area | Finding | Fix |
| --- | --- | --- |
| Structured output | Claude intermittently returns a list argument as a JSON *string* (`grades='{"grades": [...]}'`). Validation failed, which silently disabled relevance grading ("trusting retrieval") and crashed decomposition outright — a failed question for the user | A `mode="before"` validator decodes the string, unwraps the re-encoded object, and still rejects anything that isn't the expected shape |
| PDF vision | Every scanned page was read **twice**: transcribed, then "figure described" — because a scan's only embedded image is the page itself. Double the vision cost, the 20-image budget spent on scans instead of real charts, and a second looser copy whose numbers could contradict the transcript (4,081.50 transcribed vs 4,082.50 described, on a real CBE key-figures page) | The figure pass now runs only on pages that have a text layer. Figure-described pages fell 273 → 151 on the same corpus |
| Non-English scans | Removing that duplicate pass also removed the one thing it did well — an English gloss that let English questions match an Amharic scan. Two eval rows regressed from correct to not-found | The transcription prompt now ends with an `[English summary: ...]` line when the page isn't in English — same single call. Both rows returned to rank 1 |
| Eval harness | One provider timeout at question 40 aborted the whole run and discarded the 39 results before it; and a correct refusal that cites its context ("I don't have information on X; the procedure covers Y [1]") was scored as a confident answer | A failing question is scored as the failure it is and the run continues; abstention counts a citation-free answer *or* one opening with the refusal phrase the synthesis prompt mandates (shared constant, pinned by a test) |
| Judged metrics | `eval --llm-judge` reported `faithfulness: nan` and `answer_relevancy: nan` rather than failing. RAGAS assigns `temperature` onto the wrapped model before every judge call and current Claude models reject it with an HTTP 400; the judge also inherited the 12s structured-call timeout, too short for whole-row prompts. Every judged run had been silently empty | A judge model that ignores the temperature assignment, with its own 90s timeout (`JUDGE_REQUEST_TIMEOUT_SECONDS`). Faithfulness 0.970 / answer relevancy 0.939 on the first run that produced numbers at all |
| Figure budget | The per-PDF vision budget was spent on duplicates: a report's logo and header band are re-embedded on every page, so one 20-page annual report offered 320 images that cleared the size filter for perhaps a dozen distinct figures. Across the corpus, 1,043 eligible images were only **454 distinct** ones | Byte-identical images are described once per PDF, and the cap became `PDF_VISION_MAX_IMAGES` so a chart-heavy corpus can raise it deliberately |
| Chunk context | A chunk carried its heading breadcrumb but nothing naming the document it came from. Across four near-identical Ethio Re reports and four CBE ones, the passage answering "Ethiopian Reinsurance's 2020/21 profit before tax" — "During the period under review, the Company has registered Birr 220 million profit before tax" — did not reach the top 20 for that question, though its own report took the first four places | Every chunk is prefixed with its document's label before embedding. That passage now ranks 8th, and across the 50 questions MRR reached 1.000 with context recall 0.561 → 0.926 |

Verified with the full offline suite (68/68) plus a live end-to-end run: a real router call
picked the `web` route for a live-price question, a simulated web-search outage was forced, and the
resulting Research Summary (`retrieval_counts: 0`, `confidence_score: 0.0`, `citations: []`) and
synthesized answer ("No relevant sources were found...") both came out correct — confirming the
state plumbing, not just the code path in isolation.

A third pass asked a narrower question — is the multi-replica story *true*? — and read the
code behind each claim rather than the claim. Two of the four findings are bugs that no test
could have caught, because in both cases the test would have been comparing a value to itself:

| Area | Finding | Fix |
| --- | --- | --- |
| Migration locking | Both migration chains serialised themselves with `pg_advisory_lock(hash("...") % 2**31)`, and `str.__hash__` is **salted per process**. Every replica therefore took a *different* advisory lock and contended with nobody — the exact scenario the lock was written for, with a comment saying so. `CREATE TABLE IF NOT EXISTS` hid it for the migrations written so far; the first one to `ALTER` or `INSERT` would have surfaced it as a startup crash on whichever replica lost a race that was not supposed to exist | `advisory_lock.py` derives the id with CRC32 — a pure function of the bytes rather than of a runtime seed. The test spawns two interpreters with different `PYTHONHASHSEED` and compares, since a single-process assertion compares a value to itself and passes either way. A second test asserts `hash()` really is unstable, so the rationale fails loudly if that ever changes |
| Rate limiting | The limiter was built with slowapi's default in-process storage. On one container that is correct; under `DEPLOYMENT_PROFILE=multi-replica` — which shares the index, the conversations and the task registry — each replica kept private buckets, so `RATE_LIMIT_RPM_GLOBAL`, documented as a cap on aggregate load *regardless of client*, was really that number once per replica | `RATE_LIMIT_STORAGE_URI`, filled in from `REDIS_URL` by the profile. An unreachable store degrades to per-process counting rather than failing requests — a limiter that 500s the API because its bookkeeping store is down has inverted its own purpose — and `/ready` reports the degradation. Verified against a real Redis by driving two separately-built limiters and asserting the second sees the first's hits |
| Tar extraction | `bandit` flagged `extractall` on a restore archive. A false positive as written — `_safe_extract` already rejects path escapes and link members first — but the hardening was free and real | `filter="data"` as well: CPython's own sanitiser additionally rejects absolute paths, device and FIFO members, and strips setuid bits, none of which the existing loop inspects |
| Restore identifiers | Reviewing bandit's SQL findings turned up one that was not a false positive in the same way as the rest: in `_load_postgres` the *column* names are read out of the archive's JSON and interpolated into the `INSERT`, and an archive is untrusted input by this module's own reasoning. Attempts to exploit it all failed — psycopg's extended protocol refuses multiple commands per statement, and its placeholder accounting rejects the malformed statements a crafted name produces — so this was a latent reliance on driver behaviour rather than a live hole | Composed with `psycopg.sql.Identifier`, so a hostile column name fails as an unknown column. The guarantee now comes from this code rather than from an adapter detail that a different driver or a `COPY` rewrite would change. A test drives `_load_postgres` with a crafted name and asserts the table survives |
| Verification gaps | Chroma **server mode** was covered only at the construction boundary (an assertion that an `HttpClient` gets built), and behaviour at the `API_THREADPOOL_SIZE` ceiling was unmeasured because reaching it with real traffic means raising the rate limiter and spending real provider quota | Server mode now runs against a real Chroma service container in CI, asserting retrieval, tenant scoping and the cosine metric over HTTP. The ceiling is measured with a sleeping stand-in for the graph: at a ceiling of 4, peak concurrency is 4 and 12 requests take ~0.50s in three waves; with the limiter left at AnyIO's default the peak is 12 and they take ~0.18s — so the assertion has teeth and costs nothing |

Gaps identified but deliberately not yet acted on: no few-shot examples in the router/
decomposition prompts, and exact-content-hash dedup can still let the same source get cited
twice under different markers if local and web copies differ even slightly. (The third gap
listed here originally — synthesis having no token/context-length cap on however many
documents fusion returns — has since been closed; see **Context budget** in
[Production readiness](#production-readiness).)

## Known limitations

Stated plainly, because knowing where a system's edges are is more useful than pretending it
has none.

- **The committed baseline is current, and that is what makes the gate real.** It was
  re-recorded on 2026-09-22 over all 50 rows with Gemini embeddings -- the combination CI runs
  -- so `--check` compares instead of exiting "could not run": route accuracy 0.940, source
  recall 0.981, MRR 0.981, abstention 1.000. A test asserts the committed baseline is neither
  stale nor recorded over a different number of questions, because either one turns the gate
  inert while leaving it in the workflow. It still measures the five-file sample corpus; a
  private corpus keeps its own dataset and baseline (see [Evaluation](#evaluation)).
- **The eval set is 50 hand-authored questions.**
  Larger than the 28 it started at, and balanced across all five categories and all four
  routes, but still small enough that one flipped routing decision moves an aggregate by about
  two points — which is why it compares against a recorded baseline with a tolerance rather
  than against absolute thresholds. It tells you whether a change made things *worse*; it
  cannot tell you how good the system is in absolute terms. Every row is authored against the
  same five-document corpus, so it measures this system on this corpus and nothing wider. A
  naive-RAG comparison now exists for the private corpus (see [Evaluation](#evaluation)), but
  not for this one.
- **Retrieval-quality features are correctness-tested, not quality-measured.** Semantic
  chunking, reranking, small-to-big and MMR diversity all behave as specified and are covered by
  tests, but
  whether they *improve* answers on a given corpus is exactly what the eval gate answers — and
  that requires recording a baseline against real models first. The embedding model, the
  per-document labels and the chunk context lines *have* been measured this way, on a private
  30-report corpus; the three knobs under [Retrieval tuning](#retrieval-tuning) have not.
- **Vision transcription of scanned pages is sampled, not verified end to end.** Roughly 370
  scanned pages in that corpus were read by the chat provider's vision capability. A random
  sample of 12 was re-checked against the original page images: **142 figures compared, none
  disagreeing**, and two individually verified earlier (an audited income statement matched line
  for line; a key-figures page surfaced the duplicate-read bug above). The caveat is that the
  checker is the same model family that produced the transcript, so this is a careful re-read
  rather than independent ground truth, and 12 of 370 pages is a sample. A misread digit in a
  scanned table still becomes a confidently wrong answer with a correct citation. Figures beyond
  `MAX_IMAGES_PER_PDF` per file remain undescribed.
- **A self-hosted embedding server is a single point of failure, by construction.** Chat falls
  back between providers; embeddings cannot, because only the model that built the index can
  query it. `/ready` probes the server and the client fails fast, so the failure is visible and
  quick rather than silent — but while that server is unreachable the service cannot answer at
  all. A deployment that needs to survive it should embed with a hosted provider, or keep the
  embedding model in the same failure domain as the app.
- **Questions are only evaluated in English, and cross-language retrieval is uneven.** Every
  golden question is in English; non-English scanned pages carry an English summary line, which
  is what makes them retrievable from those questions at all. Probed in the other direction with
  Amharic translations of golden questions, an Amharic query found its **Amharic-source document
  at rank 1 in both cases tried**, but reached an English-source document only once out of two
  (rank 2, then absent from the top 8). So Amharic questions about the Amharic documents work;
  Amharic questions about the English reports are unreliable, and nothing in the gate measures
  it.
- **The optional backends are still verified to differing depths.** The Postgres-backed paths
  (conversations, the pgvector index) and the Redis-backed ones (answer cache, per-tenant
  budget, ingest task registry) each run against a real service container in CI, and those jobs
  fail rather than pass if their suites skip, since a suite that skips itself produces a green
  job having verified nothing. The Redis tests deliberately read back through a *second* client,
  which is the part a fake cannot check — a fake agrees with whatever the code expects of it,
  including about TTLs and about what another replica sees. Chroma **server mode** now runs
  against a real server in CI too, covering retrieval, tenant scoping and the distance metric
  over HTTP rather than only the construction boundary. What no job covers is *two replicas at
  once*: every suite runs one process against a shared service, so "these replicas agree" is
  still argued from the storage being shared rather than demonstrated by running two.
- **Tenants share one collection unless `TENANT_ISOLATION=strict`.** The default is still one
  shared collection separated by a metadata predicate, because moving an existing deployment
  is a deliberate re-index rather than something a default flip should do. `strict` gives
  Chroma one collection per tenant and pgvector always has row-level security (see
  [Strict tenant isolation](#strict-tenant-isolation)); what neither gives is per-tenant
  *capacity* isolation — one tenant's large ingest still shares the embedding provider,
  the ingest lock and the process with everyone else's. `GET /api/v1/tenant/usage` answers
  both halves of "what is this tenant costing me" — sources, chunks and corpus bytes
  alongside today's token spend — scoped to the caller, like the erasure endpoint.
  Erasure now has both units — `DELETE /api/v1/sources/{source}` removes one document across all
  five stores, and `DELETE /api/v1/tenant/data` still removes everything a tenant owns — so a
  takedown request or a retention date no longer means deleting a tenant's whole corpus and
  re-uploading the rest. Embedding and vision spend is still charged from an estimate rather
  than reported usage, because neither surfaces token counts through LangChain's callbacks.

- **Prompt-injection defense is structural, and structural is not proof.** All four prompts
  that interpolate text the pipeline did not author -- synthesis, grading, routing and
  condensation -- are now fenced with a per-request nonce and carry the trust hierarchy ahead
  of the content, and a test fails the build if a fifth is added without one. The fence and
  the instruction hierarchy make injection harder and
  attempts visible; neither can guarantee a model obeys, and no prompt-level defense can. The
  tests assert the mechanism — that documents
  are fenced, that the nonce cannot be forged, that attempts are counted and nothing is silently
  dropped — not that a given model resists, which is a property of the model and measurable only
  against a live one. What actually bounds the blast radius is elsewhere: retrieval is
  tenant-scoped, synthesis has no tools, and citations are built from the documents the pipeline
  selected rather than from anything the model claims.

- **Ingestion resumes and retries, but there is still no broker.** A restart-orphaned task is
  re-queued and resumed, transient failures retry with a bounded attempt count, duplicates are
  collapsed, and exhausted tasks land in a terminal state that serves as the dead-letter
  record. What is still absent is a real queue: no visibility timeout, no work stealing, no
  distribution across replicas. Resumption happens in the process that restarts, which means a
  replica that dies permanently takes its in-flight work with it until another one's startup
  pass notices the stale record. A broker is the right answer once ingest volume justifies
  operating one.

- **The pipeline is load-tested only at low concurrency, and concurrency is tested without a
  provider.** `/api/v1/research` has been measured end to end (12 requests at concurrency 4:
  p50 10.3s, p95 11.0s, p99 11.1s), which confirms the latency objective and that the tail is
  provider latency rather than queueing. That measurement predates the async handler, so it
  remains a valid latency figure and is no longer a concurrency one. Concurrency is tested with
  the graph replaced by an async sleep: twelve questions overlap against a two-thread pool,
  nothing deadlocks, and the in-flight gauge tracks real concurrency and returns to zero even
  when every run raises. That pins the property the change was made for. It does not pin
  *latency* under real load — a sleeping stand-in does not contend for CPU, memory or a
  provider's own rate limiter — so the ceiling discussion above is a bound rather than a
  benchmark at full concurrency. The end-to-end latency figures have not been re-measured
  since the conversion.
- **The deploy pipeline has never run against a real host.** Its logic is tested — that it
  refuses a failed CI run, deploys the tested SHA rather than the branch tip, verifies `/ready`
  and not merely `/health`, and skips rather than fails when unconfigured — by
  `tests/test_workflow_contracts.py`, which parses the workflow and asserts those properties.
  What no test can assert is that Render's API behaves as the workflow expects: the deploy
  call, the status polling and the smoke test have never executed against a live service,
  because doing so requires credentials this repo does not have. Treat the first real run as
  the test, and watch it.
- **Static analysis is not a security review.** `bandit` and `gitleaks` run on every PR and the
  baseline is zero, which means a new finding is genuinely new — but a scanner cannot find what
  it was not taught to look for, and every suppression in the tree was reviewed by the same
  person who wrote the code it suppresses. No independent review and no penetration test has
  been done. `SECURITY.md` names, per threat, what each mitigation does not cover; that table
  is a description of intent, not evidence the intent was achieved.
- **The groundedness check is a model checking a model, and it verifies support, not truth.**
  It decomposes the answer into claims and asks whether the retrieved context states them, so a
  claim supported by a document that is itself wrong scores as grounded — correctly, since the
  corpus is the ground truth here, and whether the corpus deserves to be is a question no
  runtime check can answer. Verifier and author may also share a blind spot: they are the same
  model family. It costs one structured call per question that retrieved anything (abstentions
  and the `none` route skip it), and it is the pipeline's most valuable target for prompt
  injection — a document that talks it into marking everything supported turns the score into a
  rubber stamp, which is worse than no check because the number is still reported. The prompt is
  fenced and the hierarchy stated ahead of the content, with the same caveat as everywhere else:
  that makes injection harder and attempts visible, not impossible.
- **PII detection finds formats, not people.** Emails, phone numbers, Luhn-valid cards, US
  SSNs, IBANs and cloud access keys have enough structure to match without drowning the signal
  in false positives. A name, a postal address, a date of birth, a medical detail or a national
  ID with no fixed shape passes through untouched, and no amount of pattern-writing closes that
  gap. `redact` also applies only to what is ingested *after* it is turned on — reapplying it to
  an existing corpus needs `ingest --full` — and never touches the uploaded file on disk, which
  is deliberate (it keeps the decision reversible) and means the original is still there. A
  deployment with a real regulatory obligation wants a purpose-built classifier and a human
  review step; this is a floor, not a ceiling.
- **Postgres keyword search does not rank like BM25.** `KEYWORD_BACKEND=postgres` removes the
  in-memory index's memory ceiling and changes the ranking function while it is at it:
  `ts_rank_cd` is cover-density ranking and does not model document length or term saturation
  the way BM25's `k1`/`b` do, so the two backends return different orderings for the same
  corpus — unlike the two *vector* backends, which are tested for identical ranking. Fusion
  tolerates it because RRF votes on rank position rather than score, but it means switching is
  a retrieval-quality change and wants an eval run behind it. Its text-search configuration is
  fixed at `english`, so a non-English corpus is stemmed by the wrong rules; the in-memory
  scorer has the mirror-image problem of no stemming at all.
- **The conversational eval set has no recorded baseline in this repo.** `conversational.jsonl`
  is structurally validated in CI — every row carries history, the history alternates and ends
  on an assistant turn, reference contexts are real passages from the corpus, rows expecting no
  sources are the abstention cases — but a baseline can only be recorded by running the graph
  against real models, which this repo cannot do in CI. Until someone runs
  `rag-assistant eval --dataset data/golden_eval/conversational.jsonl --baseline
  data/golden_eval/conversational-baseline.json --record-baseline --limit 12`, condensation is
  covered by structural checks and unit tests but is not *gated*. The committed
  `dataset.jsonl` baseline is unaffected: single-turn rows send an empty history, and
  `condense_question` returns early on one, so their behaviour is byte-identical.
- **The refinement pass costs a call and a wait on exactly the hardest questions.** A
  low-confidence answer now pays for a rewrite plus a second local retrieval before it may also
  pay for a web search, so the worst-case question is slower and more expensive than it was.
  That is the intended trade — those are the questions most likely to be answered badly — but
  it means `GRAPH_TIMEOUT_SECONDS` has less headroom on a route that escalates twice, and the
  budget for such a question is roughly double.

- **A re-index still costs a full re-embed, and the multi-replica catch-up has a window.**
  [Generations](#re-indexing-without-downtime) remove the downtime, not the embedding bill.
  With several replicas the ingest lock is per process, so an ingest on another replica can
  reach the old generation in the seconds before that replica notices the flip; `activate`
  runs one more catch-up after the poll interval, and a file whose ingest is still in flight
  after that is indexed by the next ingest in its tenant. Semantic chunking computed its
  boundaries with the old model, and a re-embed keeps them — `--from-corpus` recomputes them.
- **Identity across systems is only as consistent as the directory.** Document permissions
  compare strings: a Drive ACL names `finance@acme.com`, a Confluence restriction names a group
  or an Atlassian account id, and an SSO token carries whatever the IdP puts in `groups` —
  Entra ID emits group *object ids* by default. Permissions work end to end when group names
  (or emails) are synced consistently, e.g. by SCIM from the same directory, and silently
  admit nobody when they are not. That is the safe direction to fail in, and it is still a
  failure someone has to notice.
- **Connectors are verified against the APIs' documented shapes, not against live tenants.**
  Confluence and Drive are tested through a mock HTTP transport replaying their REST
  responses; the filesystem connector end to end. The first sync against a real space is the
  real test — run it with a small `interval_minutes` and watch `connectors list`.

## Service objectives and recovery

Stated as targets rather than measurements. Nothing here has been observed under production
traffic, and the alert thresholds in `ops/` were chosen to match these numbers — the point of
writing them down is that an alert nobody can justify gets silenced the first time it fires at
3am.

| Objective | Target | Measured by |
| --- | --- | --- |
| Availability | 99.5% of `/api/v1/research` returning non-5xx | `rag_http_requests_total` by status class |
| Latency | p95 under 12s, p99 under 25s for a `both`-routed question | `rag_http_request_duration_seconds` |
| Saturation | `rag_research_in_flight` below 80% of `API_THREADPOOL_SIZE` | gauge vs. configured ceiling |
| Retrieval quality | No aggregate in `rag-assistant eval --check` regressing past tolerance against the recorded baseline | CI eval gate |
| Freshness | Keyword index within `BM25_VERSION_POLL_SECONDS` of the shared index | poll interval, bounded by construction |

**Recovery objectives.** The archive now covers Postgres-backed state as well as the two
directories, so a `VECTOR_BACKEND=pgvector` deployment is genuinely recoverable from one
file. RPO is the age of the last backup archive — `rag-assistant backup`
is a manual command, so RPO equals the operator's schedule and is *not* bounded by the system
itself. That is the honest statement: an unscheduled backup is not a recovery point objective.
RTO is dominated by restore, which stages the whole archive before swapping and moves the
existing data aside rather than deleting it, so a corrupt archive fails with the deployment
untouched.

The restore path is exercised by `tests/test_backup.py`, including the case where an archive
was built with a different embedding model, and **timed once against real corpus volume**: 30
PDFs (550MB on disk, 5,421 chunks, ~370 vision-transcribed pages) backed up in **7s** into a
128MB archive and restored in **3s**, after which the manifest, chunk count and corpus file
count all matched and a real query returned the expected document. RTO is therefore a measured
number at this size rather than an estimate — dominated by archive extraction, so it grows with
corpus size, not with chunk count.

Two caveats remain. The drill restored into clean directories rather than over a *running*
service, so it measures recovery time, not the cutover; and the process-local caches (vector
store client, BM25, conversations) still hold pre-restore state, which is why the command tells
you to restart the server. RPO is untouched by any of this: `rag-assistant backup` is a manual
command, so schedule it (cron, launchd, a CI job) or RPO stays "whenever someone last
remembered".

```bash
# One line of crontab is the difference between a stated RPO and none:
0 * * * * cd /path/to/repo && .venv/bin/rag-assistant backup --output /backups --keep 24
```

## Future improvements

Deliberately scoped out as needing a concrete driving requirement before they're worth the
added complexity:

- **Qdrant (or another dedicated vector DB) instead of Chroma** — worth it once richer
  filtering or corpus size actually demand it. Per-tenant collections no longer need it
  (`TENANT_ISOLATION=strict`), and Chroma is not the bottleneck today.
- **Retrieval-quality evaluation of the optional features** — a scored comparison of
  structural vs. semantic chunking, and with vs. without reranking, on a corpus large enough
  for the difference to be measurable rather than anecdotal.
- **A work queue for ingestion and connector syncs** — both run in the process that started
  them, serialised per process. A broker would distribute them across replicas and remove the
  multi-replica catch-up window described under Known limitations.

## Testing

```bash
uv run pytest          # offline unit + node + e2e tests (no external API calls)
uv run pytest --cov    # ...with coverage (CI gates at 85%; currently ~90% with branch coverage)
uv run pytest -m live  # also exercises real Gemini/DuckDuckGo calls; requires .env and a run of `ingest` first
uv run ruff check .
```

```bash
cd frontend
npm test               # Vitest + React Testing Library — hooks and components
```

Three suites need a real service and skip cleanly without one. Each is gated on its own
environment variable, and the matching CI job **fails if the suite skips** — a suite that skips
itself produces a green job having verified nothing, which reads as coverage:

```bash
# Postgres-backed conversations and the pgvector index
RAG_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/postgres \
  uv run pytest tests/test_postgres_store.py tests/test_pgvector_store.py

# Redis-backed answer cache, per-tenant budget, ingest tasks and rate-limit counters
RAG_TEST_REDIS_URL=redis://127.0.0.1:6379/0 uv run pytest tests/test_redis_backed.py

# Chroma in server mode (CHROMA_SERVER_HOST), over HTTP rather than an on-disk SQLite file
RAG_TEST_CHROMA_HOST=127.0.0.1 RAG_TEST_CHROMA_PORT=8000 uv run pytest tests/test_chroma_server.py
```

Each has a one-line Docker equivalent if you do not have the service to hand:

```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17
docker run -d -p 6379:6379 redis:7-alpine
docker run -d -p 8000:8000 chromadb/chroma:1.5.9
```

Optionally install the pre-commit hooks so lint, lockfile drift, and accidentally-staged
`.env` files are caught before CI:

```bash
uv run pre-commit install
```

CI (`.github/workflows/ci.yml`) runs on every push and PR:

| Job | What it does |
| --- | --- |
| **backend** | ruff lint + format check, pytest with a coverage floor |
| **postgres** | the Postgres-backed backends against a `pgvector/pgvector` service container, with their own coverage gate |
| **redis** | the Redis-backed paths against a real Redis, reading back through a second client |
| **chroma-server** | Chroma server mode against a real Chroma, over HTTP |
| **static-analysis** | `bandit` over `src` (blocking, zero baseline) and `gitleaks` over the full history |
| **frontend** | oxlint, Vitest, production build |
| **audit** | `pip-audit` over the exported lockfile and `npm audit` — non-blocking, since an advisory often has no released fix |
| **eval** | the retrieval quality gate against the recorded baseline; skips on forks, which have no API keys |
| **docker** | build, Trivy image scan, then boot the real image and wait on `/health` |

`.github/workflows/private-eval.yml` gates a private corpus on its own weekly schedule, and
`.github/workflows/deploy.yml` deploys `main` after a green CI run — both skip with a notice
when unconfigured. `tests/test_workflow_contracts.py` asserts the properties these files are
relied on for: that the service-container jobs fail rather than pass when their suite skips,
that images are pinned, and that the deploy refuses a failed CI run.

## Project layout

```
src/rag_assistant/
├── config.py, llm.py, logging_conf.py   # settings, model factories, structured JSON logging
├── tracing.py, cache.py, readiness.py    # trace-ID contextvar, Redis cache, Chroma/web search health checks
├── metrics.py, auth.py                   # Prometheus collectors + LLM callback handler, API keys + principals
├── oidc.py                               # single sign-on: verifying identity-provider access tokens
├── backup.py, loadtest.py                # snapshot/restore, concurrency measurement
├── advisory_lock.py                      # deterministic Postgres lock ids (hash() is per-process salted)
├── ingestion/                            # load -> split -> embed -> index the sample corpus;
│                                          # acl.py (document permissions), generations.py +
│                                          # reindex.py (rebuild beside the serving index, then switch)
├── connectors/                           # Confluence, Google Drive, file-share sync + deletion guard
├── retrieval/                            # Chroma + pgvector stores, BM25 keyword store, DuckDuckGo web search
├── fusion/rrf.py                         # Reciprocal Rank Fusion (pure function)
├── grading/relevance_grader.py           # batched LLM relevance grading
├── graph/                                # ResearchState, one node module per concept, build_graph(),
│                                          # research_summary.py (explainability panel builder)
├── prompts/                              # prompt templates per LLM-backed node
├── eval/                                 # golden dataset loader + RAGAS eval harness
├── schemas/models.py                     # internal domain / structured-output schemas
├── schemas/api.py                        # external API request/response contracts
├── cli.py                                # Typer app: ingest / ask / serve / eval / reindex / connectors / ...
└── api.py                                # FastAPI: GET /health, GET /ready, POST /research, POST /research/stream

Dockerfile, docker-compose.yml, .dockerignore  # multi-stage build, non-root user, api + redis services

frontend/src/
├── api/client.ts                         # fetch + SSE client for the backend API
├── api/sso.ts                            # OIDC authorization code + PKCE sign-in, no library
├── hooks/useHealthStatus.ts              # polls GET /health on mount
├── hooks/useResearchStream.ts            # SSE streaming + progress/result state, testable in isolation
├── constants/exampleQuestions.ts         # example-question chip data
├── components/                           # Header, AskCard, ResultCard, ResearchSummaryPanel,
│                                          # GraphVisualization, ErrorBanner, ErrorBoundary
├── test/setup.ts                         # jest-dom matchers + RTL cleanup for Vitest
├── App.tsx                               # composition root
└── index.css                             # shared theme (light/dark)
```

## License

MIT — see [LICENSE](LICENSE).
