# Frontend

React + Vite single-page app for the Adaptive RAG Research Assistant. See the [root README](../README.md)
for how this fits together with the FastAPI backend.

## Development

```bash
cp .env.example .env   # only needed if the backend isn't on http://127.0.0.1:8000
npm install
npm run dev
```

Requires the backend running separately (`uv run rag-assistant serve`) so `/health` and `/research`
are reachable.

## Build

```bash
npm run build   # type-checks then outputs to dist/
```
