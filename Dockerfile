# Multi-stage build. Project requires Python >=3.12 (pyproject.toml), so this uses 3.12-slim
# rather than 3.11 -- 3.11 wouldn't satisfy the project's own dependency constraints.

# ---- builder: resolve + install deps with uv, isolated from the runtime image ----
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy only the dependency manifests first so this layer is cached across source-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

# ---- frontend: build the React app so the API can serve it from one container ----
FROM node:22-alpine AS frontend

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend ./
# Empty VITE_API_BASE_URL makes the built app call the same origin that served it -- exactly
# right when FastAPI serves these static files itself (see api.py's STATIC_DIR mount).
ENV VITE_API_BASE_URL=""
RUN npm run build

# ---- runtime: copy the built venv + source only, run as non-root ----
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY --from=frontend /frontend/dist ./static
COPY data/corpus ./data/corpus

# Chroma's persist directory is mounted as a volume at runtime (see docker-compose.yml); create
# it here, owned by the non-root user, so a fresh named volume inherits correct ownership instead
# of being created root-owned on first mount.
RUN mkdir -p /app/chroma_db && chown -R appuser:appuser /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STATIC_DIR=/app/static

USER appuser

EXPOSE 8000

# Index the baked-in corpus before serving (incremental: a mounted volume that's already
# indexed skips straight through; a fresh diskless instance -- e.g. Render's free tier --
# embeds the baseline corpus once at boot). `|| true` so a transient embedding-API failure
# degrades to web-search-only instead of crash-looping the container. ${PORT} is set by
# PaaS hosts like Render; defaults to 8000 for docker-compose/local runs.
# --proxy-headers + --forwarded-allow-ips: behind a PaaS load balancer (Render etc.) the
# direct peer is the LB, so without honoring X-Forwarded-For every visitor shares one
# rate-limit bucket. Trusting all proxies is right for platforms whose LB always sits in
# front; override FORWARDED_ALLOW_IPS when exposing the container directly.
CMD ["sh", "-c", "rag-assistant ingest || true; exec uvicorn rag_assistant.api:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
