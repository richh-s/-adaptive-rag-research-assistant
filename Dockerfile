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

# ---- runtime: copy the built venv + source only, run as non-root ----
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY data/corpus ./data/corpus

# Chroma's persist directory is mounted as a volume at runtime (see docker-compose.yml); create
# it here, owned by the non-root user, so a fresh named volume inherits correct ownership instead
# of being created root-owned on first mount.
RUN mkdir -p /app/chroma_db && chown -R appuser:appuser /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "rag_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
