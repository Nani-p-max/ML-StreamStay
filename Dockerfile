# ─────────────────────────────────────────────────────────────────────────────
# StreamStay W12 — Dockerfile
# Multi-stage build: keeps the final image lean (~500 MB vs 2 GB+)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools (needed by some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY features.py      .
COPY api.py           .

# Copy versioned model artefact(s) — the latest will be loaded at startup
COPY models/          ./models/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Environment defaults (override at runtime)
ENV PORT=8000 \
    WORKERS=1 \
    ENV=prod

EXPOSE 8000

# Liveness probe for Docker / Kubernetes
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
  || exit 1

# Start the API
CMD ["sh", "-c", \
     "uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]
