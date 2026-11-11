FROM python:3.11-slim

# ── Build arguments ───────────────────────────────────────────────
ARG APP_HOME=/app
WORKDIR ${APP_HOME}

# ── System dependencies ───────────────────────────────────────────
# libpq-dev: required by asyncpg native build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (layer-cached before source copy) ─────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# ── Non-root user for security ────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser ${APP_HOME}
USER appuser

# ── Runtime configuration ─────────────────────────────────────────
ENV PYTHONPATH=${APP_HOME}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/query/health').raise_for_status()"

# ── Entry point ───────────────────────────────────────────────────
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
