# syntax=docker/dockerfile:1.6

# ---------- builder ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Install only what's needed to build wheels (kept out of the runtime image).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .

# Pre-build everything into a venv so the final stage only needs to copy it.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install -r requirements-serve.txt


# ---------- runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_USE_LEGACY_KERAS=1 \
    HOST=0.0.0.0 \
    PORT=5000 \
    PATH="/opt/venv/bin:$PATH"

# Runtime libs only — no compilers. opencv-python-headless still needs libgl1
# for some codepaths, and mediapipe needs GLES/EGL for its inference graph.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libgles2 libegl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# Copy only what the runtime needs (data/ and tests/ are excluded via .dockerignore).
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser templates/ ./templates/
COPY --chown=appuser:appuser models/ ./models/

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Single gunicorn worker — model lives in worker memory; multiple workers each
# load the full TF graph (~600MB RSS). Scale with replicas, not workers.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "60", "--access-logfile", "-", "app:app"]
