# ============================================================
# Stage 1: builder — install deps and download all model weights
# ============================================================
FROM python:3.11-slim AS builder

# System packages needed for OpenCV, PaddleOCR, fitz, piexif, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgeos-dev \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------
COPY pyproject.toml ./
# Install the package in editable mode so all deps are resolved
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e ".[dev]"

# ---------------------------------------------------------------------------
# Copy source code
# ---------------------------------------------------------------------------
COPY pii_redact/ ./pii_redact/
COPY config/     ./config/
COPY scripts/    ./scripts/

# ---------------------------------------------------------------------------
# Pre-download all model weights
# ---------------------------------------------------------------------------
# Set HuggingFace cache dirs so weights land in /app/models (copied to runtime stage)
ENV TRANSFORMERS_CACHE=/app/models/trocr \
    HF_HOME=/app/models/trocr \
    HF_HUB_CACHE=/app/models/trocr

RUN mkdir -p /app/models/trocr /app/models/paddleocr /app/models/yolov8 /app/models/sam

# PaddleOCR + TrOCR + spaCy + YOLOv8
# We use || true on individual steps so a partial failure doesn't abort the build;
# CI or image verification will catch missing weights.
RUN python scripts/download_models.py || true

# spaCy en_core_web_lg (may already be done by download_models.py, idempotent)
RUN python -m spacy download en_core_web_lg || true

# ============================================================
# Stage 2: runtime — lean image, no build tools
# ============================================================
FROM python:3.11-slim AS runtime

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY --from=builder /app/pii_redact  ./pii_redact
COPY --from=builder /app/config      ./config

# Copy pre-downloaded model weights
COPY --from=builder /app/models      ./models

# Declare volume mount points
VOLUME ["/app/inbox", "/app/redacted", "/app/vault", "/app/audit_logs"]

# ---------------------------------------------------------------------------
# Offline mode — no network calls allowed at runtime
# ---------------------------------------------------------------------------
ENV TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_CACHE=/app/models/trocr \
    HF_HOME=/app/models/trocr \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

ENTRYPOINT ["python", "-m", "pii_redact.cli"]
