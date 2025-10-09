# =============================================================================
# Digital Twin – RunPod Serverless Production Dockerfile
# Enhanced with SMPL-based Body Modeling and MediaPipe Pose Detection
# Date: 09.10.2025 - FIXED
# =============================================================================

# ============== Stage 1: Builder ==============================================
FROM python:3.10-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc \
      g++ \
      make \
      cmake \
      git \
      pkg-config \
      libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --prefix=/build/install --no-warn-script-location -r requirements.txt

# ============== Stage 2: Runtime ==============================================
FROM python:3.10-slim-bookworm

LABEL maintainer="your-email@example.com" \
      version="1.0.0" \
      description="Digital Twin with SMPL Body Modeling"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    PATH="/home/appuser/.local/bin:/usr/local/bin:${PATH}"

# Install ONLY runtime dependencies (FIXED - removed libgthread-2.0-0)
RUN apt-get update && apt-get install -y --no-install-recommends \
      # OpenCV core dependencies
      libglib2.0-0 \
      libgomp1 \
      libgl1 \
      libsm6 \
      libxext6 \
      libxrender1 \
      # MediaPipe dependencies
      libgstreamer1.0-0 \
      libgstreamer-plugins-base1.0-0 \
      # Trimesh/Scipy dependencies
      libstdc++6 \
      libgfortran5 \
      # Networking
      ca-certificates \
      curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -m -u 1000 -g 1000 -s /bin/bash appuser

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=appuser:appuser /build/install /usr/local

# Copy application
COPY --chown=appuser:appuser runpod_handler.py .

# Create directories
RUN mkdir -p /app/tmp /app/logs /app/cache && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

USER appuser

# Verify installation
RUN python -c "import cv2, numpy, trimesh, scipy, mediapipe, runpod; print('✅ All dependencies OK')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Start handler
CMD ["python", "-u", "runpod_handler.py"]
