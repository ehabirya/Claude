# ===========================09.10.2025==================================================
# Digital Twin – RunPod Serverless (CPU) Dockerfile
# With MediaPipe for body pose detection
# =============================================================================
FROM python:3.10-slim-bookworm
# Environment: faster startup, cleaner images, tame BLAS threads
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libgomp1 \
      libgl1 \
      libgthread-2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libgstreamer1.0-0 \
      libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# App directory + non-root user
WORKDIR /app
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Copy requirements and install Python dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser runpod_handler.py .

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import sys; sys.exit(0)"

# Entrypoint
CMD ["python", "-u", "runpod_handler.py"]
