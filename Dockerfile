# =============================================================================
# Digital Twin — RunPod Serverless (CPU) Dockerfile
# Matches pinned deps:
# runpod==1.6.2, numpy==1.24.3, opencv-python-headless==4.8.1.78,
# Pillow==10.1.0, trimesh==4.0.5, scipy==1.11.4
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

# Only the runtime libs needed for opencv-python-headless wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# App directory + non-root user
WORKDIR /app
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Install Python deps (use your exact pins from requirements.txt)
COPY --chown=appuser:appuser requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser runpod_handler.py .

# (Optional) basic healthcheck — remove if your platform objects
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import sys; sys.exit(0)"

# Entrypoint
CMD ["python", "-u", "runpod_handler.py"]
