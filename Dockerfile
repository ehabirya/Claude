# =============================================================================
# Digital Twin – RunPod Serverless (CPU) Dockerfile
# Enhanced with SMPL-based Body Modeling and MediaPipe Pose Detection
# Date: 09.10.2025
# =============================================================================

FROM python:3.10-slim-bookworm

# Environment variables: faster startup, cleaner images, tame BLAS threads
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PATH="/home/appuser/.local/bin:${PATH}"

# System dependencies for OpenCV, MediaPipe, and mesh processing
RUN apt-get update && apt-get install -y --no-install-recommends \
      # OpenCV dependencies
      libglib2.0-0 \
      libgomp1 \
      libgl1 \
      libgthread-2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      # MediaPipe dependencies
      libgstreamer1.0-0 \
      libgstreamer-plugins-base1.0-0 \
      # Additional libraries for trimesh and scipy
      libstdc++6 \
      && rm -rf /var/lib/apt/lists/*

# Create app directory and non-root user
WORKDIR /app
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Copy and install Python dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

# Copy application code
# Note: Make sure your actual file is named 'runpod_handler.py' or update the filename
COPY --chown=appuser:appuser runpod_handler.py .

# Optional: Verify imports work correctly
RUN python -c "import cv2, numpy, trimesh, mediapipe, scipy; print('All dependencies loaded successfully')" || \
    echo "Warning: Some imports failed - check requirements.txt"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Entrypoint
CMD ["python", "-u", "runpod_handler.py"]
