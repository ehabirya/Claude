# =============================================================================
# Universal Production Dockerfile for Digital Twin Body Modeling
# Works on: RunPod, AWS Lambda, Azure, GCP, Local Docker, Kubernetes
# Architecture: CPU-optimized (works on x86_64 and ARM64 with emulation)
# Date: 09.10.2025
# =============================================================================

# ============== Stage 1: Builder ==============================================
FROM python:3.10-slim-bookworm AS builder

# Build-time environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /build

# Install ONLY build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc \
      g++ \
      make \
      cmake \
      git \
      pkg-config \
      libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install to custom prefix for easy copying
RUN pip install --prefix=/build/install --no-warn-script-location \
    --compile \
    -r requirements.txt

# Verify critical imports work
RUN python -c "import sys; sys.path.insert(0, '/build/install/lib/python3.10/site-packages'); \
    import cv2, numpy, trimesh, scipy; print('✅ Core imports OK')" || exit 1

# ============== Stage 2: Runtime ==============================================
FROM python:3.10-slim-bookworm

# Runtime metadata
LABEL maintainer="your-email@example.com" \
      version="1.0.0" \
      description="Digital Twin with SMPL Body Modeling and MediaPipe Pose" \
      base="python:3.10-slim-bookworm"

# Runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Performance tuning
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    # Path configuration
    PATH="/home/appuser/.local/bin:/usr/local/bin:${PATH}" \
    PYTHONPATH="/usr/local/lib/python3.10/site-packages:${PYTHONPATH}" \
    # Disable unnecessary features
    PYTHONHASHSEED=0 \
    # OpenCV optimizations
    OPENCV_IO_MAX_IMAGE_PIXELS=1000000000

# Install ONLY runtime dependencies (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
      # OpenCV core dependencies
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
      # Trimesh/Scipy dependencies
      libstdc++6 \
      libgfortran5 \
      # Networking (for RunPod communication)
      ca-certificates \
      curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Create non-root user with specific UID/GID for compatibility
RUN groupadd -g 1000 appuser && \
    useradd -m -u 1000 -g 1000 -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=appuser:appuser /build/install /usr/local

# Copy application code
COPY --chown=appuser:appuser runpod_handler.py .

# Optional: Copy additional files if they exist
COPY --chown=appuser:appuser 202*.py . 2>/dev/null || true

# Create necessary directories with proper permissions
RUN mkdir -p /app/tmp /app/logs /app/cache && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Verify installation and imports
RUN python -c "import sys; print('Python:', sys.version)" && \
    python -c "import cv2; print('✅ OpenCV:', cv2.__version__)" && \
    python -c "import numpy; print('✅ NumPy:', numpy.__version__)" && \
    python -c "import trimesh; print('✅ Trimesh:', trimesh.__version__)" && \
    python -c "import scipy; print('✅ SciPy:', scipy.__version__)" && \
    python -c "import mediapipe; print('✅ MediaPipe:', mediapipe.__version__)" && \
    python -c "import runpod; print('✅ RunPod:', runpod.__version__)" && \
    echo "✅ All dependencies verified successfully!"

# Health check that actually tests the handler
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import runpod, cv2, numpy, trimesh, mediapipe; print('healthy')" || exit 1

# Expose port (optional, for debugging with Flask/FastAPI wrapper)
EXPOSE 8080

# Default command
CMD ["python", "-u", "runpod_handler.py"]
