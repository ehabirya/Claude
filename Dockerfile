# ============================================================
# RunPod Serverless Container for Digital Twin Pose Estimation
# Python 3.11 slim | Small & Fast | Updated 2025-10-10
# ============================================================
FROM python:3
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # RunPod serverless worker looks up this handler (module.function)
    RPU_HANDLER=runpod_handler.run

# ---- Runtime libs only (no compilers) ----
# libspatialindex-dev supplies the shared libs Rtree needs on Bookworm
# libgl1 / libglib2.0-0 are occasionally required by OpenCV headless code paths
RUN apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libspatialindex-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps with maximum cache hits
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code (keep your original functions intact)
COPY . .

# Start RunPod serverless worker (uses RPU_HANDLER)
CMD ["python", "-m", "runpod.serverless.worker"]
