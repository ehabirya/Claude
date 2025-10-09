#!/bin/bash
set -e

IMAGE_NAME="digital-twin"
TAG="latest"

echo "🏗️  Building for LOCAL testing..."
docker build \
  --platform linux/amd64 \
  -t ${IMAGE_NAME}:${TAG} \
  --progress=plain \
  .

echo "🧪 Testing image..."
docker run --rm ${IMAGE_NAME}:${TAG} python -c "
import cv2, numpy, trimesh, mediapipe, runpod
print('✅ All imports successful!')
"

echo "✅ Build complete! Run with:"
echo "   docker run --rm -it ${IMAGE_NAME}:${TAG}"
