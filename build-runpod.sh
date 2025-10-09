#!/bin/bash
set -e

REGISTRY="your-dockerhub-username"  # Change this
IMAGE_NAME="digital-twin-runpod"
TAG="v1.0.0"

echo "🏗️  Building for RUNPOD..."
docker build \
  --platform linux/amd64 \
  -t ${REGISTRY}/${IMAGE_NAME}:${TAG} \
  -t ${REGISTRY}/${IMAGE_NAME}:latest \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  .

echo "📤 Pushing to Docker Hub..."
docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}
docker push ${REGISTRY}/${IMAGE_NAME}:latest

echo "✅ Deployed to: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
echo "🚀 Use this image in RunPod template configuration"
