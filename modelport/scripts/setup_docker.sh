#!/bin/bash

# setup_docker.sh - Configure Docker for cross-platform builds
# Used by ModelPort to ensure Docker is properly configured for multi-architecture builds

echo "🐳 Setting up Docker buildx for multi-platform builds..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop first."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Create and use a new buildx builder
echo "Creating buildx builder instance..."
docker buildx create --name modelport_builder --use

# Check platform support
echo "Verifying platform support..."
PLATFORMS=$(docker buildx inspect --bootstrap | grep "Platforms:")

echo "✅ Available platforms for builds: $PLATFORMS"
echo "🚀 Docker buildx is now configured for ModelPort!"

# Test with a simple image (optional)
# Uncomment to test buildx with a simple image
# echo "Testing with a simple multi-platform build..."
# docker buildx build --platform linux/amd64,linux/arm64 -t modelport_test --load . 