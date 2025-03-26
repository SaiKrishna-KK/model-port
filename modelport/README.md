# ModelPort 🧳
ModelPort lets you export and run your machine learning models **anywhere**, regardless of architecture or operating system.

## ✅ Features
- **Framework auto-detection** (PyTorch, ONNX, TensorFlow)
- **Export to ONNX** for cross-platform compatibility
- **Package model + inference code** in a single capsule
- **Docker container generation** for cross-platform deployment
- **GPU support** for accelerated inference
- **Model validation** during export
- **Deploy to Docker Hub** or private registries

## 📦 Getting Started

### Environment Setup with Conda

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate modelport_env

# Or install packages manually
conda create -n modelport_env python=3.10 -y
conda activate modelport_env
pip install typer[all] torch onnx onnxruntime numpy
```

### Running ModelPort

```bash
# Export your model (framework auto-detected!)
python modelport.py export model.pt

# Export with validation
python modelport.py export model.pt --test

# Customize export parameters
python modelport.py export model.pt --input-shape 1,3,224,224 --framework pytorch

# Run on desired platform
python modelport.py run modelport_export --arch linux/arm64

# Deploy to Docker Hub
python modelport.py deploy modelport_export --tag username/model:latest --push

# Build GPU-enabled Docker image
python modelport.py deploy modelport_export --tag username/model:gpu --gpu
```

## 📄 ModelPort Capsule Spec

ModelPort packages models in a consistent format with everything needed to run the model:

```
modelport_capsule/
├── model.onnx             # Model converted to ONNX format
├── inference.py           # Sample inference code
├── config.json            # Input/output metadata
├── requirements.txt       # Python dependencies
├── capsule_spec.json      # Capsule metadata
├── runtime/
│   ├── Dockerfile.x86_64  # x86_64 Docker container
│   ├── Dockerfile.arm64   # ARM64 Docker container
│   └── Dockerfile.gpu     # GPU-enabled Docker container
```

## 🔧 Architecture Support
- ✅ x86_64 (Intel, AMD)
- ✅ arm64 (Apple M1/M2, Jetson, Raspberry Pi)
- ✅ NVIDIA GPU (with CUDA support)

## 🚀 Deployment Options
- ✅ Docker Hub
- ✅ Local Docker
- ✅ Custom registries
- ✅ Multiple architectures (multi-platform)

## 🔄 Phase 1.5 Features
- ✅ `modelport deploy` command - Push Docker-based capsules to registries
- ✅ GPU-enabled Docker support - Generate and run GPU-optimized containers
- ✅ Framework auto-detection - Automatically detect PyTorch, ONNX, etc.
- ✅ `--test` flag - Validate exported models with dummy inputs
- ✅ Capsule Format V1 Spec - Standardized metadata for better interoperability

## 🐳 Docker Setup for Mac M1/M2

If you're using Apple Silicon (M1/M2), you'll need to ensure Docker is set up for multi-architecture builds:

```bash
# Enable buildx if you haven't already
docker buildx create --use

# Test with a simple build (optional)
docker buildx build --platform linux/amd64,linux/arm64 -t test-multiarch .
```
