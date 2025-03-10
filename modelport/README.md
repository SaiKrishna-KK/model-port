# ModelPort 🧳
ModelPort lets you export and run your machine learning models **anywhere** — regardless of architecture or operating system.

## ✅ Features
- Export PyTorch models to ONNX
- Package model + inference code
- Run cross-platform with Docker (`x86_64`, `arm64`, etc.)

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
# Export your model
python modelport.py export model.pt

# Run it on desired platform
python modelport.py run modelport_export --arch linux/arm64
```

## 🐳 Docker Setup for Mac M1/M2

If you're using Apple Silicon (M1/M2), you'll need to ensure Docker is set up for multi-architecture builds:

```bash
# Enable buildx if you haven't already
docker buildx create --use

# Test with a simple build (optional)
docker buildx build --platform linux/amd64,linux/arm64 -t test-multiarch .
```

## 🔧 Architecture Support
- ✅ x86_64 (Intel, AMD)
- ✅ arm64 (Apple M1/M2, Jetson, Raspberry Pi) 