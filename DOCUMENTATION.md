# ModelPort Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [API Reference](#api-reference)
7. [Capsule Format](#capsule-format)
8. [Docker Integration](#docker-integration)
9. [GPU Support](#gpu-support)
10. [Troubleshooting](#troubleshooting)

## Introduction

ModelPort is an open-source tool designed to simplify the deployment of machine learning models across different platforms and architectures. It converts models to ONNX format and packages them in a standardized format that can be easily deployed using Docker.

### Key Features

- **Framework Auto-Detection**: Automatically detects PyTorch, ONNX, and TensorFlow models
- **Export to ONNX Format**: Convert models to ONNX for maximum interoperability
- **Model Validation**: Test exported models with dummy inputs
- **Cross-Platform Containers**: Deploy on x86_64, ARM64, or with GPU acceleration
- **Docker Registry Integration**: Push capsules to Docker Hub or custom registries

## Architecture

ModelPort follows a modular architecture with these main components:

1. **Core Module** (`modelport/core/`)
   - `exporter.py`: Handles model conversion to ONNX
   - `deployer.py`: Manages Docker image creation and deployment
   - `model_utils.py`: Utilities for framework detection and model metadata

2. **CLI Module** (`modelport/cli/`)
   - `export.py`: Command-line interface for exporting models
   - `deploy.py`: Command-line interface for deploying models
   - `run.py`: Command-line interface for running models

3. **Templates** (`modelport/templates/`)
   - Dockerfile templates for different architectures

4. **Examples** (`modelport/examples/`)
   - Sample code for using ModelPort

## Installation

### Using pip (recommended)

```bash
# Install from PyPI (coming soon)
pip install modelport

# Or install directly from GitHub
pip install git+https://github.com/SaiKrishna-KK/model-port.git
```

### Development Install

```bash
# Clone the repository
git clone https://github.com/SaiKrishna-KK/model-port.git
cd model-port

# Create and activate a conda environment
conda env create -f modelport/environment.yml
conda activate modelport_env

# Install in development mode
pip install -e .
```

## Basic Usage

### Exporting Models

The simplest way to use ModelPort is to export a model:

```bash
# Export with auto-detection of framework
modelport export path/to/model.pt
```

The exported model will be saved in a directory called `modelport_export` by default.

### Deploying Models

After exporting, you can deploy the model to Docker:

```bash
# Deploy the exported model
modelport deploy modelport_export --tag username/model:latest
```

### Running Models

You can also run the model locally:

```bash
# Run the exported model
modelport run modelport_export
```

## Advanced Features

### Framework Auto-Detection

ModelPort can automatically detect the framework based on the file extension:
- `.pt`, `.pth`, `.ckpt` → PyTorch
- `.onnx` → ONNX
- `.h5`, `.pb`, `.savedmodel` → TensorFlow

```bash
# Framework will be auto-detected
modelport export path/to/model.pt
```

### Model Validation

Use the `--test` flag to validate that your model can run inference:

```bash
modelport export path/to/model.pt --test
```

This will:
1. Export the model to ONNX
2. Run inference with a dummy input
3. Record the results in `capsule_spec.json`

### GPU Support

For models that benefit from GPU acceleration:

```bash
modelport deploy modelport_export --gpu
```

This will:
1. Create a Docker image with NVIDIA CUDA support
2. Use onnxruntime-gpu instead of onnxruntime

## API Reference

### Export API

```python
from modelport.core.exporter import export_model

export_dir = export_model(
    model_path="path/to/model.pt",
    output_dir="output_directory",
    framework=None,  # Auto-detect if None
    input_shape=None,  # Optional: override input shape
    force=False,  # Whether to overwrite existing output
    test=False  # Whether to validate the model
)
```

### Deploy API

```python
from modelport.core.deployer import deploy_capsule

image_tag = deploy_capsule(
    capsule_path="path/to/capsule",
    tag="username/model:latest",
    platform="linux/amd64,linux/arm64",
    push=False,  # Whether to push to Docker Hub
    gpu=False,  # Whether to use GPU runtime
    registry="docker.io"  # Docker registry
)
```

## Capsule Format

ModelPort packages models in a standardized format with everything needed to run the model:

```
modelport_capsule/
├── model.onnx             # Model converted to ONNX format
├── inference.py           # Sample inference code
├── config.json            # Input/output metadata
├── requirements.txt       # Python dependencies
├── capsule_spec.json      # Capsule metadata & version info
├── runtime/
│   ├── Dockerfile.x86_64  # x86_64 Docker container
│   ├── Dockerfile.arm64   # ARM64 Docker container
│   └── Dockerfile.gpu     # GPU-enabled Docker container
```

### config.json

Contains model metadata:

```json
{
  "framework": "pytorch",
  "input_shape": [1, 3, 224, 224],
  "input_dtype": "float32",
  "test_results": {
    "success": true,
    "input_shape": [1, 3, 224, 224],
    "output_shapes": [[1, 1000]],
    "timestamp": "2023-06-15T14:32:15.654321"
  }
}
```

### capsule_spec.json

Contains deployment information:

```json
{
  "version": "1.0",
  "name": "resnet18",
  "framework": "pytorch",
  "created_at": "2023-06-15T14:32:10.123456",
  "input_shape": [1, 3, 224, 224],
  "input_dtype": "float32",
  "runtime": {
    "supports_gpu": true,
    "supports_cpu": true,
    "supported_platforms": ["linux/amd64", "linux/arm64"]
  },
  "deployment": {
    "image": "modelport/resnet18:latest",
    "platforms": ["linux/amd64", "linux/arm64"],
    "gpu_enabled": true
  }
}
```

## Docker Integration

ModelPort integrates with Docker to create containers for different platforms.

### Building for Multiple Platforms

```bash
modelport deploy my_model --platform "linux/amd64,linux/arm64"
```

### Pushing to Docker Registry

```bash
modelport deploy my_model --tag username/model:latest --push
```

### Using Custom Registry

```bash
modelport deploy my_model --registry ghcr.io --tag username/model:latest
```

## GPU Support

ModelPort supports GPU acceleration for inference.

### Prerequisites

1. NVIDIA GPU with compatible drivers
2. NVIDIA Container Toolkit installed

### Creating GPU-Enabled Containers

```bash
modelport deploy my_model --gpu
```

This creates a Docker image with:
- NVIDIA CUDA runtime
- onnxruntime-gpu for accelerated inference

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Make sure to install all dependencies: `pip install -r requirements.txt`

2. **Docker Build Fails**
   - Ensure Docker is properly installed and running
   - For ARM64 support, make sure buildx is set up

3. **ONNX Export Errors**
   - Check if your model is supported for ONNX export
   - Try setting a specific input shape: `--input-shape 1,3,224,224`

4. **GPU Inference Issues**
   - Verify NVIDIA drivers are properly installed
   - Check that NVIDIA Container Toolkit is installed

### Getting Help

For issues not covered in this documentation:
- Check the [GitHub issues](https://github.com/SaiKrishna-KK/model-port/issues)
- Submit a new issue with detailed information about the problem 