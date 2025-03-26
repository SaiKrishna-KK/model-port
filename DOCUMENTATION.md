# ModelPort Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Installation](#installation)
   - [TVM Installation for Native Compilation](#tvm-installation-for-native-compilation)
4. [Basic Usage](#basic-usage)
   - [Compiling Models to Native Libraries](#compiling-models-to-native-libraries-v20)
5. [Advanced Features](#advanced-features)
   - [Framework Auto-Detection](#framework-auto-detection)
   - [Model Validation](#model-validation)
   - [GPU Support](#gpu-support)
   - [Native Compilation](#native-compilation)
6. [API Reference](#api-reference)
   - [Compile API](#compile-api-v20)
7. [Capsule Format](#capsule-format)
8. [Docker Integration](#docker-integration)
9. [GPU Support](#gpu-support-1)
10. [Native Compilation](#native-compilation-1)
    - [TVM Integration](#tvm-integration)
    - [Compilation Process](#compilation-process)
    - [Supported Architectures](#supported-architectures)
    - [Command-line Usage](#command-line-usage)
    - [C++ Integration](#c-integration)
11. [Troubleshooting](#troubleshooting)

## Introduction

ModelPort is an open-source tool designed to simplify the deployment of machine learning models across different platforms and architectures. It converts models to ONNX format and packages them in a standardized format that can be easily deployed using Docker.

### Key Features

- **Framework Auto-Detection**: Automatically detects PyTorch, ONNX, and TensorFlow models
- **Export to ONNX Format**: Convert models to ONNX for maximum interoperability
- **Model Validation**: Test exported models with dummy inputs
- **Cross-Platform Containers**: Deploy on x86_64, ARM64, or with GPU acceleration
- **Docker Registry Integration**: Push capsules to Docker Hub or custom registries
- **Native Compilation**: Compile models to native shared libraries for zero-dependency deployment (v2.0+)

## Architecture

ModelPort follows a modular architecture with these main components:

1. **Core Module** (`modelport/core/`)
   - `exporter.py`: Handles model conversion to ONNX
   - `deployer.py`: Manages Docker image creation and deployment
   - `model_utils.py`: Utilities for framework detection and model metadata
   - `compiler.py`: Compiles ONNX models to native libraries using TVM (v2.0+)

2. **CLI Module** (`modelport/cli/`)
   - `export.py`: Command-line interface for exporting models
   - `deploy.py`: Command-line interface for deploying models
   - `run.py`: Command-line interface for running models
   - `compile.py`: Command-line interface for compiling models (v2.0+)
   - `run_native.py`: Command-line interface for running compiled models (v2.0+)

3. **Templates** (`modelport/templates/`)
   - Dockerfile templates for different architectures

4. **Examples** (`modelport/examples/`)
   - Sample code for using ModelPort
   - C++ example for running compiled models (v2.0+)

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

### TVM Installation for Native Compilation

If you want to use the native compilation features (v2.0+), you'll need to install TVM:

```bash
# Install TVM via pip (simplified installation)
pip install apache-tvm

# Or for custom installation with GPU support, follow the instructions at:
# https://tvm.apache.org/docs/install/index.html
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

### Compiling Models to Native Libraries (v2.0+)

For maximum performance and minimal dependencies, you can compile models to native code:

```bash
# Compile an ONNX model to a native shared library
modelport compile path/to/model.onnx

# Run the compiled model
modelport run-native modelport_native
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

### Native Compilation

Compile models to native shared libraries for maximum performance:

```bash
# Compile for the current architecture
modelport compile model.onnx

# Compile with specific optimization level
modelport compile model.onnx --opt-level 3

# Compile for a specific architecture
modelport compile model.onnx --target-arch arm64

# Compile for GPU
modelport compile model.onnx --target-device cuda
```

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

### Compile API (v2.0+)

```python
from modelport.core.compiler import compile_model

config = compile_model(
    model_path="path/to/model.onnx",
    output_dir="output_directory",
    target_arch=None,  # Auto-detect if None
    target_device="cpu",  # 'cpu', 'cuda', 'metal', 'opencl'
    opt_level=3,  # Optimization level (0-3)
    input_shapes=None  # Optional: override input shapes
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

## Native Compilation

ModelPort v2.0+ supports compiling models to native shared libraries for maximum performance and minimal dependencies.

### TVM Integration

ModelPort uses Apache TVM (Tensor Virtual Machine) as the backend for compiling models. TVM provides:

1. A high-level IR (Intermediate Representation) for ML models
2. Platform-specific code generation (via LLVM)
3. Various optimization passes for improving performance
4. Runtime libraries for multiple languages

### Compilation Process

When you compile a model with ModelPort, the following steps occur:

1. ONNX model is loaded and converted to TVM's Relay IR
2. Optimizations are applied based on the selected optimization level
3. Target-specific code generation creates native code
4. The resulting shared library and metadata are saved

The compiled output format is:

```
modelport_native/
├── model_x86_64.so         # Native shared library for x86_64
├── model_x86_64.json       # Graph JSON
├── model_x86_64.params     # Serialized parameters
├── compile_config.json     # Compilation metadata
```

### Supported Architectures

- **x86_64**: Intel/AMD 64-bit processors
- **arm64**: Apple M1/M2 (macOS)
- **aarch64**: ARM 64-bit (Linux)

### Supported Devices

- **cpu**: CPU-only compilation (default)
- **cuda**: NVIDIA GPU support via CUDA
- **metal**: Apple GPU support via Metal (macOS)
- **opencl**: OpenCL support for various GPUs

### Command-line Usage

```bash
# Basic compilation
modelport compile model.onnx

# Specify target architecture
modelport compile model.onnx --target-arch arm64

# Specify target device (GPU)
modelport compile model.onnx --target-device cuda

# Set optimization level (0-3)
modelport compile model.onnx --opt-level 3

# Override input shape
modelport compile model.onnx --input-shape "1,3,224,224"

# Test after compilation
modelport compile model.onnx --test
```

### Running Compiled Models

```bash
# Run with default settings
modelport run-native modelport_native

# Run with custom input shape
modelport run-native modelport_native --input-shape "1,3,448,448"

# Benchmark performance
modelport run-native modelport_native --iterations 100
```

### C++ Integration

ModelPort v2.0+ includes a C++ example that demonstrates how to use compiled models without Python dependencies:

```cpp
// Load the compiled model
tvm::runtime::Module mod_lib = tvm::runtime::Module::LoadFromFile("model_x86_64.so");

// Create graph runtime
tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(
    graph_json, mod_lib, 0, 0);

// Get inference functions
tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
tvm::runtime::PackedFunc run = mod.GetFunction("run");
tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");

// Run inference
run();

// Get output
tvm::runtime::NDArray output = get_output(0);
```

To build the C++ example:

```bash
cd modelport/examples
mkdir build && cd build
cmake .. -DTVM_ROOT=/path/to/tvm
make
```

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

5. **TVM Compilation Errors**
   - Make sure TVM is properly installed: `pip install apache-tvm`
   - Try a lower optimization level: `--opt-level 2`
   - Check input shape compatibility

### Getting Help

For issues not covered in this documentation:
- Check the [GitHub issues](https://github.com/SaiKrishna-KK/model-port/issues)
- Submit a new issue with detailed information about the problem 