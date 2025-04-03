# ModelPort Documentation

This comprehensive documentation covers all aspects of ModelPort, from installation to advanced usage and troubleshooting.

## Table of Contents

- [Installation](#installation)
- [Quick Start Guide](#quick-start-guide)
- [Command Line Interface](#command-line-interface)
- [Python API](#python-api)
- [Architecture Overview](#architecture-overview)
- [Docker Testing](#docker-testing)
- [Troubleshooting](#troubleshooting)
- [PyPI Publishing Guide](#pypi-publishing-guide)

## Installation

### Basic Installation

```bash
pip install modelport
```

### GPU Support

```bash
pip install modelport[gpu]
```

### Development Installation

```bash
git clone https://github.com/SaiKrishna-KK/model-port.git
cd model-port
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 1.8+
- ONNX 1.10+
- TVM 0.12.0 (installed automatically)
- ml_dtypes 0.2.0 (installed automatically)

## Quick Start Guide

### Export a PyTorch Model

```python
import torch
import modelport as mp

# Create or load your PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 2)
)
model.eval()

# Export to ONNX
mp.export.to_onnx(
    model, 
    input_shape=(1, 10),  # Batch size of 1, input size of 10
    output_path="my_model.onnx"
)
```

### Compile the Model

```python
# Compile for your target hardware
mp.compile.compile_model(
    "my_model.onnx",
    target_arch="x86_64",  # Use "aarch64" for ARM (M1/M2 Macs)
    target_device="cpu",   # Use "cuda" for NVIDIA GPUs
    opt_level=3,           # Optimization level (0-3)
    output_dir="compiled_model"
)
```

### Run Inference

```python
import numpy as np

# Prepare input data
input_data = {
    "input": np.random.randn(1, 10).astype(np.float32)
}

# Run inference
outputs = mp.inference.run("compiled_model", input_data=input_data)
print(f"Output shape: {outputs[0].shape}")
print(f"Output values: {outputs[0]}")
```

## Command Line Interface

ModelPort provides a simple command-line interface for common operations:

```bash
# Export a model to ONNX format
modelport export path/to/model.pt --output model.onnx --input-shape 1,3,224,224

# Compile a model for a specific target
modelport compile path/to/model.onnx --target-arch x86_64 --target-device cpu

# Run inference on a compiled model
modelport run path/to/compiled_model --input input.npy --output output.npy

# Print diagnostic information
modelport diagnostics
```

## Python API

### Export Module

```python
from modelport.export import to_onnx

to_onnx(
    model,                     # PyTorch model
    input_shape=(1, 3, 224, 224),  # Input shape
    output_path="model.onnx",  # Output path
    input_names=["input"],     # Input tensor names
    output_names=["output"],   # Output tensor names
    dynamic_axes=None,         # Dynamic axes for shapes
    opset_version=13           # ONNX opset version
)
```

### Compile Module

```python
from modelport.compile import compile_model

compile_model(
    model_path="model.onnx",     # Path to ONNX model
    target_arch="x86_64",        # Target architecture
    target_device="cpu",         # Target device
    opt_level=3,                 # Optimization level
    output_dir="compiled_model"  # Output directory
)
```

### Inference Module

```python
from modelport.inference import run

run(
    model_dir="compiled_model",  # Directory with compiled model
    input_data={"input": data},  # Input data dictionary
    device="cpu"                 # Device to run on
)
```

### Utilities Module

```python
from modelport.utils import check_environment

# Check the environment for required dependencies
env_info = check_environment()
print(env_info)
```

## Architecture Overview

ModelPort consists of four main components:

1. **Export**: Converts models from frameworks like PyTorch to ONNX format
2. **Compile**: Compiles ONNX models to target-specific libraries using TVM
3. **Inference**: Runs compiled models efficiently on various devices
4. **Utilities**: Provides diagnostic tools and helpers

## Docker Testing

### Quick Start

```bash
# Clone repository
git clone https://github.com/SaiKrishna-KK/model-port.git
cd model-port

# Build Docker image
docker build -t modelport-test -f Dockerfile.final .

# Run basic tests
docker run --rm modelport-test

# Run all tests (including advanced features)
docker run --rm modelport-test python3 -m tests.run_tests --all
```

### Testing Options

| Option | Command | Description |
|--------|---------|-------------|
| Basic tests | `docker run --rm modelport-test` | Run essential functionality tests |
| All tests | `docker run --rm modelport-test python3 -m tests.run_tests --all` | Run comprehensive test suite |
| Specific test | `docker run --rm modelport-test python3 -m tests.test_tvm_basic` | Run only the TVM test |
| Custom test | `docker run --rm -v $(pwd)/your_test.py:/app/custom_test.py modelport-test python3 /app/custom_test.py` | Run your own test script |

### Platform-Specific Notes

#### ARM Architecture (M1/M2 Macs)
- The Docker image automatically handles ARM-specific adjustments
- Batch inference is limited to smaller batch sizes
- Warnings about LLVM optimization can be safely ignored

#### NVIDIA GPU Support
- To enable CUDA support, use this alternative command:
```bash
docker run --gpus all --rm modelport-test python3 -m tests.run_tests --gpu
```

## Troubleshooting

### Installation Issues

#### TVM Installation Fails

**Problem:** Error installing TVM or importing TVM after installation.

**Solution:**
- Use our pre-built Docker environment which includes a patched version of TVM
- Install a specific version: `pip install apache-tvm==0.12.0 ml_dtypes==0.2.0`
- On M1/M2 Macs, you may need to apply the TVM patch:

```python
# Create a file named patch_tvm.py
import sys
runtime_ctypes_path = "<path_to_your_site_packages>/tvm/_ffi/runtime_ctypes.py"
with open(runtime_ctypes_path, "r") as f:
    content = f.read()

if "float4_e2m1fn" in content:
    modified = content.replace(
        "DataType.NUMPY2STR[np.dtype(ml_dtypes.float4_e2m1fn)] = \"float4_e2m1fn\"", 
        "# Patched: float4_e2m1fn not available"
    )
    with open(runtime_ctypes_path, "w") as f:
        f.write(modified)
    print("Successfully patched TVM runtime_ctypes.py")
```

#### Missing Dependencies

**Problem:** ImportError or ModuleNotFoundError when using ModelPort.

**Solution:**
- Check requirements: `pip install -r requirements.txt`
- Ensure you have the compatible versions:
  ```
  torch>=1.8.0
  onnx>=1.10.0
  onnxruntime>=1.8.0
  apache-tvm==0.12.0
  ml_dtypes==0.2.0
  ```

### Export Errors

#### Model Export Fails

**Problem:** Error when exporting a PyTorch model to ONNX.

**Solution:**
- Ensure your model is in `eval()` mode
- Check input shapes match expected model inputs
- For dynamic shapes, specify dynamic axes:
  ```python
  mp.export.to_onnx(
      model, 
      input_shape=(1, 10), 
      output_path="model.onnx",
      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
  )
  ```

### Compilation Issues

#### Compilation Fails for Target Architecture

**Problem:** Error when compiling for a specific architecture.

**Solution:**
- Check supported architectures: 
  - "x86_64" for Intel/AMD processors
  - "aarch64" for ARM processors (M1/M2 Macs)
- For ARM processors, use TVM 0.12.0 or our Docker environment
- Lower the optimization level: `opt_level=2`

#### CUDA Compilation Issues

**Problem:** Error when compiling for CUDA targets.

**Solution:**
- Verify CUDA is installed and detected
- Check TVM was built with CUDA support
- Use our GPU-enabled Docker image:
  ```bash
  docker run --gpus all --rm modelport-test-gpu
  ```

### Inference Problems

#### Incorrect Inference Results

**Problem:** Model outputs unexpected values or incorrect shapes.

**Solution:**
- Verify input data has correct shape and type (typically `np.float32`)
- Check input tensor names match the exported model
- Use the validation utility:
  ```python
  mp.utils.check_environment()
  ```

#### Batch Inference Fails

**Problem:** Batch inference works with batch_size=1 but fails with larger batches.

**Solution:**
- This is a known issue on ARM architectures
- Try reducing batch size to 1
- Use the non-batched inference API and process inputs sequentially
- If you must use batch processing, use our x86_64 Docker image

## PyPI Publishing Guide

### Prerequisites

- Python 3.8 or later
- pip 21.3 or later (supports pyproject.toml)
- Build dependencies: `pip install build twine`
- Test PyPI account: https://test.pypi.org/account/register/
- PyPI account: https://pypi.org/account/register/

### Building and Publishing

1. Build the package:
   ```bash
   python -m build
   ```

2. Upload to TestPyPI first:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

3. Test the installation:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ modelport
   ```

4. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

### Automated Publishing with GitHub Actions

ModelPort uses GitHub Actions for automated publishing to PyPI:

1. Set up GitHub Trusted Publishers:
   - Go to https://test.pypi.org/manage/account/publishing/
   - Add a publisher with:
     - Project: model-port
     - Owner: SaiKrishna-KK
     - Repository: model-port
     - Workflow: publish.yml
     - Environment: testpypi or pypi

2. Create a GitHub release to trigger publishing:
   - For TestPyPI: Mark as pre-release
   - For PyPI: Create a regular release 