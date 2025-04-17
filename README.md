# ModelPort 🚀

[![PyPI version](https://badge.fury.io/py/model-port.svg)](https://badge.fury.io/py/model-port)
[![Python Version](https://img.shields.io/pypi/pyversions/model-port)](https://pypi.org/project/model-port/)
[![License](https://img.shields.io/pypi/l/model-port)](https://github.com/SaiKrishna-KK/model-port/blob/main/LICENSE)

**ModelPort** streamlines machine learning model deployment, making it simple, portable, and architecture-independent.

Easily deploy your ML models anywhere — from cloud servers to edge devices — regardless of the underlying hardware or operating system. ModelPort simplifies exporting models to ONNX and provides tools for compiling them into efficient, platform-specific executables (requires separate TVM installation).

## 📣 Key Features (Originating from v2.0)

ModelPort 2.0 features native model compilation! This release introduces Apache TVM integration for compiling models to platform-specific shared libraries that run without dependencies like Python or ONNX Runtime.

### What's New in v2.0:
- 🔥 **Native Compilation** - Compile ONNX models to platform-specific native libraries
- 🚀 **Zero-Dependency Execution** - Run models without Python or ONNX Runtime
- 🖥️ **Cross-Platform Support** - Compile for x86_64, ARM64, and more
- 🎮 **GPU Acceleration** - CUDA, Metal, and OpenCL support for compiled models
- 🧰 **C++ Integration** - Run compiled models from C++ applications
- 📊 **Benchmark Tools** - Performance testing and optimization

## 📦 Installation

```bash
pip install modelport
```

For GPU support:

```bash
pip install modelport[gpu]
```

Development installation:

```bash
git clone https://github.com/SaiKrishna-KK/model-port.git
cd model-port
pip install -e .
```

## 🚀 Quick Start

### Command Line Interface

```bash
# Export a model to ONNX (framework auto-detected)
modelport export path/to/model.pt

# Compile model to native code 
modelport compile path/to/model.onnx

# Run inference on compiled model
modelport run path/to/compiled_model --input data.npy
```

### Python API

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
    input_shape=(1, 10),
    output_path="my_model.onnx"
)

# Compile for your target hardware
mp.compile.compile_model(
    "my_model.onnx",
    target_arch="x86_64",  # Use "aarch64" for ARM (M1/M2 Macs)
    target_device="cpu",   # Use "cuda" for NVIDIA GPUs
    output_dir="compiled_model"
)

# Run inference
import numpy as np
input_data = {"input": np.random.randn(1, 10).astype(np.float32)}
outputs = mp.inference.run("compiled_model", input_data)
print(f"Output shape: {outputs[0].shape}")
```

## 🔧 Supported Architectures

- ✅ **x86_64** (Intel, AMD processors)
- ✅ **arm64** (Apple M1/M2, AWS Graviton)
- ✅ **aarch64** (Jetson, Raspberry Pi, ARM Linux)
- ✅ **NVIDIA GPU** (via CUDA)
- ✅ **Apple GPU** (via Metal)
- ✅ **OpenCL** devices

## ⚠️ Known Issues

- Batch inference on ARM architecture (M1/M2 Macs) may have limitations
- Some TVM optimizations may show warnings on ARM platforms
- TVM compatibility requires specific versions (0.12.0 with ml_dtypes==0.2.0)

## 📚 Documentation

For detailed documentation, see the following resources:

- [Documentation Home](docs/index.md) - Documentation overview
- [Full Documentation](docs/full_documentation.md) - Comprehensive guide with detailed instructions
- [Release Notes](docs/release_notes.md) - Version history and changes

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The PyTorch team for their excellent work on ONNX export
- The ONNX community for creating a powerful standard for model interoperability
- The Apache TVM team for their amazing compiler infrastructure
- All contributors who have helped make this project better

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/SaiKrishna-KK">SaiKrishna-KK</a>
</p>
