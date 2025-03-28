# ModelPort 🚀

**ModelPort** makes machine learning model deployment simple, portable, and architecture-independent.

![ModelPort Banner](https://img.shields.io/badge/ModelPort-v2.0-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

**Deploy your ML models anywhere** — regardless of architecture or operating system. ModelPort simplifies the process of exporting models to ONNX format and packaging them for deployment on different platforms.

## 📣 Version 2.0 Release 

ModelPort 2.0 features native model compilation! This release introduces Apache TVM integration for compiling models to platform-specific shared libraries that run without dependencies like Python or ONNX Runtime.

### What's New in v2.0:
- 🔥 **Native Compilation** - Compile ONNX models to platform-specific native libraries
- 🚀 **Zero-Dependency Execution** - Run models without Python or ONNX Runtime
- 🖥️ **Cross-Platform Support** - Compile for x86_64, ARM64, and more
- 🎮 **GPU Acceleration** - CUDA, Metal, and OpenCL support for compiled models
- 🧰 **C++ Integration** - Run compiled models from C++ applications
- 📊 **Benchmark Tools** - Performance testing and optimization

## 🌟 Features

- ✅ **Framework Auto-Detection** - Automatically detect PyTorch, ONNX, or TensorFlow models
- ✅ **Export to ONNX Format** - Convert models to ONNX for maximum interoperability
- ✅ **Model Validation** - Test exported models with dummy inputs
- ✅ **Cross-Platform Containers** - Deploy on x86_64, ARM64, or with GPU acceleration
- ✅ **Docker Registry Integration** - Push capsules to Docker Hub or custom registries
- ✅ **Native Compilation** - Compile models to native code for maximum performance
- ✅ **Simple CLI Interface** - Quick exports, deployments, and compilations with smart defaults
- ✅ **Programmatic API** - Integration into your ML workflows

## 📚 Documentation

For comprehensive documentation on ModelPort, please refer to the [Documentation Guide](DOCUMENTATION.md). It includes:

- Detailed architecture overview
- Step-by-step installation instructions
- Complete API reference
- Examples and advanced use cases
- Native compilation guide
- Troubleshooting tips

## 📦 Quick Installation

```bash
# Install from PyPI (coming soon)
pip install modelport

# Or install directly from GitHub
pip install git+https://github.com/SaiKrishna-KK/model-port.git

# For native compilation features, install TVM
pip install apache-tvm
```

## 🚀 Quick Start

### Command Line Interface

```bash
# Export a model to ONNX (framework auto-detected)
modelport export path/to/model.pt

# Export with validation
modelport export path/to/model.pt --test

# Deploy to Docker Hub
modelport deploy my_exported_model --tag username/model:latest --push

# Build GPU-accelerated container
modelport deploy my_exported_model --gpu

# Compile model to native code (v2.0+)
modelport compile path/to/model.onnx

# Run compiled model (v2.0+)
modelport run-native modelport_native

# Benchmark compiled model (v2.0+)
modelport run-native modelport_native --benchmark --iterations 100
```

### Python API

```python
import torch
import modelport

# Load your model
model = torch.load("path/to/model.pt") 

# Export to ONNX and package it
export_dir = modelport.export_model(
    "path/to/model.pt", 
    "my_exported_model",
    test=True  # Validate the model
)

# Deploy to Docker Hub
modelport.deploy_capsule(
    "my_exported_model",
    tag="username/model:latest",
    push=True
)

# Compile model to native code (v2.0+)
config = modelport.compile_model(
    "path/to/model.onnx",
    output_dir="native_model",
    target_arch="x86_64",
    target_device="cpu",
    opt_level=3
)
```

## 📄 Output Format Overview

### Export format

ModelPort packages models in a standardized format with everything needed to run them:

```
modelport_export/
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

### Native compilation format (v2.0+)

Compiled models are packaged in a lightweight format:

```
modelport_native/
├── model_x86_64.so         # Native shared library for x86_64
├── model_x86_64.json       # Graph JSON
├── model_x86_64.params     # Serialized parameters
└── compile_config.json     # Compilation metadata
```

For more details about the formats, see the [Documentation](DOCUMENTATION.md).

## 🔧 Supported Architectures

- ✅ **x86_64** (Intel, AMD processors)
- ✅ **arm64** (Apple M1/M2, AWS Graviton)
- ✅ **aarch64** (Jetson, Raspberry Pi, ARM Linux)
- ✅ **NVIDIA GPU** (via CUDA)
- ✅ **Apple GPU** (via Metal)
- ✅ **OpenCL** devices

## 🔮 Future Roadmap (v2.5 and beyond)

ModelPort is continuously evolving. Here's what we're planning for future releases:

### 🔜 Coming in v2.5: Advanced Native Features
- **WASM Compilation**: Compile models to WebAssembly for browser deployment
- **Quantization**: INT8 and Mixed-Precision Support for compiled models
- **More Hardware Targets**: Specialized acceleration for additional devices

### ⚠️ What Might Break or Need Tuning (to be fixed in v2.1.0)
| Case | Why |
|------|-----|
| ❌ Custom Layers (Python-only logic) | Need to be rewritten or converted to supported ops |
| ❌ Rare ONNX ops (experimental) | TVM may not support obscure or very new ONNX ops |
| ❌ Very dynamic input shapes | TVM prefers fixed shapes or limited shape ranges |
| ❌ Training or backprop logic | ModelPort is for inference only, not training |

For the complete roadmap, see the [Documentation](DOCUMENTATION.md#future-roadmap).

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The PyTorch team for their excellent work on ONNX export
- The ONNX community for creating a powerful standard for model interoperability
- The Apache TVM team for their amazing compiler infrastructure
- All contributors who have helped make this project better

## 📝 Changelog

### v2.0
- Added native compilation using Apache TVM
- Added run-native command for compiled models
- Added benchmark capabilities for performance testing
- C++ integration for compiled models
- Support for multiple target architectures and devices
- Comprehensive documentation for native compilation

### v1.5
- Added deploy command for Docker registry integration
- Added GPU support for Docker containers
- Improved framework auto-detection
- Added test flag for model validation
- Standardized capsule format

### v0.1.0
- Initial release
- Basic ONNX export functionality
- Docker container generation
- Cross-platform support

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/SaiKrishna-KK">SaiKrishna-KK</a>
</p>
