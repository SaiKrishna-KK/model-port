# ModelPort 🚀

**ModelPort** makes machine learning model deployment simple, portable, and architecture-independent.

![ModelPort Banner](https://img.shields.io/badge/ModelPort-v1.5-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

**Deploy your ML models anywhere** — regardless of architecture or operating system. ModelPort simplifies the process of exporting models to ONNX format and packaging them for deployment on different platforms.

## 📣 Version 1.5 Release 
> *June 2023 - New release with enhanced capabilities and extensive testing*

We're excited to announce **ModelPort 1.5**, our newest release with significant improvements focused on model portability and deployment flexibility! This release adds framework auto-detection, model validation, Docker deployment capabilities, and GPU acceleration support.

### What's New in v1.5:
- ✨ **Framework Auto-Detection** - ModelPort now automatically detects PyTorch, ONNX, and TensorFlow models
- 🧪 **Model Validation with `--test` Flag** - One-step validation that your model works correctly  
- 🐳 **Docker Deployment Command** - Push your models to any Docker registry
- 🚀 **GPU-Enabled Docker Support** - Built-in CUDA support for accelerated inference
- 📋 **Standardized Capsule Format V1** - Well-defined metadata structure for better interoperability

### Comprehensive Testing
This release has undergone extensive testing including:
- ✅ Comprehensive unit tests covering all functionality
- ✅ Edge case handling for tiny models, complex models, and mixed data types
- ✅ Stress testing with multiple model outputs and custom dtypes
- ✅ End-to-end validation with ResNet18 and other common architectures

## 🌟 Features

- ✅ **Framework Auto-Detection** - Automatically detect PyTorch, ONNX, or TensorFlow models
- ✅ **Export to ONNX Format** - Convert models to ONNX for maximum interoperability
- ✅ **Model Validation** - Test exported models with dummy inputs
- ✅ **Cross-Platform Containers** - Deploy on x86_64, ARM64, or with GPU acceleration
- ✅ **Docker Registry Integration** - Push capsules to Docker Hub or custom registries
- ✅ **Simple CLI Interface** - Quick exports and deployments with smart defaults
- ✅ **Programmatic API** - Integration into your ML workflows

## 📚 Documentation

For comprehensive documentation on ModelPort, please refer to the [Documentation Guide](DOCUMENTATION.md). It includes:

- Detailed architecture overview
- Step-by-step installation instructions
- Complete API reference
- Examples and advanced use cases
- Troubleshooting tips

## 📦 Quick Installation

```bash
# Install from PyPI (coming soon)
pip install modelport

# Or install directly from GitHub
pip install git+https://github.com/SaiKrishna-KK/model-port.git
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
```

## 📄 Capsule Format Overview

ModelPort packages models in a standardized format with everything needed to run them:

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

For more details about the capsule format, see the [Documentation](DOCUMENTATION.md#capsule-format).

## 🔧 Supported Architectures

- ✅ **x86_64** (Intel, AMD processors)
- ✅ **arm64** (Apple M1/M2, AWS Graviton, Jetson, Raspberry Pi)
- ✅ **NVIDIA GPU** (via CUDA)

## 🔮 Future Roadmap (v2.0 and beyond)

ModelPort is continuously evolving. Here's what we're planning for future releases:

### 🔜 Coming in v2.0: Advanced Compilation & Deployment
- **TensorFlow Direct Support**: Export TensorFlow models without ONNX conversion
- **Hugging Face Integration**: Seamless export of HF Transformers models
- **Native Compilation**: Compile models to native code for maximum performance

For the complete roadmap, see the [Documentation](DOCUMENTATION.md#future-roadmap).

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The PyTorch team for their excellent work on ONNX export
- The ONNX community for creating a powerful standard for model interoperability
- All contributors who have helped make this project better

## 📝 Changelog

### v1.5 (March 2025)
- Added framework auto-detection for PyTorch, ONNX, and TensorFlow models
- Added model validation with `--test` flag
- Implemented Docker deployment command
- Added GPU-enabled Docker support
- Created standardized Capsule Format V1 spec
- Fixed PyTorch dtype handling issues
- Comprehensive stress testing and unit tests

### v0.1.0 (March 2025)
- Initial release
- Basic PyTorch to ONNX export
- Simple Docker container generation

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/SaiKrishna-KK">SaiKrishna-KK</a>
</p>
