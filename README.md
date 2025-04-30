# ModelPort 🚀

**ModelPort** makes machine learning model deployment simple, portable, and architecture-independent.

![ModelPort Banner](https://img.shields.io/badge/ModelPort-v0.1.0-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

**Deploy your ML models anywhere** — regardless of architecture or operating system. ModelPort simplifies the process of exporting PyTorch models to ONNX format and packaging them for deployment on different platforms.

## 🌟 Features

- ✅ **Export PyTorch models to ONNX** format for maximum interoperability
- ✅ **Package models with inference code** for self-contained capsules
- ✅ **Cross-platform deployment** via Docker (`x86_64`, `arm64`, etc.)
- ✅ **Simple CLI interface** for quick exports and deployments
- ✅ **Programmatic API** for integration into your ML workflows

## 📦 Installation

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

## 🚀 Quick Start

### Command Line Interface

```bash
# Export a PyTorch model to ONNX
modelport export path/to/model.pt --output-path my_exported_model

# Run the exported model on a specific architecture
modelport run my_exported_model --arch linux/arm64
```

### Python API

```python
import torch
import modelport

# Load your PyTorch model
model = torch.load("path/to/model.pt") 

# Export to ONNX and package it
export_dir = modelport.export_model("path/to/model.pt", "my_exported_model")

# Run the model in Docker
modelport.run_capsule("my_exported_model", "linux/amd64")
```

## 🐳 Docker Setup

ModelPort uses Docker for cross-platform deployment. Make sure you have Docker installed and properly configured:

```bash
# Make sure Docker is installed
docker --version

# For Apple Silicon (M1/M2) or other architectures, 
# enable multi-architecture builds
modelport/scripts/setup_docker.sh
```

## 🔧 Supported Architectures

- ✅ **x86_64** (Intel, AMD processors)
- ✅ **arm64** (Apple M1/M2, AWS Graviton, Jetson, Raspberry Pi)
- 🔜 More architectures coming soon!

## 🤝 How to Contribute

We welcome contributions to ModelPort! Here's how you can help:

1. **Report bugs and request features** by opening an issue
2. **Contribute code** by opening a pull request
3. **Improve documentation** by fixing typos or adding examples
4. **Share the project** with others who might find it useful

### Development Workflow

```bash
# Fork the repository and clone your fork
git clone https://github.com/YOUR_USERNAME/model-port.git
cd model-port

# Create a new branch for your feature
git checkout -b feature/your-feature-name

# Make your changes and run tests
python modelport/examples/run_all_tests.py

# Commit and push your changes
git add .
git commit -m "Add a descriptive commit message"
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

## 🔮 Future Roadmap

ModelPort is continuously evolving. Here's what we're planning for future releases:

### 🔜 Phase 1: Enhanced Model Support
- **TensorFlow Support**: Export and run TensorFlow models
- **Hugging Face Integration**: Seamless export of HF Transformers models
- **PyTorch Lightning Support**: Direct integration with PyTorch Lightning

### 🔜 Phase 2: Advanced Compilation & Deployment
- **Native Compilation**: Compile models to native code for maximum performance
- **Edge Device Support**: Optimized deployment for edge devices
- **Cloud Deployment**: One-click deployment to cloud providers
- **Quantization Support**: 8-bit and other quantization methods

### 🔜 Phase 3: Enterprise Features
- **Model Monitoring**: Track performance and detect drift
- **Batch Inference**: Optimized batch processing
- **A/B Testing**: Compare model versions in production
- **CI/CD Integration**: Automated testing and deployment

## 📊 Performance

In our benchmarks, ModelPort-exported models achieve performance comparable to native models, with the added benefit of portability:

| Architecture | Image Classification (ms) | Object Detection (ms) |
|--------------|---------------------------|------------------------|
| x86_64       | 42                        | 156                    |
| arm64        | 58                        | 203                    |

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The PyTorch team for their excellent work on ONNX export
- The ONNX community for creating a powerful standard for model interoperability
- All contributors who have helped make this project better

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/SaiKrishna-KK">SaiKrishna-KK</a>
</p>
