# ModelPort 🚀

[![PyPI version](https://badge.fury.io/py/model-port.svg)](https://badge.fury.io/py/model-port)
[![Python Version](https://img.shields.io/pypi/pyversions/model-port)](https://pypi.org/project/model-port/)
[![License](https://img.shields.io/pypi/l/model-port)](https://github.com/SaiKrishna-KK/model-port/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/SaiKrishna-KK/model-port)](https://github.com/SaiKrishna-KK/model-port/issues)
[![GitHub forks](https://img.shields.io/github/forks/SaiKrishna-KK/model-port)](https://github.com/SaiKrishna-KK/model-port/network)
[![GitHub stars](https://img.shields.io/github/stars/SaiKrishna-KK/model-port)](https://github.com/SaiKrishna-KK/model-port/stargazers)

**ModelPort addresses the challenges of deploying machine learning models across diverse hardware and software environments.** It provides a streamlined workflow to convert models into standard formats (like ONNX) and optionally compile them into high-performance, dependency-free executables suitable for servers, edge devices, and various operating systems.

**Goal:** Simplify ML model deployment, making it portable and architecture-independent.

## Key Features

*   🔥 **Framework Agnostic Export:** Automatically detect and export models from popular frameworks (e.g., PyTorch) to ONNX format.
*   🚀 **Native Compilation (Optional, via TVM):** Compile ONNX models into optimized, platform-specific shared libraries using Apache TVM integration. This enables:
    *   **Zero-Dependency Execution:** Run compiled models without requiring Python or the original ML framework installed.
    *   **Cross-Platform Deployment:** Target diverse architectures like x86_64, ARM64 (including Apple Silicon), AArch64, and more.
    *   **Hardware Acceleration:** Leverage GPU acceleration via CUDA, Metal, or OpenCL for compiled models.
*   ⚙️ **Simplified Inference:** Run inference using either the standard ONNX Runtime or the compiled native libraries through a consistent Python API or command-line interface.
*    C++ Integration: Load and execute compiled model artifacts directly from C++ applications.
*   📊 **Benchmarking Tools:** Facilitates performance analysis of exported and compiled models.

## 📦 Installation

**Prerequisites:**
*   Python >= 3.8

**Standard Installation:**

```bash
pip install model-port
```

**Optional Dependencies:**

*   **GPU Support (ONNX Runtime):** If using ONNX Runtime for inference and require GPU acceleration:
    ```bash
    pip install model-port[gpu]
    ```
*   **Development:** To install for development, including testing tools:
    ```bash
    git clone https://github.com/SaiKrishna-KK/model-port.git
    cd model-port
    pip install -e ".[dev]"
    ```

### Installing TVM Support (Required for Native Compilation)

The native compilation features of `model-port` (`modelport compile`, `mp.compile`, etc.) rely on Apache TVM. Due to the complexities of TVM installation and the lack of universal pre-built packages on PyPI, **TVM is NOT installed automatically.**

**To enable native compilation:**

1.  Install `model-port` first (as shown above).
2.  Install Apache TVM separately into the same Python environment by following the **official TVM installation instructions**:
    *   **Official Guide:** [https://tvm.apache.org/docs/install/](https://tvm.apache.org/docs/install/)
    *   **Build from Source (Recommended for specific versions/maximum compatibility):** [https://tvm.apache.org/docs/install/from_source.html](https://tvm.apache.org/docs/install/from_source.html)
    *   **Potentially Available Pre-built Wheels (Check TVM Docs First):**
        *   Linux/macOS: `pip install apache-tvm` (May not have desired version/architecture)
        *   Windows: `pip install tlcpack -f https://tlcpack.ai/wheels` (Community-provided)

**Important:** Always refer to the official TVM documentation for the most accurate and up-to-date installation method for your specific OS, Python version, and hardware requirements.

If you encounter installation issues, please refer to the [Troubleshooting](#-troubleshooting) section below.

## 🚀 Quick Start

Get started quickly with ModelPort using either the CLI or the Python API.

### Command Line Interface (CLI)

```bash
# Export a model (e.g., PyTorch) to ONNX - framework is auto-detected
modelport export /path/to/your/model.pt --output-path my_model.onnx

# (Optional) Compile the ONNX model to a native shared library
# Assumes TVM is installed separately - see Installation section
modelport compile my_model.onnx --target-arch x86_64 --output-dir compiled_model

# Run inference using the compiled model artifact
# Assumes input data is a .npy file matching model input name 'input'
modelport run compiled_model --input /path/to/input_data.npy

# Run inference using the original ONNX file (via ONNX Runtime)
modelport run my_model.onnx --input /path/to/input_data.npy
```

### Python API

```python
import torch
import numpy as np
import modelport as mp

# --- 1. Prepare or Load Your Model ---
# Example: Simple PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 2)
)
model.eval()
dummy_input = torch.randn(1, 10) # Example input for tracing/shape inference

# --- 2. Export to ONNX ---
onnx_path = "my_model.onnx"
mp.export.to_onnx(
    model=model,
    input_data=dummy_input, # Provide sample input data
    output_path=onnx_path
)
print(f"Model exported to: {onnx_path}")

# --- 3. (Optional) Compile to Native Artifact ---
# Ensure TVM is installed separately first!
compiled_model_dir = "compiled_model"
try:
    mp.compile.compile_model(
        model_path=onnx_path,
        target_arch="x86_64",  # e.g., "x86_64", "aarch64", "arm64" (for Apple Silicon)
        target_device="cpu",   # e.g., "cpu", "cuda", "metal"
        output_dir=compiled_model_dir
    )
    print(f"Model compiled to: {compiled_model_dir}")
    use_compiled = True
except ImportError:
    print("TVM not found. Skipping native compilation.")
    print("Install TVM separately to enable this feature (see README).")
    use_compiled = False
except Exception as e:
    print(f"TVM compilation failed: {e}")
    use_compiled = False


# --- 4. Run Inference ---
input_name = "input0" # Default or check your model's input name
input_data_np = dummy_input.numpy().astype(np.float32)
inference_input = {input_name: input_data_np}

model_to_run = compiled_model_dir if use_compiled else onnx_path

outputs = mp.inference.run(model_to_run, inference_input)

# Process outputs (structure depends on your model)
output_data = outputs[0] # Assuming first output tensor
print(f"Inference executed using: {model_to_run}")
print(f"Output shape: {output_data.shape}")
print(f"Output sample: {output_data[0, :5]}") # Print first 5 elements
```

## 🔧 Supported Target Platforms (for Native Compilation via TVM)

ModelPort, via TVM, aims to compile models for a wide range of targets:

*   ✅ **CPUs:** x86_64 (Intel/AMD), arm64 (Apple Silicon), aarch64 (Generic ARM Linux)
*   ✅ **GPUs:** NVIDIA (CUDA), AMD (ROCm - Linux only), Apple (Metal), Vulkan, OpenCL

*Note: Successful compilation depends on having the correct TVM version and build configuration installed, along with necessary drivers/SDKs (like CUDA toolkit for NVIDIA GPUs).*

## 📚 Documentation & Resources

*   **Release Notes:** [docs/release_notes.md](docs/release_notes.md) - Track changes and updates.
*   **Index:** [docs/index.md](docs/index.md) - Overview of documentation files.
*   **Detailed Guide:** [docs/full_documentation.md](docs/full_documentation.md) - In-depth usage instructions.
*   **Contribution Guide:** See [CONTRIBUTING.md](CONTRIBUTING.md) (if available) or refer to the section below.

## 🛠️ Troubleshooting

If you encounter problems during installation or usage:

1.  **Check Prerequisites:** Ensure you have Python >= 3.8 installed.
2.  **TVM Issues:** If facing issues with native compilation (`modelport compile`):
    *   Verify that you have installed Apache TVM separately **after** installing `model-port`.
    *   Confirm your installed TVM version meets any requirements mentioned for specific features.
    *   Consult the official Apache TVM documentation and installation guides for your platform.
3.  **Search Issues:** Check the [GitHub Issues](https://github.com/SaiKrishna-KK/model-port/issues) page for similar problems.
4.  **Report New Issues:** If your issue is new, please [open a new issue](https://github.com/SaiKrishna-KK/model-port/issues/new/choose). Include:
    *   Your operating system and version.
    *   Your Python version.
    *   `model-port` version (`pip show model-port`).
    *   TVM version (if relevant: `python -c "import tvm; print(tvm.__version__)"`).
    *   The exact command or code snippet causing the issue.
    *   The full error message and traceback.

## 🤝 Contributing

Contributions are welcome! Whether it's reporting bugs, suggesting features, improving documentation, or submitting code changes, your help is appreciated.

Please check the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines (if it exists). If not, feel free to open an issue to discuss your idea or submit a pull request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

*   The PyTorch, ONNX, and Apache TVM communities for their foundational work.
*   All contributors and users of ModelPort.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/SaiKrishna-KK">SaiKrishna-KK</a>
</p>
