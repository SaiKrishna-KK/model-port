# ModelPort Release Notes

## Version 2.0.6 (YYYY-MM-DD)

*   **Docs:** Overhauled `README.md` for improved clarity, structure, and user guidance. Added sections for troubleshooting and contributing.

## Version 2.0.5 

*   **Refactor:** Made `apache-tvm` an optional dependency.
*   **Info:** Users needing TVM features must now install `apache-tvm` separately, preferably by following the official TVM installation guide (often requires building from source) due to the lack of universal pre-built wheels on PyPI.

## Version 2.0.4 

*   **Fix (Attempted):** Changed `apache-tvm` dependency from `0.12.0` to `0.11.1` due to unavailability of `0.12.0` on PyPI for some architectures. (Note: `0.11.1` was also found to be unavailable for many platforms via pip.) 

## Version 2.0.3

This release introduces a package name change from "modelport" to "model-port" to improve PyPI compatibility.

### What's New in v2.0.3:
- Changed package name to "model-port" to avoid conflicts on PyPI/TestPyPI
- Retained all functionality from version 2.0.2
- Improved installation reliability

## Version 2.0.2

This maintenance release addresses packaging and PyPI publishing configuration issues.

### What's New in v2.0.2:
- Fixed license format in pyproject.toml to comply with PEP 621 specifications
- Improved package metadata for PyPI compatibility
- Resolved build issues affecting PyPI deployment
- Updated GitHub Actions workflows for more reliable publishing

All features and functionality from version 2.0.0 remain unchanged.

## Version 2.0.0

ModelPort 2.0 features native model compilation! This release introduces Apache TVM integration for compiling models to platform-specific shared libraries that run without dependencies like Python or ONNX Runtime.

### What's New in v2.0:
- **Native Compilation** - Compile ONNX models to platform-specific native libraries
- **Zero-Dependency Execution** - Run models without Python or ONNX Runtime
- **Cross-Platform Support** - Compile for x86_64, ARM64, and more
- **GPU Acceleration** - CUDA, Metal, and OpenCL support for compiled models
- **C++ Integration** - Run compiled models from C++ applications
- **Benchmark Tools** - Performance testing and optimization
- **TVM Integration** - Apache TVM support for model compilation
- **Improved Testing** - Docker-based testing infrastructure
- **Comprehensive Documentation** - Detailed guides and examples

### Known Issues
- Batch inference on ARM architecture (M1/M2 Macs) may have limitations
- Some TVM optimizations may show warnings on ARM platforms
- TVM compatibility requires specific versions (0.12.0 with ml_dtypes==0.2.0)

## Version 1.5.0

- Added deploy command for Docker registry integration
- Added GPU support for Docker containers
- Improved framework auto-detection
- Added test flag for model validation
- Standardized capsule format

## Version 0.1.0

- Initial release
- Basic ONNX export functionality
- Docker container generation
- Cross-platform support

