# ModelPort Release Notes

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