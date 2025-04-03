# ModelPort v1.5 Release Notes

We're excited to announce **ModelPort 1.5**, our newest release with significant improvements focused on model portability and deployment flexibility!

## What's New in v1.5:

- ✨ **Framework Auto-Detection** - ModelPort now automatically detects PyTorch, ONNX, and TensorFlow models
- 🧪 **Model Validation with `--test` Flag** - One-step validation that your model works correctly  
- 🐳 **Docker Deployment Command** - Push your models to any Docker registry
- 🚀 **GPU-Enabled Docker Support** - Built-in CUDA support for accelerated inference
- 📋 **Standardized Capsule Format V1** - Well-defined metadata structure for better interoperability

## Comprehensive Testing

This release has undergone extensive testing including:
- ✅ Comprehensive unit tests covering all functionality
- ✅ Edge case handling for tiny models, complex models, and mixed data types
- ✅ Stress testing with multiple model outputs and custom dtypes
- ✅ End-to-end validation with ResNet18 and other common architectures

## Full Changelog

- Added framework auto-detection for PyTorch, ONNX, and TensorFlow models
- Added model validation with `--test` flag
- Implemented Docker deployment command
- Added GPU-enabled Docker support
- Created standardized Capsule Format V1 spec
- Fixed PyTorch dtype handling issues
- Comprehensive stress testing and unit tests

## Breaking Changes

None! This release maintains backward compatibility with previous versions.

## Getting Started

Check out the README.md for installation and usage instructions. 