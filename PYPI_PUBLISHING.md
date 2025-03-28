# Publishing ModelPort to PyPI

This document explains how to publish ModelPort to PyPI so users can install it using `pip install modelport`.

## About ModelPort

ModelPort is a versatile tool that allows you to export and run machine learning models anywhere. It supports:

- **Framework auto-detection** for PyTorch, ONNX, and other formats
- **ONNX export** for cross-platform compatibility 
- **Docker container generation** for deployment
- **Native compilation** via TVM (when available)
- **Cross-platform support** for x86_64, ARM64, etc.

## Prerequisites

Make sure you have the necessary tools installed:

```bash
pip install build twine
```

## Building the Package

To build the package, run:

```bash
python -m build
```

This will generate both source distribution (`.tar.gz`) and wheel (`.whl`) files in the `dist/` directory.

## Testing the Package

Before publishing to PyPI, you can test the package using TestPyPI:

1. Register an account on [TestPyPI](https://test.pypi.org/account/register/)
2. Upload your package to TestPyPI:

```bash
twine upload --repository testpypi dist/*
```

3. Install from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ modelport
```

4. Test the installation:

```bash
# Test basic functionality
python -c "import modelport; print(modelport.__version__)"

# Test CLI if available
modelport --help
```

## Publishing to PyPI

Once you're satisfied with the package, you can publish it to PyPI:

1. Register an account on [PyPI](https://pypi.org/account/register/)
2. Upload your package:

```bash
twine upload dist/*
```

## User Installation Guide

After publishing, users can install ModelPort with:

```bash
# Basic installation
pip install modelport

# Installation with optional TVM support
pip install modelport apache-tvm
```

## Usage Examples

Include this section in the README to help users:

```python
# Basic export usage
import modelport

# Export a PyTorch model to ONNX
modelport.export_model("model.pt", "output_dir", test=True)

# Deploy as a Docker container
modelport.deploy_capsule("output_dir", tag="username/model:latest", push=True)
```

CLI usage:

```bash
# Export a model (framework auto-detected)
modelport export model.pt

# Run the model in a Docker container
modelport run output_dir
```

## Versioning

When releasing a new version:

1. Update the version number in both `pyproject.toml` and `setup.py`
2. Update the changelog in `RELEASE_NOTES.md`
3. Tag the release in git:

```bash
git tag -a v2.0.0 -m "Release v2.0.0"
git push origin v2.0.0
```

## Troubleshooting

### TVM Not Available Warning

If users see "TVM not found. Native compilation will not be available", they can:

1. Install TVM: `pip install apache-tvm`
2. Or ignore it if they don't need native compilation features

### Docker Issues

If users encounter Docker-related errors:

1. Make sure Docker is installed and running
2. Check permissions (may need to add user to docker group)
3. Verify Docker Buildx is available for cross-platform builds

### Import Errors

If users get import errors after installation:

1. Check if all dependencies are correctly installed
2. Verify Python version (requires Python 3.8+)
3. Try reinstalling with `pip install --force-reinstall modelport`

## Continuous Integration

For automated publishing, consider setting up GitHub Actions to publish to PyPI when a new release is created. Here's a sample workflow file (`.github/workflows/publish.yml`):

```yaml
name: Publish to PyPI

on:
  release:
    types: [created]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build and publish
      env:
        TWINE_USERNAME: ${{ secrets.PYPI_USERNAME }}
        TWINE_PASSWORD: ${{ secrets.PYPI_PASSWORD }}
      run: |
        python -m build
        twine upload dist/*
```

Remember to add `PYPI_USERNAME` and `PYPI_PASSWORD` as secrets in your GitHub repository settings. 