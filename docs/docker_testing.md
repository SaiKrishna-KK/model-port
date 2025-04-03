# ModelPort Docker Testing Guide

This guide explains how to use Docker for testing ModelPort.

## Quick Start

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

## Setup Requirements

- Docker installed on your system (any recent version)
- Git for cloning the repository
- Internet connection for downloading dependencies
- Works on all platforms (Linux, macOS including M1/M2, Windows)

## Testing Options

| Option | Command | Description |
|--------|---------|-------------|
| Basic tests | `docker run --rm modelport-test` | Run essential functionality tests |
| All tests | `docker run --rm modelport-test python3 -m tests.run_tests --all` | Run comprehensive test suite |
| Specific test | `docker run --rm modelport-test python3 -m tests.test_tvm_basic` | Run only the TVM test |
| Custom test | `docker run --rm -v $(pwd)/your_test.py:/app/custom_test.py modelport-test python3 /app/custom_test.py` | Run your own test script |

## Troubleshooting

### Common Issues

1. **Docker Build Fails**
   - Make sure Docker has enough resources allocated
   - Try using `--no-cache` to force a clean build

2. **TVM-Related Errors**
   - The image includes a patched TVM 0.12.0 which works on all platforms
   - ARM architecture (M1/M2 Macs) may show warnings about optimization - these can be ignored

3. **Batch Inference Issues**
   - Batch processing is platform-dependent
   - Single inference works reliably on all platforms

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

## Customizing the Environment

To modify the Docker environment:

1. Edit `Dockerfile.final` to add custom dependencies
2. Rebuild the image with:
   ```bash
   docker build -t modelport-test-custom -f Dockerfile.final .
   ```
3. Run your customized image:
   ```bash
   docker run --rm modelport-test-custom
   ```

## Development Testing Workflow

1. Make changes to your code
2. Run the Docker tests to verify changes:
   ```bash
   docker run --rm -v $(pwd):/app modelport-test python3 -m tests.run_tests --basic
   ```
3. Fix any issues that appear in the test output
4. Run comprehensive tests before submitting:
   ```bash
   docker run --rm -v $(pwd):/app modelport-test python3 -m tests.run_tests --all
   ``` 