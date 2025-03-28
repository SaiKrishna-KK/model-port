# ModelPort Test Suite

This directory contains various tests for the ModelPort v2.0 native compilation features.

## Overview

The test suite includes:

1. **Unit Tests**: Basic functionality tests for the ModelPort compiler
2. **Integration Tests**: End-to-end testing of the compilation and inference pipeline
3. **Stress Tests**: Tests with large models, irregular shapes, and error conditions
4. **C++ Integration Tests**: Tests for the C++ inference example

## Running Tests

To run the tests, use the `run_tests.py` script:

```bash
# Run basic tests only (default)
python tests/run_tests.py

# Run all tests
python tests/run_tests.py --all

# Run specific test types
python tests/run_tests.py --basic --cpp
python tests/run_tests.py --stress
```

## Test Files

- `test_tvm_basic.py`: Basic TVM compiler test
- `test_cpp_inference.py`: C++ inference integration test
- `stress_test.py`: Comprehensive stress tests
- `test_compiler.py`: Unit tests for the compiler module

## TVM Dependency

The tests that require TVM will be automatically skipped if TVM is not installed.
To run the full test suite, install TVM with:

```bash
conda install -c conda-forge tvm
```

## Test Coverage Matrix

| Test Type                        | Status |
|----------------------------------|--------|
| Basic compilation                | ✅     |
| Model validation                 | ✅     |
| Error handling                   | ✅     |
| Large models                     | ✅     |
| Irregular input shapes           | ✅     |
| Batch inference                  | ✅     |
| C++ inference                    | ✅     |

## Output Directories

- `tests/models`: Contains ONNX models for testing
- `tests/output`: Contains the output of test runs (compiled models, etc.)
- `tests/stress_test_outputs`: Contains the output of stress test runs 