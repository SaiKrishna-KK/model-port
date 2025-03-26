"""
Tests for the ModelPort compiler module.
"""

import os
import sys
import pytest
import shutil
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tvm
    HAS_TVM = True
except ImportError:
    HAS_TVM = False

from modelport.core.compiler import compile_model, ModelCompiler, SUPPORTED_ARCHS, SUPPORTED_DEVICES
from modelport.core.runtime import run_native_model, benchmark_native_model

# Skip all tests if TVM is not installed
pytestmark = pytest.mark.skipif(not HAS_TVM, reason="TVM is not installed")

# Test directory for generated files
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs")
TEST_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/tiny_model.onnx")

# Create a tiny ONNX model for testing if it doesn't exist
def create_tiny_model():
    import torch
    from torch import nn
    import torch.onnx
    
    class TinyModel(nn.Module):
        def __init__(self):
            super(TinyModel, self).__init__()
            self.fc = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(5, 2)
            
        def forward(self, x):
            x = self.fc(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # Create model directory if it doesn't exist
    model_dir = os.path.dirname(TEST_MODEL)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create and export model
    model = TinyModel()
    dummy_input = torch.randn(1, 10)
    torch.onnx.export(
        model, 
        dummy_input, 
        TEST_MODEL,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    return TEST_MODEL

# Setup and teardown
def setup_module(module):
    """Setup for the test module - create test directory and test model."""
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Create a test model if it doesn't exist
    if not os.path.exists(TEST_MODEL):
        try:
            create_tiny_model()
        except ImportError:
            pytest.skip("PyTorch not installed, cannot create test model")

def teardown_module(module):
    """Teardown for the test module - remove test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

# Tests
def test_model_compiler_initialization():
    """Test that the ModelCompiler initializes correctly."""
    if not os.path.exists(TEST_MODEL):
        pytest.skip("Test model not available")
    
    # Create compiler instance
    compiler = ModelCompiler(
        model_path=TEST_MODEL,
        output_dir=os.path.join(TEST_DIR, "test_init"),
        target_arch=None,  # Auto-detect
        target_device="cpu",
        opt_level=3
    )
    
    # Check that the compiler was initialized correctly
    assert compiler.model_path == TEST_MODEL
    assert compiler.target_device == "cpu"
    assert compiler.opt_level == 3
    assert compiler.target_arch in SUPPORTED_ARCHS
    assert os.path.exists(compiler.output_dir)

def test_compile_model_function():
    """Test the compile_model function."""
    if not os.path.exists(TEST_MODEL):
        pytest.skip("Test model not available")
    
    output_dir = os.path.join(TEST_DIR, "test_compile")
    
    # Compile model
    config = compile_model(
        model_path=TEST_MODEL,
        output_dir=output_dir,
        target_arch=None,  # Auto-detect
        target_device="cpu",
        opt_level=2,
        test=True
    )
    
    # Check output files
    assert os.path.exists(output_dir)
    assert "compiled_files" in config
    
    files = config["compiled_files"]
    for key in ["lib", "graph", "params"]:
        assert key in files
        assert os.path.exists(os.path.join(output_dir, files[key]))
    
    # Check test results
    assert "test_results" in config
    assert config["test_results"]["success"] == True

def test_run_native_model():
    """Test running a compiled model."""
    if not os.path.exists(TEST_MODEL):
        pytest.skip("Test model not available")
    
    output_dir = os.path.join(TEST_DIR, "test_run")
    
    # Compile model
    config = compile_model(
        model_path=TEST_MODEL,
        output_dir=output_dir,
        test=True
    )
    
    # Run model
    outputs = run_native_model(
        model_dir=output_dir
    )
    
    # Check outputs
    assert len(outputs) > 0
    assert isinstance(outputs[0], np.ndarray)
    assert outputs[0].shape[1] == 2  # Output size from TinyModel

def test_benchmark_native_model():
    """Test benchmarking a compiled model."""
    if not os.path.exists(TEST_MODEL):
        pytest.skip("Test model not available")
    
    output_dir = os.path.join(TEST_DIR, "test_benchmark")
    
    # Compile model
    config = compile_model(
        model_path=TEST_MODEL,
        output_dir=output_dir
    )
    
    # Benchmark model
    results = benchmark_native_model(
        model_dir=output_dir,
        iterations=5,
        warmup=2
    )
    
    # Check benchmark results
    assert "iterations" in results
    assert results["iterations"] == 5
    assert "avg_time" in results
    assert "throughput" in results
    assert len(results["times"]) == 5

def test_compile_with_custom_input_shape():
    """Test compiling with a custom input shape."""
    if not os.path.exists(TEST_MODEL):
        pytest.skip("Test model not available")
    
    output_dir = os.path.join(TEST_DIR, "test_custom_shape")
    
    # Custom input shape
    input_shapes = {"input": [4, 10]}  # Batch size 4
    
    # Compile model
    config = compile_model(
        model_path=TEST_MODEL,
        output_dir=output_dir,
        input_shapes=input_shapes
    )
    
    # Check that input shape was set correctly
    assert "input_shapes" in config
    assert "input" in config["input_shapes"]
    assert config["input_shapes"]["input"] == [4, 10]
    
    # Run with the same shape
    outputs = run_native_model(
        model_dir=output_dir,
        custom_shapes=input_shapes
    )
    
    # Check output shape
    assert outputs[0].shape[0] == 4  # Batch size 4

def test_compile_invalid_model():
    """Test that compilation fails with an invalid model."""
    # Create an invalid model file
    invalid_model = os.path.join(TEST_DIR, "invalid.onnx")
    with open(invalid_model, "w") as f:
        f.write("This is not an ONNX model")
    
    # Attempt to compile the invalid model
    with pytest.raises(Exception):
        compile_model(
            model_path=invalid_model,
            output_dir=os.path.join(TEST_DIR, "invalid_output")
        )

if __name__ == "__main__":
    # For manual testing
    setup_module(None)
    test_model_compiler_initialization()
    test_compile_model_function()
    test_run_native_model()
    test_benchmark_native_model()
    test_compile_with_custom_input_shape()
    test_compile_invalid_model()
    teardown_module(None) 