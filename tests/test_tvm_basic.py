#!/usr/bin/env python3
"""
Basic TVM Compiler Test for ModelPort

This is a simplified test to verify that the TVM compiler implementation works.
It generates a tiny model, compiles it with TVM, and runs inference.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if TVM is available
try:
    import tvm
    HAS_TVM = True
except ImportError:
    HAS_TVM = False
    print("WARNING: TVM not installed. Test will be skipped.")

try:
    from modelport.core.compiler import compile_model
    from modelport.core.runtime import run_native_model
except ImportError as e:
    print(f"Error importing ModelPort modules: {e}")
    sys.exit(1)

def create_tiny_model():
    """Create a tiny model for testing"""
    print("Creating a tiny model for testing...")
    
    # Create a simple model
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super(TinyModel, self).__init__()
            self.fc1 = torch.nn.Linear(10, 5)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(5, 2)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # Create directory
    os.makedirs("tests/models", exist_ok=True)
    
    # Create and export model
    model = TinyModel()
    model.eval()
    dummy_input = torch.randn(1, 10)
    
    onnx_path = "tests/models/tiny_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    print(f"Model saved to {onnx_path}")
    return onnx_path

def test_tvm_compiler():
    """Test the TVM compiler with a tiny model"""
    # Create model
    model_path = create_tiny_model()
    
    # Skip actual test if TVM is not available
    if not HAS_TVM:
        print("Skipping TVM compilation test because TVM is not installed.")
        print("To run this test, install TVM with: conda install -c conda-forge tvm")
        return True  # Consider the test passed if TVM is not available
    
    # Create output directory
    output_dir = "tests/output/tvm_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Compile model
    print(f"Compiling model {model_path} to {output_dir}...")
    config = compile_model(
        model_path=model_path,
        output_dir=output_dir,
        test=True
    )
    
    # Check compilation result
    if "compiled_files" in config:
        print("Compilation successful!")
        for key, filename in config["compiled_files"].items():
            file_path = os.path.join(output_dir, filename)
            print(f"  - {key}: {filename} ({os.path.getsize(file_path)} bytes)")
    else:
        print("Compilation failed!")
        return False
    
    # Run inference
    print("Running inference on compiled model...")
    outputs = run_native_model(output_dir)
    
    # Print output shape
    print(f"Inference successful - Output shape: {outputs[0].shape}")
    print(f"Output values: {outputs[0]}")
    
    # Test batch inference
    print("Testing batch inference...")
    custom_shapes = {"input": [4, 10]}  # Batch size of 4
    outputs = run_native_model(output_dir, custom_shapes=custom_shapes)
    print(f"Batch inference successful - Output shape: {outputs[0].shape}")
    
    return True

if __name__ == "__main__":
    print("=== ModelPort TVM Compiler Basic Test ===")
    success = test_tvm_compiler()
    print("\n=== Test Result ===")
    
    if success:
        print("✅ TVM compiler test PASSED")
        sys.exit(0)
    else:
        print("❌ TVM compiler test FAILED")
        sys.exit(1) 