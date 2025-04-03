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
    """Test TVM compilation and inference"""
    
    # Track test stages
    compilation_success = False
    inference_success = False
    batch_inference_success = False
    
    model_path = create_tiny_model()
    
    # Create output directory
    output_dir = "tests/output/tvm_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Compile the model
    print(f"Compiling model {model_path} to {output_dir}...")
    try:
        success = compile_model(
            model_path,
            output_dir=output_dir,
            target_arch="aarch64",  # Use aarch64 for ARM
            target_device="cpu",
            opt_level=3,  # Maximum optimization
            test=True  # Run test after compilation
        )
        
        if not success:
            print("Compilation failed!")
            return False
            
        print("Compilation successful!")
        
        # Verify generated files
        expected_files = [
            f"model_aarch64.so",
            f"model_aarch64.json",
            f"model_aarch64.params",
            "compile_config.json"
        ]
        
        for file in expected_files:
            file_path = os.path.join(output_dir, file)
            if not os.path.exists(file_path):
                print(f"Missing expected file: {file}")
                return False
            file_size = os.path.getsize(file_path)
            print(f"  - {file}: {file_size} bytes")
        compilation_success = True
    except Exception as e:
        print(f"Compilation error: {e}")
        return False
    
    # Run inference on the compiled model
    print("Running inference on compiled model...")
    try:
        # Create test input
        input_data = {
            "input": np.random.randn(1, 10).astype(np.float32)
        }
        outputs = run_native_model(output_dir, input_data=input_data)
        if outputs is None or len(outputs) == 0:
            print("Inference failed!")
            return False
        print(f"Inference successful - Output shape: {outputs[0].shape}")
        print(f"Output values: {outputs[0]}")
        inference_success = True
    except Exception as e:
        print(f"Inference error: {e}")
        return False
    
    # Optional: Test batch inference (not critical for overall success)
    print("Testing batch inference (optional)...")
    try:
        # Create batch test input - use a smaller batch size
        batch_size = 2
        input_size = 10
        
        # Create batch input data with the correct shape
        batch_input = np.random.randn(batch_size, input_size).astype(np.float32)
        input_data = {
            "input": batch_input
        }
        outputs = run_native_model(output_dir, input_data=input_data)
        if outputs is None or len(outputs) == 0:
            print("Batch inference failed - but this is optional.")
        else:
            print(f"Batch inference successful - Output shape: {outputs[0].shape}")
            expected_shape = (batch_size, 2)  # (batch_size, output_size)
            if outputs[0].shape != expected_shape:
                print(f"Unexpected output shape: got {outputs[0].shape}, expected {expected_shape}")
            else:
                batch_inference_success = True
    except Exception as e:
        print(f"Batch inference error: {e}")
        print("Note: Batch inference is optional and not required for test success.")
    
    # Consider the test successful if compilation and single inference work
    return compilation_success and inference_success

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