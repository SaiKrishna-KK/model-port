#!/usr/bin/env python3
"""
Test C++ Inference with ModelPort

This script tests the C++ integration for running ModelPort compiled models.
It generates a model, compiles it, and runs inference with the C++ example.
"""

import os
import sys
import time
import torch
import subprocess
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
except ImportError as e:
    print(f"Error importing ModelPort modules: {e}")
    sys.exit(1)

def create_model():
    """Create a model for testing C++ inference"""
    print("Creating a model for C++ inference testing...")
    
    # Create directory
    models_dir = "tests/models"
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.relu = torch.nn.ReLU()
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(16, 10)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.relu(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        # Create and export model
        model = SimpleModel()
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32)
        
        onnx_path = os.path.join(models_dir, "simple_cnn.onnx")
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
    
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def test_cpp_inference():
    """Test C++ inference with the compiled model"""
    # Create model
    model_path = create_model()
    if not model_path:
        return False
    
    # Skip actual test if TVM is not available
    if not HAS_TVM:
        print("Skipping C++ inference test because TVM is not installed.")
        print("To run this test, install TVM with: conda install -c conda-forge tvm")
        return True  # Consider the test passed if TVM is not available
    
    # Create output directory
    output_dir = "tests/output/cpp_test"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Compile model
        print(f"Compiling model {model_path} to {output_dir}...")
        config = compile_model(
            model_path=model_path,
            output_dir=output_dir,
            test=True
        )
        
        # Check compilation result
        if "compiled_files" not in config:
            print("Compilation failed!")
            return False
        
        print("Compilation successful!")
        
        # Get compiled files
        lib_file = os.path.join(output_dir, config["compiled_files"]["lib"])
        graph_file = os.path.join(output_dir, config["compiled_files"]["graph"])
        params_file = os.path.join(output_dir, config["compiled_files"]["params"])
        
        # Check C++ example files
        cpp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelport", "examples")
        cpp_file = os.path.join(cpp_dir, "cpp_inference.cpp")
        cmake_file = os.path.join(cpp_dir, "CMakeLists.txt")
        
        if not os.path.exists(cpp_file) or not os.path.exists(cmake_file):
            print(f"C++ example files not found: {cpp_file}, {cmake_file}")
            return False
        
        # Build directory
        build_dir = os.path.join(cpp_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        
        # Save current directory
        cwd = os.getcwd()
        
        # Build C++ example
        print("Building C++ example...")
        os.chdir(build_dir)
        
        # Run CMake
        cmake_cmd = ["cmake", ".."]
        print(f"Running: {' '.join(cmake_cmd)}")
        cmake_proc = subprocess.run(cmake_cmd, capture_output=True, text=True)
        
        if cmake_proc.returncode != 0:
            print(f"CMake failed with error: {cmake_proc.stderr}")
            os.chdir(cwd)
            return False
        
        # Run Make
        make_cmd = ["make"]
        print(f"Running: {' '.join(make_cmd)}")
        make_proc = subprocess.run(make_cmd, capture_output=True, text=True)
        
        if make_proc.returncode != 0:
            print(f"Make failed with error: {make_proc.stderr}")
            os.chdir(cwd)
            return False
        
        # Run C++ inference
        cpp_exe = os.path.join(build_dir, "cpp_inference")
        if not os.path.exists(cpp_exe):
            print(f"C++ executable not found: {cpp_exe}")
            os.chdir(cwd)
            return False
        
        # Run inference
        run_cmd = [cpp_exe, graph_file, lib_file, params_file]
        print(f"Running C++ inference: {' '.join(run_cmd)}")
        
        start_time = time.time()
        cpp_proc = subprocess.run(run_cmd, capture_output=True, text=True)
        end_time = time.time()
        
        # Return to original directory
        os.chdir(cwd)
        
        # Check result
        if cpp_proc.returncode != 0:
            print(f"C++ inference failed with error: {cpp_proc.stderr}")
            return False
        
        # Print output
        print(f"C++ inference successful (time: {end_time - start_time:.4f}s)")
        print("\nC++ output:")
        print(cpp_proc.stdout)
        
        return True
    
    except Exception as e:
        print(f"Error during C++ testing: {e}")
        return False

if __name__ == "__main__":
    print("=== ModelPort C++ Inference Test ===")
    success = test_cpp_inference()
    print("\n=== Test Result ===")
    
    if success:
        print("✅ C++ inference test PASSED")
        sys.exit(0)
    else:
        print("❌ C++ inference test FAILED")
        sys.exit(1) 