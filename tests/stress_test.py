#!/usr/bin/env python3
"""
ModelPort Native Compilation Stress Test

This script performs comprehensive stress testing of the ModelPort compiler
with various models, input shapes, architectures, and error conditions.

Usage:
    python -m tests.stress_test [--all] [--large-models] [--irregular-shapes] 
                               [--error-cases] [--cpp-test] [--batch-test]
"""

import os
import sys
import time
import argparse
import logging
import platform
import shutil
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stress_test_results.log")
    ]
)
logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import torch
    import torchvision
    import onnx
    import tvm
    HAS_DEPS = True
    HAS_TVM = True
except ImportError as e:
    if "tvm" in str(e):
        logger.error(f"TVM not installed: {e}")
        logger.error("To run these tests, install TVM with: conda install -c conda-forge tvm")
        HAS_TVM = False
        HAS_DEPS = torch and torchvision and onnx
    else:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install: torch, torchvision, onnx, tvm")
        HAS_DEPS = False
        HAS_TVM = False

# Import ModelPort modules
try:
    from modelport.core.compiler import compile_model, ModelCompiler, SUPPORTED_ARCHS, SUPPORTED_DEVICES
    from modelport.core.runtime import run_native_model, benchmark_native_model, ModelRunner
except ImportError as e:
    logger.error(f"Error importing ModelPort modules: {e}")
    sys.exit(1)

# Test constants
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress_test_outputs")
MODELS_DIR = os.path.join(TEST_DIR, "models")

def setup():
    """Create directories for test output"""
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"Created test directories: {TEST_DIR}")
    logger.info(f"Current system architecture: {platform.machine()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"TVM installed: {HAS_TVM}")
    if HAS_TVM:
        logger.info(f"TVM version: {tvm.__version__ if hasattr(tvm, '__version__') else 'Unknown'}")
    else:
        logger.warning("TVM is not installed. Tests will be skipped.")
        logger.warning("To run these tests, install TVM with: conda install -c conda-forge tvm")

def cleanup():
    """Clean up test directories"""
    logger.info("Cleaning up test directories...")
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

def generate_resnet_model(size=18, input_shape=(1, 3, 224, 224)):
    """Generate a ResNet model for testing"""
    if not HAS_DEPS:
        logger.error("Missing dependencies, cannot generate model")
        return None

    try:
        logger.info(f"Generating ResNet{size} model with input shape {input_shape}...")
        
        # Get the appropriate model function
        if size == 18:
            model_fn = torchvision.models.resnet18
        elif size == 50:
            model_fn = torchvision.models.resnet50
        elif size == 101:
            model_fn = torchvision.models.resnet101
        else:
            logger.error(f"Unsupported ResNet size: {size}")
            return None
        
        # Load model
        model = model_fn(pretrained=False)
        model.eval()
        
        # Generate dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export to ONNX
        onnx_path = os.path.join(MODELS_DIR, f"resnet{size}_{input_shape[2]}x{input_shape[3]}.onnx")
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        
        logger.info(f"Successfully generated model: {onnx_path}")
        return onnx_path
    
    except Exception as e:
        logger.error(f"Error generating model: {str(e)}")
        return None

def generate_tiny_model(input_shape=(1, 10)):
    """Generate a tiny MLP model for quick testing"""
    if not HAS_DEPS:
        logger.error("Missing dependencies, cannot generate model")
        return None
    
    try:
        logger.info(f"Generating tiny MLP model with input shape {input_shape}...")
        
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super(TinyModel, self).__init__()
                self.fc1 = torch.nn.Linear(input_shape[1], 5)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(5, 2)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
        
        # Create model
        model = TinyModel()
        model.eval()
        
        # Generate dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export to ONNX
        onnx_path = os.path.join(MODELS_DIR, f"tiny_mlp_{input_shape[1]}.onnx")
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            input_names=["input"],
            output_names=["output"]
        )
        
        logger.info(f"Successfully generated model: {onnx_path}")
        return onnx_path
    
    except Exception as e:
        logger.error(f"Error generating model: {str(e)}")
        return None

def create_corrupt_model():
    """Create a corrupted ONNX file for testing error handling"""
    corrupt_path = os.path.join(MODELS_DIR, "corrupt.onnx")
    with open(corrupt_path, "w") as f:
        f.write("This is not a valid ONNX model file")
    logger.info(f"Created corrupt model file: {corrupt_path}")
    return corrupt_path

def test_large_models():
    """Test ModelPort with large models"""
    logger.info("=== Testing Large Models ===")
    
    # Skip test if TVM is not available
    if not HAS_TVM:
        logger.warning("Skipping large models test because TVM is not installed")
        return True
    
    # Generate models of different sizes
    resnet18_path = generate_resnet_model(size=18)
    resnet50_path = generate_resnet_model(size=50)
    resnet101_path = generate_resnet_model(size=101)
    
    if not all([resnet18_path, resnet50_path, resnet101_path]):
        logger.error("Failed to generate all models")
        return False
    
    models = [
        ("ResNet18", resnet18_path),
        ("ResNet50", resnet50_path),
        ("ResNet101", resnet101_path)
    ]
    
    success = True
    for name, path in models:
        logger.info(f"Testing compilation of {name}...")
        try:
            output_dir = os.path.join(TEST_DIR, f"compiled_{name.lower()}")
            
            # Measure compilation time
            start_time = time.time()
            config = compile_model(
                model_path=path,
                output_dir=output_dir,
                opt_level=3,
                test=True
            )
            compile_time = time.time() - start_time
            
            logger.info(f"{name} compilation time: {compile_time:.2f} seconds")
            
            # Check if compilation was successful
            if "compiled_files" in config:
                logger.info(f"{name} compilation successful")
                
                # Test inference
                try:
                    logger.info(f"Testing inference on compiled {name}...")
                    outputs = run_native_model(output_dir)
                    logger.info(f"{name} inference successful - Output shape: {outputs[0].shape}")
                    
                    # Test benchmark
                    logger.info(f"Benchmarking compiled {name}...")
                    results = benchmark_native_model(output_dir, iterations=5, warmup=2)
                    logger.info(f"{name} benchmark results: {results['avg_time']:.4f} seconds/inference")
                    
                except Exception as e:
                    logger.error(f"Error during {name} inference: {str(e)}")
                    success = False
            else:
                logger.error(f"{name} compilation failed")
                success = False
                
        except Exception as e:
            logger.error(f"Error compiling {name}: {str(e)}")
            success = False
    
    return success

def test_irregular_shapes():
    """Test ModelPort with irregular input shapes"""
    logger.info("=== Testing Irregular Input Shapes ===")
    
    # Skip test if TVM is not available
    if not HAS_TVM:
        logger.warning("Skipping irregular shapes test because TVM is not installed")
        return True
    
    # Define a list of irregular shapes to test
    shapes = [
        (1, 1, 128, 128),  # Single channel
        (1, 5, 224, 224),  # 5 channels
        (1, 3, 17, 31),    # Non-standard dimensions
        (4, 3, 224, 224),  # Batch size > 1
        (1, 10),           # MLP input
        (8, 5)             # MLP with batch
    ]
    
    success = True
    for shape in shapes:
        logger.info(f"Testing shape: {shape}")
        
        try:
            # Generate appropriate model based on shape
            if len(shape) == 4:  # CNN input
                model_path = generate_resnet_model(size=18, input_shape=shape)
                model_type = "ResNet18"
            else:  # MLP input
                model_path = generate_tiny_model(input_shape=shape)
                model_type = "TinyMLP"
            
            if model_path is None:
                logger.error(f"Failed to generate model for shape {shape}")
                success = False
                continue
            
            # Compile the model
            shape_str = "_".join(str(dim) for dim in shape)
            output_dir = os.path.join(TEST_DIR, f"{model_type}_{shape_str}")
            
            config = compile_model(
                model_path=model_path,
                output_dir=output_dir,
                test=True
            )
            
            # Verify the compilation was successful
            if "compiled_files" in config:
                logger.info(f"Compilation successful for shape {shape}")
                
                # Test inference
                outputs = run_native_model(output_dir)
                logger.info(f"Inference successful for shape {shape} - Output shape: {outputs[0].shape}")
                
                # Verify batch dimension is preserved
                if len(shape) == 4 and shape[0] > 1:
                    assert outputs[0].shape[0] == shape[0], f"Batch dimension not preserved: {outputs[0].shape}"
                    logger.info(f"Batch dimension correctly preserved: {outputs[0].shape[0]}")
            else:
                logger.error(f"Compilation failed for shape {shape}")
                success = False
        
        except Exception as e:
            logger.error(f"Error testing shape {shape}: {str(e)}")
            success = False
    
    return success

def test_error_handling():
    """Test ModelPort error handling with invalid inputs"""
    logger.info("=== Testing Error Handling ===")
    
    # Generate a valid model for architecture and device tests
    model_path = generate_tiny_model()
    if not model_path:
        logger.error("Failed to generate model for error handling tests")
        return False
    
    success = True
    
    # Test with corrupt ONNX file
    corrupt_path = create_corrupt_model()
    logger.info("Testing compilation of corrupt ONNX file...")
    try:
        output_dir = os.path.join(TEST_DIR, "corrupt_output")
        compile_model(
            model_path=corrupt_path,
            output_dir=output_dir
        )
        logger.error("Compilation of corrupt file did not fail as expected")
        success = False
    except Exception as e:
        logger.info(f"Correctly failed on corrupt file with error: {str(e)}")
    
    # Test with non-existent file
    logger.info("Testing compilation of non-existent file...")
    try:
        output_dir = os.path.join(TEST_DIR, "nonexistent_output")
        compile_model(
            model_path="nonexistent.onnx",
            output_dir=output_dir
        )
        logger.error("Compilation of non-existent file did not fail as expected")
        success = False
    except Exception as e:
        logger.info(f"Correctly failed on non-existent file with error: {str(e)}")
    
    # Skip TVM-specific tests if TVM is not available
    if not HAS_TVM:
        logger.warning("Skipping TVM-specific error handling tests because TVM is not installed")
        return success
    
    # Test with invalid architecture
    logger.info("Testing compilation with invalid architecture...")
    try:
        output_dir = os.path.join(TEST_DIR, "invalid_arch_output")
        compile_model(
            model_path=model_path,
            output_dir=output_dir,
            target_arch="invalid_arch"
        )
        logger.error("Compilation with invalid architecture did not fail as expected")
        success = False
    except Exception as e:
        logger.info(f"Correctly failed on invalid architecture with error: {str(e)}")
    
    # Test with invalid device
    logger.info("Testing compilation with invalid device...")
    try:
        output_dir = os.path.join(TEST_DIR, "invalid_device_output")
        compile_model(
            model_path=model_path,
            output_dir=output_dir,
            target_device="invalid_device"
        )
        logger.error("Compilation with invalid device did not fail as expected")
        success = False
    except Exception as e:
        logger.info(f"Correctly failed on invalid device with error: {str(e)}")
    
    return success

def test_batch_inference():
    """Test batch inference with compiled models"""
    logger.info("=== Testing Batch Inference ===")
    
    # Skip test if TVM is not available
    if not HAS_TVM:
        logger.warning("Skipping batch inference test because TVM is not installed")
        return True
    
    # Generate a model with batch support
    batch_model_path = generate_resnet_model(size=18, input_shape=(1, 3, 224, 224))
    if not batch_model_path:
        logger.error("Failed to generate model for batch testing")
        return False
    
    success = True
    try:
        # Compile the model
        output_dir = os.path.join(TEST_DIR, "batch_model")
        config = compile_model(
            model_path=batch_model_path,
            output_dir=output_dir,
            test=True
        )
        
        if "compiled_files" not in config:
            logger.error("Compilation failed for batch testing")
            return False
        
        # Create a runner
        runner = ModelRunner(output_dir)
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32]
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Create batch input
            custom_shapes = {"input": [batch_size, 3, 224, 224]}
            
            # Run inference
            start_time = time.time()
            outputs = runner.run(custom_shapes=custom_shapes)
            inference_time = time.time() - start_time
            
            # Check output batch dimension
            if outputs[0].shape[0] != batch_size:
                logger.error(f"Output batch size mismatch: expected {batch_size}, got {outputs[0].shape[0]}")
                success = False
            else:
                logger.info(f"Batch size {batch_size} - Inference time: {inference_time:.4f}s - Throughput: {batch_size/inference_time:.2f} images/s")
        
        # Test increasing batch sizes and measure performance
        logger.info("Measuring batch performance scaling:")
        results = {}
        for batch_size in batch_sizes:
            custom_shapes = {"input": [batch_size, 3, 224, 224]}
            
            # Warmup
            for _ in range(3):
                runner.run(custom_shapes=custom_shapes)
            
            # Benchmark
            times = []
            for _ in range(5):
                start_time = time.time()
                runner.run(custom_shapes=custom_shapes)
                times.append(time.time() - start_time)
            
            avg_time = sum(times) / len(times)
            results[batch_size] = {
                "time": avg_time,
                "throughput": batch_size / avg_time,
                "throughput_per_instance": 1 / avg_time * batch_size
            }
        
        # Log results
        logger.info("Batch performance results:")
        prev_throughput = None
        for batch_size, result in sorted(results.items()):
            logger.info(f"Batch {batch_size}: {result['time']:.4f}s, Throughput: {result['throughput']:.2f} images/s")
            if prev_throughput is not None:
                scaling = result['throughput'] / prev_throughput
                logger.info(f"  Scaling from previous batch: {scaling:.2f}x")
            prev_throughput = result['throughput']
    
    except Exception as e:
        logger.error(f"Error during batch testing: {str(e)}")
        success = False
    
    return success

def test_cpp_inference():
    """Test C++ inference with the example code"""
    logger.info("=== Testing C++ Inference ===")
    
    # Skip test if TVM is not available
    if not HAS_TVM:
        logger.warning("Skipping C++ inference test because TVM is not installed")
        return True
    
    # Generate a model
    model_path = generate_tiny_model()
    if not model_path:
        logger.error("Failed to generate model for C++ testing")
        return False
    
    success = True
    try:
        # Compile the model
        output_dir = os.path.join(TEST_DIR, "cpp_model")
        config = compile_model(
            model_path=model_path,
            output_dir=output_dir,
            test=True
        )
        
        if "compiled_files" not in config:
            logger.error("Compilation failed for C++ testing")
            return False
        
        # Get the compiled files
        lib_file = os.path.join(output_dir, config["compiled_files"]["lib"])
        graph_file = os.path.join(output_dir, config["compiled_files"]["graph"])
        params_file = os.path.join(output_dir, config["compiled_files"]["params"])
        
        # Check the C++ example directory
        cpp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelport", "examples")
        cpp_file = os.path.join(cpp_dir, "cpp_inference.cpp")
        cmake_file = os.path.join(cpp_dir, "CMakeLists.txt")
        
        if not os.path.exists(cpp_file) or not os.path.exists(cmake_file):
            logger.error(f"C++ example files not found: {cpp_file}, {cmake_file}")
            return False
        
        # Create build directory
        build_dir = os.path.join(cpp_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        
        # Run CMake and make
        logger.info("Building C++ example...")
        os.chdir(build_dir)
        
        cmake_cmd = f"cmake .."
        logger.info(f"Running: {cmake_cmd}")
        cmake_result = os.system(cmake_cmd)
        
        if cmake_result != 0:
            logger.error("CMake failed")
            return False
        
        make_cmd = "make"
        logger.info(f"Running: {make_cmd}")
        make_result = os.system(make_cmd)
        
        if make_result != 0:
            logger.error("Make failed")
            return False
        
        # Run the C++ example
        cpp_exe = os.path.join(build_dir, "cpp_inference")
        if not os.path.exists(cpp_exe):
            logger.error(f"C++ executable not found: {cpp_exe}")
            return False
        
        run_cmd = f"{cpp_exe} {graph_file} {lib_file} {params_file}"
        logger.info(f"Running C++ inference: {run_cmd}")
        run_result = os.system(run_cmd)
        
        if run_result != 0:
            logger.error("C++ inference failed")
            return False
        else:
            logger.info("C++ inference successful")
        
        # Clean up
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    except Exception as e:
        logger.error(f"Error during C++ testing: {str(e)}")
        success = False
    
    return success

def main():
    parser = argparse.ArgumentParser(description="ModelPort Native Compilation Stress Test")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--large-models", action="store_true", help="Test with large models")
    parser.add_argument("--irregular-shapes", action="store_true", help="Test with irregular input shapes")
    parser.add_argument("--error-cases", action="store_true", help="Test error handling")
    parser.add_argument("--batch-test", action="store_true", help="Test batch inference")
    parser.add_argument("--cpp-test", action="store_true", help="Test C++ inference")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up test files")
    
    args = parser.parse_args()
    
    # If no args specified, default to --all
    if not any([args.all, args.large_models, args.irregular_shapes, args.error_cases, args.batch_test, args.cpp_test]):
        args.all = True
    
    setup()
    
    results = {}
    
    if args.all or args.large_models:
        results["large_models"] = test_large_models()
    
    if args.all or args.irregular_shapes:
        results["irregular_shapes"] = test_irregular_shapes()
    
    if args.all or args.error_cases:
        results["error_handling"] = test_error_handling()
    
    if args.all or args.batch_test:
        results["batch_inference"] = test_batch_inference()
    
    if args.all or args.cpp_test:
        results["cpp_inference"] = test_cpp_inference()
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    all_success = True
    for test_name, success in results.items():
        logger.info(f"{test_name}: {'✅ SUCCESS' if success else '❌ FAILED'}")
        all_success = all_success and success
    
    logger.info(f"\nOverall: {'✅ ALL TESTS PASSED' if all_success else '❌ SOME TESTS FAILED'}")
    
    # Clean up unless --no-cleanup is specified
    if not args.no_cleanup:
        cleanup()
    else:
        logger.info(f"Test files preserved in: {TEST_DIR}")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 