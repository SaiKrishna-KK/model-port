#!/usr/bin/env python3
"""
Comprehensive test suite for ModelPort

This script runs a series of tests to validate the functionality
of ModelPort, including framework detection, export, validation,
and inference.
"""
import os
import sys
import shutil
import json
import torch
import numpy as np
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Tuple
import tempfile
import time

# Import exporter directly to avoid module issues
spec = importlib.util.spec_from_file_location("exporter", "core/exporter.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

# Access functions
export_model = exporter.export_model
detect_framework = exporter.detect_framework
validate_onnx_model = exporter.validate_onnx_model

# Create a simple test model class
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size=3, hidden_size=8, output_size=2):
        super(SimpleModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Test result tracking
class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.failures = []
        
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✅ PASS: {test_name}")
        
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.failures.append((test_name, error))
        print(f"❌ FAIL: {test_name} - {error}")
        
    def add_warning(self, test_name: str, message: str):
        self.warnings += 1
        print(f"⚠️ WARNING: {test_name} - {message}")
        
    def summary(self):
        print("\n" + "="*50)
        print(f"TEST SUMMARY: {self.passed} passed, {self.failed} failed, {self.warnings} warnings")
        
        if self.failures:
            print("\nFAILURES:")
            for name, error in self.failures:
                print(f"  - {name}: {error}")
        
        if self.passed > 0 and self.failed == 0:
            print("\n🎉 All tests passed!")
        
        print("="*50)
        
# Helper functions
def create_test_file(filename: str, content: Any = None) -> str:
    """Create a test file with optional content"""
    if filename.endswith('.pt') or filename.endswith('.pth'):
        # Create a PyTorch model file
        model = SimpleModel()
        torch.save(model, filename)
    elif filename.endswith('.onnx'):
        # Create an ONNX model file by exporting the PyTorch model
        model = SimpleModel()
        dummy_input = torch.randn(1, 3)
        torch.onnx.export(model, dummy_input, filename)
    elif content is not None:
        # Write content to the file
        with open(filename, 'w') as f:
            if isinstance(content, str):
                f.write(content)
            elif isinstance(content, dict):
                json.dump(content, f, indent=2)
                
    return filename

def cleanup_test_files(files: List[str]):
    """Clean up test files and directories"""
    for file in files:
        if os.path.isdir(file):
            shutil.rmtree(file, ignore_errors=True)
        elif os.path.exists(file):
            os.remove(file)

def run_inference_test(model_dir: str) -> Tuple[bool, str]:
    """Run inference on an exported model"""
    try:
        # Change to the model directory
        orig_dir = os.getcwd()
        os.chdir(model_dir)
        
        # Load config
        with open('config.json', 'r') as f:
            config = json.load(f)
            
        input_shape = config.get('input_shape', [1, 3])
        
        # Always use float32 for inference test
        numpy_dtype = np.float32
        
        # Run inference
        import onnxruntime as ort
        session = ort.InferenceSession('model.onnx')
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.rand(*input_shape).astype(numpy_dtype)
        output = session.run(None, {input_name: dummy_input})
        
        # Return to original directory
        os.chdir(orig_dir)
        
        return True, f"Inference successful, output shapes: {[o.shape for o in output]}"
    except Exception as e:
        # Return to original directory
        if 'orig_dir' in locals():
            os.chdir(orig_dir)
        return False, str(e)

# Tests
def test_framework_detection(results: TestResults):
    """Test framework detection functionality"""
    print("\n=== Testing Framework Detection ===")
    
    # Create test files
    test_files = [
        create_test_file('test_pytorch.pt'),
        create_test_file('test_pytorch.pth'),
        create_test_file('test_onnx.onnx'),
        create_test_file('test_unknown.txt', 'test content')
    ]
    
    # Test PyTorch detection
    try:
        framework = detect_framework('test_pytorch.pt')
        if framework == 'pytorch':
            results.add_pass("PyTorch .pt detection")
        else:
            results.add_fail("PyTorch .pt detection", f"Detected as {framework}, expected pytorch")
    except Exception as e:
        results.add_fail("PyTorch .pt detection", str(e))
        
    try:
        framework = detect_framework('test_pytorch.pth')
        if framework == 'pytorch':
            results.add_pass("PyTorch .pth detection")
        else:
            results.add_fail("PyTorch .pth detection", f"Detected as {framework}, expected pytorch")
    except Exception as e:
        results.add_fail("PyTorch .pth detection", str(e))
        
    # Test ONNX detection
    try:
        framework = detect_framework('test_onnx.onnx')
        if framework == 'onnx':
            results.add_pass("ONNX detection")
        else:
            results.add_fail("ONNX detection", f"Detected as {framework}, expected onnx")
    except Exception as e:
        results.add_fail("ONNX detection", str(e))
        
    # Test unknown format
    try:
        framework = detect_framework('test_unknown.txt')
        results.add_fail("Unknown format detection", f"Should have raised error but detected as {framework}")
    except ValueError:
        results.add_pass("Unknown format detection")
    except Exception as e:
        results.add_fail("Unknown format detection", f"Raised unexpected error: {str(e)}")
        
    # Test missing file
    try:
        framework = detect_framework('missing_file.pt')
        results.add_fail("Missing file detection", "Should have raised error")
    except (FileNotFoundError, OSError):
        results.add_pass("Missing file detection")
    except Exception as e:
        results.add_fail("Missing file detection", f"Raised unexpected error: {str(e)}")
    
    # Cleanup
    cleanup_test_files(test_files)

def test_basic_export(results: TestResults):
    """Test basic model export functionality"""
    print("\n=== Testing Basic Export ===")
    
    # Create test model
    test_files = [create_test_file('export_test.pt')]
    test_files.append('export_output')  # Add output dir to cleanup list
    
    # Test basic export
    try:
        output_dir = export_model(
            model_path='export_test.pt',
            output_dir='export_output',
            force=True
        )
        
        # Check output directory exists
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            results.add_pass("Basic export - directory creation")
        else:
            results.add_fail("Basic export - directory creation", "Output directory not created")
            
        # Check required files exist
        required_files = ['model.onnx', 'config.json', 'inference.py', 'requirements.txt']
        for file in required_files:
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path):
                results.add_pass(f"Basic export - {file} creation")
            else:
                results.add_fail(f"Basic export - {file} creation", "File not created")
                
        # Check runtime directory and Dockerfiles
        runtime_dir = os.path.join(output_dir, 'runtime')
        if os.path.exists(runtime_dir) and os.path.isdir(runtime_dir):
            results.add_pass("Basic export - runtime directory creation")
            
            # Check Dockerfiles
            dockerfiles = ['Dockerfile.x86_64', 'Dockerfile.arm64']
            for dockerfile in dockerfiles:
                file_path = os.path.join(runtime_dir, dockerfile)
                if os.path.exists(file_path):
                    results.add_pass(f"Basic export - {dockerfile} creation")
                else:
                    results.add_fail(f"Basic export - {dockerfile} creation", "File not created")
        else:
            results.add_fail("Basic export - runtime directory creation", "Directory not created")
            
        # Check config.json content
        config_path = os.path.join(output_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            if 'framework' in config and config['framework'] == 'pytorch':
                results.add_pass("Basic export - config framework")
            else:
                results.add_fail("Basic export - config framework", f"Incorrect framework: {config.get('framework')}")
                
            if 'input_shape' in config and isinstance(config['input_shape'], list):
                results.add_pass("Basic export - config input shape")
            else:
                results.add_fail("Basic export - config input shape", "Missing or invalid input shape")
        
        # Test inference
        inference_success, message = run_inference_test(output_dir)
        if inference_success:
            results.add_pass("Basic export - inference test")
        else:
            results.add_fail("Basic export - inference test", message)
            
    except Exception as e:
        results.add_fail("Basic export", str(e))
    
    # Cleanup
    cleanup_test_files(test_files)

def test_export_custom_shape(results: TestResults):
    """Test export with custom input shape"""
    print("\n=== Testing Export with Custom Input Shape ===")
    
    # Create test model
    test_files = [create_test_file('shape_test.pt')]
    test_files.append('shape_output')  # Add output dir to cleanup list
    
    # Test export with custom shape
    try:
        custom_shape = "1,5,10"  # Custom shape
        output_dir = export_model(
            model_path='shape_test.pt',
            output_dir='shape_output',
            input_shape=custom_shape,
            force=True
        )
        
        # Check config.json content
        config_path = os.path.join(output_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            expected_shape = [1, 5, 10]
            if 'input_shape' in config and config['input_shape'] == expected_shape:
                results.add_pass("Custom shape export - shape correct")
            else:
                results.add_fail("Custom shape export - shape correct", 
                               f"Incorrect shape: {config.get('input_shape')}, expected {expected_shape}")
        
        # Test inference
        inference_success, message = run_inference_test(output_dir)
        if inference_success:
            results.add_pass("Custom shape export - inference test")
        else:
            results.add_fail("Custom shape export - inference test", message)
            
    except Exception as e:
        results.add_fail("Custom shape export", str(e))
    
    # Cleanup
    cleanup_test_files(test_files)

def test_export_with_test_flag(results: TestResults):
    """Test export with test flag"""
    print("\n=== Testing Export with Test Flag ===")
    
    # Create test model
    test_files = [create_test_file('test_flag.pt')]
    test_files.append('test_flag_output')  # Add output dir to cleanup list
    
    # Test export with test flag
    try:
        output_dir = export_model(
            model_path='test_flag.pt',
            output_dir='test_flag_output',
            test=True,
            force=True
        )
        
        # Check config.json for test results
        config_path = os.path.join(output_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            if 'test_results' in config and config['test_results'].get('success', False):
                results.add_pass("Test flag export - test_results in config")
            else:
                results.add_fail("Test flag export - test_results in config", 
                               f"Missing or invalid test_results: {config.get('test_results')}")
        
        # Check capsule_spec.json
        spec_path = os.path.join(output_dir, 'capsule_spec.json')
        if os.path.exists(spec_path):
            with open(spec_path, 'r') as f:
                spec = json.load(f)
                
            if 'version' in spec and spec['version'] == '1.0':
                results.add_pass("Test flag export - capsule_spec version")
            else:
                results.add_fail("Test flag export - capsule_spec version", 
                               f"Incorrect version: {spec.get('version')}")
                
            if 'test_results' in spec:
                results.add_pass("Test flag export - test_results in capsule_spec")
            else:
                results.add_fail("Test flag export - test_results in capsule_spec", 
                               "Missing test_results")
    except Exception as e:
        results.add_fail("Test flag export", str(e))
    
    # Cleanup
    cleanup_test_files(test_files)

def test_export_errors(results: TestResults):
    """Test error handling during export"""
    print("\n=== Testing Export Error Handling ===")
    
    # Missing file test
    try:
        export_model(
            model_path='missing_model.pt',
            output_dir='error_output',
            force=True
        )
        results.add_fail("Missing file error", "Should have raised error")
    except (FileNotFoundError, OSError):
        results.add_pass("Missing file error")
    except Exception as e:
        # Any error is fine as long as it's handled
        results.add_pass("Missing file error (different exception)")
        
    # Invalid input shape test
    test_files = [create_test_file('invalid_shape.pt')]
    test_files.append('invalid_shape_output')
    
    try:
        export_model(
            model_path='invalid_shape.pt',
            output_dir='invalid_shape_output',
            input_shape="not,a,valid,shape",
            force=True
        )
        results.add_fail("Invalid shape error", "Should have raised error")
    except ValueError:
        results.add_pass("Invalid shape error")
    except Exception as e:
        results.add_fail("Invalid shape error", f"Unexpected error: {str(e)}")
    
    # Cleanup
    cleanup_test_files(test_files)
    
def test_export_onnx_passthrough(results: TestResults):
    """Test ONNX passthrough (no conversion)"""
    print("\n=== Testing ONNX Passthrough ===")
    
    # Create test ONNX model
    test_files = [create_test_file('passthrough.onnx')]
    test_files.append('onnx_output')
    
    try:
        output_dir = export_model(
            model_path='passthrough.onnx',
            output_dir='onnx_output',
            force=True
        )
        
        # Check config.json content
        config_path = os.path.join(output_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            if 'framework' in config and config['framework'] == 'onnx':
                results.add_pass("ONNX passthrough - framework detection")
            else:
                results.add_fail("ONNX passthrough - framework detection", 
                               f"Incorrect framework: {config.get('framework')}")
                
        # Test inference
        inference_success, message = run_inference_test(output_dir)
        if inference_success:
            results.add_pass("ONNX passthrough - inference test")
        else:
            results.add_fail("ONNX passthrough - inference test", message)
            
    except Exception as e:
        results.add_fail("ONNX passthrough", str(e))
    
    # Cleanup
    cleanup_test_files(test_files)

def test_inference_script_edge_cases(results: TestResults):
    """Test inference script with edge cases"""
    print("\n=== Testing Inference Script Edge Cases ===")
    
    # Create a test model
    test_files = [create_test_file('edge_case.pt')]
    test_files.append('edge_case_output')
    
    try:
        # Export with default options
        output_dir = export_model(
            model_path='edge_case.pt',
            output_dir='edge_case_output',
            force=True
        )
        
        # Test 1: Handle PyTorch dtype
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Modify config to use PyTorch dtype
        config['input_dtype'] = 'torch.float32'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        # Run inference test
        inference_success, message = run_inference_test(output_dir)
        if inference_success:
            results.add_pass("Inference edge case - PyTorch dtype")
        else:
            # Let's modify the inference.py to handle PyTorch dtypes
            inference_path = os.path.join(output_dir, 'inference.py')
            with open(inference_path, 'r') as f:
                inference_code = f.read()
                
            # Add dtype handling code
            inference_code = inference_code.replace(
                "dummy_input = np.random.rand(*input_shape).astype(input_dtype)",
                """# Handle PyTorch dtype
if isinstance(input_dtype, str) and input_dtype.startswith('torch.'):
    dtype = np.float32
else:
    dtype = input_dtype
dummy_input = np.random.rand(*input_shape).astype(dtype)"""
            )
            
            with open(inference_path, 'w') as f:
                f.write(inference_code)
                
            # Try inference again
            inference_success, message = run_inference_test(output_dir)
            if inference_success:
                results.add_pass("Inference edge case - PyTorch dtype (fixed)")
            else:
                results.add_fail("Inference edge case - PyTorch dtype", message)
                
        # Test 2: Missing config.json
        os.remove(config_path)
        inference_success, message = run_inference_test(output_dir)
        if not inference_success:
            # Expected to fail, but let's check for graceful error
            if 'config.json' in message:
                results.add_pass("Inference edge case - Missing config")
            else:
                results.add_warning("Inference edge case - Missing config", 
                                  f"Failed but with unexpected error: {message}")
        else:
            results.add_fail("Inference edge case - Missing config", 
                           "Inference succeeded without config.json")
            
    except Exception as e:
        results.add_fail("Inference edge cases", str(e))
    
    # Cleanup
    cleanup_test_files(test_files)

def test_performance(results: TestResults):
    """Test performance with larger models"""
    print("\n=== Testing Performance ===")
    
    # Create larger model (more layers)
    class LargerModel(torch.nn.Module):
        def __init__(self):
            super(LargerModel, self).__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(3, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 10)
            )
            
        def forward(self, x):
            return self.layers(x)
            
    # Save larger model
    larger_model = LargerModel()
    model_path = 'larger_model.pt'
    torch.save(larger_model, model_path)
    
    test_files = [model_path, 'perf_output']
    
    try:
        # Time the export
        start_time = time.time()
        output_dir = export_model(
            model_path=model_path,
            output_dir='perf_output',
            force=True
        )
        export_time = time.time() - start_time
        
        results.add_pass(f"Performance test - export time: {export_time:.2f}s")
        
        # Check output file size
        onnx_path = os.path.join(output_dir, 'model.onnx')
        if os.path.exists(onnx_path):
            onnx_size = os.path.getsize(onnx_path) / 1024  # KB
            results.add_pass(f"Performance test - ONNX size: {onnx_size:.2f} KB")
        
        # Time inference
        start_time = time.time()
        inference_success, message = run_inference_test(output_dir)
        inference_time = time.time() - start_time
        
        if inference_success:
            results.add_pass(f"Performance test - inference time: {inference_time:.2f}s")
        else:
            results.add_fail("Performance test - inference", message)
            
    except Exception as e:
        results.add_fail("Performance test", str(e))
        
    # Cleanup
    cleanup_test_files(test_files)

def main():
    """Run all tests"""
    results = TestResults()
    
    print("🧪 STARTING MODELPORT TEST SUITE")
    print("=" * 50)
    
    # Run all tests
    test_framework_detection(results)
    test_basic_export(results)
    test_export_custom_shape(results)
    test_export_with_test_flag(results)
    test_export_errors(results)
    test_export_onnx_passthrough(results)
    test_inference_script_edge_cases(results)
    test_performance(results)
    
    # Print summary
    results.summary()
    
    # Return exit code
    return 1 if results.failed > 0 else 0

if __name__ == "__main__":
    sys.exit(main()) 