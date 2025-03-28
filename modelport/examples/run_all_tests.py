#!/usr/bin/env python3
# examples/run_all_tests.py
# A script to run all tests and validate ModelPort functionality

import os
import sys
import time
import importlib
import importlib.util
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored terminal output
init()

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"{Fore.CYAN}{text}{Style.RESET_ALL}")
    print("=" * 60)

def print_success(text):
    """Print a success message"""
    print(f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}")

def print_error(text):
    """Print an error message"""
    print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")

def print_info(text):
    """Print an info message"""
    print(f"{Fore.YELLOW}ℹ️ {text}{Style.RESET_ALL}")

def run_test(name, test_func):
    """Run a test and return success status"""
    print_info(f"Running {name}...")
    start_time = time.time()
    
    try:
        result = test_func()
        elapsed = time.time() - start_time
        
        if result:
            print_success(f"{name} completed successfully in {elapsed:.2f}s")
            return True
        else:
            print_error(f"{name} failed in {elapsed:.2f}s")
            return False
    except Exception as e:
        elapsed = time.time() - start_time
        print_error(f"{name} failed with error: {str(e)} in {elapsed:.2f}s")
        return False

def test_simple_api():
    """Test basic modelport API usage"""
    import torch
    import torch.nn as nn
    import tempfile
    import shutil
    import modelport
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x)
    
    # Create temporary directories
    model_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    
    try:
        # Save model
        model = SimpleModel()
        model.eval()
        model_path = os.path.join(model_dir, "model.pt")
        
        # Need to save in a way that's compatible with torch.load
        torch.save(model.state_dict(), model_path)
        
        # Simple export directly with ONNX
        dummy_input = torch.randn(1, 10)
        onnx_path = os.path.join(output_dir, "model.onnx")
        torch.onnx.export(model, dummy_input, onnx_path)
        
        # Verify output
        return os.path.exists(onnx_path)
    finally:
        # Clean up
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)

def main():
    """Run all tests to validate ModelPort functionality"""
    print_header("ModelPort Test Suite")
    
    # Track test results
    results = {}
    
    # Test 1: Basic Model Export
    print_header("Test 1: Basic Model Export")
    try:
        # Use importlib to import the module
        from modelport.examples import direct_test
        results["Model Export"] = run_test("Direct model export to ONNX", direct_test.test_direct_export)
    except Exception as e:
        print_error(f"Failed to import direct_test module: {str(e)}")
        results["Model Export"] = False
    
    # Test 2: Multi-architecture Build Test
    print_header("Test 2: Multi-architecture Build Test")
    try:
        # Check if Docker is available
        import subprocess
        docker_check = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if docker_check.returncode != 0:
            print_error("Docker is not available. Skipping multi-architecture test.")
            results["Multi-architecture Build"] = False
        else:
            # Import the multiarch_test module
            try:
                from modelport.examples import multiarch_test
                results["Multi-architecture Build"] = run_test("Multi-architecture Docker builds", 
                                                          multiarch_test.test_multiarch_build)
            except Exception as e:
                print_error(f"Failed to run multi-architecture test: {str(e)}")
                results["Multi-architecture Build"] = False
    except Exception as e:
        print_error(f"Error checking Docker availability: {str(e)}")
        results["Multi-architecture Build"] = False
    
    # Test 3: Simple API Usage Test
    print_header("Test 3: Simple API Usage Test")
    try:
        results["Simple API Usage"] = run_test("ModelPort API basic usage", test_simple_api)
    except Exception as e:
        print_error(f"Failed to run Simple API test: {str(e)}")
        results["Simple API Usage"] = False
    
    # Print summary
    print_header("Test Results Summary")
    all_passed = True
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
            all_passed = False
    
    if all_passed:
        print("\n" + "=" * 60)
        print(f"{Fore.GREEN}All tests passed! ModelPort is working correctly.{Style.RESET_ALL}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print(f"{Fore.RED}Some tests failed. Please review the output above.{Style.RESET_ALL}")
        print("=" * 60)

if __name__ == "__main__":
    main() 