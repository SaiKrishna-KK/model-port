#!/usr/bin/env python3
# examples/run_all_tests.py
# A script to run all tests and validate ModelPort functionality

import os
import sys
import time
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

def import_module_from_file(file_path):
    """Import a Python module from a file path"""
    module_name = os.path.basename(file_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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

def main():
    """Run all tests to validate ModelPort functionality"""
    print_header("ModelPort Test Suite")
    
    # Track test results
    results = {}
    
    # Test 1: Basic Model Export
    print_header("Test 1: Basic Model Export")
    try:
        direct_test = import_module_from_file("examples/direct_test.py")
        results["Model Export"] = run_test("Direct model export to ONNX", direct_test.test_direct_export)
    except Exception as e:
        print_error(f"Failed to import direct_test.py: {str(e)}")
        results["Model Export"] = False
    
    # Test 2: Docker Container Test
    print_header("Test 2: Docker Container Test")
    try:
        docker_test = import_module_from_file("examples/direct_test_docker.py")
        # First make sure the test directory is set up
        test_dir = docker_test.setup_docker_test()
        results["Docker Container"] = run_test("Docker container build and run", 
                                              lambda: docker_test.run_docker_test(test_dir))
    except Exception as e:
        print_error(f"Failed to run Docker test: {str(e)}")
        results["Docker Container"] = False
    
    # Test 3: Multi-architecture Build Test
    print_header("Test 3: Multi-architecture Build Test")
    try:
        multiarch_test = import_module_from_file("examples/multiarch_test.py")
        results["Multi-architecture Build"] = run_test("Multi-architecture Docker builds", 
                                                      multiarch_test.test_multiarch_build)
    except Exception as e:
        print_error(f"Failed to run multi-architecture test: {str(e)}")
        results["Multi-architecture Build"] = False
    
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