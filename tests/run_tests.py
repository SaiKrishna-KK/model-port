#!/usr/bin/env python3
"""
ModelPort Test Runner

This script runs various tests for the ModelPort v2.0 native compilation features.
It can run:
1. Basic TVM compiler tests
2. C++ inference tests
3. Comprehensive stress tests

Usage:
    python -m tests.run_tests [--all] [--basic] [--cpp] [--stress]
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

def run_test(test_script, args=None):
    """Run a test script and return success/failure"""
    start_time = time.time()
    
    cmd = [sys.executable, test_script]
    if args:
        cmd.extend(args)
    
    print(f"\n=== Running: {' '.join(cmd)} ===\n")
    
    try:
        proc = subprocess.run(cmd, capture_output=False)
        success = proc.returncode == 0
    except Exception as e:
        print(f"Error running test: {e}")
        success = False
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n=== Test completed in {duration:.2f} seconds ===")
    return success

def main():
    parser = argparse.ArgumentParser(description="ModelPort Test Runner")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--basic", action="store_true", help="Run basic TVM compiler test")
    parser.add_argument("--cpp", action="store_true", help="Run C++ inference test")
    parser.add_argument("--stress", action="store_true", help="Run stress tests")
    parser.add_argument("--stress-args", type=str, help="Arguments to pass to stress test", default="")
    
    args = parser.parse_args()
    
    # If no args specified, default to --basic
    if not any([args.all, args.basic, args.cpp, args.stress]):
        args.basic = True
    
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test scripts
    basic_test = os.path.join(base_dir, "test_tvm_basic.py")
    cpp_test = os.path.join(base_dir, "test_cpp_inference.py")
    stress_test = os.path.join(base_dir, "stress_test.py")
    
    # Results
    results = {}
    
    # Run tests
    if args.all or args.basic:
        print("\n=== Running Basic TVM Compiler Test ===")
        results["basic"] = run_test(basic_test)
    
    if args.all or args.cpp:
        print("\n=== Running C++ Inference Test ===")
        results["cpp"] = run_test(cpp_test)
    
    if args.all or args.stress:
        print("\n=== Running Comprehensive Stress Tests ===")
        stress_args = args.stress_args.split() if args.stress_args else ["--all"]
        results["stress"] = run_test(stress_test, stress_args)
    
    # Print summary
    print("\n=== Test Summary ===")
    all_pass = True
    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name.upper()}: {status}")
        all_pass = all_pass and success
    
    # Final result
    print(f"\nOVERALL: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main()) 