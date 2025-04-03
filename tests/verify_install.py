#!/usr/bin/env python3
"""
Simple verification script to check ModelPort installation.
"""

import sys
import platform
import torch
import onnx
import onnxruntime
import numpy as np

try:
    import tvm
    HAS_TVM = True
except ImportError:
    HAS_TVM = False

try:
    import modelport
    HAS_MODELPORT = True
except ImportError:
    HAS_MODELPORT = False

def print_system_info():
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")

def print_package_versions():
    print("\n=== Package Versions ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"ONNX: {onnx.__version__}")
    print(f"ONNX Runtime: {onnxruntime.__version__}")
    print(f"NumPy: {np.__version__}")
    if HAS_TVM:
        print(f"TVM: {tvm.__version__}")
    else:
        print("TVM: Not installed")
    
    if HAS_MODELPORT:
        print(f"ModelPort: {modelport.__version__}")
    else:
        print("ModelPort: Not installed")

def verify_basic_functionality():
    if not HAS_MODELPORT:
        print("\n❌ ModelPort not installed. Cannot verify functionality.")
        return False
    
    print("\n=== Verifying Basic Functionality ===")
    
    # Check if key modules are available
    modules = ["core", "export", "compile", "inference", "utils"]
    for module in modules:
        try:
            getattr(modelport, module)
            print(f"✅ Module {module} available")
        except AttributeError:
            print(f"❌ Module {module} not available")
            return False
    
    print("\n=== ModelPort Verification Complete ===")
    return True

if __name__ == "__main__":
    print("=== ModelPort Installation Verification ===")
    print_system_info()
    print_package_versions()
    verify_basic_functionality()
    
    if HAS_MODELPORT:
        print("\n✅ ModelPort seems to be installed correctly.")
    else:
        print("\n❌ ModelPort is not installed. Please install with 'pip install modelport'") 