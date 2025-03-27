#!/usr/bin/env python3
# Test that the modelport package can be installed and imported
import os
import sys
import subprocess
import importlib

def test_install_and_import():
    """Test that modelport can be installed and imported"""
    print("Testing ModelPort as a Python package...")
    
    # Step 1: Install the package in development mode
    print("\n1. Installing modelport package...")
    install_cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    
    try:
        subprocess.run(install_cmd, check=True)
        print("✅ Installation successful")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    
    # Step 2: Test importing the package
    print("\n2. Testing package import...")
    
    try:
        import modelport
        print(f"✅ Import successful - ModelPort version: {modelport.__version__}")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Step 3: Check that key functions are available
    print("\n3. Checking package functionality...")
    
    # Check export_model function
    if hasattr(modelport, "export_model") and callable(modelport.export_model):
        print("✅ export_model function available")
    else:
        print("❌ export_model function not available")
        return False
    
    # Check run_capsule function
    if hasattr(modelport, "run_capsule") and callable(modelport.run_capsule):
        print("✅ run_capsule function available")
    else:
        print("❌ run_capsule function not available")
        return False
    
    print("\n✅ All tests passed! ModelPort is successfully packaged as a library.")
    return True

if __name__ == "__main__":
    test_install_and_import() 