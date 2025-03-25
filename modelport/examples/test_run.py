#!/usr/bin/env python3
# examples/test_run.py
# A script to test the Docker run functionality of ModelPort

import os
import sys
import subprocess
import argparse

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.docker_runner import run_capsule
from examples.test_export import test_export

def ensure_export_exists():
    """Make sure we have an exported model to test with"""
    export_dir = 'test_export'
    if not os.path.exists(export_dir) or not os.path.isdir(export_dir):
        print("📦 No export found. Creating a test export first...")
        test_export()
    else:
        print(f"✅ Using existing export at {export_dir}")
    
    return os.path.abspath(export_dir)

def verify_docker():
    """Verify Docker is installed and running"""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        subprocess.run(["docker", "info"], check=True, capture_output=True)
        print("✅ Docker is installed and running")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not installed or not running")
        print("Please install Docker Desktop and start it before running this script")
        return False

def test_run(arch="linux/amd64"):
    """Test running the exported model in Docker"""
    print(f"🧪 Testing ModelPort run functionality with arch: {arch}")
    
    # Verify Docker is available
    if not verify_docker():
        return False
    
    # Make sure we have an export
    export_path = ensure_export_exists()
    
    print(f"🚀 Running exported model with Docker ({arch})...")
    try:
        run_capsule(export_path, arch)
        print("✅ Test run completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error running model: {e}")
        return False

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test ModelPort's Docker run functionality")
    parser.add_argument("--arch", default="linux/amd64", help="Target architecture (linux/amd64 or linux/arm64)")
    args = parser.parse_args()
    
    # Run the test
    success = test_run(args.arch)
    sys.exit(0 if success else 1) 