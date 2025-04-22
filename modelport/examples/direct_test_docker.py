#!/usr/bin/env python3
# examples/direct_test_docker.py
# A simple script that tests running an ONNX model in Docker

import os
import subprocess
import shutil
import sys

def setup_docker_test():
    """Set up the Docker test directory with the exported model and inference script"""
    test_dir = 'direct_test_export'
    if not os.path.exists(test_dir) or not os.path.isfile(os.path.join(test_dir, 'model.onnx')):
        print("❌ No exported model found. Please run direct_test.py first.")
        sys.exit(1)
    
    # Create a Dockerfile directly in the test directory
    dockerfile_path = os.path.join(test_dir, 'Dockerfile')
    with open(dockerfile_path, 'w') as f:
        f.write("""FROM python:3.10-slim

RUN pip install onnxruntime numpy

COPY . /app
WORKDIR /app

CMD ["python", "inference.py"]
""")
    
    # Create a simple inference script
    inference_path = os.path.join(test_dir, 'inference.py')
    with open(inference_path, 'w') as f:
        f.write("""import onnxruntime as ort
import numpy as np

print("🧪 Running inference on model.onnx...")
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {input_name: dummy_input})

print("✅ Inference output shape:", output[0].shape)
print("✅ Inference successful!")
""")
    
    print(f"✅ Docker test directory set up at: {test_dir}")
    return test_dir

def run_docker_test(test_dir):
    """Build and run the Docker container"""
    # Verify Docker is running
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not installed or not running")
        return False
    
    # Build the Docker image
    print("🔄 Building Docker image...")
    build_result = subprocess.run(
        ["docker", "build", "-t", "modelport_direct_test", test_dir],
        capture_output=True,
        text=True
    )
    
    if build_result.returncode != 0:
        print(f"❌ Docker build failed: {build_result.stderr}")
        return False
    
    print("✅ Docker build successful")
    
    # Run the Docker container
    print("🚀 Running Docker container...")
    run_result = subprocess.run(
        ["docker", "run", "--rm", "modelport_direct_test"],
        capture_output=True,
        text=True
    )
    
    if run_result.returncode != 0:
        print(f"❌ Docker run failed: {run_result.stderr}")
        return False
    
    print("📋 Container output:")
    print(run_result.stdout)
    
    print("✅ Docker test completed successfully!")
    return True

if __name__ == "__main__":
    print("🧪 Testing ModelPort with Docker directly...")
    test_dir = setup_docker_test()
    run_docker_test(test_dir) 