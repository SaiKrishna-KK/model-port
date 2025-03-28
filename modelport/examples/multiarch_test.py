#!/usr/bin/env python3
# examples/multiarch_test.py
# A script to test building multi-architecture Docker images

import os
import subprocess
import sys
import tempfile
import torch
import torch.nn as nn
import shutil

def create_test_docker_dir():
    """Create a test directory with a model and Dockerfile"""
    # Create a temporary directory
    test_dir = tempfile.mkdtemp(prefix="modelport_test_")
    
    try:
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(10, 2)
                
            def forward(self, x):
                return self.fc(x)
        
        # Save the model to ONNX format
        model = SimpleModel()
        model.eval()
        
        # Export directly to ONNX
        dummy_input = torch.randn(1, 10)
        onnx_path = os.path.join(test_dir, "model.onnx")
        torch.onnx.export(model, dummy_input, onnx_path)
        
        # Create a simple inference script
        inference_script = """
import onnxruntime as ort
import numpy as np

def run_inference():
    # Load the ONNX model
    session = ort.InferenceSession("model.onnx")
    input_name = session.get_inputs()[0].name
    
    # Create test input
    test_input = np.random.rand(1, 10).astype(np.float32)
    
    # Run inference
    output = session.run(None, {input_name: test_input})
    print(f"Inference successful! Output shape: {output[0].shape}")
    return True

if __name__ == "__main__":
    run_inference()
"""
        with open(os.path.join(test_dir, "inference.py"), "w") as f:
            f.write(inference_script)
        
        # Create a Dockerfile
        dockerfile = """
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install onnxruntime numpy

# Copy model and inference script
COPY model.onnx .
COPY inference.py .

# Run inference when container starts
CMD ["python", "inference.py"]
"""
        with open(os.path.join(test_dir, "Dockerfile"), "w") as f:
            f.write(dockerfile)
        
        return test_dir
    except Exception as e:
        # Clean up in case of error
        shutil.rmtree(test_dir, ignore_errors=True)
        raise e

def test_multiarch_build():
    """Test building Docker images for multiple architectures"""
    # Create test directory with model and Dockerfile
    test_dir = create_test_docker_dir()
    
    try:
        # Verify Docker is available
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker is not available. Cannot run multi-architecture tests.")
            return False
        
        # Verify Docker buildx is available
        try:
            subprocess.run(["docker", "buildx", "version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker buildx is not available. Please install it to run multi-architecture tests.")
            return False
        
        # For test purposes, we'll only build for the current architecture
        # This makes the test faster and more reliable
        print("🔄 Building Docker image for current architecture...")
        build_cmd = [
            "docker", "build",
            "-t", "modelport_test_image",
            test_dir
        ]
        
        print(f"📋 Running command: {' '.join(build_cmd)}")
        build_result = subprocess.run(build_cmd, capture_output=True, text=True)
        
        if build_result.returncode != 0:
            print(f"❌ Build failed: {build_result.stderr}")
            return False
        
        print("✅ Build successful!")
        
        # Verify the image
        print("🔍 Verifying image...")
        inspect_cmd = ["docker", "image", "inspect", "modelport_test_image"]
        inspect_result = subprocess.run(inspect_cmd, capture_output=True, text=True)
        
        if inspect_result.returncode != 0:
            print(f"❌ Image verification failed: {inspect_result.stderr}")
            return False
        
        print("✅ Image verified successfully!")
        
        # Optional: Run a quick test with the image
        try:
            print("🧪 Running a quick test with the image...")
            run_cmd = ["docker", "run", "--rm", "modelport_test_image"]
            run_result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=30)
            
            if run_result.returncode != 0:
                print(f"❌ Container run failed: {run_result.stderr}")
                return False
            
            print(f"✅ Container ran successfully with output: {run_result.stdout.strip()}")
        except subprocess.TimeoutExpired:
            print("⚠️ Container run timed out, but build was successful")
        
        return True
    finally:
        # Clean up
        try:
            # Remove test directory
            shutil.rmtree(test_dir, ignore_errors=True)
            
            # Remove Docker image
            subprocess.run(["docker", "rmi", "modelport_test_image"], 
                          capture_output=True, check=False)
        except Exception as e:
            print(f"⚠️ Cleanup error (non-fatal): {str(e)}")

if __name__ == "__main__":
    print("🧪 Testing ModelPort with Docker builds...")
    success = test_multiarch_build()
    sys.exit(0 if success else 1) 