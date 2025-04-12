# core/exporter.py
import torch
import os
import shutil
import sys
import importlib.util
import inspect
import pathlib

def export_model(model_path, output_dir):
    """
    Export a PyTorch model to ONNX format and prepare a portable capsule.
    
    Args:
        model_path (str): Path to the PyTorch model file (.pt)
        output_dir (str): Directory where the exported model and assets will be stored
        
    Returns:
        str: Path to the output directory containing the exported model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to determine if we're in a module context or script context
    # Check if the environment has the model's class already
    
    # Load the model
    try:
        # First try direct loading with weights_only=False
        model = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        # If that fails, try with weights_only=True for newer PyTorch versions
        print(f"Warning: Could not load model with weights_only=False: {e}")
        print("Trying with weights_only=True...")
        model = torch.load(model_path, map_location="cpu", weights_only=True)
        
    model.eval()

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(output_dir, "model.onnx")
    torch.onnx.export(model, dummy_input, onnx_path)

    # Copy inference resources
    # Find the templates directory relative to this file
    module_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(os.path.dirname(module_dir))
    
    # Copy inference script
    inference_src = os.path.join(package_dir, "modelport", "examples", "inference.py")
    inference_dst = os.path.join(output_dir, "inference.py")
    
    if os.path.exists(inference_src):
        shutil.copy(inference_src, inference_dst)
    else:
        # Try a relative path if the package structure is different
        alt_src = os.path.join(os.path.dirname(module_dir), "examples", "inference.py")
        if os.path.exists(alt_src):
            shutil.copy(alt_src, inference_dst)
        else:
            # Create a default inference script
            with open(inference_dst, 'w') as f:
                f.write("""import onnxruntime as ort
import numpy as np

print("Running inference on model.onnx...")
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {input_name: dummy_input})

print("Inference output shape:", output[0].shape)
print("Inference successful!")
""")

    # Copy Docker templates
    templates_src = os.path.join(package_dir, "modelport", "templates")
    templates_dst = os.path.join(output_dir, "runtime")
    
    if os.path.exists(templates_src) and os.path.isdir(templates_src):
        shutil.copytree(templates_src, templates_dst, dirs_exist_ok=True)
    else:
        # Try a relative path if the package structure is different
        alt_src = os.path.join(os.path.dirname(module_dir), "templates")
        if os.path.exists(alt_src) and os.path.isdir(alt_src):
            shutil.copytree(alt_src, templates_dst, dirs_exist_ok=True)
        else:
            # Create a default Dockerfile
            os.makedirs(templates_dst, exist_ok=True)
            with open(os.path.join(templates_dst, "Dockerfile.x86_64"), 'w') as f:
                f.write("""FROM python:3.10-slim

RUN pip install onnxruntime numpy

COPY . /app
WORKDIR /app

CMD ["python", "inference.py"]
""")
            # Copy the same content for ARM64
            shutil.copy(
                os.path.join(templates_dst, "Dockerfile.x86_64"),
                os.path.join(templates_dst, "Dockerfile.arm64")
            )
            
    return output_dir 