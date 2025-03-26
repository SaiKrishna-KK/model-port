#!/usr/bin/env python3
# Standalone model export script for ResNet18 with framework auto-detection and test features
import os
import torch
import shutil
import numpy as np
import json
import importlib.util
from pathlib import Path

# Import the exporter.py directly without module imports
spec = importlib.util.spec_from_file_location("exporter", "core/exporter.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

# Now we can access export_model and detect_framework functions
export_model = exporter.export_model
detect_framework = exporter.detect_framework

# Constants
MODEL_PATH = "resnet18.pt"
OUTPUT_DIR = "modelport_capsule"
INPUT_SHAPE = [1, 3, 224, 224]
TEST_MODEL = True  # Set to True to test the model after export

print("🧩 ModelPort Standalone Export")
print("------------------------------")

# Ensure model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    print("Please run examples/train_resnet18.py first")
    exit(1)

# Auto-detect framework
detected_framework = detect_framework(MODEL_PATH)
print(f"🔍 Auto-detected framework: {detected_framework}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📂 Created output directory: {OUTPUT_DIR}")

# Export the model with test
print(f"📦 Exporting model: {MODEL_PATH}")
print(f"🧪 Running validation: {TEST_MODEL}")

try:
    output_dir = export_model(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        # No need to specify framework - it's auto-detected
        input_shape=",".join(str(x) for x in INPUT_SHAPE),
        force=True,
        test=TEST_MODEL
    )
    
    # Load the config to confirm settings
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"✅ Exported with framework: {config.get('framework')}")
            print(f"✅ Input shape: {config.get('input_shape')}")
            print(f"✅ Input dtype: {config.get('input_dtype')}")
            
            # Show test results if available
            if 'test_results' in config:
                print(f"✅ Test results: {config['test_results']}")
    
    # Check if capsule_spec.json exists
    spec_path = os.path.join(output_dir, "capsule_spec.json")
    if os.path.exists(spec_path):
        with open(spec_path, 'r') as f:
            spec = json.load(f)
            print(f"✅ Capsule spec version: {spec.get('version')}")
            print(f"✅ Created at: {spec.get('created_at')}")
            
    print("\n🎉 Export completed successfully!")
    print(f"📁 ModelPort capsule is ready at: {OUTPUT_DIR}")
    print("\nTo run inference:")
    print(f"    python {OUTPUT_DIR}/inference.py")
    print("\nTo build and run Docker container:")
    print(f"    cd {OUTPUT_DIR}")
    print("    docker buildx build -f runtime/Dockerfile.arm64 -t modelport_container --platform linux/arm64 .")
    print("    docker run --rm modelport_container")
    print("\nTo deploy to Docker Hub:")
    print("    modelport deploy modelport_capsule --tag username/modelport:latest --push")
    
except Exception as e:
    print(f"❌ Export failed: {str(e)}")
    exit(1) 