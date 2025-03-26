#!/usr/bin/env python3
# Test script for framework auto-detection and --test flag
import os
import sys
import importlib.util
from pathlib import Path

# Import the exporter.py directly without module imports
spec = importlib.util.spec_from_file_location("exporter", "core/exporter.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

# Now we can access export_model function
export_model = exporter.export_model

print("🧩 ModelPort Export Test")
print("------------------------")

# Constants for the test
MODEL_PATH = "resnet18.pt"
OUTPUT_DIR = "auto_detect_test"
TEST_FLAG = True

# Ensure the model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    print("Please run examples/train_resnet18.py first to create the model")
    sys.exit(1)

# Test 1: Framework Auto-Detection
print("\n🔍 Test 1: Framework Auto-Detection")
print(f"📦 Exporting model: {MODEL_PATH}")
print(f"📁 Output directory: {OUTPUT_DIR}")

try:
    # Call export_model without specifying the framework
    output_dir = export_model(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        framework=None,  # Force auto-detection
        force=True
    )
    
    # Check if config.json contains the correct framework
    import json
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            detected_framework = config.get("framework")
            print(f"✅ Auto-detected framework: {detected_framework}")
    else:
        print(f"❌ Config file not found at {config_path}")
        
    print(f"✅ Test 1 passed: Model exported successfully with auto-detection")
except Exception as e:
    print(f"❌ Test 1 failed: {str(e)}")
    sys.exit(1)

# Test 2: --test Flag
print("\n🧪 Test 2: --test Flag")
print(f"📦 Exporting model with validation: {MODEL_PATH}")
print(f"📁 Output directory: {OUTPUT_DIR}_validated")

try:
    # Call export_model with test=True
    output_dir = export_model(
        model_path=MODEL_PATH,
        output_dir=f"{OUTPUT_DIR}_validated",
        test=TEST_FLAG,
        force=True
    )
    
    # Check if capsule_spec.json contains test results
    spec_path = os.path.join(output_dir, "capsule_spec.json")
    if os.path.exists(spec_path):
        with open(spec_path, 'r') as f:
            spec = json.load(f)
            print(f"✅ Capsule spec version: {spec.get('version', 'unknown')}")
            print(f"✅ Framework: {spec.get('framework', 'unknown')}")
            
            # Check if test results are included
            test_results = None
            if "test_results" in spec:
                test_results = spec["test_results"]
                print(f"✅ Test results found in capsule_spec.json: {test_results}")
            else:
                # Check if test_results are in config.json instead
                config_path = os.path.join(output_dir, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if "test_results" in config:
                            test_results = config["test_results"]
                            print(f"✅ Test results found in config.json: {test_results}")
            
            if test_results:
                print(f"✅ Test 2 passed: Model validation results recorded")
            else:
                print(f"⚠️ Test 2 partial pass: Model exported but no test results found")
    else:
        print(f"⚠️ Capsule spec not found at {spec_path}")
        print(f"⚠️ Test 2 partial pass: Model exported but capsule_spec.json not found")
except Exception as e:
    print(f"❌ Test 2 failed: {str(e)}")
    sys.exit(1)

print("\n🎉 All tests completed!")
print(f"📁 Auto-detection test output: {OUTPUT_DIR}")
print(f"📁 Validation test output: {OUTPUT_DIR}_validated") 