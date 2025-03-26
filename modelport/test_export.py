#!/usr/bin/env python3
# Simple test script to run export directly
from core.exporter import export_model
from core.model_utils import detect_framework, get_model_metadata, validate_onnx_model

print("🧩 ModelPort Test Export")
print("------------------------")

MODEL_PATH = "resnet18.pt"
OUTPUT_DIR = "modelport_capsule"
INPUT_SHAPE = "1,3,224,224"

print(f"📦 Exporting model: {MODEL_PATH}")
print(f"🔍 Auto-detecting framework...")

framework = detect_framework(MODEL_PATH)
print(f"✅ Detected framework: {framework}")

print(f"📊 Extracting model metadata...")
metadata = get_model_metadata(MODEL_PATH, framework)
print(f"✅ Input shape: {metadata['input_shape']}")
print(f"✅ Input dtype: {metadata['input_dtype']}")

# Convert input shape from string
input_shape = [int(x) for x in INPUT_SHAPE.split(",")]
print(f"📐 Using input shape: {input_shape}")

# Run export
try:
    output_path = export_model(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        framework=framework,
        input_shape=INPUT_SHAPE,
        force=True
    )
    print(f"✅ Model exported to: {output_path}")
    
    # Validate ONNX model
    onnx_path = f"{output_path}/model.onnx"
    success, error = validate_onnx_model(onnx_path)
    if success:
        print(f"✅ ONNX model validation successful")
    else:
        print(f"❌ ONNX model validation failed: {error}")
except Exception as e:
    print(f"❌ Export failed: {str(e)}") 