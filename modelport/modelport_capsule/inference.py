# examples/inference.py
import onnxruntime as ort
import numpy as np
import json
import os

# Check if config.json exists and load it
if os.path.exists('config.json'):
    with open('config.json', 'r') as f:
        config = json.load(f)
    input_shape = config.get('input_shape', [1, 3, 224, 224])
    input_dtype = config.get('input_dtype', 'float32')
else:
    # Default values if config doesn't exist
    input_shape = [1, 3, 224, 224]
    input_dtype = 'float32'

print("Running inference on model.onnx...")
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

# Create dummy input based on config
dummy_input = np.random.rand(*input_shape).astype(np.dtype(input_dtype))
output = session.run(None, {input_name: dummy_input})

print("✅ Inference output shapes:", [o.shape for o in output])
print("✅ Inference successful!") 