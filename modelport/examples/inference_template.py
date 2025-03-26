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

# Convert torch dtype to numpy dtype if necessary
if isinstance(input_dtype, str) and input_dtype.startswith('torch.'):
    # Map torch dtypes to numpy dtypes
    if 'float32' in input_dtype or 'float' in input_dtype:
        numpy_dtype = np.float32
    elif 'float64' in input_dtype or 'double' in input_dtype:
        numpy_dtype = np.float64
    elif 'int64' in input_dtype or 'long' in input_dtype:
        numpy_dtype = np.int64
    elif 'int32' in input_dtype or 'int' in input_dtype:
        numpy_dtype = np.int32
    else:
        # Default to float32
        numpy_dtype = np.float32
else:
    # Assume it's already a numpy-compatible string or use float32 as fallback
    try:
        numpy_dtype = input_dtype
    except:
        numpy_dtype = np.float32

# Create dummy input with correct shape and dtype
dummy_input = np.random.rand(*input_shape).astype(numpy_dtype)
output = session.run(None, {input_name: dummy_input})

print("✅ Inference output shapes:", [o.shape for o in output])
print("✅ Inference successful!") 