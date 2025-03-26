import onnxruntime as ort
import numpy as np
import json

# Load model configuration
with open('config.json', 'r') as f:
    config = json.load(f)

print("Running inference on model.onnx...")
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

# Create input with correct shape and dtype
input_shape = config['input_shape']
input_dtype = config['input_dtype']

# Convert torch dtype to numpy dtype if necessary
if input_dtype.startswith('torch.'):
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
    # Assume it's already a numpy-compatible string
    numpy_dtype = input_dtype

# Create random input data
dummy_input = np.random.rand(*input_shape).astype(numpy_dtype)
output = session.run(None, {input_name: dummy_input})

print("✅ Inference output shapes:", [o.shape for o in output])
print("✅ Inference successful!") 