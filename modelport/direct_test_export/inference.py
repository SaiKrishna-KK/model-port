import onnxruntime as ort
import numpy as np

print("🧪 Running inference on model.onnx...")
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {input_name: dummy_input})

print("✅ Inference output shape:", output[0].shape)
print("✅ Inference successful!")
