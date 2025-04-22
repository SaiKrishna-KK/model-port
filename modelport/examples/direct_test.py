#!/usr/bin/env python3
# examples/direct_test.py
# A simple script that tests the ONNX export without saving a model first

import torch
import torch.nn as nn
import os
import onnx
import onnxruntime as ort
import numpy as np

class SimpleModel(nn.Module):
    """A simple model for testing purposes"""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 112 * 112, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def test_direct_export():
    """Test direct ONNX export without saving the model first"""
    print("🔄 Creating a simple model...")
    model = SimpleModel()
    model.eval()

    # Create output directory
    output_dir = 'direct_test_export'
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    print("🔄 Exporting to ONNX...")
    torch.onnx.export(model, dummy_input, onnx_path)
    
    # Verify the ONNX model
    print("🔍 Verifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Test inference with ONNX Runtime
    print("🧪 Testing inference with ONNX Runtime...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    # Run inference
    test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    output = session.run(None, {input_name: test_input})
    
    print(f"✅ Model exported and tested successfully! Output shape: {output[0].shape}")
    return True

if __name__ == "__main__":
    test_direct_export() 