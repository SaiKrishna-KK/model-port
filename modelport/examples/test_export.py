#!/usr/bin/env python3
# examples/test_export.py
# A simple test script to demonstrate model export with ModelPort

import torch
import torch.nn as nn
import os
import sys

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.exporter import export_model

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

def create_and_save_model():
    """Create a sample model and save it"""
    model = SimpleModel()
    model.eval()
    
    # Save model
    output_dir = 'test_model'
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'simple_model.pt')
    
    # Save the model
    torch.save(model, model_path)
    print(f"✅ Model saved to {model_path}")
    
    return model_path

def test_export():
    """Test the export functionality"""
    print("📦 Creating and saving a test model...")
    model_path = create_and_save_model()
    
    print("🔄 Exporting model with ModelPort...")
    export_dir = 'test_export'
    export_model(model_path, export_dir)
    
    print(f"✅ Model exported successfully to {export_dir}")
    print(f"🔍 Files in export directory: {os.listdir(export_dir)}")

if __name__ == "__main__":
    test_export() 