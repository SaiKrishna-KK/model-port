#!/usr/bin/env python3
# examples/simple_usage.py
# A simple example showing how to use ModelPort as an imported library

import torch
import torch.nn as nn
import os
import modelport

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

def main():
    # Create model directory
    model_dir = 'library_test_model'
    os.makedirs(model_dir, exist_ok=True)
    
    # Create and save a simple model
    print("🔄 Creating and saving a test model...")
    model = SimpleModel()
    model.eval()
    
    model_path = os.path.join(model_dir, 'simple_model.pt')
    # Register the SimpleModel class before saving
    torch.serialization.add_safe_globals([SimpleModel])
    torch.save(model, model_path)
    
    # Export the model using ModelPort
    print("🔄 Exporting model with ModelPort...")
    output_dir = 'library_test_export'
    
    # Use the imported modelport.export_model function
    modelport.export_model(model_path, output_dir, force=True)
    
    print(f"✅ Model exported successfully to: {output_dir}")
    
    # Run the model using Docker
    if input("Do you want to run the model in Docker? (y/n): ").lower().startswith('y'):
        print("🔄 Running model with Docker...")
        
        # Use the imported modelport.run_capsule function
        modelport.run_capsule(output_dir, "linux/amd64")
    
    print("✅ Example completed successfully!")

if __name__ == "__main__":
    main() 