#!/usr/bin/env python3
"""
Stress test for ModelPort

This script performs a series of stress tests on ModelPort to ensure
it can handle various edge cases.
"""
import os
import sys
import torch
import numpy as np
import importlib.util
import time
import shutil
from pathlib import Path

# Import exporter directly to avoid module issues
spec = importlib.util.spec_from_file_location("exporter", "core/exporter.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

# Access functions
export_model = exporter.export_model
detect_framework = exporter.detect_framework

# Define test models
class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)

class ComplexModel(torch.nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        # Complex model with multiple input/output tensors
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 224 * 224, 100)
        self.fc2 = torch.nn.Linear(100, 10)
        
    def forward(self, x):
        # Intermediate values for multiple outputs
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        flat = c2.view(-1, 32 * 224 * 224)
        f1 = self.fc1(flat)
        out = self.fc2(f1)
        return out, f1, c2  # Multiple outputs

class LargeModelWithCustomDtype(torch.nn.Module):
    def __init__(self):
        super(LargeModelWithCustomDtype, self).__init__()
        # Use half precision for some operations
        self.half_linear = torch.nn.Linear(100, 100).half()
        # Use full precision for others
        self.full_linear = torch.nn.Linear(100, 10)
        
    def forward(self, x):
        # Convert to half precision for some operations
        x_half = x.half()
        h = self.half_linear(x_half)
        # Convert back to full precision
        h_full = h.float()
        return self.full_linear(h_full)

def stress_test_tiny_model():
    """Test with a tiny model with minimal parameters"""
    print("\n=== Stress Test: Tiny Model ===")
    model = TinyModel()
    
    # Save model
    model_path = "tiny_model.pt"
    torch.save(model, model_path)
    
    # Run export
    output_dir = "tiny_model_output"
    try:
        start_time = time.time()
        export_model(
            model_path=model_path,
            output_dir=output_dir,
            force=True,
            test=True
        )
        elapsed = time.time() - start_time
        print(f"✅ Tiny model exported successfully in {elapsed:.2f}s")
        
        # Test inference
        onnx_path = os.path.join(output_dir, "model.onnx")
        config_path = os.path.join(output_dir, "config.json")
        inference_path = os.path.join(output_dir, "inference.py")
        
        # Check files exist
        for path in [onnx_path, config_path, inference_path]:
            if os.path.exists(path):
                print(f"✅ {os.path.basename(path)} exists")
            else:
                print(f"❌ {os.path.basename(path)} missing")
                
        # Run simple inference test
        orig_dir = os.getcwd()
        os.chdir(output_dir)
        
        try:
            import onnxruntime as ort
            import json
            
            # Load config
            with open("config.json", "r") as f:
                config = json.load(f)
                
            # Run inference
            session = ort.InferenceSession("model.onnx")
            input_name = session.get_inputs()[0].name
            input_shape = config.get("input_shape", [1, 1])
            
            dummy_input = np.random.rand(*input_shape).astype(np.float32)
            output = session.run(None, {input_name: dummy_input})
            
            print(f"✅ Inference successful, output shapes: {[o.shape for o in output]}")
        except Exception as e:
            print(f"❌ Inference failed: {str(e)}")
        finally:
            os.chdir(orig_dir)
        
    except Exception as e:
        print(f"❌ Tiny model export failed: {str(e)}")
    
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def stress_test_complex_model():
    """Test with a complex model with multiple outputs"""
    print("\n=== Stress Test: Complex Model with Multiple Outputs ===")
    model = ComplexModel()
    
    # Save model
    model_path = "complex_model.pt"
    torch.save(model, model_path)
    
    # Run export
    output_dir = "complex_model_output"
    input_shape = "1,3,224,224"  # Standard image shape
    
    try:
        start_time = time.time()
        export_model(
            model_path=model_path,
            output_dir=output_dir,
            input_shape=input_shape,
            force=True,
            test=True
        )
        elapsed = time.time() - start_time
        print(f"✅ Complex model exported successfully in {elapsed:.2f}s")
        
        # Test inference
        onnx_path = os.path.join(output_dir, "model.onnx")
        config_path = os.path.join(output_dir, "config.json")
        
        # Check files exist
        for path in [onnx_path, config_path]:
            if os.path.exists(path):
                print(f"✅ {os.path.basename(path)} exists")
            else:
                print(f"❌ {os.path.basename(path)} missing")
                
        # Run simple inference test
        orig_dir = os.getcwd()
        os.chdir(output_dir)
        
        try:
            import onnxruntime as ort
            import json
            
            # Load config
            with open("config.json", "r") as f:
                config = json.load(f)
                
            # Run inference
            session = ort.InferenceSession("model.onnx")
            input_name = session.get_inputs()[0].name
            input_shape = config.get("input_shape", [1, 3, 224, 224])
            
            dummy_input = np.random.rand(*input_shape).astype(np.float32)
            output = session.run(None, {input_name: dummy_input})
            
            print(f"✅ Inference successful, output count: {len(output)}")
            print(f"✅ Output shapes: {[o.shape for o in output]}")
        except Exception as e:
            print(f"❌ Inference failed: {str(e)}")
        finally:
            os.chdir(orig_dir)
        
    except Exception as e:
        print(f"❌ Complex model export failed: {str(e)}")
    
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def stress_test_custom_dtype():
    """Test with a model using custom/mixed data types"""
    print("\n=== Stress Test: Model with Custom Data Types ===")
    model = LargeModelWithCustomDtype()
    
    # Save model
    model_path = "dtype_model.pt"
    torch.save(model, model_path)
    
    # Run export
    output_dir = "dtype_model_output"
    input_shape = "1,100"
    
    try:
        start_time = time.time()
        export_model(
            model_path=model_path,
            output_dir=output_dir,
            input_shape=input_shape,
            force=True,
            test=True
        )
        elapsed = time.time() - start_time
        print(f"✅ Custom dtype model exported successfully in {elapsed:.2f}s")
        
        # Test inference
        onnx_path = os.path.join(output_dir, "model.onnx")
        config_path = os.path.join(output_dir, "config.json")
        
        # Check files exist
        for path in [onnx_path, config_path]:
            if os.path.exists(path):
                print(f"✅ {os.path.basename(path)} exists")
            else:
                print(f"❌ {os.path.basename(path)} missing")
                
        # Run simple inference test
        orig_dir = os.getcwd()
        os.chdir(output_dir)
        
        try:
            import onnxruntime as ort
            import json
            
            # Load config
            with open("config.json", "r") as f:
                config = json.load(f)
                
            # Run inference
            session = ort.InferenceSession("model.onnx")
            input_name = session.get_inputs()[0].name
            input_shape = config.get("input_shape", [1, 100])
            
            dummy_input = np.random.rand(*input_shape).astype(np.float32)
            output = session.run(None, {input_name: dummy_input})
            
            print(f"✅ Inference successful, output shapes: {[o.shape for o in output]}")
        except Exception as e:
            print(f"❌ Inference failed: {str(e)}")
        finally:
            os.chdir(orig_dir)
        
    except Exception as e:
        print(f"❌ Custom dtype model export failed: {str(e)}")
    
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def stress_test_empty_directory():
    """Test creating a capsule in an empty directory"""
    print("\n=== Stress Test: Empty Directory ===")
    
    # Create a simple model
    model = TinyModel()
    model_path = "empty_dir_model.pt"
    torch.save(model, model_path)
    
    # Create an empty directory
    output_dir = "empty_directory_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Run export to the empty directory
    try:
        export_model(
            model_path=model_path,
            output_dir=output_dir,
            force=True
        )
        print(f"✅ Export to empty directory successful")
        
        # Check if files were created
        essential_files = ["model.onnx", "inference.py", "config.json"]
        for file in essential_files:
            path = os.path.join(output_dir, file)
            if os.path.exists(path):
                print(f"✅ {file} created")
            else:
                print(f"❌ {file} missing")
    except Exception as e:
        print(f"❌ Export to empty directory failed: {str(e)}")
    
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def stress_test_overwrite_existing():
    """Test overwriting an existing capsule"""
    print("\n=== Stress Test: Overwrite Existing Capsule ===")
    
    # Create a simple model
    model = TinyModel()
    model_path = "overwrite_model.pt"
    torch.save(model, model_path)
    
    output_dir = "overwrite_test"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # First export
    try:
        export_model(
            model_path=model_path,
            output_dir=output_dir,
            force=True
        )
        print(f"✅ First export successful")
        
        # Add a custom file to the output directory
        custom_file = os.path.join(output_dir, "custom_file.txt")
        with open(custom_file, "w") as f:
            f.write("This is a custom file that should be preserved")
        
        # Try to export again without force
        try:
            export_model(
                model_path=model_path,
                output_dir=output_dir,
                force=False
            )
            print(f"❌ Second export without force succeeded (should have failed)")
        except ValueError:
            print(f"✅ Second export without force failed as expected")
        
        # Try to export again with force
        export_model(
            model_path=model_path,
            output_dir=output_dir,
            force=True
        )
        print(f"✅ Second export with force succeeded")
        
        # Check if custom file was removed (it should be)
        if os.path.exists(custom_file):
            print(f"⚠️ Custom file still exists after force export")
        else:
            print(f"✅ Custom file was removed as expected")
    except Exception as e:
        print(f"❌ Overwrite test failed: {str(e)}")
    
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def main():
    """Run all stress tests"""
    print("🧪 STARTING MODELPORT STRESS TESTS")
    print("=" * 50)
    
    # Run stress tests
    try:
        stress_test_tiny_model()
        stress_test_complex_model()
        stress_test_custom_dtype()
        stress_test_empty_directory()
        stress_test_overwrite_existing()
        
        print("\n🎉 All stress tests completed!")
    except Exception as e:
        print(f"\n❌ Stress tests failed: {str(e)}")
    
if __name__ == "__main__":
    main() 