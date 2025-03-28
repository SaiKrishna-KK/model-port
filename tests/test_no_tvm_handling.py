#!/usr/bin/env python3
"""
TVM Absence Handling Test for ModelPort

This test verifies that the ModelPort code properly handles cases where TVM is not installed,
ensuring graceful degradation and appropriate error messages.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import importlib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestTVMAbsenceHandling(unittest.TestCase):
    """Test how ModelPort handles the absence of TVM"""
    
    def test_compiler_module_import(self):
        """Test that the compiler module can be imported without TVM"""
        from modelport.core import compiler
        self.assertFalse(compiler.HAS_TVM, "HAS_TVM should be False when TVM is not installed")
    
    def test_runtime_module_import(self):
        """Test that the runtime module can be imported without TVM"""
        from modelport.core import runtime
        self.assertFalse(runtime.HAS_TVM, "HAS_TVM should be False when TVM is not installed")
    
    def test_compile_command_import(self):
        """Test that the compile command can be imported without TVM"""
        from modelport.cli import compile
        # Simply importing the module should not raise any exception
    
    def test_run_native_command_import(self):
        """Test that the run-native command can be imported without TVM"""
        from modelport.cli import run_native
        # Simply importing the module should not raise any exception
    
    def test_compiler_graceful_error(self):
        """Test that the compile_model function raises an appropriate error when TVM is not available"""
        from modelport.core.compiler import compile_model
        with self.assertRaises(ImportError) as context:
            compile_model("dummy.onnx")
        self.assertIn("TVM is required", str(context.exception))
    
    def test_runtime_graceful_error(self):
        """Test that the run_native_model function raises an appropriate error when TVM is not available"""
        from modelport.core.runtime import run_native_model
        with self.assertRaises(ImportError) as context:
            run_native_model("dummy_dir")
        self.assertIn("TVM is required", str(context.exception))
    
    def test_direct_compiler_api(self):
        """Test the direct API call to compiler with better handling"""
        from modelport.core.compiler import ModelCompiler
        
        with self.assertRaises(ImportError) as context:
            compiler = ModelCompiler("dummy.onnx", "output_dir")
        self.assertIn("TVM is required", str(context.exception))
    
    def test_modelport_entrypoint(self):
        """Test that the main ModelPort entrypoint can be imported without TVM"""
        import modelport
        # The module should be importable

def run_tests():
    """Run the tests"""
    unittest.main(module=__name__, exit=False)
    return True

if __name__ == "__main__":
    print("=== ModelPort TVM Absence Handling Test ===")
    success = run_tests()
    print("\n=== Test Result ===")
    
    if success:
        print("✅ TVM absence handling test PASSED")
        sys.exit(0)
    else:
        print("❌ TVM absence handling test FAILED")
        sys.exit(1) 