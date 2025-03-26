#!/bin/bash
# Cleanup script for the ModelPort repository

echo "Cleaning up ModelPort repository..."

# Remove large model files
echo "Removing model binary files..."
find . -name "*.pt" -o -name "*.onnx" | xargs rm -f

# Remove test artifacts
echo "Removing test artifacts..."
rm -rf library_test_model
rm -rf library_test_export
rm -rf modelport/auto_detect_test
rm -rf modelport/auto_detect_test_validated
rm -rf modelport/direct_test_export
rm -rf modelport/test_export
rm -rf modelport/test_model
rm -rf modelport/test_cli_export
rm -rf modelport/modelport_capsule

# Remove Python cache directories
echo "Removing Python cache directories..."
find . -name "__pycache__" -type d | xargs rm -rf
rm -rf modelport.egg-info

# Remove redundant test files
echo "Removing redundant test files..."
rm -f test_library_import.py
rm -f modelport/test_export.py
rm -f modelport/examples/test_run.py
rm -f modelport/examples/test_export.py
rm -f modelport/examples/direct_test_docker.py
rm -f fix_inference.py
rm -f modelport/deploy_test.py
rm -f modelport/export_test.py
rm -f modelport/fix_exporter.py
rm -f modelport/run_all_tests.py
rm -f modelport/stress_test.py

# Remove Git release prep files
echo "Removing Git release preparation files..."
rm -f prepare_github_release.sh
rm -f GITHUB_RELEASE_INSTRUCTIONS.md

echo "Cleanup complete!" 