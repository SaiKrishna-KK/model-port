#!/usr/bin/env python3
# examples/multiarch_test.py
# A script to test building multi-architecture Docker images

import os
import subprocess
import sys

def test_multiarch_build():
    """Test building Docker images for multiple architectures"""
    test_dir = 'direct_test_export'
    if not os.path.exists(test_dir) or not os.path.isfile(os.path.join(test_dir, 'Dockerfile')):
        print("❌ Docker test directory not set up. Please run direct_test_docker.py first.")
        sys.exit(1)
    
    # Verify Docker buildx is available
    try:
        subprocess.run(["docker", "buildx", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker buildx is not available. Please run scripts/setup_docker.sh first.")
        return False
    
    # Test building for multiple architectures separately
    platforms = ["linux/amd64", "linux/arm64"]
    success = True
    
    for platform in platforms:
        arch_name = platform.split('/')[1]
        image_name = f"modelport_test_{arch_name}"
        
        print(f"🔄 Building Docker image for {platform}...")
        build_cmd = [
            "docker", "buildx", "build",
            "--platform", platform,
            "-t", image_name,
            "--load",  # Load the image into Docker's local image store
            test_dir
        ]
        
        print(f"📋 Running command: {' '.join(build_cmd)}")
        build_result = subprocess.run(build_cmd, capture_output=True, text=True)
        
        if build_result.returncode != 0:
            print(f"❌ Build for {platform} failed: {build_result.stderr}")
            success = False
            continue
        
        print(f"✅ Build for {platform} successful!")
        
        # Verify the image
        print(f"🔍 Verifying image for {platform}...")
        inspect_cmd = ["docker", "image", "inspect", image_name]
        inspect_result = subprocess.run(inspect_cmd, capture_output=True, text=True)
        
        if inspect_result.returncode != 0:
            print(f"❌ Image verification for {platform} failed: {inspect_result.stderr}")
            success = False
            continue
        
        print(f"✅ Image for {platform} verified successfully!")
    
    if success:
        print("✅ All architecture tests completed successfully!")
    else:
        print("⚠️ Some architecture tests failed. See above for details.")
    
    return success

if __name__ == "__main__":
    print("🧪 Testing ModelPort with multi-architecture Docker builds...")
    test_multiarch_build() 