#!/usr/bin/env python3
# Standalone deploy test script
import os
import sys
import subprocess
import json
from pathlib import Path

# Import the deployer.py directly without module imports
import importlib.util
spec = importlib.util.spec_from_file_location("deployer", "core/deployer.py")
deployer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployer)

# Now we can access deploy_capsule function
deploy_capsule = deployer.deploy_capsule

print("🧩 ModelPort Deploy Test")
print("------------------------")

# Path to the capsule we want to deploy
CAPSULE_PATH = "modelport_capsule"
TAG = "modelport/resnet18:latest"
PLATFORM = "linux/arm64"  # Use platform appropriate for Mac M1

# Ensure the capsule exists
if not os.path.exists(CAPSULE_PATH):
    print(f"❌ Capsule not found at {CAPSULE_PATH}")
    print("Please run export_resnet.py first to create the capsule")
    sys.exit(1)

# Deploy the capsule
print(f"📦 Deploying capsule: {CAPSULE_PATH}")
print(f"🏷️ Using tag: {TAG}")
print(f"🖥️ Target platform: {PLATFORM}")

try:
    # Call the deploy function
    image_tag = deploy_capsule(
        capsule_path=CAPSULE_PATH,
        tag=TAG,
        platform=PLATFORM,
        push=False,
        gpu=False
    )
    
    print(f"✅ Successfully built Docker image: {image_tag}")
    print("\nTo run the container, use:")
    print(f"    docker run --rm {image_tag}")
    
    # Verify the Docker image exists
    result = subprocess.run(
        ["docker", "image", "inspect", image_tag], 
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        print("\n🔍 Docker image details:")
        # Extract some basic info from the image inspect output
        output = result.stdout
        image_id = output.split('"Id": "')[1].split('"')[0] if '"Id": "' in output else "unknown"
        created_date = output.split('"Created": "')[1].split('"')[0] if '"Created": "' in output else "unknown"
        print(f"    Image ID: {image_id}")
        print(f"    Created: {created_date}")
        
        # Check if there is a capsule_spec.json file
        spec_path = os.path.join(CAPSULE_PATH, "capsule_spec.json")
        if os.path.exists(spec_path):
            with open(spec_path, 'r') as f:
                spec = json.load(f)
                print(f"    Capsule spec version: {spec.get('version', 'unknown')}")
                print(f"    Framework: {spec.get('framework', 'unknown')}")
                platforms = spec.get('deployment', {}).get('platforms', [])
                if platforms:
                    print(f"    Deployment platforms: {', '.join(platforms)}")
    else:
        print(f"❌ Docker image verification failed: {result.stderr}")
    
except Exception as e:
    print(f"❌ Deployment failed: {str(e)}")
    sys.exit(1) 