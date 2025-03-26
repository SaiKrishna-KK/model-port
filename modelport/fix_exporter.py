#!/usr/bin/env python3
"""
Fix script for exporter.py

This script updates the exporter.py file to use the improved inference template
that correctly handles PyTorch datatypes.
"""
import os
import shutil
import importlib.util

def read_template():
    """Read the improved inference template"""
    template_path = os.path.join('examples', 'inference_template.py')
    if os.path.exists(template_path):
        with open(template_path, 'r') as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Template file not found: {template_path}")

def update_exporter():
    """Update the exporter.py file"""
    exporter_path = os.path.join('core', 'exporter.py')
    
    # Make a backup
    backup_path = f"{exporter_path}.bak"
    shutil.copy2(exporter_path, backup_path)
    print(f"✅ Created backup at {backup_path}")
    
    # Read the exporter file
    with open(exporter_path, 'r') as f:
        content = f.read()
    
    # Find the section where the inference script is defined
    # This is a bit hacky, but it should work
    inference_start = content.find('with open(inference_dst, \'w\') as f:')
    if inference_start == -1:
        print("❌ Could not find inference script section in exporter.py")
        return False
    
    # Find the start of the triple-quoted string
    script_start = content.find('"""import', inference_start)
    if script_start == -1:
        print("❌ Could not find inference script content in exporter.py")
        return False
    
    # Find the end of the triple-quoted string
    script_end = content.find('"""', script_start + 3)
    if script_end == -1:
        print("❌ Could not find end of inference script content in exporter.py")
        return False
    
    # Get the template content
    template = read_template()
    
    # Replace the inference script
    new_content = (
        content[:script_start] + 
        '"""' + template + '"""' + 
        content[script_end + 3:]
    )
    
    # Write the updated content back
    with open(exporter_path, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Updated exporter.py with improved inference template")
    return True

def main():
    """Main function"""
    print("🔧 Updating exporter.py with improved inference template")
    try:
        success = update_exporter()
        if success:
            print("✅ Successfully updated exporter.py")
        else:
            print("❌ Failed to update exporter.py")
    except Exception as e:
        print(f"❌ Error updating exporter.py: {str(e)}")

if __name__ == "__main__":
    main() 