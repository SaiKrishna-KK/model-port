#!/bin/bash
# Script to prepare and push ModelPort v1.5 to GitHub while preserving the previous version

# Ensure we're in the correct directory
echo "Current directory: $(pwd)"

# 1. Make sure repository is initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
fi

# 2. Check current branch and ensure we have the latest code
current_branch=$(git branch --show-current)
echo "Current branch: $current_branch"

# 3. Add remote if not already added (replace with your actual GitHub repository URL)
if ! git remote | grep -q origin; then
    echo "Adding GitHub remote..."
    git remote add origin https://github.com/SaiKrishna-KK/model-port.git
fi

# 4. Commit current changes to preserve the previous version if not already committed
echo "Committing current state as previous version..."
git add .
git commit -m "Previous version before v1.5 upgrade"

# 5. Create a new branch for v1.5
echo "Creating branch for v1.5..."
git branch v1.5-release
git checkout v1.5-release

# 6. The README.md has already been updated, now commit those changes
echo "Committing v1.5 changes..."
git add README.md
git commit -m "Release v1.5 with framework auto-detection, model validation, and Docker deployment"

# 7. Add a tag for the release
echo "Adding release tag..."
git tag -a v1.5 -m "ModelPort v1.5 Release"

# 8. Instructions for pushing to GitHub
echo ""
echo "=== NEXT STEPS ==="
echo "To push these changes to GitHub, run the following commands:"
echo ""
echo "# Push the main branch to preserve previous version"
echo "git push origin main"
echo ""
echo "# Push the new release branch"
echo "git push origin v1.5-release"
echo ""
echo "# Push the tag"
echo "git push origin v1.5"
echo ""
echo "After pushing, create a release on GitHub:"
echo "1. Go to https://github.com/SaiKrishna-KK/model-port/releases"
echo "2. Click 'Draft a new release'"
echo "3. Select the 'v1.5' tag"
echo "4. Title it 'ModelPort v1.5'"
echo "5. Add release notes (copy from the README.md changelog section)"
echo "6. Publish the release" 