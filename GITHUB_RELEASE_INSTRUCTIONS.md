# GitHub Release Instructions for ModelPort v1.5

These instructions will guide you through pushing ModelPort v1.5 to GitHub while preserving the previous version.

## Prerequisites

- Git installed on your machine
- GitHub account with push access to the repository
- The updated files already on your local machine

## Step 1: Ensure your repository is set up

```bash
# Navigate to your repository directory
cd /path/to/model-port

# Check if git is initialized
git status
```

If you get an error that it's not a git repository, initialize it:

```bash
git init
```

## Step 2: Set up the remote

```bash
# Add your GitHub repository as the origin (if not already done)
git remote add origin https://github.com/SaiKrishna-KK/model-port.git

# Check that it was added correctly
git remote -v
```

## Step 3: Preserve the previous version

```bash
# Create and switch to a branch for the old version (if not already on a branch)
git checkout -b v0.1.5-main

# Add all current files
git add .

# Commit them as the previous version
git commit -m "ModelPort v0.1.5 - Previous version"

# Push this branch to GitHub
git push -u origin v0.1.5-main
```

## Step 4: Create the new v1.5 release branch

```bash
# Create and switch to a new branch for v1.5
git checkout -b v1.5-release

# If the README is already updated, add it
git add README.md RELEASE_NOTES.md

# Commit the changes
git commit -m "Release v1.5 with framework auto-detection, model validation, and Docker deployment"

# Push the branch to GitHub
git push -u origin v1.5-release
```

## Step 5: Create a release tag

```bash
# Create an annotated tag
git tag -a v1.5 -m "ModelPort v1.5 Release"

# Push the tag to GitHub
git push origin v1.5
```

## Step 6: Create a GitHub Release

1. Go to your repository on GitHub
2. Click on "Releases" in the right sidebar
3. Click "Draft a new release"
4. Select the tag "v1.5"
5. Title: "ModelPort v1.5"
6. Description: Copy the contents of RELEASE_NOTES.md
7. Click "Publish release"

## Step 7: Set the default branch (optional)

If you want the v1.5 branch to be the default when people visit your repo:

1. Go to your repository on GitHub
2. Click "Settings"
3. Click "Branches" in the left sidebar
4. Change the default branch to "v1.5-release"
5. Click "Update"

## Step 8: Create a Pull Request (optional)

If you plan to merge the changes into your main branch later:

1. Go to your repository on GitHub
2. Click "Pull requests"
3. Click "New pull request"
4. Set base branch to "main" and compare branch to "v1.5-release"
5. Click "Create pull request"
6. Add a title and description
7. Click "Create pull request"

You now have your previous version preserved in the v0.1.5-main branch, and your new version in the v1.5-release branch, with a proper GitHub release created from the tag. 