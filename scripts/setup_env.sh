#!/bin/bash
# ==============================================================================
# Setup Development Environment
# ==============================================================================
#
# This script updates the repositories and sets up the python environment using uv.
# It assumes:
# 1. You are running this from the multitabfm project (or its scripts dir).
# 2. fastdfs repo is located at ../fastdfs relative to multitabfm root.
#
# Usage:
#   ./scripts/setup_env.sh
#
# ==============================================================================

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FASTDFS_ROOT="${PROJECT_ROOT}/../fastdfs"

echo "============================================================"
echo "Setting up RDBLearn Environment"
echo "============================================================"
echo "  Project root: ${PROJECT_ROOT}"
echo "  FastDFS root: ${FASTDFS_ROOT}"
echo "============================================================"

# 1. Update fastdfs repo
if [ -d "${FASTDFS_ROOT}" ]; then
    echo "[1/4] Updating fastdfs repository..."
    cd "${FASTDFS_ROOT}"
    # Verify it is a git repo
    if [ -d ".git" ]; then
        echo "  - Checking out master..."
        git checkout master
        echo "  - Pulling latest changes..."
        git pull
    else
        echo "Warning: ${FASTDFS_ROOT} is not a git repository. Skipping update."
    fi
else
    echo "Error: fastdfs directory not found at ${FASTDFS_ROOT}"
    exit 1
fi

# 2. Update multitabfm repo
echo ""
echo "[2/4] Updating multitabfm repository..."
cd "${PROJECT_ROOT}"
# Verify it is a git repo
if [ -d ".git" ]; then
    echo "  - Checking out main..."
    git checkout main
    echo "  - Pulling latest changes..."
    git pull
else
    echo "Warning: Current directory is not a git repository. Skipping update."
fi

# 3. Install uv if not present
echo ""
echo "[3/4] Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "  - uv not found. Installing..."
    # Try different installation methods if needed, but pip usually works if python is present
    pip install uv || curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "  - uv is already installed ($(uv --version))"
fi

# 4. Sync environment
echo ""
echo "[4/4] Syncing environment with uv..."
cd "${PROJECT_ROOT}"
# Use --locked to ensure lock file is respected and not updated
uv sync --locked

echo ""
echo "============================================================"
echo "Environment setup complete!"
echo "============================================================"
