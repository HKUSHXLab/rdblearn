#!/bin/bash
# ==============================================================================
# Setup Development Environment
# ==============================================================================
#
# This script updates the repositories and sets up the python environment using uv.
# It assumes:
# - You are running this from the multitabfm project (or its scripts dir).
#
# Usage:
#   ./scripts/setup_env.sh
#
# ==============================================================================

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "Setting up RDBLearn Environment"
echo "============================================================"
echo "  Project root: ${PROJECT_ROOT}"
echo "============================================================"

# 1. Update multitabfm repo
echo ""
echo "[1/3] Updating multitabfm repository..."
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

# 2. Clone LimiX repository
echo ""
echo "[2/4] Cloning LimiX repository..."
if [ -d "${PROJECT_ROOT}/LimiX" ]; then
    echo "  - LimiX already exists at ${PROJECT_ROOT}/LimiX. Pulling latest..."
    cd "${PROJECT_ROOT}/LimiX" && git pull
else
    echo "  - Cloning LimiX into ${PROJECT_ROOT}/LimiX..."
    git clone https://github.com/limix-ldm/LimiX.git "${PROJECT_ROOT}/LimiX"
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
