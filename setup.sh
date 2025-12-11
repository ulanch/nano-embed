#!/bin/bash

# This script sets up the environment for running nano-embed on a Lambda Labs instance.

# 1. Install uv (if not already installed)
echo "--- Installing uv ---"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# Add uv to the PATH
export PATH="$HOME/.local/bin:$PATH"
echo "uv installed."

# 2. Create a virtual environment using uv
echo "--- Creating virtual environment ---"
uv venv
echo "Virtual environment created."

# 3. Activate the virtual environment
source .venv/bin/activate
echo "Virtual environment activated."

# 4. Install dependencies using uv sync
echo "--- Installing dependencies ---"
# We assume a Lambda instance will have GPUs, so we install with the 'gpu' extra
uv sync --extra gpu 
echo "Dependencies installed."

echo -e "\nSetup complete. You can now run the training script, for example:\nbash run_embedding.sh"
