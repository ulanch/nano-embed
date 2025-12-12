#!/bin/bash
set -euo pipefail

# This script orchestrates the training and evaluation of nano-embed,
# an embedding model based on the llm2vec paper.

# Default intermediate artifacts directory is in ~/.cache/nano-embed
export OMP_NUM_THREADS=1
export NANO_EMBED_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANO_EMBED_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
if [[ "$(uname)" == "Linux" ]]; then
    uv sync --extra gpu
else
    uv sync --extra cpu
fi
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d34_embed bash run_embedding.sh`
if [ -z "${WANDB_RUN:-}" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nano_embed.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# The user has indicated they want to use a pre-trained tokenizer.
# The following lines for building and training a new tokenizer are commented out.
# # Install Rust / Cargo
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# source "$HOME/.cargo/env"

# # Build the rustbpe Tokenizer
# uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# # Download the first ~2B characters of pretraining dataset
# # look at dev/repackage_data_reference.py for details on how this data was prepared
# # each data shard is ~250M chars
# # so we download 2e9 / 250e6 = 8 data shards at this point
# # each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# python -m nano_embed.dataset -n 8
# # Immediately also kick off downloading more shards in the background while tokenizer trains
# # See comment below for why 240 is the right number here
# python -m nano_embed.dataset -n 240 &
# DATASET_DOWNLOAD_PID=$!
# # train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
# python -m scripts.tok_train --max_chars=2000000000
# # evaluate the tokenizer (report compression ratio etc.)
# python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Embedding Model Training (Stage 1: MNTP)

# Number of processes/GPUs to use
# Dynamically determine NPROC_PER_NODE based on available GPUs
if python -c "import torch; print(torch.cuda.is_available())" | grep -q True; then
    NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
else
    NPROC_PER_NODE=1 # Default to 1 if no CUDA devices or CUDA not available
fi

if [ "$NPROC_PER_NODE" -eq 0 ]; then
    echo "No CUDA devices found. Running on CPU (NPROC_PER_NODE=1)."
    NPROC_PER_NODE=1
fi
echo "NPROC_PER_NODE set to $NPROC_PER_NODE"

# Train the model with Masked Next Token Prediction (initialize from pretrained base)
BASE_PRETRAIN_DIR="$NANO_EMBED_BASE_DIR/base_checkpoints/d34"
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train_mntp -- \
  --depth=34 \
  --run=$WANDB_RUN \
  --bidirectional=True \
  --use_lora=True \
  --mask_ratio=0.15 \
  --model_tag=d34_mntp \
  --load_from_base_dir="$BASE_PRETRAIN_DIR" \
  --device_batch_size=64 \
  --save_every=500 \
  --num_iterations=1000

# -----------------------------------------------------------------------------
# Merge LoRA weights from MNTP training
# Note: you need to find the last checkpoint path from the previous run
# and provide it to the merge script.
# For example: MNTP_CKPT_PATH="$NANO_EMBED_BASE_DIR/base_checkpoints/d20/ckpt_001000.pt"
# Find the latest MNTP checkpoint (LoRA-only weights)
MNTP_CKPT_DIR="$NANO_EMBED_BASE_DIR/base_checkpoints/d34_mntp"
MNTP_CKPT_FILE=$(ls -v "$MNTP_CKPT_DIR"/model_*.pt | tail -n 1)
if [ -z "$MNTP_CKPT_FILE" ]; then
    echo "Error: No MNTP checkpoint found in $MNTP_CKPT_DIR"
    exit 1
fi
echo "Latest MNTP checkpoint: $MNTP_CKPT_FILE"

# Define path for merged checkpoint
MERGED_CKPT_DIR="$NANO_EMBED_BASE_DIR/base_checkpoints/d34_merged"
mkdir -p "$MERGED_CKPT_DIR"
MERGED_CKPT_FILE="${MERGED_CKPT_DIR}/merged_$(basename "$MNTP_CKPT_FILE")"

echo "Merging LoRA weights from MNTP training into base pretrained..."
python -m scripts.merge_lora --lora_checkpoint_path "$MNTP_CKPT_FILE" --base_model_dir "$BASE_PRETRAIN_DIR" --output_dir "$MERGED_CKPT_DIR"

# -----------------------------------------------------------------------------
# Embedding Model Training (Stage 2: Contrastive Learning)

echo "Starting Contrastive Learning (SimCSE) training..."
# Train the model with Contrastive Learning (SimCSE)
# This script will automatically find the latest checkpoint in the specified directory.
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train_contrastive -- \
  --depth=34 \
  --run=$WANDB_RUN \
  --bidirectional=True \
  --use_lora=True \
  --load_from_base_dir="$MERGED_CKPT_DIR" \
  --model_tag=d34_contrastive \
  --device_batch_size=64 \
  --save_every=500 \
  --num_iterations=1000

# -----------------------------------------------------------------------------
# Embedding Model Evaluation

echo "Evaluating embedding model..."
# After contrastive training, merge the LoRA deltas into the merged base to get final weights
CONTRASTIVE_CKPT_DIR="$NANO_EMBED_BASE_DIR/base_checkpoints/d34_contrastive"
CONTRASTIVE_CKPT_FILE=$(ls -v "$CONTRASTIVE_CKPT_DIR"/model_*.pt | tail -n 1)
if [ -z "$CONTRASTIVE_CKPT_FILE" ]; then
    echo "Error: No contrastive LoRA checkpoint found in $CONTRASTIVE_CKPT_DIR"
    exit 1
fi
FINAL_CKPT_DIR="$NANO_EMBED_BASE_DIR/base_checkpoints/d34_final"
mkdir -p "$FINAL_CKPT_DIR"
echo "Merging final LoRA from contrastive into merged base..."
python -m scripts.merge_lora --lora_checkpoint_path "$CONTRASTIVE_CKPT_FILE" --base_model_dir "$MERGED_CKPT_DIR" --output_dir "$FINAL_CKPT_DIR"

# Evaluate the final merged embedding model on downstream tasks
python -m scripts.embed_eval --checkpoint_path "$FINAL_CKPT_DIR"

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nano_embed.report generate
