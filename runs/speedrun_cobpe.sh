#!/bin/bash

# This script is the CoBPE counterpart to runs/speedrun.sh.
# It is designed to run on a blank 8XH100 GPU node.

# 1) Example launch (simplest):
# bash runs/speedrun_cobpe.sh
# 2) Example launch in a screen session:
# screen -L -Logfile runs/speedrun_cobpe.log -S speedrun_cobpe bash runs/speedrun_cobpe.sh
# 3) Example launch with wandb logging:
# WANDB_RUN=speedrun_cobpe screen -L -Logfile runs/speedrun_cobpe.log -S speedrun_cobpe bash runs/speedrun_cobpe.sh

# Keep CoBPE artifacts separate from the baseline speedrun.
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat_cobpe"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Build the CoBPE Rust tokenizer runtime into the active environment.
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    source "$HOME/.cargo/env"
fi
uv pip install maturin
maturin develop --release --manifest-path rust_ext/compositional_runtime/Cargo.toml

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging, first run `wandb login`, then set
# WANDB_RUN when launching this script.
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Reset the report and record system information.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# CoBPE tokenizer

# Download the first ~2B characters of pretraining data for tokenizer training.
python -m nanochat.dataset -n 8
# Continue downloading pretraining shards while the tokenizer trains.
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# Train the normalized buffered base vocabulary and write compositional.json.
python -m scripts.tok_train --cobpe
# Evaluate modifier-aware tokenizer compression and roundtrip behavior.
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 CoBPE model with the same speedrun training budget as runs/speedrun.sh.
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN --model-tag=cobpe_d24
# Evaluate CORE, joint CoBPE BPB, and modifier-aware samples.
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --model-tag=cobpe_d24 --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT

# chat_sft.py does not yet carry modifier IDs through its batches, so running it
# here would train and evaluate only the base-token stream. Leave SFT out until
# that path is modifier-aware.

# -----------------------------------------------------------------------------
# Generate the report.
python -m nanochat.report generate
