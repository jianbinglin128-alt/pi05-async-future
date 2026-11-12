#!/usr/bin/env bash
set -euo pipefail

source "$HOME/data/openpi/scripts/env_a6000.sh"
export OPENPI_REPO="$OPENPI_ROOT/repo-async-context"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTHONPATH="$OPENPI_REPO/src:$OPENPI_REPO/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$OPENPI_REPO"

experiment="${ASYNC_CONTEXT_EXP_NAME:-async_context_sync_targets_20261111}"

exec python -u scripts/train.py pi05_libero_async_context \
    --exp-name="$experiment" \
    --assets-base-dir="$OPENPI_ROOT/assets" \
    --checkpoint-base-dir="$OPENPI_ROOT/outputs" \
    --weight-loader.params-path="$OPENPI_ROOT/checkpoints/pi05_base/params" \
    --batch-size=12 \
    --num-workers=8 \
    --num-train-steps=30000 \
    --log-interval=10 \
    --save-interval=1000 \
    --no-wandb-enabled
