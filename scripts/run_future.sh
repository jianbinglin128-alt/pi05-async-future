#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-help}
if [[ "$mode" == help ]]; then
    printf '%s\n' 'Usage: bash scripts/run_future.sh server GPU PORT [extra server options]' '       bash scripts/run_future.sh eval GPU PORT [--suite libero_spatial|libero_object|libero_goal|libero_10|all] [--trials 50] [--output NEW_DIRECTORY]'
    exit 0
fi
gpu=${2:?Specify the physical GPU index}
port=${3:?Specify the port}
shift 3
[[ "$gpu" =~ ^[0-9]+$ ]] || { printf '%s\n' 'GPU must be one physical index'; exit 2; }
export CUDA_VISIBLE_DEVICES="$gpu"
libero_source=${LIBERO_SOURCE:-/data1/lin/LIBERO}
export PYTHONPATH="$repo_dir:$repo_dir/src:$repo_dir/packages/openpi-client/src:$libero_source${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/data1/lin/openpi_cache}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.60}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
cd "$repo_dir"
case "$mode" in
    server)
        export JAX_PLATFORMS=cuda
        exec "$repo_dir/.venv/bin/python" scripts/serve_policy.py --port "$port" "$@"
        ;;
    eval)
        export JAX_PLATFORMS=cpu
        export MUJOCO_GL=egl
        export PYOPENGL_PLATFORM=egl
        export MUJOCO_EGL_DEVICE_ID="$gpu"
        export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/.libero}
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        exec "$repo_dir/.venv/bin/python" examples/libero/main.py --port "$port" "$@"
        ;;
    *) printf 'Unknown mode: %s\n' "$mode"; exit 2 ;;
esac
