# Future-only JAX runtime

This snapshot provides inference for the Orbax/JAX checkpoint
[`Dream-LIN/pi05_libero_async_future`](https://huggingface.co/Dream-LIN/pi05_libero_async_future).
It does not include model weights, a virtual environment, evaluation videos, or
machine-specific audit logs. It is not a future-training reproduction recipe.

## Install on another machine

Use Linux, an NVIDIA GPU with drivers compatible with the JAX CUDA dependencies,
Python 3.11, and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/jianbinglin128-alt/pi05-async-future.git
cd pi05-async-future
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

Install LIBERO and the evaluation dependencies in the same environment using
the [upstream LIBERO setup](https://github.com/Lifelong-Robot-Learning/LIBERO).
The prepared server uses `/data1/lin/LIBERO`; on another machine set
`LIBERO_SOURCE` to your LIBERO checkout and configure its BDDL/init-state assets.
`LIBERO_CONFIG_PATH` selects the directory containing LIBERO's configuration.
The evaluator's imports must resolve with `.venv/bin/python` before launch.

Download the checkpoint, including its manifest, params, and assets:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Dream-LIN/pi05_libero_async_future",
    local_dir="checkpoints/pi05_libero_async_future",
)
PY
```

## Start the server and client

In the server terminal, for physical GPU 0 and port 18900:

```bash
bash scripts/run_future.sh server 0 18900 \
  --checkpoint "$PWD/checkpoints/pi05_libero_async_future"
```

In the client terminal, for ten Spatial tasks and three trials per task:

```bash
export LIBERO_SOURCE=/absolute/path/to/LIBERO
bash scripts/run_future.sh eval 0 18900 \
  --suite libero_spatial --trials 3 \
  --output data/libero/future_spatial_3trials
```

For all four suites and fifty trials per task, use `--suite all --trials 50` and
a new output directory. The `eval_async_future.py` filename remains an alias for
`main.py` with the same CLI. The old context evaluator has been removed.

The server accepts only the future JAX checkpoint manifest, verifies every
listed file's size and SHA256, and loads parameters with full key/shape checking.
The prepared server defaults to `/data1/lin/Apaper2/model/pi05_libero_async_future`;
pass `--checkpoint` as above elsewhere. No split-stage profiler is enabled.

## Exact action schedule

1. Bootstrap synchronously using the actual observation and empty context.
2. Receive ten predicted actions; keep the first five for execution.
3. Execute three actions, then acquire the current observation.
4. Submit that observation plus the two remaining fixed actions as
   `action_context`, while executing those same two actions.
5. At the five-action boundary, wait if necessary and use the next output's
   first five predictions. Repeat from step 3.

The checkpoint predicts actions **after** its supplied context. The client must
not skip the first two new predictions, extrapolate future robot state, or
replace the queued actions while their request is in flight. `action_context`
is an input required by the future model, not a legacy context-checkpoint loader.
Ten predicted actions and ten diffusion/flow iterations are separate settings;
this runtime uses both, with only five predicted actions executed per chunk.

## Validation and measurement

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m unittest discover \
  -s tests -p 'test_future_schedule.py' -v
JAX_PLATFORMS=cpu .venv/bin/python scripts/check_future_setup.py \
  --checkpoint "$PWD/checkpoints/pi05_libero_async_future"
```

The second command accepts the downloaded checkpoint directory. It verifies
manifest hashes, imported module paths, the complete parameter tree,
and the abstract JAX sampler output shape without running a GPU benchmark.

Raw outputs include per-request `inferences.jsonl`, per-action `actions.jsonl`,
and per-episode `episodes.jsonl`. Client latency brackets `client.infer` after
inputs are prepared. Action latency brackets `env.step`. Episode time starts
after reset and ten settling steps and stops at the final action's return.
Three warmup requests, disk writes, and pending-request drain after termination
are excluded from episode time. The evaluator does not save videos or impose a
20 Hz wall-clock sleep. A request returning after episode termination is marked
`terminal_prefetch`.

## Attribution

This runtime derives from [OpenPI](https://github.com/Physical-Intelligence/openpi).
The original Apache license and Gemma license files are retained. Check the
checkpoint model card and upstream model/data terms before using the weights.
