# Reflex inference in OpenPI

The PyTorch PI0/PI0.5 path supports fused projections, bounded streaming KV
history, CUDA Graph denoising, an optional visual worker on a separate CUDA stream,
and an optional state-conditioned AdaRMS branch. These are inference adaptations
for OpenPI checkpoints; changing the attention mask or adding untrained
conditioning does not establish task-quality equivalence to the original model.

## Environment

RTX 5090 requires the CUDA 12.8 PyTorch wheels selected in `pyproject.toml` and
`uv.lock` (torch 2.7.1+cu128, torchvision 0.22.1+cu128). After installing the
environment, install the repository's Transformers replacements:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/models/. \
  .venv/lib/python3.11/site-packages/transformers/models/
```

LIBERO additionally needs a populated LIBERO checkout on `PYTHONPATH` and its
configured assets. On this machine the populated checkout is
`/data1/lin/Apaper2/openpi/third_party/libero`. The copy in this repository is
an empty submodule. This environment also has a `libero_local_source.pth` pointing
to the populated checkout, so imports work without `PYTHONPATH`; the command below
keeps the source location explicit for reproducibility. For EGL rendering set `MUJOCO_GL=egl` and
`PYOPENGL_PLATFORM=egl`; use `JAX_PLATFORMS=cpu` for the PyTorch evaluation.
The existing official LIBERO initial-state files contain NumPy arrays; the
evaluation command sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for compatibility
with their loader on PyTorch 2.6+.

## Server and model options

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=6 JAX_PLATFORMS=cpu \
OPENPI_DATA_HOME=/data1/lin/openpi_cache \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python scripts/serve_policy.py \
  --port 18105 --profile-inference \
  --streaming-window 1 --cuda-graph --concurrent-vision \
  policy:checkpoint --policy.config pi05_libero \
  --policy.dir /data1/lin/openpi_cache/openpi-assets/checkpoints/pi05_libero_pytorch
```

`--streaming-window` counts observations, each containing all configured camera
views. Language tokens remain pinned; each frame attends to language and its own
tokens, and the action expert attends to the retained visual window. Eviction
overwrites one ring slot without moving retained KV. A chronological snapshot is
copied once per chunk into static decode storage. Empty slots are masked from the
first chunk so the captured attention shape remains constant while the ring fills.
The suffix scratch is overwritten at every denoising step, never appended.
This is not a claim of zero allocations across the entire Python control loop.
Use window 1 for the corrected single-observation path: each frame overwrites the
previous frame at fixed RoPE positions immediately after the valid language
tokens. The suffix offset is recomputed from that frame's valid mask. Window sizes
above 1 retain the experimental history and increasing position scheme; resetting
only a counter would be invalid for retained KV that already contains old RoPE.

`--cuda-graph` captures all ten denoising steps by default. The first call includes
warmup/capture overhead. Subsequent calls copy new state, prefix KV, masks, RoPE
offsets and noise into the same addresses and replay the graph. A shape or step
count change rebuilds its one bounded graph entry. This flag disables the separate
`torch.compile` route. Do not modify model weights while retaining a captured graph.

`--concurrent-vision` requires `--streaming-window` and retains its CLI name for
compatibility. It runs this request's visual prefill on a worker CUDA stream and
waits for publication before snapshotting and decoding. Without the flag, prefill
runs on the calling stream. Both paths use this request's image, including during
CUDA Graph replay. Vision and action decoding within one request are sequential;
the external broker still overlaps the entire request with action execution.
Worker errors propagate to the caller. There is no speculative next-frame job
left pending when inference returns.

`policy.reset()` drains pending vision work and resets the episode history. Prompt,
token mask or window changes bootstrap a fresh history. Configurations with state
discretized into prompt tokens will also reset history when those tokens change.
`pi05_libero` already uses `discrete_state_input=False`. Call `policy.close()` at
shutdown to join the producer. The server closes it on normal shutdown.

Projection fusion happens after loading the original checkpoint, packing QKV into
one linear and gate/up into another. The original checkpoint file is unchanged.
The inference model's in-memory state dict uses fused keys; do not overwrite an
original checkpoint with it and expect the original unfused loader to read it.

## State conditioning and normalization

`--state-conditioning` explicitly adds Reflex's state projection and two-layer
MLP to the timestep conditioning used by PI0.5's expert AdaRMSNorm. Its final
linear is initialized to zero, preserving the pretrained timestep conditioning
until this new branch is trained. Enabling it alone does not provide learned
state modulation or demonstrate the paper's long-horizon stability gains.

RMS variance and normalization are computed in FP32; the condition is cast to
the actual modulation layer's weight dtype. OpenPI's pretrained scale/shift/gate
normalization is retained. Reflex's custom single-scale norm cannot replace
those checkpoint tensors directly because their parameter shapes differ. This
port provides the state-conditioning path while preserving OpenPI's norm layout;
it is not a literal replacement of every Reflex normalization class. New state
MLP weights require explicit saving/loading by a fine-tuning workflow; the server
flag initializes a fresh branch and is not an adapter checkpoint loader.

## LIBERO action overlap and future-state prediction

```bash
CUDA_VISIBLE_DEVICES=5 JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
PYTHONPATH=/data1/lin/Apaper2/openpi/third_party/libero \
.venv/bin/python examples/libero/main.py --args.host 127.0.0.1 --args.port 18105 \
  --args.task-suite-name libero_spatial,libero_object \
  --args.async-inference --args.replan-steps 5 --args.overlap-steps 2 \
  --args.metrics-out-path data/libero/new_reflex_eval.jsonl --args.no-save-video
```

The model returns ten actions; the broker retains the first five. After executing
three actions, the fourth call launches the next inference using the current
observation, then returns action four. Actions four and five execute during
inference. If inference is incomplete at the chunk boundary, the next call waits.
The first chunk blocks. An episode-ending reset drains pending work and resets
the server cache. Optional `control_hz` in the broker API adapts overlap from
measured inference latency; the LIBERO command above fixes it at two steps.
Thus requests use observations O3, O8, O13, ... for the next five-action chunks.
At chunk start the newest image is two control steps old (100 ms at 20 Hz), with
no additional model-internal five-step visual delay. If inference takes longer
than the overlap, the broker waits at the boundary.

Future-state prediction applies only the remaining commands. LIBERO's eight
observed coordinates are position, rotation vector and two finger joint positions;
its seven action coordinates are OSC deltas and one gripper command. The LIBERO
callback clips/scales deltas using the actual controller, composes world-axis
rotations by left multiplication, and uses the previous executed chunk's final
gripper command as the future open/close target. For Panda, negative means open
`[0.04, -0.04]`, positive means closed `[0, 0]`, zero retains sensed finger values.
It does not add gripper commands to joint positions. This predicts commanded
targets, not contact dynamics or tracking errors. The generic broker predictor
only supports a seven-dimensional state with deltas in matching units; other
robots must provide `state_predictor`.

## Timing and validation

- `policy_timing.infer_ms`: model sampling plus materializing outputs on the CPU,
  excluding input/output transforms and WebSocket overhead.
- `server_timing.infer_ms`: the complete server policy call, including transforms.
- Client inference time: request through decoded response, including transport.
- `llm_ms`: synchronous prefix stage in standard mode; GPU duration of the consumed
  visual producer publication in streaming mode (includes image encoding and KV
  prefill). This publication always belongs to the current call.
- Streaming profiling additionally reports `vision_encoder_ms` (all configured
  camera image encoders, projection and visual token packing), `llm_prefill_ms`
  (language initialization when needed plus visual transformer prefill),
  `language_init_ms` and `visual_prefill_ms`. These use CUDA events on the stream
  that executes the work, without an extra synchronization between stages.
  `vlm_total_ms` aliases the historical combined `llm_ms`; `llm_ms` is retained
  for compatibility and must not be interpreted as the split LLM-only time.
  Split values are propagated to broker records and LIBERO episode metrics.
- `prefix_wait_ms`: host critical-path preparation/wait and snapshot staging in
  streaming mode; it is not the visual producer's full compute time.
- `action_expert_ms`: all configured denoising steps, including decode preparation
  as applicable; divide by `denoising_steps` for an amortized per-step figure.
- `async_hidden_ms` and `switch_wait_ms`: broker overlap and chunk-boundary wait
  totals. These measure inference versus environment execution, separately from
  overlap between the visual producer and action expert on the GPU.

Visual prefill and action decoding are sequential in the corrected path. Their
stage times exclude some host and transform overhead; use complete policy/client
timing to assess latency. Broker overlap metrics separately describe how much
inference overlaps environment execution.

`scripts/check_reflex_inference.py` checks a real checkpoint with synthetic images,
fixed noise, eager/graph parity, graph reuse, streaming and state-conditioning
initialization. It is not a LIBERO accuracy benchmark. See
[the migration audit](reflex_migration_audit.md) for tests, evidence and limits.
