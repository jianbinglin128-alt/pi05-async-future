# Reflex migration audit

Update (2026-11-08): the producer/decode concurrency and lag descriptions below
record the audited implementation before the current-frame fix. Streaming decode
now waits for this request's visual publication; window 1 overwrites at fixed
RoPE coordinates. See [current usage](reflex_streaming.md) and
[the correction evaluation](reflex_single_frame_fix_results.md) for current behavior.

Audit date: 2026-11-07 UTC. Target: `/data1/lin/Apaper2/openpi-copy`.
Reference implementation: `/data1/lin/Reflex`; reference paper:
`/data1/lin/paper/Guo和Liu - 2026 - Reflex Real-Time VLA Control through Streaming Inference.pdf`.

This audit covers the requested inference modules and their OpenPI integration.
It does not claim reproduction of the paper's reported accuracy, throughput or
long-horizon stability, nor a port of every optional experimental utility in the
reference repository. The user explicitly deferred the partition-attention versus
original pretrained-model equivalence issue.


Follow-up logic and behavioral audit (2026-11-08):
[success-rate diagnosis](reflex_success_rate_diagnosis.md) records 27 deterministic
paired diagnostic rollouts. Additional visual request lag and drifting visual
RoPE coordinates relative to pinned language both have direct behavioral evidence.
A fixed observation/noise test also demonstrates action drift with a one-frame
ring. State prediction has zero effect with this checkpoint configuration and the
zero-initialized branch. These findings supersede any implication that runtime
module coverage establishes checkpoint semantic equivalence.

## Source-to-target comparison

| Requested capability | Reference | Target and result |
| --- | --- | --- |
| Independent vision producer and action consumer | Paper §3.3; `reflex/policies/pi05/concurrent_executor.py`, `streaming_manager.py` | `reflex_runtime.VisionProducer` uses one worker and a separate CUDA stream. `PI0Pytorch.sample_actions_streaming` snapshots completed KV, stages decode inputs and launches current vision alongside decode. Per-job events, exceptions, reset and close are handled. Bootstrap/capture are isolated. Steady state uses one-observation-old vision. |
| QKV projection fusion | Paper §3.4; `reflex/layers/linear.py`, `modeling_pi05.py:fuse` | `GemmaAttention.fuse_qkv` packs loaded weights, supports grouped-query attention and optional bias, deletes separate projections. Both backbones and the joint forward support the fused representation. |
| Gate/up projection fusion | Same sources | `GemmaMLP.fuse_gate_up` packs loaded projections, retains the pretrained activation and down projection, deletes originals. Fusion is enabled after checkpoint loading. This follows the source's packed `F.linear` implementation. |
| Static ring KV storage | Paper §3.4 and Appendix B; streaming attention/manager | `StreamingPrefixCache` preallocates one tensor for layer/KV/batch/head/capacity/dimension at first frame. Slot overwrite does not shift retained KV. A stable chronological snapshot is copied once per chunk; this copy is intentional and bounded. |
| Static action suffix | Three cache zones in paper §3.1 | `StaticPrefixSnapshot` owns prefix plus reusable suffix scratch. Each flow step overwrites suffix KV rather than growing a DynamicCache or concatenating prefix/suffix per layer. Masked empty history slots keep decode shape stable before the ring fills. |
| CUDA Graph | `reflex/optimizations/single_frame_optimizer.py:CUDAGraphWrapper` | `DenoisingRunner` captures the complete denoising loop. Every chunk updates state, prefix, mask, noise and monotonic RoPE offset in fixed buffers. One bounded entry is rebuilt on shape/step/mode changes. Returned actions are cloned to prevent later replay from mutating old outputs. |
| State-conditioned precision-aware norm | Paper §3.2; `AdaRMSNorm`, `PI05SuffixEmbedding` in reference model | Optional state projection and two-layer MLP augment the timestep condition. RMS computation uses FP32 and the condition is cast to the modulation layer dtype. OpenPI's pretrained scale/shift/gate norm layout is retained because the source's single-scale layout has different tensor shapes. The new MLP output is zero initialized and requires training; enabling it alone leaves pretrained conditioning unchanged. |
| Action execution versus inference overlap | Paper Algorithm 1; reference broker/scheduler | `ReflexActionChunkBroker` is called once per environment step. With horizon 5 / overlap 2 it submits after 3 executed actions, before returning action 4, then executes actions 4 and 5 while inference runs. Optional control frequency enables adaptive overlap. |
| Future-state prediction and gripper | Paper §3.3 heuristic; user-specified gripper behavior | Generic 7D callback adds remaining deltas only and retains the previous chunk's last gripper command. LIBERO callback handles actual OSC clipping/scaling, world rotation composition, and maps that command to Panda finger targets. No raw command is added to finger joint positions. This is a target heuristic, not a learned dynamics model. |
| Episode isolation and timing | OpenPI WebSocket/client integration | Reset propagates through WebSocket and drains both broker and vision work. Timings distinguish producer compute, prefix critical-path wait, complete denoising, policy and server/client calls. Model timing materializes CPU outputs before stopping the timer. |

The target uses OpenPI's existing RoPE, image encoder, action transforms and
checkpoint normalization statistics. Optional quantized KV, similarity-based
frame skipping and other experimental source utilities are outside this module
migration. No claim of a new pretrained state-MLP or full-loop zero allocation is
made. See [usage and timing definitions](reflex_streaming.md).

## Review fixes

- Removed per-step CPU-list-to-CUDA mask creation, which is incompatible with graph
  capture; masks now use device-side positions.
- Fixed the visual projection width being hardcoded to 2048, exposed by the small
  model integration check. Real 2B checkpoint dimensions are unchanged.
- Changed timing barriers at the concurrent decode boundary to wait on the decode
  stream, preserving visual overlap. Producer timing comes from CUDA events for
  the consumed publication, rather than reporting wait time as VLM compute.
- Updated all mutable captured inputs per chunk, reset graph state when enabling
  the state MLP, and protected returned outputs from replay aliasing.
- Corrected the broker launch boundary so the observation and remaining-action
  prediction do not count the next action twice.
- Used actual LIBERO controller units and rotation composition for prediction.
- Fixed norm dtype handling, the adaptive norm's missing-weight fallback and repr.
- Pinned CUDA 12.8 wheels in the environment lock for RTX 5090 / sm120.

## Validation artifacts

All new artifacts are separate from the old full evaluation, under
`data/libero/reflex_migration_check/`.

- `all_tests.log`: **16 passed**, including CPU/CUDA fusion and static suffix
  parity, partition-cache reference/eviction and stable addresses, real small-model
  graph parity with changed observations/noise, graph reuse and output ownership,
  streaming reset/prompt change, state-MLP identity initialization, worker error
  propagation, broker scheduling, WebSocket reset, and LIBERO controller prediction.
  The FP32 numerical comparison disables TF32 only within the test fixture;
  production OpenPI precision remains unchanged.
- `checkpoint_smoke.json` / `.log`: real `pi05_libero_pytorch` weights on GPU 6,
  synthetic images, fixed noise, ten denoising steps. Standard and graph outputs
  match exactly for this sample. Graph captures remain at two after switching
  once from standard-prefix shape to streaming-prefix shape; subsequent streaming
  calls replay the same graph. State-conditioning initialization produces finite
  actions. This is not an accuracy measurement.
- `libero_smoke.jsonl` / `.log`: spatial/object smoke episodes through the WebSocket
  server and per-step asynchronous broker, with replan 5, overlap 2, window 2,
  CUDA Graph, concurrent vision and the zero-initialized state branch. Results are
  recorded separately; see the final records for success and episode metrics.
- `server.log`: temporary test server settings and inference/reset activity.
- `libero_baseline.jsonl` / `.log` and `baseline_server.log`: an additional
  standard-prefix comparison with the same environment seed, first task/episode,
  asynchronous broker and CUDA Graph, but without streaming vision. Server-side
  random noise is not paired between these rollouts; this comparison is an
  environment/behavior check, not a controlled accuracy ablation.

### Task-behavior result: not accepted

| Route | Spatial task 0, episode 1 | Object task 0, episode 1 |
| --- | --- | --- |
| Streaming window 2 + concurrent vision + graph | Failed at 220 steps; 45 inferences; 25.599 s | Failed at 280 steps; 57 inferences; 14.570 s |
| Standard prefix + graph, same broker | Succeeded at 94 steps; 20 inferences; 12.107 s | Succeeded at 142 steps; 29 inferences; 10.457 s |

Both routes completed without runtime exceptions. The streaming task failures
are a regression signal, not a successful task-quality validation. There is only
one episode per suite, and the server-side random flow noise is not paired, so
this does not establish a statistically valid accuracy difference or isolate a
cause. The deferred attention-mask equivalence issue and one-observation lag
remain relevant. These semantics were documented rather than silently changed.
`libero_comparison.json` contains both routes and their latency summaries.

For the streaming runs, uniformly excluding the first two calls (bootstrap and
its reused visual publication), mean client latency is 42.248 / 42.455 ms,
producer compute 64.352 / 64.268 ms, and ten-step expert duration 31.946 / 32.355 ms
(spatial / object). Amortized expert steps are 3.195 / 3.236 ms. These are only
smoke timing samples, not a speedup claim. The complete raw records retain cold
calls and the final inference drained at episode shutdown.

Reproduce checkpoint validation with:

```bash
CUDA_VISIBLE_DEVICES=6 JAX_PLATFORMS=cpu OPENPI_DATA_HOME=/data1/lin/openpi_cache \
.venv/bin/python scripts/check_reflex_inference.py \
  --checkpoint /data1/lin/openpi_cache/openpi-assets/checkpoints/pi05_libero_pytorch \
  --output data/libero/reflex_migration_check/checkpoint_smoke.json
```

Do not compare the cold first call to warmed graph replay as a speedup claim.
Producer and expert timings overlap in concurrent mode and cannot be summed as
end-to-end time. Dividing the expert duration by ten gives an amortized flow step,
not an individually instrumented GPU kernel latency.

## Existing evaluation and process handling

The previous full results are retained in
`data/libero/async_eval_20261107/full.jsonl`: spatial 370/500 (74%), object
426/500 (85.2%). These belong to the previous implementation and must not be
reported as accuracy of this migration. The file has 1000 episode records, 20
task records and one summary record (3,266,632 bytes).

The old server PID 2691354 on GPU 6 / port 18104 was terminated and confirmed gone.
Temporary smoke processes are stopped after validation; no long-running full
benchmark is launched by this audit.
