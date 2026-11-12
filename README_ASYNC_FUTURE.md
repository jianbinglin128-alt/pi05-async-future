# Async future-chunk experiment

Server root: `/home/lin/data/openpi`; SSH user: `lin`.

The previous training process was stopped at approximately logged step 10560.
Its code, step-5000/10000 checkpoints, and 30/30 Spatial evaluation remain intact.

## Definition

New repository: `/home/lin/data/openpi/repo-async-future`

Configuration: `pi05_libero_async_future`

Initialization: official `/home/lin/data/openpi/checkpoints/pi05_libero/params`.
Neither old trained action-expert weights nor old optimizer state are restored.

For observation O_t, sample K uniformly from 0..9. Clean fixed context is
`a[t:t+K]`; the ten future target actions are `a[t+K:t+K+10]`.
Source windows have 19 actions and are filtered against episode boundaries.
Both context and targets come from recorded LIBERO demonstrations; this run
does not generate teacher labels or use previous policy predictions as context.

The existing suffix contains nine masked context slots and ten generated slots.
Context is not noised. Flow-matching loss applies only to the ten generated
future slots; it does not reconstruct context. Action dimensions retain the
existing 7-to-32 padding and 32-dimensional loss convention.

Train action expert, action projections, time MLP and context projection; freeze
vision/language experts. Keep discrete_state_input=False. Use all four LIBERO
suites, batch 12, physical GPU 1/2, FSDP 2, 8 workers, 30000 steps, AdamW,
gradient clipping 1, EMA .999, warmup 1000, peak LR 1e-5, cosine end LR 1e-6.

## Runtime isolation

Reuse the previously created private lin Python environment; no shared or
private environment packages/.pth files are changed. Source the future wrapper
to choose the new repository through PYTHONPATH (including worker processes):

```bash
. /home/lin/data/openpi/scripts/env_future.sh
```

## Logs and output

- Supervisor: `/home/lin/data/openpi/logs/future_run_status.json`
- Verification: `/home/lin/data/openpi/logs/verify_future.json`
- Smoke: `/home/lin/data/openpi/logs/future_smoke_20261111.log`
- Formal training: `/home/lin/data/openpi/logs/future_full_20261111.log`
- Formal checkpoints: `/home/lin/data/openpi/outputs/pi05_libero_async_future/future_full_20261111/`
- Code/supervision manifest: `/home/lin/data/openpi/repo-async-future/ASYNC_FUTURE_MANIFEST.json`

Save every 1000 steps. Retention remains latest plus multiples of 5000; early
1000/2000/etc checkpoints will be removed when replaced unless copied elsewhere.

```bash
tail -f /home/lin/data/openpi/logs/future_full_20261111.log
cat /home/lin/data/openpi/logs/future_run_status.json
```

The launch supervisor verifies alignment and real normalized data, then requires
a successful two-step training run and finalized checkpoint before full training.

## Future evaluation

Use `examples/libero/eval_async_future.py` with this new model/config. It checks
server metadata `action_target_alignment=after_context`. With 3+2 scheduling,
execute the two fixed queued actions during inference, then take new predictions
`actions[:5]`. Do not discard the first two new predictions. The prior
`eval_async_context.py` remains the legacy overlapping-output evaluator.

Example once the new experiment has a finalized checkpoint (replace STEP):

```bash
. /home/lin/data/openpi/scripts/env_future.sh
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python scripts/serve_policy.py \
  --port 18111 policy:checkpoint --policy.config pi05_libero_async_future \
  --policy.dir /home/lin/data/openpi/outputs/pi05_libero_async_future/future_full_20261111/STEP
```

In another shell:

```bash
. /home/lin/data/openpi/scripts/env_libero_lin.sh
. /home/lin/data/openpi/scripts/env_future.sh
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu \
  /home/lin/data/openpi/libero-eval-env/bin/python examples/libero/eval_async_future.py \
  --args.port 18111 --args.task-suite-name libero_spatial \
  --args.limit-tasks 10 --args.num-trials-per-task 3 \
  --args.metrics-out-path /home/lin/data/openpi/evaluations/future_STEP_spatial.jsonl
```
