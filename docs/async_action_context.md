# Asynchronous action context

The `pi05_libero_async_context` configuration trains the 10-step synchronous action chunk conditioned on the unexecuted prefix of the previous chunk. Context length varies from 0 to 9 at training time; the model always reserves nine context slots and masks unused slots.

For an observation at timestep `t`, the supervised target remains the ordinary synchronous chunk `a[t:t+10]`, regardless of context length. Training samples `K` uniformly from 0 to 9 and copies the first `K` actions into the context without changing their values; these represent the actions that remain queued from the previous chunk and are executing while inference runs. They are not used as target replacements. Samples crossing an episode boundary are excluded. Context values are normalized with the same action statistics as targets, then padded to nine slots with a boolean validity mask.

In the action expert, the context block follows the image/language prefix and precedes the noisy target-action block. Context tokens attend to the prefix and valid context tokens. Target tokens attend to the prefix, valid context, and the whole target block. The flow-matching loss remains on the ten target actions only.

At inference, include `action_context` in the Libero observation dict as the unnormalized queued action array with shape `[K, 7]`, where `0 <= K <= 9`. These are the previous chunk's remaining actions, in execution order. The controller must keep executing them unchanged while the new chunk is computed. Once it is ready, discard the new chunk's first `K` overlapping predictions and enqueue its remaining actions. `LiberoInputs`, normalization, padding, and fixed-slot packing prepare the model input. When the queue is empty, omit `action_context`; all nine slots are masked automatically.

The action expert, action input/output projections, timestep MLP, and new context projection are trainable. The image encoder and language expert are frozen. Existing checkpoints are loaded into matching parameters; the new context projection starts from its model initialization.

Example smoke launch on the prepared A6000 setup:

```bash
source ~/data/openpi/scripts/env_a6000.sh
python scripts/train.py pi05_libero_async_context \
  --exp-name=async-context-smoke \
  --assets-base-dir="$OPENPI_ROOT/assets" \
  --checkpoint-base-dir="$OPENPI_ROOT/outputs" \
  --weight-loader.params-path="$OPENPI_ROOT/checkpoints/pi05_base/params" \
  --batch-size=3 --num-workers=0 --num-train-steps=1 \
  --log-interval=1 --save-interval=1 --no-wandb-enabled
```
