"""CPU-only audit: imports, future config, checkpoint hashes and abstract shapes."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import flax.nnx as nnx
from flax import traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp

from openpi.models import pi0
from openpi.policies import policy_config
from openpi.policies.future_policy import CHECKPOINT, CONFIG_NAME, verify_checkpoint
from openpi.training import config
from openpi_client import future_action_chunk_broker, websocket_client_policy

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=str(CHECKPOINT))
args = parser.parse_args()
cfg = config.get_config(CONFIG_NAME)
assert cfg.model.action_horizon == 10 and cfg.model.action_context_horizon == 9
assert cfg.model.discrete_state_input is False
assert not any("async_context" in name for name in config._CONFIGS_DICT)
for module in [pi0, policy_config, future_action_chunk_broker, websocket_client_policy]:
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
checkpoint, manifest = verify_checkpoint(args.checkpoint)
with ocp.PyTreeCheckpointer() as checkpointer:
    metadata = checkpointer.metadata(checkpoint / "params")["params"]
got = traverse_util.flatten_dict(metadata)
if all(key[-1] == "value" for key in got):
    got = {key[:-1]: value for key, value in got.items()}
abstract_model = nnx.eval_shape(cfg.model.create, jax.random.key(0))
expected = traverse_util.flatten_dict(nnx.state(abstract_model).to_pure_dict())
assert set(got) == set(expected), dict(missing=list(set(expected)-set(got)),extra=list(set(got)-set(expected)))
for key in expected:
    assert tuple(got[key].shape) == tuple(expected[key].shape), key
assert any("action_context_in_proj" in key for key in got)
from openpi.shared import nnx_utils
sample_shape = jax.eval_shape(
    nnx_utils.module_jit(abstract_model.sample_actions),
    jax.random.key(7), cfg.model.fake_obs(), num_steps=10)
assert sample_shape.shape == (1, 10, 32)
obs_spec, action_spec = cfg.model.inputs_spec()
assert obs_spec.action_context.shape == (1, 9, 32)
assert action_spec.shape == (1, 10, 32)
libero_action_dim = 7
for length in [0, 2]:
    from openpi import transforms
    packed = transforms.PackActionContext(9, 32)({"action_context": np.ones((length, libero_action_dim), dtype=np.float32)})
    assert packed["action_context"].shape == (9, 32)
    assert int(packed["action_context_mask"].sum()) == length
result = dict(passed=True, backend=jax.default_backend(), config=CONFIG_NAME,
    checkpoint=str(checkpoint), manifest_files=len(manifest["files"]),
    all_manifest_sha256_verified=True, full_parameter_keys_and_shapes_match=True,
    parameter_leaves=len(got), sampler_abstract_output_shape=list(sample_shape.shape), python=sys.executable,
    module_paths={m.__name__:m.__file__ for m in [pi0,policy_config,future_action_chunk_broker,websocket_client_policy]},
    prediction_steps=10,execute_steps=5,overlap_steps=2,pre_inference_steps=3,
    legacy_context_config_removed=True, gpu_evaluation_started=False)
(root / "FUTURE_VALIDATION.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
