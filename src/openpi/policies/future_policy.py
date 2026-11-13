"""Load only the future-target JAX checkpoint with strict parameter matching."""
import hashlib
import json
from pathlib import Path

import jax

from openpi.policies import policy_config
from openpi.training import config

CONFIG_NAME = "pi05_libero_async_future"
CHECKPOINT = Path("/data1/lin/Apaper2/model/pi05_libero_async_future")
REPO_ID = "Dream-LIN/pi05_libero_async_future"


def verify_checkpoint(checkpoint=CHECKPOINT, *, full_hash=True):
    checkpoint = Path(checkpoint).expanduser().resolve()
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    if manifest.get("repo_id") != REPO_ID or manifest.get("format") != "Orbax JAX":
        raise ValueError("Expected the Dream-LIN/pi05_libero_async_future Orbax JAX checkpoint")
    if (checkpoint / "model.safetensors").exists():
        raise ValueError("Future entry point requires the JAX checkpoint")
    paths = {item["path"] for item in manifest["files"]}
    if not {"params/_METADATA", "assets/physical-intelligence/libero/norm_stats.json"} <= paths:
        raise ValueError("Incomplete checkpoint manifest")
    for item in manifest["files"]:
        path = (checkpoint / item["path"]).resolve()
        if not path.is_relative_to(checkpoint) or not path.is_file():
            raise ValueError(f"Invalid or missing manifest file: {item['path']}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Weight size mismatch: {item['path']}")
        if full_hash or item["path"] in {"params/_METADATA", "assets/physical-intelligence/libero/norm_stats.json"}:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Weight SHA256 mismatch: {item['path']}")
    return checkpoint, manifest


def create_future_policy(checkpoint=CHECKPOINT, *, seed=7):
    checkpoint, manifest = verify_checkpoint(checkpoint)
    cfg = config.get_config(CONFIG_NAME)
    assert cfg.model.pi05 and cfg.model.action_horizon == 10
    assert cfg.model.action_context_horizon == 9 and cfg.model.discrete_state_input is False
    policy = policy_config.create_trained_policy(cfg, checkpoint, sample_kwargs={"num_steps": 10})
    assert not policy._is_pytorch_model
    policy._rng = jax.random.key(seed)
    policy._metadata = dict(policy.metadata, config=CONFIG_NAME, checkpoint=str(checkpoint),
        checkpoint_repo_id=manifest["repo_id"], checkpoint_step=manifest["checkpoint_step"],
        backend="jax", sampler="monolithic_jit", action_target_alignment="future",
        predicted_steps=10, executed_steps=5, overlap_steps=2, pre_inference_steps=3,
        num_denoise_steps=10, policy_seed=seed)
    return policy
