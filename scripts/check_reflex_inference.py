"""Check eager/graph parity and streaming with a real PyTorch checkpoint.

Uses synthetic observations, so this checks integration, not task success.
Run from the repository root with the same environment as serve_policy.py.
"""

import argparse
import dataclasses
import json
from pathlib import Path
import time

import numpy as np
import torch

from openpi.policies import policy_config
from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    train = config.get_config(args.config)
    train = dataclasses.replace(train, model=dataclasses.replace(train.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(train, args.checkpoint)
    policy.enable_inference_profiling()
    obs = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.full((224, 224, 3), 127, dtype=np.uint8),
        "observation/state": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.02, -0.02], np.float32),
        "prompt": "pick up the black bowl and place it on the plate",
    }
    noise = (
        np.random.default_rng(42).normal(size=(train.model.action_horizon, train.model.action_dim)).astype(np.float32)
    )
    results = []
    modes = [
        ("standard", {}),
        ("graph", {"cuda_graph": True}),
        ("graph_repeat", {"cuda_graph": True}),
        *[
            (name, {"streaming_window": 2, "cuda_graph": True, "concurrent_vision": True})
            for name in ("streaming", "streaming_repeat", "streaming_repeat2")
        ],
    ]
    try:
        model = policy._model  # noqa: SLF001
        for mode, kwargs in modes:
            # This test deliberately switches routes without loading the 2B model again.
            policy._sample_kwargs = kwargs  # noqa: SLF001
            policy._sample_actions = (  # noqa: SLF001
                model.sample_actions_streaming if "streaming_window" in kwargs else model.sample_actions
            )
            started = time.perf_counter()
            result = policy.infer(obs, noise=noise)
            record = {
                "mode": mode,
                "wall_ms": (time.perf_counter() - started) * 1000,
                "shape": list(result["actions"].shape),
                "finite": bool(np.isfinite(result["actions"]).all()),
                "timing": result["policy_timing"],
                "captures": model._denoising_runner.captures,  # noqa: SLF001
                "replays": model._denoising_runner.replays,  # noqa: SLF001
            }
            assert record["finite"]
            if mode == "standard":
                baseline = result["actions"].copy()
            if mode.startswith("graph"):
                record["max_abs_diff_from_standard"] = float(np.max(np.abs(result["actions"] - baseline)))
                np.testing.assert_allclose(result["actions"], baseline, atol=2e-4, rtol=2e-4)
            results.append(record)
            print(json.dumps(record), flush=True)
        policy.enable_state_conditioning()
        model.reset_streaming_cache()
        finite = bool(np.isfinite(policy.infer(obs, noise=noise)["actions"]).all())
        assert finite
        results.append({"state_conditioning": "enabled_zero_initialized", "finite": finite})
    finally:
        policy.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
