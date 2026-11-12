"""Deterministic prefix comparisons and paired LIBERO ablations for Reflex.

Diagnostic only: does not change the deployed model implementation.
"""

# ruff: noqa: SLF001
import argparse
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import jax
from libero.libero import benchmark
import numpy as np
from openpi_client import image_tools
from openpi_client.reflex_action_chunk_broker import ReflexActionChunkBroker
import torch

from examples.libero.main import LIBERO_DUMMY_ACTION
from examples.libero.main import _get_libero_env
from examples.libero.main import _make_state_predictor
from examples.libero.main import _quat2axisangle
from openpi.models.model import Observation
from openpi.policies import policy_config
from openpi.training import config


def element(obs, prompt):
    return {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224)
        ),
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"])
        ),
        "prompt": str(prompt),
    }


def difference(actual, reference):
    delta = np.asarray(actual, np.float64) - np.asarray(reference, np.float64)
    return {
        "max_abs": float(np.abs(delta).max()),
        "mean_abs": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt((delta**2).mean())),
    }


@torch.no_grad()
def prefix_forensics(policy, raw, out):
    model = policy._model
    inputs = policy._input_transform(copy.deepcopy(raw))
    obs = Observation.from_dict(jax.tree.map(lambda x: torch.from_numpy(np.array(x)).cuda()[None], inputs))
    images, image_masks, tokens, token_masks, state = model._preprocess_observation(obs, train=False)
    embeddings, masks, _ = model.embed_prefix(images, image_masks, tokens, token_masks)
    positions = masks.long().cumsum(1) - 1
    image_count = embeddings.shape[1] - tokens.shape[1]
    permutation = torch.cat(
        (torch.arange(image_count, embeddings.shape[1], device="cuda"), torch.arange(image_count, device="cuda"))
    )
    reorder_embeddings, reorder_masks = embeddings[:, permutation], masks[:, permutation]
    noise = torch.from_numpy(np.random.default_rng(42).normal(size=(1, 10, 32)).astype(np.float32)).cuda()
    offset = masks.long().sum(1, keepdim=True)
    lm = model.paligemma_with_expert.paligemma.language_model
    # Match sample_actions: its explicit float32 additive mask uses eager attention.
    lm.config._attn_implementation = "eager"
    result = {
        "configuration": {
            "discrete_state_input": model.config.discrete_state_input,
            "pi05": model.pi05,
            "image_physical_tokens": image_count,
            "valid_visual_tokens": int(masks[:, :image_count].sum()),
            "language_physical_tokens": tokens.shape[1],
            "valid_language_tokens": int(token_masks.sum()),
        },
        "variants": [],
    }
    arrays = {}
    prefix_caches = {}

    def decode(kv, mask, position_offset=offset):
        entry = model._denoising_runner.prepare(state, mask, kv, noise, position_offset, 10, cuda_graph=False)
        actions = model._denoising_runner.run(entry, 10, cuda_graph=False)
        return actions.cpu().numpy().copy()

    for name, emb, mask, pos, partition in [
        ("standard", embeddings, masks, positions, False),
        ("reorder_preserve_positions", reorder_embeddings, reorder_masks, positions[:, permutation], False),
        ("reorder_reassign_positions", reorder_embeddings, reorder_masks, reorder_masks.long().cumsum(1) - 1, False),
        ("partition_preserve_positions", reorder_embeddings, reorder_masks, positions[:, permutation], True),
        ("partition_reassign_positions", reorder_embeddings, reorder_masks, reorder_masks.long().cumsum(1) - 1, True),
    ]:
        allowed = mask[:, None, :, None] & mask[:, None, None, :]
        if partition:
            allowed[:, :, : tokens.shape[1], tokens.shape[1] :] = False
        attn = torch.where(allowed, 0.0, -2.3819763e38)
        kv = lm(inputs_embeds=emb, attention_mask=attn, position_ids=pos, use_cache=True).past_key_values
        prefix_caches[name] = kv
        arrays[name] = decode(kv, mask)
        reference = arrays["standard"]
        record = {"name": name, "raw_model_action_difference": difference(arrays[name][..., :7], reference[..., :7])}
        unnorm = policy._output_transform({"state": state[0].cpu().numpy(), "actions": arrays[name][0]})["actions"]
        arrays[name + "_robot_actions"] = unnorm
        record["robot_action_difference"] = difference(
            unnorm, arrays["standard_robot_actions"] if name != "standard" else unnorm
        )
        record["first_robot_action"] = unnorm[0].tolist()
        result["variants"].append(record)
    for layer in (0, 1, 8, 17):
        standard = prefix_caches["standard"][layer][1][:, :, permutation]
        partition = prefix_caches["partition_preserve_positions"][layer][1]
        result.setdefault("same_positions_value_cache_difference_by_layer", {})[str(layer)] = difference(
            partition.float().cpu(), standard.float().cpu()
        )
    model.reset_streaming_cache()
    a = (
        model.sample_actions_streaming("cuda", obs, noise=noise, streaming_window=1, concurrent_vision=False)
        .cpu()
        .numpy()
    )
    result["incremental_vs_full_partition"] = difference(a, arrays["partition_reassign_positions"])
    result["repeated_identical_observation_window1"] = []
    for chunk in range(1, 45):
        current = (
            model.sample_actions_streaming(
                "cuda", obs, noise=noise, streaming_window=1, concurrent_vision=False, cuda_graph=True
            )
            .cpu()
            .numpy()
        )
        if chunk in (1, 4, 9, 19, 29, 44):
            result["repeated_identical_observation_window1"].append(
                {
                    "chunk": chunk + 1,
                    "next_position": int(model._streaming_prefix.next_position.item()),
                    "difference_from_first": difference(current[..., :7], a[..., :7]),
                }
            )
    model.reset_streaming_cache()
    baseline = model.sample_actions("cuda", obs, noise=noise).cpu().numpy()
    altered_obs = dataclasses.replace(obs, state=obs.state + 5)
    changed = model.sample_actions("cuda", altered_obs, noise=noise).cpu().numpy()
    result["state_perturbation_before_state_mlp"] = difference(changed, baseline)
    model.enable_state_conditioning()
    baseline_with_branch = model.sample_actions("cuda", obs, noise=noise).cpu().numpy()
    changed_with_branch = model.sample_actions("cuda", altered_obs, noise=noise).cpu().numpy()
    result["zero_initialized_state_branch_vs_original"] = difference(baseline_with_branch, baseline)
    result["state_perturbation_with_zero_state_mlp"] = difference(changed_with_branch, baseline_with_branch)
    (out / "prefix_forensics.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(out / "initial_action_arrays.npz", **arrays)
    print("PREFIX_FORENSICS", json.dumps(result), flush=True)


class DiagnosticPolicy:
    def __init__(self, policy, mode):
        self.policy = policy
        self.mode = mode
        self.index = 0
        self.previous = None
        self.timings = []
        model = policy._model
        model.reset_streaming_cache()
        model._denoising_runner.entry = None
        streaming = mode.startswith("partition")
        policy._sample_actions = model.sample_actions_streaming if streaming else model.sample_actions
        policy._sample_kwargs = {"cuda_graph": True}
        if streaming:
            policy._sample_kwargs.update(
                streaming_window=1
                if mode in ("partition_fresh", "partition_window1", "partition_rebase1", "partition_fixed1")
                else 2,
                concurrent_vision=mode == "partition_fixed1",
            )

    def infer(self, obs):
        obs = copy.deepcopy(obs)
        if self.mode == "standard_lag":
            current = copy.deepcopy(obs)
            if self.previous is not None:
                for name in ("observation/image", "observation/wrist_image"):
                    obs[name] = self.previous[name]
            self.previous = current
        if self.mode == "partition_fresh":
            self.policy.reset()
        if self.mode == "partition_rebase1":
            # Diagnostic for a one-frame ring only: overwrite the evicted frame
            # at the original coordinates, retaining the same language KV.
            # Rebasing only a counter is NOT valid for retained multi-frame KV.
            cache = self.policy._model._streaming_prefix
            if cache is not None:
                cache.next_position = cache.token_masks.long().sum(dim=1, keepdim=True)
        noise = np.random.default_rng(42 + self.index).normal(size=(10, 32)).astype(np.float32)
        self.index += 1
        result = self.policy.infer(obs, noise=noise)
        self.timings.append(result.get("policy_timing", {}).copy())
        return result

    def reset(self):
        self.policy.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 6, 8])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--skip-forensics", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "standard",
            "partition_fresh",
            "partition_history",
            "standard_lag",
            "partition_window1",
            "partition_rebase1",
            "partition_fixed1",
        ],
        default=["standard", "partition_fixed1"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    train = config.get_config("pi05_libero")
    train = dataclasses.replace(train, model=dataclasses.replace(train.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(train, args.checkpoint)
    policy.enable_inference_profiling()
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    records = []
    try:
        for task_id, trial in ((task, trial) for task in args.tasks for trial in range(args.trials)):
            task = suite.get_task(task_id)
            initial = suite.get_task_init_states(task_id)[trial]
            for mode in args.modes:
                env, prompt = _get_libero_env(task, 256, 7)
                wrapper = None
                broker = None
                try:
                    np.random.seed(7)
                    obs = env.reset()
                    obs = env.set_init_state(initial)
                    for _ in range(10):
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    raw = element(obs, prompt)
                    if not args.skip_forensics and not (args.output / "prefix_forensics.json").exists():
                        prefix_forensics(policy, raw, args.output)
                    initial_hash = hashlib.sha256(
                        raw["observation/image"].tobytes()
                        + raw["observation/wrist_image"].tobytes()
                        + raw["observation/state"].tobytes()
                    ).hexdigest()
                    wrapper = DiagnosticPolicy(policy, mode)
                    broker = ReflexActionChunkBroker(
                        wrapper,
                        5,
                        overlap_steps=2,
                        predict_future_state=True,
                        state_key="observation/state",
                        state_predictor=_make_state_predictor(env),
                    )
                    start = time.perf_counter()
                    done = False
                    action_trace = []
                    for step in range(220):  # noqa: B007 - used after loop for the executed step count
                        action = broker.infer(element(obs, prompt))["actions"]
                        action_trace.append(np.asarray(action).copy())
                        obs, _, done, _ = env.step(np.asarray(action).tolist())
                        if done:
                            break
                    broker.reset()
                    record = {
                        "task_id": task_id,
                        "task": str(prompt),
                        "episode": trial + 1,
                        "mode": mode,
                        "success": bool(done),
                        "sim_steps": step + 1,
                        "wall_s": time.perf_counter() - start,
                        "inferences": wrapper.index,
                        "flow_noise_seed": "42+inference_index",
                        "environment_seed": 7,
                        "initial_observation_sha256": initial_hash,
                        "inference_timings": wrapper.timings,
                        "broker_records": broker.inference_records,
                        "error": None,
                    }
                    np.save(
                        args.output / f"actions_task{task_id}_trial{trial + 1}_{mode}.npy", np.asarray(action_trace)
                    )
                except Exception as exc:
                    record = {"task_id": task_id, "mode": mode, "error": repr(exc)}
                    raise
                finally:
                    if broker is not None:
                        broker.close()
                    env.close()
                records.append(record)
                with (args.output / "ablations.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print("ABLATION", json.dumps(record), flush=True)
        (args.output / "ablations.json").write_text(json.dumps(records, indent=2) + "\n")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
