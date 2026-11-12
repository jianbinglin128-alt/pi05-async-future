"""Evaluate a future-chunk policy; output begins AFTER the clean fixed context."""

import concurrent.futures
import dataclasses
import json
import logging
import pathlib
import time

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from scipy.spatial.transform import Rotation
import tyro

from examples.libero.main import LIBERO_DUMMY_ACTION
from examples.libero.main import LIBERO_ENV_RESOLUTION
from examples.libero.main import _quat2axisangle


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 18005
    task_suite_name: str = "libero_spatial"
    limit_tasks: int = 1
    num_trials_per_task: int = 1
    pre_inference_steps: int = 3
    async_execute_steps: int = 2
    executed_chunk_steps: int = 5
    num_steps_wait: int = 10
    resize_size: int = 224
    seed: int = 7
    metrics_out_path: str = "data/libero/async_context_eval/metrics.jsonl"
    save_video: bool = False
    video_out_path: str = "data/libero/async_context_eval/videos"


def _make_env(task, seed):
    task_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def _observation(obs, prompt, resize_size, context=None):
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_image, resize_size, resize_size))
    result = {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(prompt),
    }
    if context is not None:
        result["action_context"] = np.asarray(context, dtype=np.float32)
    return result


def _infer(client, observation):
    started = time.perf_counter()
    result = client.infer(observation)
    return np.asarray(result["actions"]), time.perf_counter() - started


def _execute(env, obs, actions, max_steps, step_count, done):
    for action in actions:
        if step_count >= max_steps or done:
            break
        obs, _, done, _ = env.step(np.asarray(action).tolist())
        step_count += 1
    return obs, step_count, done


def eval_libero(args: Args):
    if args.pre_inference_steps != 3 or args.async_execute_steps != 2 or args.executed_chunk_steps != 5:
        raise ValueError("This evaluation path is configured for 3 synchronous + 2 asynchronous action steps")
    np.random.seed(args.seed)
    max_steps_by_suite = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if args.task_suite_name not in max_steps_by_suite:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    metrics_path = pathlib.Path(args.metrics_out_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_count = min(suite.n_tasks, args.limit_tasks)
    successes = 0
    episodes = 0
    client = WebsocketClientPolicy(args.host, args.port)
    if client.get_server_metadata().get("action_target_alignment") != "after_context":
        raise ValueError("This evaluator requires an after_context policy checkpoint")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="async-policy") as executor, metrics_path.open(
        "w", encoding="utf-8"
    ) as metrics:
        for task_id in range(task_count):
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            for trial in range(args.num_trials_per_task):
                env = _make_env(task, args.seed)
                episode_start = time.perf_counter()
                obs = env.reset()
                obs = env.set_init_state(initial_states[trial])
                done = False
                step_count = 0
                inference_ms = []
                hidden_ms = []
                try:
                    for _ in range(args.num_steps_wait):
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        step_count += 1
                    max_steps = max_steps_by_suite[args.task_suite_name] + args.num_steps_wait

                    # Bootstrap synchronously, then start async inference after three actions.
                    actions, latency = _infer(
                        client, _observation(obs, task.language, args.resize_size)
                    )
                    if actions.shape[0] != 10:
                        raise ValueError(f"Expected a 10-action chunk, got {actions.shape}")
                    inference_ms.append(latency * 1000)
                    obs, step_count, done = _execute(
                        env, obs, actions[: args.pre_inference_steps], max_steps, step_count, done
                    )
                    overlap_actions = actions[
                        args.pre_inference_steps : args.pre_inference_steps + args.async_execute_steps
                    ]

                    while step_count < max_steps and not done:
                        observation = _observation(obs, task.language, args.resize_size, overlap_actions)
                        started = time.perf_counter()
                        future = executor.submit(_infer, client, observation)
                        obs, step_count, done = _execute(
                            env, obs, overlap_actions, max_steps, step_count, done
                        )
                        overlap_elapsed = time.perf_counter() - started
                        if done or step_count >= max_steps:
                            future.cancel()
                            break
                        actions, latency = future.result()
                        inference_ms.append(latency * 1000)
                        hidden_ms.append(min(latency, overlap_elapsed) * 1000)
                        if actions.shape[0] != 10:
                            raise ValueError(f"Expected a 10-action chunk, got {actions.shape}")
                        # Every predicted action is AFTER overlap_actions. Execute
                        # the new chunk from index zero; do not discard two steps.
                        next_chunk = actions[: args.executed_chunk_steps]
                        obs, step_count, done = _execute(
                            env,
                            obs,
                            next_chunk[: args.pre_inference_steps],
                            max_steps,
                            step_count,
                            done,
                        )
                        overlap_actions = next_chunk[
                            args.pre_inference_steps : args.pre_inference_steps + args.async_execute_steps
                        ]

                    success = bool(done)
                    record = {
                        "type": "episode",
                        "suite": args.task_suite_name,
                        "task_id": task_id,
                        "task": task.language,
                        "episode": trial + 1,
                        "success": success,
                        "sim_steps": max(0, step_count - args.num_steps_wait),
                        "inference_count": len(inference_ms),
                        "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else None,
                        "mean_hidden_ms": float(np.mean(hidden_ms)) if hidden_ms else 0.0,
                        "episode_wall_s": time.perf_counter() - episode_start,
                        "error": None,
                    }
                except Exception as exc:
                    logging.exception("LIBERO episode failed: %s task %s trial %s", args.task_suite_name, task_id, trial)
                    record = {
                        "type": "episode",
                        "suite": args.task_suite_name,
                        "task_id": task_id,
                        "task": task.language,
                        "episode": trial + 1,
                        "success": False,
                        "sim_steps": max(0, step_count - args.num_steps_wait),
                        "inference_count": len(inference_ms),
                        "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else None,
                        "mean_hidden_ms": float(np.mean(hidden_ms)) if hidden_ms else 0.0,
                        "episode_wall_s": time.perf_counter() - episode_start,
                        "error": repr(exc),
                    }
                finally:
                    env.close()

                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                episodes += 1
                successes += int(record["success"])
                logging.info("EPISODE_METRICS_JSON %s", json.dumps(record))

        summary = {
            "type": "summary",
            "suite": args.task_suite_name,
            "tasks": task_count,
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else None,
        }
        metrics.write(json.dumps(summary) + "\n")
        metrics.flush()
        logging.info("SUMMARY_JSON %s", json.dumps(summary))
    client._ws.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(eval_libero)
