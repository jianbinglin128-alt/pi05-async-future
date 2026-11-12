import collections
import dataclasses
import json
import logging
import math
import pathlib
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.reflex_action_chunk_broker import ReflexActionChunkBroker
from scipy.spatial.transform import Rotation
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    async_inference: bool = False
    overlap_steps: int = 2
    metrics_out_path: str = "data/libero/async_metrics.jsonl"
    save_video: bool = True
    limit_tasks: int | None = None
    resume: bool = False

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    max_steps_by_suite = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    suites = [name.strip() for name in args.task_suite_name.split(",")]
    if any(name not in max_steps_by_suite for name in suites):
        raise ValueError(f"Unknown task suite in {args.task_suite_name}")
    if args.async_inference and not 1 <= args.overlap_steps < args.replan_steps:
        raise ValueError("Require 1 <= overlap_steps < replan_steps")
    metrics_path = pathlib.Path(args.metrics_out_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    previous_records = []
    if args.resume and metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as existing_file:
            previous_records = [json.loads(line) for line in existing_file if line.strip()]
    if args.save_video:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    prior_episodes = [
        record for record in previous_records if record["type"] == "episode" and record["suite"] in suites
    ]
    completed_tasks = {
        (record["suite"], record["task_id"])
        for record in previous_records
        if record["type"] == "task" and record["suite"] in suites
    }
    total_episodes = len(prior_episodes)
    total_successes = sum(record["success"] for record in prior_episodes)
    with metrics_path.open("a" if args.resume else "w", encoding="utf-8") as metrics_file:
        for suite_name in suites:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            task_count = min(suite.n_tasks, args.limit_tasks or suite.n_tasks)
            logging.info("Task suite: %s (%s tasks)", suite_name, task_count)
            for task_id in tqdm.tqdm(range(task_count), desc=suite_name):
                if (suite_name, task_id) in completed_tasks:
                    continue
                task = suite.get_task(task_id)
                initial_states = suite.get_task_init_states(task_id)
                env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
                episode_records = [
                    record
                    for record in prior_episodes
                    if record["suite"] == suite_name and record["task_id"] == task_id
                ]
                completed_episodes = {record["episode"] for record in episode_records}
                try:
                    for episode_idx in range(args.num_trials_per_task):
                        if episode_idx + 1 in completed_episodes:
                            continue
                        episode_start = time.perf_counter()
                        obs = env.reset()
                        obs = env.set_init_state(initial_states[episode_idx])
                        broker = (
                            ReflexActionChunkBroker(
                                client,
                                args.replan_steps,
                                overlap_steps=args.overlap_steps,
                                predict_future_state=True,
                                state_key="observation/state",
                                state_predictor=_make_state_predictor(env),
                            )
                            if args.async_inference
                            else None
                        )
                        action_plan = collections.deque()
                        synchronous_records = []
                        replay_images = []
                        done = False
                        error = None
                        t = 0
                        try:
                            while t < max_steps_by_suite[suite_name] + args.num_steps_wait:
                                if t < args.num_steps_wait:
                                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                                    t += 1
                                    continue
                                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                                img = image_tools.convert_to_uint8(
                                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                                )
                                wrist_img = image_tools.convert_to_uint8(
                                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                                )
                                if args.save_video:
                                    replay_images.append(img)
                                if broker is not None or not action_plan:
                                    element = {
                                        "observation/image": img,
                                        "observation/wrist_image": wrist_img,
                                        "observation/state": np.concatenate(
                                            (
                                                obs["robot0_eef_pos"],
                                                _quat2axisangle(obs["robot0_eef_quat"]),
                                                obs["robot0_gripper_qpos"],
                                            )
                                        ),
                                        "prompt": str(task_description),
                                    }
                                    if broker is not None:
                                        action = broker.infer(element)["actions"]
                                    else:
                                        start = time.perf_counter()
                                        result = client.infer(element)
                                        synchronous_records.append(
                                            {
                                                **{
                                                    name: result.get("policy_timing", {}).get(name)
                                                    for name in (
                                                        "vlm_total_ms",
                                                        "vision_encoder_ms",
                                                        "llm_prefill_ms",
                                                        "language_init_ms",
                                                        "visual_prefill_ms",
                                                    )
                                                },
                                                "client_infer_ms": (time.perf_counter() - start) * 1000,
                                                "server_infer_ms": result.get("server_timing", {}).get("infer_ms"),
                                                "llm_ms": result.get("policy_timing", {}).get("llm_ms"),
                                                "action_expert_ms": result.get("policy_timing", {}).get(
                                                    "action_expert_ms"
                                                ),
                                            }
                                        )
                                        action_chunk = result["actions"]
                                        if len(action_chunk) < args.replan_steps:
                                            raise ValueError("Model action horizon is shorter than replan_steps")
                                        action_plan.extend(action_chunk[: args.replan_steps])
                                if broker is None:
                                    action = action_plan.popleft()
                                obs, _, done, _ = env.step(np.asarray(action).tolist())
                                t += 1
                                if done:
                                    break
                        except Exception as exc:
                            error = repr(exc)
                            logging.exception(
                                "LIBERO episode failed: %s task %s episode %s", suite_name, task_id, episode_idx
                            )
                        finally:
                            if broker is not None:
                                broker.reset()
                                broker.close()
                        episode_wall_s = time.perf_counter() - episode_start
                        records = broker.inference_records if broker is not None else synchronous_records
                        entry = {
                            "type": "episode",
                            "suite": suite_name,
                            "task_id": task_id,
                            "task": task_description,
                            "episode": episode_idx + 1,
                            "success": bool(done),
                            "sim_steps": max(0, t - args.num_steps_wait),
                            "episode_wall_s": episode_wall_s,
                            "inference_count": len(records),
                            "client_infer_ms": [r["client_infer_ms"] for r in records],
                            "server_infer_ms": [r["server_infer_ms"] for r in records],
                            "llm_ms": [r["llm_ms"] for r in records],
                            "action_expert_ms": [r["action_expert_ms"] for r in records],
                            **{
                                name: [r.get(name) for r in records]
                                for name in (
                                    "vlm_total_ms",
                                    "vision_encoder_ms",
                                    "llm_prefill_ms",
                                    "language_init_ms",
                                    "visual_prefill_ms",
                                )
                            },
                            "async_hidden_ms": broker.overlap_hidden_ms if broker is not None else 0.0,
                            "switch_wait_ms": broker.switch_wait_ms if broker is not None else 0.0,
                            "error": error,
                        }
                        episode_records.append(entry)
                        _write_metric(metrics_file, entry)
                        logging.info("EPISODE_METRICS_JSON %s", json.dumps(entry))
                        total_episodes += 1
                        total_successes += int(done)
                        if args.save_video and replay_images:
                            suffix = "success" if done else "failure"
                            task_segment = task_description.replace(" ", "_")
                            imageio.mimwrite(
                                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                                replay_images,
                                fps=10,
                            )
                finally:
                    env.close()
                task_record = {
                    "type": "task",
                    "suite": suite_name,
                    "task_id": task_id,
                    "task": task_description,
                    "episodes": len(episode_records),
                    "successes": sum(record["success"] for record in episode_records),
                    "success_rate": sum(record["success"] for record in episode_records) / len(episode_records),
                    "inference_count": sum(record["inference_count"] for record in episode_records),
                    "total_episode_wall_s": sum(record["episode_wall_s"] for record in episode_records),
                    "mean_episode_wall_s": sum(record["episode_wall_s"] for record in episode_records)
                    / len(episode_records),
                    "async_hidden_ms": sum(record["async_hidden_ms"] for record in episode_records),
                    "switch_wait_ms": sum(record["switch_wait_ms"] for record in episode_records),
                }
                for field in (
                    "client_infer_ms",
                    "server_infer_ms",
                    "llm_ms",
                    "action_expert_ms",
                    "vlm_total_ms",
                    "vision_encoder_ms",
                    "llm_prefill_ms",
                    "language_init_ms",
                    "visual_prefill_ms",
                ):
                    values = [value for record in episode_records for value in record[field] if value is not None]
                    task_record[f"mean_{field}"] = sum(values) / len(values) if values else None
                _write_metric(metrics_file, task_record)
                logging.info("TASK_METRICS_JSON %s", json.dumps(task_record))
        _write_metric(
            metrics_file,
            {
                "type": "summary",
                "episodes": total_episodes,
                "successes": total_successes,
                "success_rate": total_successes / total_episodes if total_episodes else None,
            },
        )


def _make_state_predictor(env):
    """Propagate OSC commands in controller units; retain the final gripper target.

    This predicts commanded targets, not contact dynamics or tracking errors.
    Rotation updates use the same left multiplication as robosuite OSC.
    """
    controller = env.env.robots[0].controller
    low, high = np.array(controller.input_min), np.array(controller.input_max)
    output_low, output_high = np.array(controller.output_min), np.array(controller.output_max)
    scale = np.abs(output_high - output_low) / np.abs(high - low)
    input_center, output_center = (high + low) / 2, (output_high + output_low) / 2

    def predict(state, remaining, last_action):
        if state.shape != (8,) or remaining.ndim != 2 or remaining.shape[1] != 7:
            raise ValueError("LIBERO prediction requires 8D observed state and 7D actions")
        orientation = Rotation.from_rotvec(state[3:6])
        for action in remaining:
            delta = (np.clip(action[:6], low, high) - input_center) * scale + output_center
            state[:3] += delta[:3]
            orientation = Rotation.from_rotvec(delta[3:6]) * orientation
        state[3:6] = orientation.as_rotvec()
        # Panda observations use two finger qpos; commands use one open/close
        # value (-1 open, +1 close). Never put a +/-1 command into a joint qpos.
        if last_action[6] != 0:
            state[6:] = [0.04, -0.04] if last_action[6] < 0 else [0.0, 0.0]
        return state

    return predict


def _write_metric(metrics_file, record):
    metrics_file.write(json.dumps(record) + "\n")
    metrics_file.flush()


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
