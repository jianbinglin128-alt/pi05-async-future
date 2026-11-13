"""LIBERO future-only evaluation: predict 10, execute 5, overlap 2."""
import argparse
import json
import logging
import math
from pathlib import Path
import time

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from openpi_client.future_action_chunk_broker import FutureActionChunkBroker

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)


def _quat2axisangle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def _get_libero_env(task, resolution, seed):
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task.language


def _observation(obs, prompt):
    result = {}
    for source, target in [("agentview_image", "observation/image"),
                           ("robot0_eye_in_hand_image", "observation/wrist_image")]:
        image = np.ascontiguousarray(obs[source][::-1, ::-1])
        result[target] = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    result["observation/state"] = np.concatenate((obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    result["prompt"] = str(prompt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--suite", choices=[*MAX_STEPS, "all"], default="libero_spatial")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--limit-tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="data/libero/future_chunk10_exec5_overlap2")
    args = parser.parse_args()
    if args.trials < 1 or not 1 <= args.limit_tasks <= 10:
        parser.error("trials must be positive and limit-tasks must be 1..10")
    logging.basicConfig(level=logging.INFO, force=True)
    np.random.seed(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ["run.json", "episodes.jsonl"]):
        raise FileExistsError(f"Use a fresh output directory: {output}")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    expected = dict(config="pi05_libero_async_future", action_target_alignment="future",
                    predicted_steps=10, executed_steps=5, overlap_steps=2, pre_inference_steps=3)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"This evaluator requires the future 10/5/2 server: {metadata}")
    (output / "run.json").write_text(json.dumps(dict(
        args=vars(args), server_metadata=metadata, schedule=expected,
        timing=dict(client="before/after client.infer; input already prepared",
                    action="before/after env.step; list conversion excluded",
                    rollout="after reset and 10 settling steps, before first observation processing, through final env.step return",
                    excluded="three warmup requests, reset, settling, terminal pending request drain, disk writes")), indent=2))
    episodes = []
    warmed = False
    suites = list(MAX_STEPS) if args.suite == "all" else [args.suite]
    try:
        for suite_name in suites:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            for task_id in range(min(args.limit_tasks, suite.n_tasks)):
                task = suite.get_task(task_id)
                states = suite.get_task_init_states(task_id)
                if len(states) < args.trials:
                    raise ValueError(f"Only {len(states)} initial states for task {task_id}")
                env, description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
                try:
                    for trial in range(args.trials):
                        env.reset()
                        obs = env.set_init_state(states[trial])
                        for _ in range(10):
                            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                        client.reset()
                        if not warmed:
                            warm_obs = _observation(obs, description)
                            warm_obs["action_context"] = np.zeros((0, 7), dtype=np.float32)
                            warm = []
                            for index in range(3):
                                before = time.perf_counter_ns()
                                response = client.infer(warm_obs)
                                after = time.perf_counter_ns()
                                assert np.asarray(response["actions"]).shape == (10, 7)
                                warm.append(dict(index=index, duration_ns=after-before))
                            (output / "warmup.json").write_text(json.dumps(warm, indent=2))
                            client.reset()
                            warmed = True
                        broker = FutureActionChunkBroker(client)
                        actions = []
                        done = False
                        error = None
                        start = time.perf_counter_ns()
                        terminal = None
                        try:
                            for step in range(MAX_STEPS[suite_name]):
                                action = broker.infer(_observation(obs, description))["actions"].tolist()
                                before = time.perf_counter_ns()
                                obs, _, done, _ = env.step(action)
                                after = time.perf_counter_ns()
                                actions.append(dict(step=step, start_ns=before, end_ns=after,
                                                    duration_ns=after-before))
                                if done:
                                    break
                            terminal = after
                        except Exception as exception:
                            error = repr(exception)
                            terminal = time.perf_counter_ns()
                            raise
                        finally:
                            # A terminal pending prefetch may finish here; its drain
                            # is excluded from rollout time and identified in logs.
                            if terminal is None:
                                terminal = time.perf_counter_ns()
                            try:
                                broker.reset()
                            finally:
                                broker.close()
                            entry = dict(suite=suite_name, task_id=task_id, task=description,
                                trial=trial, success=bool(done and error is None), error=error,
                                steps=len(actions), start_ns=start, end_ns=terminal,
                                rollout_wall_s=(terminal-start)/1e9,
                                action_total_ms=sum(x["duration_ns"] for x in actions)/1e6,
                                switch_wait_ms=broker.switch_wait_ms, overlap_hidden_ms=broker.overlap_hidden_ms)
                            episodes.append(entry)
                            for filename, records in [("episodes.jsonl", [entry]),
                                    ("actions.jsonl", actions), ("inferences.jsonl", broker.inference_records)]:
                                with (output / filename).open("a") as file:
                                    for row in records:
                                        row = dict(row, suite=suite_name, task_id=task_id, trial=trial)
                                        if filename == "inferences.jsonl":
                                            row["terminal_prefetch"] = row["end_ns"] > terminal
                                        file.write(json.dumps(row) + "\n")
                            logging.info("EPISODE %s", json.dumps(entry))
                finally:
                    env.close()
    finally:
        client._ws.close()
    (output / "summary.json").write_text(json.dumps(dict(
        episodes=len(episodes), successes=sum(x["success"] for x in episodes),
        mean_rollout_wall_s=sum(x["rollout_wall_s"] for x in episodes)/len(episodes),
        schedule=expected), indent=2))


if __name__ == "__main__":
    main()
