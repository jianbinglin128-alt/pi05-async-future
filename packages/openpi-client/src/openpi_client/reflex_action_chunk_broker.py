"""Overlapped action-chunk inference for the OpenPI client runtime."""

# The client package supports Python 3.7, which cannot parse PEP 604 unions.
# ruff: noqa: UP045
from __future__ import annotations

import copy
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable, Optional

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ReflexActionChunkBroker(_base_policy.BasePolicy):
    """Infer the next chunk while the current chunk is being executed.

    Delta-state prediction requires matching coordinates and units. For robot
    specific controller scaling and rotation conventions supply a state_predictor.
    The broker bootstraps synchronously, then uses one background worker.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        *,
        overlap_steps: int,
        control_hz: Optional[float] = None,
        predict_future_state: bool = False,
        state_key: str = "state",
        action_key: str = "actions",
        state_predictor: Optional[Callable] = None,
        action_context: bool = False,
    ):
        if action_horizon < 2 or not 1 <= overlap_steps < action_horizon:
            raise ValueError("Require action_horizon >= 2 and 1 <= overlap_steps < action_horizon")
        if control_hz is not None and control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self._policy = policy
        self._horizon = action_horizon
        self._overlap = overlap_steps
        self._control_hz = control_hz
        self._predict_future_state = predict_future_state
        self._state_key = state_key
        self._action_key = action_key
        self._state_predictor = state_predictor
        self._action_context = action_context
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reflex-inference")
        self._pending: Optional[Future] = None
        self._current: Optional[dict] = None
        self._index = 0
        self._latency = None
        self._metrics_lock = Lock()
        self.inference_records = []
        self.overlap_hidden_ms = 0.0
        self.switch_wait_ms = 0.0
        self._pending_started = None

    def _infer_timed(self, observation):
        start = time.monotonic()
        result = self._policy.infer(observation)
        finished = time.monotonic()
        latency = finished - start
        actions = np.asarray(result[self._action_key])
        if actions.ndim < 2 or actions.shape[0] < self._horizon:
            raise ValueError("Policy must return at least action_horizon actions")
        skip = len(observation.get("action_context", [])) if self._action_context else 0
        if actions.shape[0] < self._horizon + skip:
            raise ValueError("Policy horizon must include overlap plus executable chunk")
        result[self._action_key] = actions[skip : skip + self._horizon]
        with self._metrics_lock:
            self.inference_records.append(
                {
                    "client_infer_ms": latency * 1000,
                    "server_infer_ms": result.get("server_timing", {}).get("infer_ms"),
                    "llm_ms": result.get("policy_timing", {}).get("llm_ms"),
                    "action_expert_ms": result.get("policy_timing", {}).get("action_expert_ms"),
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
                    "started_at": start,
                    "finished_at": finished,
                }
            )
        return result, latency

    def _prepare_observation(self, obs):
        observation = copy.deepcopy(obs)
        if self._action_context:
            observation["action_context"] = np.asarray(self._current[self._action_key])[self._index :].copy()
        if self._predict_future_state:
            state = np.asarray(observation[self._state_key])
            actions = np.asarray(self._current[self._action_key])
            remaining = actions[self._index :]
            if self._state_predictor is not None:
                observation[self._state_key] = self._state_predictor(state.copy(), remaining.copy(), actions[-1].copy())
                return observation
            if state.shape != (7,) or actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError("Future-state prediction requires a 7D delta state or a state_predictor")
            projected = state.copy()
            projected[:6] += remaining[:, :6].sum(axis=0)
            # The gripper command is a discrete open/close target, not a delta.
            projected[6] = actions[-1, 6]
            observation[self._state_key] = projected
        return observation

    def _update_latency(self, latency):
        self._latency = latency if self._latency is None else 0.75 * self._latency + 0.25 * latency
        if self._control_hz is not None:
            estimated = math.ceil(self._latency * self._control_hz) + 1
            self._overlap = min(self._horizon - 1, max(1, estimated))

    @override
    def infer(self, obs: dict) -> dict:
        if self._current is None:
            self._current, latency = self._infer_timed(copy.deepcopy(obs))
            self._update_latency(latency)
        elif self._index == self._horizon:
            if self._pending is None:
                self._current, latency = self._infer_timed(copy.deepcopy(obs))
            else:
                switch_at = time.monotonic()
                try:
                    self._current, latency = self._pending.result()
                finally:
                    self._pending = None
                self.switch_wait_ms += (time.monotonic() - switch_at) * 1000
                self.overlap_hidden_ms += min(latency * 1000, max(0.0, (switch_at - self._pending_started) * 1000))
                self._pending_started = None
            self._update_latency(latency)
            self._index = 0

        # obs reflects all previously returned/executed actions. Launch before
        # returning the next action, so remaining includes that action exactly once.
        if self._index >= self._horizon - self._overlap and self._pending is None:
            self._pending_started = time.monotonic()
            self._pending = self._executor.submit(self._infer_timed, self._prepare_observation(obs))

        def slicer(value):
            if isinstance(value, np.ndarray) and value.shape[:1] == (self._horizon,):
                return value[self._index, ...]
            return value

        result = tree.map_structure(slicer, self._current)
        self._index += 1
        return result

    @override
    def reset(self) -> None:
        try:
            if self._pending is not None:
                self._pending.result()
        finally:
            self._pending = None
            self._pending_started = None
            self._policy.reset()
            self._current = None
            self._index = 0
            self._latency = None

    def close(self) -> None:
        self._executor.shutdown(wait=True)
