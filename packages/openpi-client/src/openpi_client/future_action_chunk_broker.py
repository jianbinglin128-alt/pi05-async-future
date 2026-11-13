"""Future-target action chunks: predict 10, execute 5, overlap the final 2."""
from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openpi_client import base_policy

PREDICTED_STEPS = 10
EXECUTED_STEPS = 5
OVERLAP_STEPS = 2
PRE_INFERENCE_STEPS = EXECUTED_STEPS - OVERLAP_STEPS


class FutureActionChunkBroker(base_policy.BasePolicy):
    """Use current observations and queued actions; never extrapolate robot state.

    The future checkpoint predicts AFTER the supplied queued actions. Its first
    predictions must therefore be retained, rather than skipped by overlap.
    Call infer once before every env.step, with the latest actual observation.
    """

    def __init__(self, policy):
        self._policy = policy
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="future-inference")
        self._current = None
        self._index = 0
        self._pending = None
        self._submitted_at = None
        self._closed = False
        self.inference_records = []
        self.switch_wait_ms = 0.0
        self.overlap_hidden_ms = 0.0

    def _infer(self, observation):
        started = time.perf_counter_ns()
        response = self._policy.infer(observation)
        finished = time.perf_counter_ns()
        actions = np.asarray(response["actions"])
        if actions.shape != (PREDICTED_STEPS, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Future policy must return 10 finite 7D actions, got {actions.shape}")
        self.inference_records.append(dict(
            index=len(self.inference_records), start_ns=started, end_ns=finished,
            client_e2e_ms=(finished-started)/1e6,
            context_steps=len(observation["action_context"]),
            predicted_steps=PREDICTED_STEPS, executed_steps=EXECUTED_STEPS,
            server_infer_ms=response.get("server_timing", {}).get("infer_ms")))
        # Future output is already aligned after context: do NOT use actions[2:7].
        return actions[:EXECUTED_STEPS].copy()

    def infer(self, obs):
        if self._closed:
            raise RuntimeError("Broker is closed")
        if self._current is None:
            observation = copy.deepcopy(obs)
            observation["action_context"] = np.zeros((0, 7), dtype=np.float32)
            self._current = self._infer(observation)
        elif self._index == EXECUTED_STEPS:
            if self._pending is None:
                raise RuntimeError("Missing future request at chunk boundary")
            switch_at = time.perf_counter_ns()
            try:
                self._current = self._pending.result()
            finally:
                self._pending = None
            self.switch_wait_ms += (time.perf_counter_ns()-switch_at)/1e6
            record = self.inference_records[-1]
            # Only time when the request was actually running can be hidden.
            self.overlap_hidden_ms += max(0, min(switch_at, record["end_ns"])-record["start_ns"])/1e6
            self._index = 0

        if self._index == PRE_INFERENCE_STEPS and self._pending is None:
            # Three actions have finished; obs is current. Execute these same two
            # queued actions while the worker generates the next future chunk.
            observation = copy.deepcopy(obs)
            observation["action_context"] = self._current[self._index:].copy()
            assert observation["action_context"].shape == (OVERLAP_STEPS, 7)
            self._submitted_at = time.perf_counter_ns()
            self._pending = self._executor.submit(self._infer, observation)
        action = self._current[self._index].copy()
        self._index += 1
        return {"actions": action}

    def reset(self):
        try:
            if self._pending is not None:
                self._pending.result()
        finally:
            self._pending = None
            self._current = None
            self._index = 0
            self._policy.reset()

    def close(self):
        if not self._closed:
            self._executor.shutdown(wait=True)
            self._closed = True
