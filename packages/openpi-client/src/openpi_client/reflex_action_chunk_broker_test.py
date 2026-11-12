import threading

import numpy as np
import pytest

from openpi_client.reflex_action_chunk_broker import ReflexActionChunkBroker


class FakePolicy:
    def __init__(self):
        self.observations = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.reset_count = 0

    def infer(self, obs):
        self.observations.append(obs)
        if len(self.observations) == 2:
            self.started.set()
            assert self.release.wait(timeout=5)
        value = len(self.observations)
        return {"actions": np.full((4, 7), value, dtype=np.float32)}

    def reset(self):
        self.reset_count += 1


def test_overlaps_chunks_and_predicts_future_state():
    policy = FakePolicy()
    broker = ReflexActionChunkBroker(policy, 4, overlap_steps=2, predict_future_state=True)
    obs = {"state": np.zeros(7, dtype=np.float32)}
    try:
        assert np.all(broker.infer(obs)["actions"] == 1)
        assert np.all(broker.infer(obs)["actions"] == 1)
        assert np.all(broker.infer(obs)["actions"] == 1)
        assert policy.started.wait(timeout=5)
        np.testing.assert_array_equal(policy.observations[1]["state"], [2, 2, 2, 2, 2, 2, 1])
        assert np.all(broker.infer(obs)["actions"] == 1)
        policy.release.set()
        assert np.all(broker.infer(obs)["actions"] == 2)
        broker.reset()
        assert policy.reset_count == 1
        assert np.all(broker.infer(obs)["actions"] == 3)
    finally:
        policy.release.set()
        broker.close()


def test_rejects_invalid_overlap_and_state_mapping():
    policy = FakePolicy()
    with pytest.raises(ValueError, match="overlap_steps"):
        ReflexActionChunkBroker(policy, 4, overlap_steps=4)
    broker = ReflexActionChunkBroker(policy, 4, overlap_steps=2, predict_future_state=True)
    try:
        broker.infer({"state": np.zeros(3, dtype=np.float32)})
        broker.infer({"state": np.zeros(3, dtype=np.float32)})
        with pytest.raises(ValueError, match="7D delta state or a state_predictor"):
            broker.infer({"state": np.zeros(3, dtype=np.float32)})
    finally:
        broker.close()


def test_truncates_predicted_horizon_to_executed_horizon():
    class TenStepPolicy:
        def infer(self, obs):
            return {"actions": np.arange(10, dtype=np.float32)[:, None]}

        def reset(self):
            pass

    broker = ReflexActionChunkBroker(TenStepPolicy(), 5, overlap_steps=2)
    try:
        actions = [broker.infer({})["actions"].item() for _ in range(5)]
        assert actions == [0, 1, 2, 3, 4]
        assert len(broker.inference_records) >= 1
    finally:
        broker.close()


@pytest.mark.parametrize("overlap", [1, 2])
def test_five_step_chunks_submit_observations_before_remaining_actions(overlap):
    class RecordingPolicy:
        def __init__(self):
            self.observed_steps = []

        def infer(self, obs):
            self.observed_steps.append(obs["step"])
            return {"actions": np.full((10, 1), obs["step"], dtype=np.float32)}

    policy = RecordingPolicy()
    broker = ReflexActionChunkBroker(policy, 5, overlap_steps=overlap)
    try:
        values = [broker.infer({"step": step})["actions"].item() for step in range(16)]
        assert values == [0] * 5 + [5 - overlap] * 5 + [10 - overlap] * 5 + [15 - overlap]
        assert policy.observed_steps == [0, 5 - overlap, 10 - overlap, 15 - overlap]
    finally:
        broker.close()
