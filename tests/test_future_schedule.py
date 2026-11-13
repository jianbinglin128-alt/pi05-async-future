"""Check actual future alignment and asynchronous scheduling without a GPU."""
import threading
import unittest

import numpy as np
from openpi_client.future_action_chunk_broker import FutureActionChunkBroker


class Policy:
    def __init__(self):
        self.requests = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.reset_count = 0

    def infer(self, obs):
        index = len(self.requests)
        self.requests.append(obs)
        if index:
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("Test did not release asynchronous request")
        return {"actions": np.full((10, 7), index * 100.0) + np.arange(10)[:, None]}

    def reset(self):
        self.reset_count += 1


class FutureScheduleTest(unittest.TestCase):
    def test_three_plus_two_no_output_skip_or_state_prediction(self):
        policy = Policy()
        broker = FutureActionChunkBroker(policy)
        observation = lambda step: {"observation/state": np.full(8, step), "prompt": "test"}
        try:
            received = [broker.infer(observation(i))["actions"][0] for i in range(3)]
            self.assertEqual(received, [0, 1, 2])
            self.assertEqual(len(policy.requests), 1)
            self.assertEqual(policy.requests[0]["action_context"].shape, (0, 7))
            received.append(broker.infer(observation(3))["actions"][0])
            self.assertTrue(policy.started.wait(timeout=5))
            # Inference is still blocked; actions 4 and 5 continue executing.
            received.append(broker.infer(observation(4))["actions"][0])
            self.assertEqual(received, [0, 1, 2, 3, 4])
            np.testing.assert_array_equal(policy.requests[1]["action_context"],
                                          np.full((2, 7), [3]*7) + np.arange(2)[:, None])
            np.testing.assert_array_equal(policy.requests[1]["observation/state"], np.full(8, 3))
            # The submitted observation/context own copies, unaffected by the caller.
            policy.release.set()
            received.extend(broker.infer(observation(i))["actions"][0] for i in range(5, 10))
            self.assertEqual(received[5:], [100, 101, 102, 103, 104])
            broker.reset()
            self.assertEqual(policy.reset_count, 1)
            self.assertEqual(len(policy.requests), 3)
            np.testing.assert_array_equal(policy.requests[2]["action_context"][:, 0], [103, 104])
        finally:
            policy.release.set()
            broker.close()

    def test_reject_wrong_prediction_length(self):
        policy = Policy()
        policy.infer = lambda obs: {"actions": np.zeros((8, 7))}
        broker = FutureActionChunkBroker(policy)
        try:
            with self.assertRaisesRegex(ValueError, "10 finite 7D"):
                broker.infer({})
        finally:
            broker.close()

    def test_worker_failure_is_not_silently_replaced(self):
        class Broken(Policy):
            def infer(self, obs):
                if len(obs["action_context"]):
                    raise RuntimeError("worker failed")
                return super().infer(obs)
        broker = FutureActionChunkBroker(Broken())
        try:
            for _ in range(5):
                broker.infer({})
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                broker.infer({})
        finally:
            broker.close()


if __name__ == "__main__":
    unittest.main()
