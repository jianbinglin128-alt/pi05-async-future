import asyncio

import numpy as np
from openpi_client import msgpack_numpy
import websockets.exceptions

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class _FakePolicy:
    def __init__(self):
        self.reset_count = 0
        self.observations = []

    def reset(self):
        self.reset_count += 1

    def infer(self, observation):
        self.observations.append(observation)
        return {"actions": np.zeros((1, 7), dtype=np.float32)}


class _FakeWebsocket:
    remote_address = ("localhost", 1234)

    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []

    async def recv(self):
        try:
            return next(self.messages)
        except StopIteration:
            raise websockets.exceptions.ConnectionClosedOK(None, None) from None

    async def send(self, message):
        self.sent.append(message)


def test_reset_command_resets_policy_without_running_inference():
    packer = msgpack_numpy.Packer()
    policy = _FakePolicy()
    websocket = _FakeWebsocket(
        [
            packer.pack({"_policy_command": "reset"}),
            packer.pack({"observation": np.array([1.0], dtype=np.float32)}),
        ]
    )

    asyncio.run(WebsocketPolicyServer(policy, metadata={"test": True})._handler(websocket))  # noqa: SLF001

    responses = [msgpack_numpy.unpackb(message) for message in websocket.sent]
    assert responses[0] == {"test": True}
    assert responses[1] == {"_policy_command": "reset_ack"}
    assert "server_timing" in responses[2]
    assert policy.reset_count == 1
    assert len(policy.observations) == 1
