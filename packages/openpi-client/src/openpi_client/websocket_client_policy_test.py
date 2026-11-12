from threading import Lock

from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class _FakeWebsocket:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        return msgpack_numpy.Packer().pack({"_policy_command": "reset_ack"})


def test_reset_sends_command_and_waits_for_acknowledgement():
    policy = WebsocketClientPolicy.__new__(WebsocketClientPolicy)
    policy._ws = _FakeWebsocket()
    policy._packer = msgpack_numpy.Packer()
    policy._request_lock = Lock()

    policy.reset()

    assert msgpack_numpy.unpackb(policy._ws.sent[0]) == {"_policy_command": "reset"}
