"""Future-only inference server. Defaults: chunk=10, execute=5, overlap=2."""
import argparse
import json
import logging

from openpi.policies.future_policy import CHECKPOINT, create_future_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    policy = create_future_policy(args.checkpoint, seed=args.seed)
    logging.info("FUTURE_POLICY_READY %s", json.dumps(policy.metadata))
    try:
        WebsocketPolicyServer(policy, host=args.host, port=args.port,
                              metadata=policy.metadata).serve_forever()
    finally:
        policy.close()


if __name__ == "__main__":
    main()
