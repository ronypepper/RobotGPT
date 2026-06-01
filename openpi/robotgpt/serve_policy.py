"""Serve a policy over a websocket connection.

This script is copied from the "scripts/serve_policy.py" script of the openpi repository and
modified to allow modification of the used TrainConfig's repo_id at the command line.
Default environments have also been removed.

Modifications: Copyright (c) 2026 ronypepper.

License: Apache 2.0
"""

import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Training config name (e.g., "pi0_aloha_sim").
    config: str | None = None
    # Checkpoint number (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    checkpoint_dir: str | None = None

    # Repository id of the asset data (same as repo id of the training dataset, e.g. "HF_USERNAME/dataset")
    repo_id: str | None = None


def main(args: Args) -> None:
    config = _config.get_config(args.config)
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=args.repo_id))
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir, default_prompt=args.default_prompt)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
