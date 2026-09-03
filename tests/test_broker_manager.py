"""Exact Docker contract tests for the Workspace egress broker."""

from json import dumps
from subprocess import CompletedProcess
from unittest.mock import patch

from pytest import raises

from tapestry.workspace.broker_manager import (
    EgressBrokerConfig,
    broker_command,
    inspect_broker_address,
)


def test_broker_command_uses_only_the_docker_dialect(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")
    server = tmp_path / "server.py"
    server.write_text("", encoding="utf-8")
    config = EgressBrokerConfig(
        image="python:3.12-slim-bookworm",
        port=8090,
        policy_file=str(policy),
        audit_database=str(tmp_path / "audit.sqlite3"),
        server_file=str(server),
        memory_limit="512m",
        cpu_limit=0.5,
        host_port=48090,
        auth_token="a" * 43,
        session_id="workspace-test",
    )

    command = broker_command(config)

    assert command[0:2] == ["docker", "create"]
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--cpus") + 1] == "0.5"
    assert "--uid" not in command
    assert "--gid" not in command


def test_broker_inspection_reads_the_exact_docker_shape():
    payload = [
        {
            "NetworkSettings": {
                "Networks": {
                    "bridge": {"IPAddress": "172.17.0.4"},
                }
            }
        }
    ]
    result = CompletedProcess(args=[], returncode=0, stdout=dumps(payload), stderr="")

    with patch("tapestry.workspace.broker_manager.run_docker", return_value=result):
        address = inspect_broker_address("broker-id")

    assert address == "172.17.0.4"


def test_broker_inspection_rejects_an_alternate_shape():
    result = CompletedProcess(args=[], returncode=0, stdout=dumps({"networks": [{"address": "192.168.64.2"}]}), stderr="")

    with (
        patch("tapestry.workspace.broker_manager.run_docker", return_value=result),
        raises(RuntimeError, match="no IPv4 address"),
    ):
        inspect_broker_address("broker-id")
