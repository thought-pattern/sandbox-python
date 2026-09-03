"""Unit tests for Tapestry's Docker sandbox manager."""

from subprocess import CompletedProcess
from unittest.mock import patch

from pytest import mark as pytest_mark
from pytest import raises as pytest_raises

from tapestry.workspace.sandbox_manager import (
    ContainerConfig,
    create_container,
    container_command,
    parse_memory,
    sandbox_session,
)


BASE_CONFIG = {
    "image": "python-sandbox:latest",
    "port": 8080,
    "workspace": "/workspace",
    "memory_limit": "2g",
    "cpu_limit": 1.0,
}


def test_parse_memory_gigabytes():
    assert parse_memory("2g") == 2048


def test_parse_memory_megabytes():
    assert parse_memory("512m") == 512


@pytest_mark.parametrize("value", ["1024", "0g", "1.5g", "-1m"])
def test_parse_memory_rejects_invalid_values(value):
    with pytest_raises(ValueError):
        parse_memory(value)


def test_config_allocates_loopback_port_and_session_token():
    config = ContainerConfig(**BASE_CONFIG)
    assert 1 <= config.host_port <= 65535
    assert config.host_bind == "127.0.0.1"
    assert len(config.auth_token) >= 32
    assert config.base_url == f"http://127.0.0.1:{config.host_port}"


def test_create_command_is_the_exact_docker_boundary():
    config = ContainerConfig(
        **(
            BASE_CONFIG
            | {
                "cpu_limit": 0.5,
                "host_port": 48080,
                "auth_token": "a" * 43,
            }
        )
    )
    command = container_command(config)

    assert command[0:2] == ["docker", "create"]
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--cap-add") + 1] == "NET_ADMIN"
    assert command[command.index("--pids-limit") + 1] == "256"
    assert command[command.index("--publish") + 1] == "127.0.0.1:48080:8080"
    assert command[command.index("--memory") + 1] == "2G"
    assert command[command.index("--cpus") + 1] == "0.5"
    assert command[command.index("--user") + 1] == "root"
    assert command[command.index("--auth-token") + 1] == "a" * 43
    assert command[command.index("--egress-policy") + 1] == "deny"


def test_config_rejects_non_loopback_bind():
    with pytest_raises(ValueError, match="host_bind"):
        ContainerConfig(
            image="python-sandbox:latest",
            port=8080,
            workspace="/workspace",
            memory_limit="2g",
            cpu_limit=1.0,
            host_bind="0.0.0.0",
        )


def test_config_rejects_short_auth_token():
    with pytest_raises(ValueError, match="auth_token"):
        ContainerConfig(
            image="python-sandbox:latest",
            port=8080,
            workspace="/workspace",
            memory_limit="2g",
            cpu_limit=1.0,
            auth_token="short",
        )


def test_create_container_raises_on_nonzero_exit():
    completed = CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
    with patch(
        "tapestry.workspace.sandbox_manager.imported_subprocess_run",
        return_value=completed,
    ):
        with pytest_raises(RuntimeError, match="boom"):
            create_container(ContainerConfig(**BASE_CONFIG))


def test_create_container_rejects_empty_identifier():
    completed = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch(
        "tapestry.workspace.sandbox_manager.imported_subprocess_run",
        return_value=completed,
    ):
        with pytest_raises(RuntimeError, match="no container identifier"):
            create_container(ContainerConfig(**BASE_CONFIG))


def test_sandbox_session_destroys_after_success():
    with (
        patch(
            "tapestry.workspace.sandbox_manager.create_container",
            return_value="cid",
        ),
        patch("tapestry.workspace.sandbox_manager.start_container") as start,
        patch("tapestry.workspace.sandbox_manager.destroy_container") as destroy,
    ):
        with sandbox_session(ContainerConfig(**BASE_CONFIG)) as container_id:
            assert container_id == "cid"

    start.assert_called_once_with("cid")
    destroy.assert_called_once_with("cid")


def test_sandbox_session_preserves_body_error_and_notes_cleanup_failure():
    with (
        patch(
            "tapestry.workspace.sandbox_manager.create_container",
            return_value="cid",
        ),
        patch("tapestry.workspace.sandbox_manager.start_container"),
        patch(
            "tapestry.workspace.sandbox_manager.destroy_container",
            side_effect=RuntimeError("cleanup failed"),
        ),
    ):
        with pytest_raises(ValueError, match="work failed") as caught:
            with sandbox_session(ContainerConfig(**BASE_CONFIG)):
                raise ValueError("work failed")

    assert any("cleanup also failed" in note for note in caught.value.__notes__)


def test_sandbox_session_surfaces_cleanup_failure_after_success():
    with (
        patch(
            "tapestry.workspace.sandbox_manager.create_container",
            return_value="cid",
        ),
        patch("tapestry.workspace.sandbox_manager.start_container"),
        patch(
            "tapestry.workspace.sandbox_manager.destroy_container",
            side_effect=RuntimeError("cleanup failed"),
        ),
    ):
        with pytest_raises(RuntimeError, match="cleanup failed"):
            with sandbox_session(ContainerConfig(**BASE_CONFIG)):
                pass


def test_broker_policy_installs_firewall_and_passes_only_broker_endpoint():
    config = ContainerConfig(
        **(
            BASE_CONFIG
            | {
                "egress_policy": "broker",
                "broker_url": "http://192.168.64.9:8090",
                "broker_token": "b" * 43,
                "host_port": 48080,
            }
        )
    )
    command = container_command(config)

    assert command[command.index("--user") + 1] == "root"
    assert command[command.index("--broker-url") + 1] == "http://192.168.64.9:8090"
    assert command[command.index("--broker-token") + 1] == "b" * 43
