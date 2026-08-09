"""Unit tests for the sandbox container manager."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from manager import (
    ContainerConfig,
    create_container,
    create_container_command,
    create_fargate_task,
    parse_memory,
    sandbox_session,
)


def test_parse_memory_gigabytes():
    assert parse_memory("2g") == "2048"


def test_parse_memory_megabytes():
    assert parse_memory("512m") == "512"


@pytest.mark.parametrize("value", ["1024", "0g", "1.5g", "-1m"])
def test_parse_memory_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_memory(value)


def test_config_allocates_loopback_port_and_session_token():
    config = ContainerConfig()
    assert 1 <= config.host_port <= 65535
    assert config.host_bind == "127.0.0.1"
    assert len(config.auth_token) >= 32
    assert config.base_url == f"http://127.0.0.1:{config.host_port}"


def test_create_command_container_runtime_boundary():
    config = ContainerConfig(
        memory_limit="2g",
        cpu_limit=1.0,
        port=8080,
        host_port=48080,
        auth_token="a" * 43,
    )
    command = create_container_command(config)

    assert command[0] == "container"
    assert "--security-opt" not in command
    assert "--cap-drop" not in command
    assert command[command.index("--publish") + 1] == "127.0.0.1:48080:8080"
    assert command[command.index("--memory") + 1] == "2G"
    assert command[command.index("--cpus") + 1] == "1"
    assert command[command.index("--user") + 1] == "root"
    assert command[command.index("--auth-token") + 1] == "a" * 43
    assert command[command.index("--egress-policy") + 1] == "deny"
    assert command[command.index("--max-file-bytes") + 1] == "10485760"
    assert command[command.index("--max-response-bytes") + 1] == "12582912"
    assert "--env" not in command


def test_create_command_docker_adds_shared_kernel_hardening():
    config = ContainerConfig(host_port=48080, cpu_limit=0.5, auth_token="a" * 43)
    command = create_container_command(config, runtime="docker")

    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--cap-add") + 1] == "NET_ADMIN"
    assert command[command.index("--pids-limit") + 1] == "256"
    assert command[command.index("--cpus") + 1] == "0.5"


def test_apple_runtime_rejects_fractional_cpu_limit():
    config = ContainerConfig(host_port=48080, cpu_limit=0.5, auth_token="a" * 43)
    with pytest.raises(ValueError, match="whole-number"):
        create_container_command(config, runtime="container")


def test_config_rejects_non_loopback_bind():
    with pytest.raises(ValueError, match="host_bind"):
        ContainerConfig(host_bind="0.0.0.0")


def test_config_rejects_short_auth_token():
    with pytest.raises(ValueError, match="auth_token"):
        ContainerConfig(auth_token="short")


def test_config_rejects_response_limit_too_small_for_structured_error():
    with pytest.raises(ValueError, match="max_response_bytes"):
        ContainerConfig(max_response_bytes=511)


def test_create_container_raises_on_nonzero_exit():
    completed = SimpleNamespace(returncode=1, stdout="", stderr="boom")
    with patch("manager.subprocess.run", return_value=completed):
        with pytest.raises(RuntimeError, match="boom"):
            create_container(ContainerConfig())


def test_create_container_rejects_empty_identifier():
    completed = SimpleNamespace(returncode=0, stdout="", stderr="")
    with patch("manager.subprocess.run", return_value=completed):
        with pytest.raises(RuntimeError, match="no container identifier"):
            create_container(ContainerConfig())


def test_sandbox_session_destroys_after_success():
    with (
        patch("manager.create_container", return_value="cid"),
        patch("manager.start_container") as start,
        patch("manager.destroy_container") as destroy,
    ):
        with sandbox_session(ContainerConfig()) as container_id:
            assert container_id == "cid"

    start.assert_called_once_with("cid", runtime="container")
    destroy.assert_called_once_with("cid", runtime="container")


def test_sandbox_session_preserves_body_error_and_notes_cleanup_failure():
    with (
        patch("manager.create_container", return_value="cid"),
        patch("manager.start_container"),
        patch("manager.destroy_container", side_effect=RuntimeError("cleanup failed")),
    ):
        with pytest.raises(ValueError, match="work failed") as caught:
            with sandbox_session(ContainerConfig()):
                raise ValueError("work failed")

    assert any("cleanup also failed" in note for note in caught.value.__notes__)


def test_sandbox_session_surfaces_cleanup_failure_after_success():
    with (
        patch("manager.create_container", return_value="cid"),
        patch("manager.start_container"),
        patch("manager.destroy_container", side_effect=RuntimeError("cleanup failed")),
    ):
        with pytest.raises(RuntimeError, match="cleanup failed"):
            with sandbox_session(ContainerConfig()):
                pass


def test_fargate_rejects_non_broker_policy():
    with pytest.raises(RuntimeError, match="broker egress"):
        create_fargate_task(ContainerConfig(egress_policy="deny"), "cluster", ["subnet"], ["sg"])


def test_fargate_reports_run_task_failures():
    ecs = MagicMock()
    ecs.register_task_definition.return_value = {"taskDefinition": {"taskDefinitionArn": "task-def"}}
    ecs.run_task.return_value = {"tasks": [], "failures": [{"reason": "capacity"}]}
    config = ContainerConfig(
        egress_policy="broker",
        broker_url="http://192.168.64.9:8090",
        broker_token="b" * 43,
    )
    with patch("manager.create_ecs_client", return_value=ecs):
        with pytest.raises(RuntimeError, match="capacity"):
            create_fargate_task(config, "cluster", ["subnet"], ["sg"])


def test_containerfile_uses_privilege_dropping_entrypoint():
    root = Path(__file__).resolve().parent.parent
    containerfile = (root / "container" / "Containerfile").read_text()
    entrypoint = (root / "container" / "entrypoint.py").read_text()

    assert "USER root" in containerfile
    assert "entrypoint.py" in containerfile
    assert "os.setgid" in entrypoint
    assert "os.setuid" in entrypoint
    assert entrypoint.index("enforce_egress_policy()") < entrypoint.index("drop_privileges()")


def test_broker_runtime_installs_firewall_and_passes_only_broker_endpoint():
    config = ContainerConfig(
        egress_policy="broker",
        broker_url="http://192.168.64.9:8090",
        broker_token="b" * 43,
        host_port=48080,
    )
    command = create_container_command(config)

    assert command[command.index("--user") + 1] == "root"
    assert command[command.index("--broker-url") + 1] == "http://192.168.64.9:8090"
    assert command[command.index("--broker-token") + 1] == "b" * 43
