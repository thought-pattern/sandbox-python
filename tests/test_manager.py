"""Unit tests for the sandbox container manager."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from manager import ContainerConfig, create_container, parse_memory, sandbox_session


def test_parse_memory_gigabytes():
    assert parse_memory("2g") == "2048"


def test_parse_memory_megabytes():
    assert parse_memory("512m") == "512"


def test_parse_memory_rejects_unitless():
    with pytest.raises(ValueError):
        parse_memory("1024")


def test_create_container_container_runtime_resource_flags():
    config = ContainerConfig(memory_limit="2g", cpu_limit=1.0, port=8080)
    completed = SimpleNamespace(returncode=0, stdout="cid\n", stderr="")
    with patch("manager.subprocess.run", return_value=completed) as run:
        container_id = create_container(config)

    assert container_id == "cid"
    cmd = run.call_args[0][0]
    assert cmd[0] == "container"
    # Apple's VM-isolated runtime neither exposes nor needs the shared-kernel flags.
    assert "--security-opt" not in cmd
    assert "--cap-drop" not in cmd
    # container wants an uppercase memory suffix and an integer CPU count.
    assert cmd[cmd.index("--memory") + 1] == "2G"
    assert cmd[cmd.index("--cpus") + 1] == "1"
    assert cmd[cmd.index("--publish") + 1] == "8080:8080"
    assert cmd[-1] == config.image


def test_create_container_docker_runtime_adds_shared_kernel_hardening():
    completed = SimpleNamespace(returncode=0, stdout="cid\n", stderr="")
    with patch("manager.subprocess.run", return_value=completed) as run:
        create_container(ContainerConfig(), runtime="docker")

    cmd = run.call_args[0][0]
    assert cmd[cmd.index("--security-opt") + 1] == "no-new-privileges"
    assert cmd[cmd.index("--cap-drop") + 1] == "ALL"


def test_create_container_raises_on_nonzero_exit():
    completed = SimpleNamespace(returncode=1, stdout="", stderr="boom")
    with patch("manager.subprocess.run", return_value=completed):
        with pytest.raises(RuntimeError, match="boom"):
            create_container(ContainerConfig())


def test_sandbox_session_destroys_after_success():
    with patch("manager.create_container", return_value="cid"), patch("manager.start_container") as start, patch(
        "manager.destroy_container"
    ) as destroy:
        with sandbox_session(ContainerConfig()) as container_id:
            assert container_id == "cid"

    start.assert_called_once_with("cid", runtime="container")
    destroy.assert_called_once_with("cid", runtime="container")


def test_sandbox_session_destroys_after_error():
    with patch("manager.create_container", return_value="cid"), patch("manager.start_container"), patch(
        "manager.destroy_container"
    ) as destroy:
        with pytest.raises(ValueError):
            with sandbox_session(ContainerConfig()):
                raise ValueError("work failed")

    destroy.assert_called_once_with("cid", runtime="container")


def test_containerfile_declares_non_root_user():
    containerfile = Path(__file__).resolve().parent.parent / "container" / "Containerfile"
    text = containerfile.read_text()
    assert "USER sandbox" in text
    assert "useradd" in text
    assert "chown -R sandbox:sandbox" in text
    assert "USER root" not in text
