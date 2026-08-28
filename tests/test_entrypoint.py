"""Network-boundary tests for deny and broker sandbox startup policies."""

from unittest.mock import patch

from entrypoint import Path as entrypoint_Path
from entrypoint import enforce_egress_policy as entrypoint_enforce_egress_policy
from entrypoint import main as entrypoint_main
from entrypoint import sys as entrypoint_sys
from pytest import mark as pytest_mark
from pytest import raises as pytest_raises


def test_broker_policy_allows_only_broker_before_default_drop():
    calls = []
    with (
        patch(
            "entrypoint.shutil.which",
            side_effect=lambda name: "/sbin/iptables" if "iptables" in name else "",
        ),
        patch(
            "entrypoint.run_firewall",
            side_effect=lambda binary, arguments: calls.append(arguments),
        ),
        patch.object(entrypoint_Path, "exists", return_value=False),
    ):
        entrypoint_enforce_egress_policy("http://192.168.64.9:8090")

    broker_rule = next(arguments for arguments in calls if "192.168.64.9" in arguments)
    assert broker_rule[broker_rule.index("--dport") + 1] == "8090"
    assert broker_rule[broker_rule.index("--ctstate") + 1] == "NEW"
    assert calls[-1] == ["-P", "OUTPUT", "DROP"]
    assert sum("-d" in arguments for arguments in calls) == 1
    return False


@pytest_mark.parametrize(
    "url", ["https://192.168.64.9:8090", "http://broker:8090", "http://192.168.64.9"]
)
def test_broker_policy_requires_explicit_http_ipv4_and_port(url):
    with (
        patch("entrypoint.shutil.which", return_value="/sbin/iptables"),
        pytest_raises(RuntimeError, match="broker"),
    ):
        entrypoint_enforce_egress_policy(url)
    return False


def test_main_forwards_process_command_line(monkeypatch):
    arguments = [
        "entrypoint.py",
        "--egress-policy",
        "unrestricted",
        "--auth-token",
        "a" * 43,
    ]
    monkeypatch.setattr(entrypoint_sys, "argv", arguments)
    with (
        patch("entrypoint.drop_privileges"),
        patch("entrypoint.os.execv") as execute,
    ):
        entrypoint_main()

    forwarded = execute.call_args.args[1]
    assert forwarded[-4:] == arguments[-4:]
    return False
