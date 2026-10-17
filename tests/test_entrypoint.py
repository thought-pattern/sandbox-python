# Copyright 2025-2026 Thought Pattern AI, LLC
# SPDX-License-Identifier: Apache-2.0

"""Firewall ordering and descriptor-only service-key handoff at the existing entrypoint boundary."""

from os import close, environ, read

from entrypoint import (
    KEY_BOOTSTRAP_VARIABLE,
    NON_PUBLIC_NETWORKS,
    enforce_egress_policy,
    main as entrypoint_main,
)
from pytest import fixture

# Stand-in for the resolver the entrypoint reads from /etc/resolv.conf at startup.
RESOLVER = "192.0.2.53"
FLUSH = ["-F", "OUTPUT"]
LOOPBACK = ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"]
REPLIES = [
    "-A",
    "OUTPUT",
    "-m",
    "conntrack",
    "--ctstate",
    "ESTABLISHED,RELATED",
    "-j",
    "ACCEPT",
]
DROP_POLICY = ["-P", "OUTPUT", "DROP"]


@fixture
def installed_rules(monkeypatch):
    rules = []
    monkeypatch.setattr("entrypoint.shutil_which", lambda name: f"/usr/sbin/{name}")
    monkeypatch.setattr("entrypoint.nameserver_addresses", lambda: [RESOLVER])
    monkeypatch.setattr(
        "entrypoint.run_firewall",
        lambda binary, arguments: rules.append((binary, arguments)),
    )
    return rules


def ipv4_rules(installed_rules):
    rules = [arguments for binary, arguments in installed_rules if binary == "/usr/sbin/iptables"]
    return rules


def test_web_egress_passes_replies_before_dropping_non_public_networks(installed_rules):
    enforce_egress_policy(allow_web=True)
    assert ipv4_rules(installed_rules) == [
        FLUSH,
        LOOPBACK,
        REPLIES,
        ["-A", "OUTPUT", "-p", "udp", "-d", RESOLVER, "--dport", "53", "-j", "ACCEPT"],
        ["-A", "OUTPUT", "-p", "tcp", "-d", RESOLVER, "--dport", "53", "-j", "ACCEPT"],
        *[["-A", "OUTPUT", "-d", network, "-j", "DROP"] for network in NON_PUBLIC_NETWORKS.get("ipv4", ())],
        ["-A", "OUTPUT", "-p", "tcp", "--dport", "80", "-j", "ACCEPT"],
        ["-A", "OUTPUT", "-p", "tcp", "--dport", "443", "-j", "ACCEPT"],
        DROP_POLICY,
    ]
    assert "169.254.0.0/16" in NON_PUBLIC_NETWORKS.get("ipv4", ())
    assert "fc00::/7" in NON_PUBLIC_NETWORKS.get("ipv6", ())


def test_deny_egress_allows_only_loopback_and_replies(installed_rules):
    enforce_egress_policy()
    assert ipv4_rules(installed_rules) == [FLUSH, LOOPBACK, REPLIES, DROP_POLICY]


def test_main_maps_each_container_policy_to_its_firewall(monkeypatch, tmp_path):
    key_file = tmp_path / "workspace-key"
    key_file.write_text("k" * 32 + "\n")
    key_file.chmod(0o600)
    calls = []
    monkeypatch.setattr(
        "entrypoint.enforce_egress_policy",
        lambda allow_web=False: calls.append(("firewall", allow_web)),
    )
    monkeypatch.setattr("entrypoint.drop_privileges", lambda: calls.append(("drop",)))
    monkeypatch.setattr("entrypoint.protect_key_directory", lambda arguments: calls.append(("key_boundary",)))
    monkeypatch.setattr(
        "entrypoint.os_execv",
        lambda path, arguments: calls.append(("exec", arguments[2:])),
    )
    for policy, firewall in (
        ("web", [("firewall", True)]),
        ("deny", [("firewall", False)]),
        ("direct", []),
    ):
        calls.clear()
        entrypoint_main(["--egress-policy", policy, "--auth-token-file", str(key_file)])
        # The server receives the descriptor opened before the privilege drop, never the file's path or the key.
        descriptor = int(calls[-1][1][-1])
        try:
            assert calls == [
                *firewall,
                ("key_boundary",),
                ("drop",),
                ("exec", ["--egress-policy", policy, "--auth-token-fd", str(descriptor)]),
            ]
            assert read(descriptor, 64) == b"k" * 32 + b"\n"
        finally:
            close(descriptor)


def test_deployment_passes_its_key_by_descriptor_and_retires_bootstrap_file(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("entrypoint.verify_deployment_identity", lambda: calls.append(("verified",)))
    monkeypatch.setattr("entrypoint.os_execv", lambda path, arguments: calls.append(("exec", arguments[2:])))
    monkeypatch.setitem(environ, KEY_BOOTSTRAP_VARIABLE, "b" * 32)
    bootstrap_directory = tmp_path / "bootstrap"
    bootstrap_directory.mkdir()
    monkeypatch.setattr("entrypoint.tempfile_mkdtemp", lambda prefix: str(bootstrap_directory))
    arguments = ["--egress-policy", "deny", "--egress-enforcement", "deployment", "--auth-token-file", "/run/secrets/key"]
    entrypoint_main(arguments)
    executed = calls[-1][1]
    descriptor = int(executed[-1])
    try:
        assert calls[0] == ("verified",) and executed[:-2] == arguments[:-2]
        assert executed[-2] == "--auth-token-fd"
        assert KEY_BOOTSTRAP_VARIABLE not in environ
        assert not bootstrap_directory.exists()
        assert read(descriptor, 64) == b"b" * 32 + b"\n"
    finally:
        close(descriptor)
