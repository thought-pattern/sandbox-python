# Copyright 2025-2026 Jason E. Robinson.
# SPDX-License-Identifier: Apache-2.0

"""Firewall rules the root entrypoint installs for each egress policy, recorded instead of executed."""

from pytest import fixture

import entrypoint

# Stand-in for the resolver the entrypoint reads from /etc/resolv.conf at startup.
RESOLVER = "192.0.2.53"
FLUSH = ["-F", "OUTPUT"]
LOOPBACK = ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"]
REPLIES = ["-A", "OUTPUT", "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"]
DROP_POLICY = ["-P", "OUTPUT", "DROP"]


@fixture
def installed_rules(monkeypatch):
    rules = []
    monkeypatch.setattr(entrypoint, "shutil_which", lambda name: f"/usr/sbin/{name}")
    monkeypatch.setattr(entrypoint, "nameserver_addresses", lambda: [RESOLVER])
    monkeypatch.setattr(entrypoint, "run_firewall", lambda binary, arguments: rules.append((binary, arguments)))
    return rules


def ipv4_rules(installed_rules):
    return [arguments for binary, arguments in installed_rules if binary == "/usr/sbin/iptables"]


def test_web_egress_passes_replies_before_dropping_non_public_networks(installed_rules):
    entrypoint.enforce_egress_policy(allow_web=True)
    assert ipv4_rules(installed_rules) == [
        FLUSH,
        LOOPBACK,
        REPLIES,
        ["-A", "OUTPUT", "-p", "udp", "-d", RESOLVER, "--dport", "53", "-j", "ACCEPT"],
        ["-A", "OUTPUT", "-p", "tcp", "-d", RESOLVER, "--dport", "53", "-j", "ACCEPT"],
        *[["-A", "OUTPUT", "-d", network, "-j", "DROP"] for network in entrypoint.NON_PUBLIC_NETWORKS.get("ipv4", ())],
        ["-A", "OUTPUT", "-p", "tcp", "--dport", "80", "-j", "ACCEPT"],
        ["-A", "OUTPUT", "-p", "tcp", "--dport", "443", "-j", "ACCEPT"],
        DROP_POLICY,
    ]
    assert "169.254.0.0/16" in entrypoint.NON_PUBLIC_NETWORKS.get("ipv4", ())
    assert "fc00::/7" in entrypoint.NON_PUBLIC_NETWORKS.get("ipv6", ())


def test_deny_egress_allows_only_loopback_and_replies(installed_rules):
    entrypoint.enforce_egress_policy()
    assert ipv4_rules(installed_rules) == [FLUSH, LOOPBACK, REPLIES, DROP_POLICY]


def test_main_maps_each_container_policy_to_its_firewall(monkeypatch):
    calls = []
    monkeypatch.setattr(entrypoint, "enforce_egress_policy", lambda allow_web=False: calls.append(("firewall", allow_web)))
    monkeypatch.setattr(entrypoint, "drop_privileges", lambda: calls.append(("drop",)))
    monkeypatch.setattr(entrypoint, "os_execv", lambda path, arguments: calls.append(("exec", arguments[2:])))
    for policy, firewall in (("web", [("firewall", True)]), ("deny", [("firewall", False)]), ("direct", [])):
        calls.clear()
        arguments = ["--egress-policy", policy, "--auth-token", "t" * 32]
        entrypoint.main(arguments)
        assert calls == [*firewall, ("drop",), ("exec", arguments)]
