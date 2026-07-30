"""Configure the sandbox network boundary, drop privilege, and start the server."""

import argparse
import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).with_name("server.py")


def parse_policy(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--egress-policy", choices=("deny", "broker", "unrestricted"), default="deny")
    return parser.parse_known_args(argv)[0]


def run_firewall(binary, arguments):
    result = subprocess.run([binary, *arguments], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{binary} {' '.join(arguments)} failed: {result.stderr.strip()}")


def deny_egress():
    """Allow loopback and replies to inbound calls, but reject new outbound flows."""
    # Apple's container VM currently exposes the legacy xtables kernel API but
    # not nf_tables. Debian's unqualified iptables command selects nft, so
    # prefer the legacy frontend and retain the generic names as a fallback for
    # Docker/podman hosts whose images omit the alternatives.
    ipv4 = shutil.which("iptables-legacy") or shutil.which("iptables")
    ipv6 = shutil.which("ip6tables-legacy") or shutil.which("ip6tables")
    if not ipv4:
        raise RuntimeError("deny egress policy requires iptables")

    def apply_rules(binary):
        run_firewall(binary, ["-F", "OUTPUT"])
        run_firewall(binary, ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"])
        run_firewall(
            binary,
            ["-A", "OUTPUT", "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"],
        )
        run_firewall(binary, ["-P", "OUTPUT", "DROP"])

    apply_rules(ipv4)

    interfaces = Path("/proc/net/if_inet6")
    has_external_ipv6 = interfaces.exists() and any(
        line.split()[-1] != "lo"
        for line in interfaces.read_text().splitlines()
        if line.split()
    )
    if not has_external_ipv6:
        return
    if ipv6:
        try:
            apply_rules(ipv6)
            return
        except RuntimeError:
            # Apple's current guest kernel assigns link-local IPv6 but omits
            # the ip6tables filter table. Disable IPv6 at the kernel boundary
            # rather than silently leaving an unfiltered route.
            pass
    Path("/proc/sys/net/ipv6/conf/all/disable_ipv6").write_text("1")
    Path("/proc/sys/net/ipv6/conf/default/disable_ipv6").write_text("1")
    if interfaces.exists() and any(
        line.split()[-1] != "lo"
        for line in interfaces.read_text().splitlines()
        if line.split()
    ):
        raise RuntimeError("could not enforce deny policy for IPv6")


def drop_privileges(username="sandbox"):
    if os.geteuid() != 0:
        raise RuntimeError("sandbox entrypoint must start as root so it can install the network boundary")
    account = pwd.getpwnam(username)
    os.setgroups([])
    os.setgid(account.pw_gid)
    os.setuid(account.pw_uid)
    os.environ["HOME"] = account.pw_dir


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    policy = parse_policy(argv).egress_policy
    if policy == "deny":
        deny_egress()
    drop_privileges()
    os.execv(sys.executable, [sys.executable, str(SERVER), *argv])


if __name__ == "__main__":
    main()
