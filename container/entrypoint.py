"""Configure the sandbox network boundary, drop privilege, and start the server."""

from argparse import ArgumentParser as argparse_ArgumentParser
from ctypes import CDLL, get_errno
from os import environ as os_environ
from os import execv as os_execv
from os import geteuid as os_geteuid
from os import setgid as os_setgid
from os import setgroups as os_setgroups
from os import setuid as os_setuid
from pathlib import Path
from pwd import getpwnam as pwd_getpwnam
from shutil import which as shutil_which
from subprocess import run as subprocess_run
from sys import argv as sys_argv
from sys import executable as sys_executable

SERVER = Path(__file__).with_name("server.py")
COMMAND_LINE_ARGUMENTS = []
WEB_PORTS = ("80", "443")
# Web egress never reaches private, loopback, link-local or multicast networks,
# keeping cloud metadata and credential endpoints (169.254.169.254,
# 169.254.170.2) and the host's local network out of reach.
NON_PUBLIC_NETWORKS = {
    "ipv4": (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "224.0.0.0/4",
        "240.0.0.0/4",
    ),
    "ipv6": ("::1/128", "fc00::/7", "fe80::/10", "ff00::/8"),
}


def parse_policy(argv):
    parser = argparse_ArgumentParser(add_help=False)
    parser.add_argument("--egress-policy", choices=("web", "deny", "direct"), default="direct")
    parser.add_argument("--egress-enforcement", choices=("container", "deployment"), default="container")
    computed_return_value = parser.parse_known_args(argv)[0]
    return computed_return_value


def run_firewall(binary, arguments):
    result = subprocess_run([binary, *arguments], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{binary} {' '.join(arguments)} failed: {result.stderr.strip()}")
    return False


def nameserver_addresses():
    """Read the resolvers the guest must reach to resolve web destinations."""
    resolv = Path("/etc/resolv.conf")
    lines = resolv.read_text().splitlines() if resolv.exists() else []
    result = [line.split()[1] for line in lines if line.strip().startswith("nameserver") and len(line.split()) > 1]
    return result


def enforce_egress_policy(allow_web=False):
    """Allow only loopback and replies, plus DNS and outbound TCP 80/443 to public addresses when web egress is on."""
    ipv4 = shutil_which("iptables")
    ipv6 = shutil_which("ip6tables")
    if not ipv4:
        raise RuntimeError("controlled egress policy requires iptables")
    resolvers = nameserver_addresses() if allow_web else []

    def apply_rules(binary):
        family = "ipv6" if binary == ipv6 else "ipv4"
        run_firewall(binary, ["-F", "OUTPUT"])
        run_firewall(binary, ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"])
        # Replies, including the tool server's responses to a private-network
        # client, must pass before the non-public drops below.
        run_firewall(
            binary,
            [
                "-A",
                "OUTPUT",
                "-m",
                "conntrack",
                "--ctstate",
                "ESTABLISHED,RELATED",
                "-j",
                "ACCEPT",
            ],
        )
        for resolver in resolvers:
            if (":" in resolver) == (family == "ipv6"):
                for protocol in ("udp", "tcp"):
                    run_firewall(binary, ["-A", "OUTPUT", "-p", protocol, "-d", resolver, "--dport", "53", "-j", "ACCEPT"])
        if allow_web:
            for network in NON_PUBLIC_NETWORKS.get(family, ()):
                run_firewall(binary, ["-A", "OUTPUT", "-d", network, "-j", "DROP"])
            for port in WEB_PORTS:
                run_firewall(binary, ["-A", "OUTPUT", "-p", "tcp", "--dport", port, "-j", "ACCEPT"])
        run_firewall(binary, ["-P", "OUTPUT", "DROP"])
        return False

    apply_rules(ipv4)

    interfaces = Path("/proc/net/if_inet6")
    has_external_ipv6 = interfaces.exists() and any(
        line.split()[-1] != "lo" for line in interfaces.read_text().splitlines() if line.split()
    )
    if not has_external_ipv6:
        return False
    if not ipv6:
        raise RuntimeError("controlled egress policy requires ip6tables when IPv6 is active")
    apply_rules(ipv6)
    return False


def drop_privileges(username="sandbox"):
    if os_geteuid() != 0:
        raise RuntimeError("sandbox entrypoint must start as root so it can switch to the sandbox account")
    account = pwd_getpwnam(username)
    os_setgroups([])
    os_setgid(account.pw_gid)
    os_setuid(account.pw_uid)
    os_environ["HOME"] = account.pw_dir
    return False


def verify_deployment_identity():
    """Start a deployment-enforced guest without retaining privilege or gaining it on exec."""
    if os_geteuid() == 0:
        raise RuntimeError("deployment-enforced Workspace must start as nonroot")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    if any(int(status.get(name, "1").strip(), 16) for name in ("CapEff", "CapPrm", "CapAmb")):
        raise RuntimeError("deployment-enforced Workspace must have zero guest capabilities")
    libc = CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise RuntimeError(f"Workspace could not install no-new-privileges: errno {get_errno()}")


def main(argv=COMMAND_LINE_ARGUMENTS):
    selected_arguments = sys_argv[1:] if argv is COMMAND_LINE_ARGUMENTS else argv
    arguments = list(selected_arguments)
    policy = parse_policy(arguments)
    if policy.egress_enforcement == "deployment":
        if policy.egress_policy not in {"web", "deny"}:
            raise RuntimeError("deployment-enforced Workspace requires controlled egress")
        verify_deployment_identity()
        os_execv(sys_executable, [sys_executable, str(SERVER), *arguments])
        return False
    if policy.egress_policy == "deny":
        enforce_egress_policy()
    elif policy.egress_policy == "web":
        enforce_egress_policy(allow_web=True)
    drop_privileges()
    os_execv(sys_executable, [sys_executable, str(SERVER), *arguments])
    return False


if __name__ == "__main__":
    main()
