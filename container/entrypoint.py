"""Configure the sandbox network boundary, drop privilege, and start the server."""

from argparse import ArgumentParser as argparse_ArgumentParser
from os import environ as os_environ
from os import execv as os_execv
from os import geteuid as os_geteuid
from os import setgid as os_setgid
from os import setgroups as os_setgroups
from os import setuid as os_setuid
from pathlib import Path
from pwd import getpwnam as pwd_getpwnam
from shutil import which as shutil_which
from socket import AF_INET6, IPPROTO_TCP
from socket import getaddrinfo as socket_getaddrinfo
from subprocess import run as subprocess_run
from sys import argv as sys_argv
from sys import executable as sys_executable

SERVER = Path(__file__).with_name("server.py")
COMMAND_LINE_ARGUMENTS = []


def parse_policy(argv):
    parser = argparse_ArgumentParser(add_help=False)
    parser.add_argument("--egress-policy", choices=("allowlist", "deny", "direct"), default="direct")
    parser.add_argument("--egress-allowlist", default="")
    computed_return_value = parser.parse_known_args(argv)[0]
    return computed_return_value


def run_firewall(binary, arguments):
    result = subprocess_run([binary, *arguments], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{binary} {' '.join(arguments)} failed: {result.stderr.strip()}")
    return False


def nameserver_addresses():
    """Read the resolvers the guest must reach to resolve allowlisted destinations."""
    resolv = Path("/etc/resolv.conf")
    lines = resolv.read_text().splitlines() if resolv.exists() else []
    result = [line.split()[1] for line in lines if line.strip().startswith("nameserver") and len(line.split()) > 1]
    return result


def allowlist_addresses(hosts):
    """Resolve each allowlisted destination once, before privileges drop; unresolvable hosts fail closed."""
    addresses = {"ipv4": set(), "ipv6": set()}
    for host in hosts:
        try:
            records = socket_getaddrinfo(host, 443, proto=IPPROTO_TCP)
        except OSError as err:
            raise RuntimeError(f"allowlisted egress destination {host} does not resolve: {err}") from err
        for family, _, _, _, address in records:
            addresses["ipv6" if family == AF_INET6 else "ipv4"].add(address[0])
    return addresses


def enforce_egress_policy(allowed_hosts=()):
    """Allow only loopback and replies, plus DNS and HTTP(S) to resolved allowlisted destinations."""
    ipv4 = shutil_which("iptables")
    ipv6 = shutil_which("ip6tables")
    if not ipv4:
        raise RuntimeError("controlled egress policy requires iptables")
    destinations = allowlist_addresses(allowed_hosts) if allowed_hosts else {"ipv4": set(), "ipv6": set()}
    resolvers = nameserver_addresses() if allowed_hosts else []

    def apply_rules(binary):
        family = "ipv6" if binary == ipv6 else "ipv4"
        run_firewall(binary, ["-F", "OUTPUT"])
        run_firewall(binary, ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"])
        for resolver in resolvers:
            if (":" in resolver) == (family == "ipv6"):
                for protocol in ("udp", "tcp"):
                    run_firewall(binary, ["-A", "OUTPUT", "-p", protocol, "-d", resolver, "--dport", "53", "-j", "ACCEPT"])
        for address in sorted(destinations.get(family, set())):
            for port in ("443", "80"):
                run_firewall(binary, ["-A", "OUTPUT", "-p", "tcp", "-d", address, "--dport", port, "-j", "ACCEPT"])
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


def main(argv=COMMAND_LINE_ARGUMENTS):
    selected_arguments = sys_argv[1:] if argv is COMMAND_LINE_ARGUMENTS else argv
    arguments = list(selected_arguments)
    policy = parse_policy(arguments)
    if policy.egress_policy == "deny":
        enforce_egress_policy()
    elif policy.egress_policy == "allowlist":
        hosts = tuple(host for host in policy.egress_allowlist.split(",") if host)
        if not hosts:
            raise RuntimeError("allowlist egress policy requires at least one destination")
        enforce_egress_policy(hosts)
    drop_privileges()
    os_execv(sys_executable, [sys_executable, str(SERVER), *arguments])
    return False


if __name__ == "__main__":
    main()
