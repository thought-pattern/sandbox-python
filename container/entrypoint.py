"""Configure the sandbox network boundary, drop privilege, and start the server."""

from argparse import ArgumentParser as argparse_ArgumentParser
from ipaddress import AddressValueError as ipaddress_AddressValueError
from ipaddress import IPv4Address as ipaddress_IPv4Address
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
from urllib import parse as urllib_parse

SERVER = Path(__file__).with_name("server.py")
COMMAND_LINE_ARGUMENTS = []


def parse_policy(argv):
    parser = argparse_ArgumentParser(add_help=False)
    parser.add_argument("--egress-policy", choices=("deny", "broker", "unrestricted"), default="deny")
    parser.add_argument("--broker-url", default="")
    computed_return_value = parser.parse_known_args(argv)[0]
    return computed_return_value


def run_firewall(binary, arguments):
    result = subprocess_run([binary, *arguments], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{binary} {' '.join(arguments)} failed: {result.stderr.strip()}")
    return False


def enforce_egress_policy(broker_url=""):
    """Allow loopback, replies, and optionally one broker; reject other egress."""
    ipv4 = shutil_which("iptables")
    ipv6 = shutil_which("ip6tables")
    if not ipv4:
        raise RuntimeError("deny egress policy requires iptables")

    broker_host = ""
    broker_port = 0
    if broker_url:
        parsed = urllib_parse.urlsplit(broker_url)
        if parsed.scheme != "http" or not parsed.hostname or parsed.port is None:
            raise RuntimeError("broker policy requires an explicit http broker URL and port")
        try:
            broker_host = str(ipaddress_IPv4Address(parsed.hostname))
        except ipaddress_AddressValueError as err:
            raise RuntimeError("broker URL must use an explicit IPv4 address") from err
        broker_port = parsed.port

    def apply_rules(binary):
        run_firewall(binary, ["-F", "OUTPUT"])
        run_firewall(binary, ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"])
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
        if broker_host:
            run_firewall(
                binary,
                [
                    "-A",
                    "OUTPUT",
                    "-p",
                    "tcp",
                    "-d",
                    broker_host,
                    "--dport",
                    str(broker_port),
                    "-m",
                    "conntrack",
                    "--ctstate",
                    "NEW",
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
        raise RuntimeError("deny egress policy requires ip6tables when IPv6 is active")
    apply_rules(ipv6)
    return False


def drop_privileges(username="sandbox"):
    if os_geteuid() != 0:
        raise RuntimeError("sandbox entrypoint must start as root so it can install the network boundary")
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
    elif policy.egress_policy == "broker":
        enforce_egress_policy(policy.broker_url)
    drop_privileges()
    os_execv(sys_executable, [sys_executable, str(SERVER), *arguments])
    return False


if __name__ == "__main__":
    main()
