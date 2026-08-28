"""Container lifecycle management for the authenticated Python sandbox."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module as _import_module
from ipaddress import AddressValueError as ipaddress_AddressValueError
from ipaddress import IPv4Address as ipaddress_IPv4Address
from logging import getLogger as logging_getLogger
from re import fullmatch as re_fullmatch
from secrets import token_urlsafe as secrets_token_urlsafe
from socket import AF_INET as socket_AF_INET
from socket import SOCK_STREAM as socket_SOCK_STREAM
from socket import socket as socket_socket
from subprocess import TimeoutExpired as subprocess_TimeoutExpired
from subprocess import run as subprocess_run
from sys import exc_info as sys_exc_info
from urllib import parse as urllib_parse

_DEFAULT_ARGUMENT_DICT = {}

try:
    boto3 = _import_module("boto3")
except ImportError:  # pragma: no cover - local runtimes do not need boto3
    boto3 = {}

LOGGER = logging_getLogger(__name__)
RUNTIME_COMMAND_TIMEOUT = 60
ALLOWED_RUNTIMES = {"container", "docker", "podman"}
ALLOWED_EGRESS_POLICIES = {"deny", "broker", "unrestricted"}
task_def_cache = {}


def available_port(host="127.0.0.1"):
    """Return a currently available TCP port for a short-lived sandbox."""
    with socket_socket(socket_AF_INET, socket_SOCK_STREAM) as listener:
        listener.bind((host, 0))
        _return_value = listener.getsockname()[1]
        return _return_value


@dataclass
class ContainerConfig:
    image: str = "python-sandbox:latest"
    workspace_mount: str = ""
    workspace: str = "/workspace"
    memory_limit: str = "2g"
    cpu_limit: float = 1.0
    port: int = 8080
    host_port: int = 0
    host_bind: str = "127.0.0.1"
    egress_policy: str = "deny"
    auth_token: str = field(default_factory=lambda: secrets_token_urlsafe(32))
    max_request_bytes: int = 1_048_576
    max_output_bytes: int = 1_048_576
    max_file_bytes: int = 10_485_760
    max_response_bytes: int = 12_582_912
    max_concurrent_requests: int = 4
    max_tool_timeout: int = 300
    request_read_timeout: float = 10.0
    pids_limit: int = 256
    broker_url: str = ""
    broker_token: str = ""
    broker_package_destination: str = "pypi"
    aws: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.host_port == 0:
            self.host_port = available_port(self.host_bind)
        validate_container_config(self)
        return False

    @property
    def base_url(self):
        _return_value = f"http://{self.host_bind}:{self.host_port}"
        return _return_value


def validate_container_config(config):
    """Fail before starting a container when boundary settings are unsafe."""
    if not isinstance(config.image, str) or not config.image.strip():
        raise ValueError("image must be a non-empty string")
    if not 1 <= int(config.port) <= 65535 or not 1 <= int(config.host_port) <= 65535:
        raise ValueError("container and host ports must be between 1 and 65535")
    if config.host_bind not in {"127.0.0.1"}:
        raise ValueError("MVP sandbox host_bind must be 127.0.0.1")
    if config.egress_policy not in ALLOWED_EGRESS_POLICIES:
        raise ValueError(f"unsupported egress policy: {config.egress_policy}")
    if config.egress_policy == "broker":
        parsed_broker = urllib_parse.urlsplit(config.broker_url)
        if (
            parsed_broker.scheme != "http"
            or not parsed_broker.hostname
            or parsed_broker.port is None
        ):
            raise ValueError(
                "broker policy requires an explicit http broker_url and port"
            )
        try:
            ipaddress_IPv4Address(parsed_broker.hostname)
        except ipaddress_AddressValueError as err:
            raise ValueError(
                "broker_url host must be an explicit IPv4 address"
            ) from err
        if (
            parsed_broker.path not in ("", "/")
            or parsed_broker.query
            or parsed_broker.fragment
            or parsed_broker.username
        ):
            raise ValueError(
                "broker_url must contain only scheme, IPv4 address, and port"
            )
        if len(config.broker_token) < 32:
            raise ValueError(
                "broker policy requires a broker_token of at least 32 characters"
            )
        if not re_fullmatch(
            r"[a-z][a-z0-9_-]{0,63}", config.broker_package_destination
        ):
            raise ValueError("broker_package_destination is invalid")
    elif config.broker_url or config.broker_token:
        raise ValueError("broker_url and broker_token require egress_policy='broker'")
    if len(config.auth_token) < 32:
        raise ValueError("auth_token must be at least 32 characters")
    if float(config.cpu_limit) <= 0:
        raise ValueError("cpu_limit must be greater than zero")
    parse_memory(config.memory_limit)
    for name in (
        "max_request_bytes",
        "max_output_bytes",
        "max_file_bytes",
        "max_response_bytes",
        "max_concurrent_requests",
        "max_tool_timeout",
        "pids_limit",
    ):
        if int(getattr(config, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if float(config.request_read_timeout) <= 0:
        raise ValueError("request_read_timeout must be positive")
    if config.max_response_bytes < 512:
        raise ValueError("max_response_bytes must be at least 512")
    return False


def parse_memory(mem):
    """Parse a positive memory string such as 2g or 512m to MiB."""
    match = re_fullmatch(r"([1-9][0-9]*)([gGmM])", str(mem))
    if not match:
        raise ValueError(f"Invalid memory format: {mem}. Use '2g' or '512m'.")
    amount, unit = match.groups()
    value = int(amount)
    _return_value = str(value * 1024 if unit.lower() == "g" else value)
    return _return_value


def server_arguments(config, *, auth_token=""):
    """Return the explicit server arguments shared by every runtime."""
    arguments = [
        "--workspace",
        config.workspace,
        "--port",
        str(config.port),
        "--auth-token",
        auth_token or config.auth_token,
        "--egress-policy",
        config.egress_policy,
        "--max-request-bytes",
        str(config.max_request_bytes),
        "--max-output-bytes",
        str(config.max_output_bytes),
        "--max-file-bytes",
        str(config.max_file_bytes),
        "--max-response-bytes",
        str(config.max_response_bytes),
        "--max-concurrent-requests",
        str(config.max_concurrent_requests),
        "--max-tool-timeout",
        str(config.max_tool_timeout),
        "--request-read-timeout",
        str(config.request_read_timeout),
    ]
    if config.egress_policy == "broker":
        arguments.extend(
            [
                "--broker-url",
                config.broker_url,
                "--broker-token",
                config.broker_token,
                "--broker-package-destination",
                config.broker_package_destination,
            ]
        )
    return arguments


def create_container_command(config, runtime="container"):
    """Build a local-runtime create command without executing it."""
    if runtime not in ALLOWED_RUNTIMES:
        raise ValueError(f"unsupported container runtime: {runtime}")
    validate_container_config(config)
    cpu_limit = float(config.cpu_limit)
    if runtime == "container":
        if not cpu_limit.is_integer():
            raise ValueError(
                "Apple container runtime requires a whole-number cpu_limit"
            )
        rendered_cpu_limit = str(int(cpu_limit))
    else:
        rendered_cpu_limit = str(cpu_limit)
    command = [runtime, "create"]
    if runtime in ("docker", "podman"):
        command.extend(
            [
                "--security-opt",
                "no-new-privileges",
                "--cap-drop",
                "ALL",
                "--pids-limit",
                str(config.pids_limit),
            ]
        )
        if config.egress_policy in {"deny", "broker"}:
            command.extend(["--cap-add", "NET_ADMIN"])
    if config.egress_policy in {"deny", "broker"}:
        command.extend(["--user", "root"])
    command.extend(
        [
            "--publish",
            f"{config.host_bind}:{config.host_port}:{config.port}",
            "--memory",
            config.memory_limit.upper(),
            "--cpus",
            rendered_cpu_limit,
        ]
    )
    if config.workspace_mount:
        command.extend(["--volume", f"{config.workspace_mount}:{config.workspace}"])
    command.extend([config.image, *server_arguments(config)])
    return command


def _run_runtime(command, operation, *, check=True):
    try:
        result = subprocess_run(
            command,
            capture_output=True,
            text=True,
            timeout=RUNTIME_COMMAND_TIMEOUT,
        )
    except subprocess_TimeoutExpired as err:
        raise RuntimeError(f"{operation} timed out") from err
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{operation} failed: {detail}")
    return result


def create_container(config, runtime="container"):
    result = _run_runtime(
        create_container_command(config, runtime), "container creation"
    )
    container_id = result.stdout.strip()
    if not container_id:
        raise RuntimeError("container creation returned no container identifier")
    return container_id


def start_container(container_id, runtime="container"):
    _run_runtime([runtime, "start", container_id], "container start")
    return False


def stop_container(container_id, runtime="container"):
    _run_runtime([runtime, "stop", "-t", "10", container_id], "container stop")
    return False


def destroy_container(container_id, runtime="container"):
    _run_runtime([runtime, "rm", "-f", container_id], "container destruction")
    return False


@contextmanager
def sandbox_session(config, runtime="container"):
    """Create, start, yield, and verifiably destroy a local sandbox."""
    container_id = create_container(config, runtime=runtime)
    try:
        start_container(container_id, runtime=runtime)
        yield container_id
    finally:
        active_error = sys_exc_info()[1]
        try:
            destroy_container(container_id, runtime=runtime)
        except Exception as cleanup_error:
            if active_error is None:
                raise
            if hasattr(active_error, "add_note"):
                active_error.add_note(f"sandbox cleanup also failed: {cleanup_error}")
            LOGGER.exception("Sandbox cleanup failed for %s", container_id)


def create_ecs_client(config):
    """Build an ECS client from explicit config values only."""
    if not boto3:
        return {}
    aws = config.aws or {}
    access_key = aws.get("access_key_id", "")
    secret_key = aws.get("secret_access_key", "")
    region = aws.get("region", "")
    if not (access_key and secret_key and region):
        return {}
    kwargs = {
        "region_name": region,
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }
    if aws.get("session_token", False):
        kwargs["aws_session_token"] = aws.get("session_token", False)
    client = boto3.client("ecs", **kwargs)
    return client


def task_definition_key(config):
    """Key every task-definition setting except the per-task auth token."""
    _return_value = ":".join(
        str(value)
        for value in (
            config.image,
            config.workspace,
            config.memory_limit,
            config.cpu_limit,
            config.port,
            config.egress_policy,
            config.max_request_bytes,
            config.max_output_bytes,
            config.max_file_bytes,
            config.max_response_bytes,
            config.max_concurrent_requests,
            config.max_tool_timeout,
            config.request_read_timeout,
        )
    )
    return _return_value


def create_fargate_task(config, cluster, subnet_ids, security_group_ids):
    """Start one private Fargate task with a per-task session token."""
    if config.egress_policy != "broker":
        raise RuntimeError(
            "Fargate requires an externally enforced broker egress policy"
        )
    if not subnet_ids or not security_group_ids:
        raise ValueError("Fargate requires explicit subnets and security groups")
    ecs = create_ecs_client(config)
    if not ecs:
        raise RuntimeError("Explicit AWS credentials are not configured")

    key = task_definition_key(config)
    task_def = task_def_cache.get(key, "")
    if not task_def:
        task_def = register_task_def(config, ecs)
        task_def_cache[key] = task_def

    response = ecs.run_task(
        cluster=cluster,
        taskDefinition=task_def,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": subnet_ids,
                "securityGroups": security_group_ids,
                "assignPublicIp": "DISABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": "sandbox",
                    "command": server_arguments(config),
                }
            ]
        },
    )
    failures = response.get("failures", [])
    tasks = response.get("tasks", [])
    if failures or not tasks:
        raise RuntimeError(
            f"Fargate task failed to start: {failures or 'no task returned'}"
        )
    _return_value = tasks[0].get("taskArn", "")
    return _return_value


def register_task_def(config, ecs=_DEFAULT_ARGUMENT_DICT):
    if ecs is _DEFAULT_ARGUMENT_DICT:
        ecs = _DEFAULT_ARGUMENT_DICT.copy()
    ecs = ecs or create_ecs_client(config)
    if not ecs:
        raise RuntimeError("Explicit AWS credentials are not configured")
    placeholder_token = "task-runtime-token-not-valid-for-live-requests"
    response = ecs.register_task_definition(
        family="python-sandbox",
        networkMode="awsvpc",
        requiresCompatibilities=["FARGATE"],
        cpu=str(int(config.cpu_limit * 1024)),
        memory=parse_memory(config.memory_limit),
        containerDefinitions=[
            {
                "name": "sandbox",
                "image": config.image,
                "essential": True,
                "command": server_arguments(config, auth_token=placeholder_token),
                "portMappings": [{"containerPort": config.port}],
            }
        ],
    )
    _return_value = response.get("taskDefinition", {}).get("taskDefinitionArn", "")
    return _return_value


def wait_for_fargate_task(task_arn, cluster, config):
    ecs = create_ecs_client(config)
    if not ecs:
        raise RuntimeError("Explicit AWS credentials are not configured")
    waiter = ecs.get_waiter("tasks_running")
    waiter.wait(cluster=cluster, tasks=[task_arn])
    return False


def stop_fargate_task(task_arn, cluster, config):
    ecs = create_ecs_client(config)
    if not ecs:
        raise RuntimeError("Explicit AWS credentials are not configured")
    ecs.stop_task(cluster=cluster, task=task_arn, reason="sandbox session complete")
    return False


@contextmanager
def fargate_session(config, cluster, subnet_ids, security_group_ids):
    """Start, wait for, yield, and always stop one Fargate sandbox."""
    task_arn = create_fargate_task(config, cluster, subnet_ids, security_group_ids)
    try:
        wait_for_fargate_task(task_arn, cluster, config)
        yield task_arn
    finally:
        stop_fargate_task(task_arn, cluster, config)


def get_fargate_endpoint(task_arn, cluster, port, config):
    ecs = create_ecs_client(config)
    if not ecs:
        raise RuntimeError("Explicit AWS credentials are not configured")
    response = ecs.describe_tasks(cluster=cluster, tasks=[task_arn])
    tasks = response.get("tasks", [])
    if not tasks:
        raise RuntimeError("Fargate task was not found")
    for attachment in tasks[0].get("attachments", []):
        if attachment.get("type", "") != "ElasticNetworkInterface":
            continue
        for detail in attachment.get("details", []):
            if detail.get("name", "") == "privateIPv4Address":
                _return_value = f"http://{detail.get('value', '')}:{port}"
                return _return_value
    raise RuntimeError("Could not find task IP address")
