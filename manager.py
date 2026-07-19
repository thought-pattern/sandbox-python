"""Container lifecycle management for Python sandbox."""

import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field

try:
    import boto3
except ImportError:  # pragma: no cover - local runtimes do not need boto3
    boto3 = None

task_def_cache = {}


@dataclass
class ContainerConfig:
    image: str = "python-sandbox:latest"
    workspace_mount: str = None
    workspace: str = "/workspace"
    memory_limit: str = "2g"
    cpu_limit: float = 1.0
    port: int = 8080
    aws: dict = field(default_factory=dict)


def parse_memory(mem):
    """Parse memory string like '2g' or '512m' to MB string."""
    mem = mem.lower()
    if mem.endswith("g"):
        return str(int(mem[:-1]) * 1024)
    elif mem.endswith("m"):
        return mem[:-1]
    else:
        raise ValueError(f"Invalid memory format: {mem}. Use '2g' or '512m'.")


def create_container(config, runtime="container"):
    cmd = [runtime, "create"]
    # Docker and podman share the host kernel, so the sandbox drops Linux
    # capabilities and blocks privilege escalation there. Apple's container
    # runtime gives each container its own lightweight VM -- a stronger
    # boundary that neither exposes nor needs those flags; the non-root USER
    # in the image still applies inside the guest.
    if runtime in ("docker", "podman"):
        cmd.extend(["--security-opt", "no-new-privileges", "--cap-drop", "ALL"])
    cmd.extend(
        [
            "--publish",
            f"{config.port}:{config.port}",
            "--memory",
            config.memory_limit.upper(),
            "--cpus",
            str(int(config.cpu_limit)),
        ]
    )
    if config.workspace_mount:
        cmd.extend(["--volume", f"{config.workspace_mount}:{config.workspace}"])
    cmd.extend([config.image, "--workspace", config.workspace, "--port", str(config.port)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create container: {result.stderr}")
    return result.stdout.strip()


def start_container(container_id, runtime="container"):
    result = subprocess.run([runtime, "start", container_id], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container: {result.stderr}")


def stop_container(container_id, runtime="container"):
    subprocess.run([runtime, "stop", "-t", "10", container_id], capture_output=True, text=True)


def destroy_container(container_id, runtime="container"):
    subprocess.run([runtime, "rm", "-f", container_id], capture_output=True, text=True)


@contextmanager
def sandbox_session(config, runtime="container"):
    """Create, start, and always destroy a sandbox container.

    The container is torn down on the way out even when the work inside the
    block raises, so a sandbox is disposable by construction rather than by
    the caller remembering to clean up.
    """
    container_id = create_container(config, runtime=runtime)
    try:
        start_container(container_id, runtime=runtime)
        yield container_id
    finally:
        destroy_container(container_id, runtime=runtime)


def create_ecs_client(config):
    """Build an ECS client from explicit config values only."""
    if boto3 is None:
        return None
    aws = config.aws or {}
    access_key = aws.get("access_key_id", "")
    secret_key = aws.get("secret_access_key", "")
    region = aws.get("region", "")
    if not (access_key and secret_key and region):
        return None
    kwargs = {
        "region_name": region,
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }
    if aws.get("session_token"):
        kwargs["aws_session_token"] = aws["session_token"]
    return boto3.client("ecs", **kwargs)


def create_fargate_task(config, cluster, subnet_ids, security_group_ids):
    ecs = create_ecs_client(config)
    if ecs is None:
        raise RuntimeError("Explicit AWS credentials are not configured")

    key = f"{config.image}:{config.memory_limit}:{config.cpu_limit}:{config.port}"
    if key in task_def_cache:
        task_def = task_def_cache[key]
    else:
        task_def = register_task_def(config, ecs)
        task_def_cache[key] = task_def

    resp = ecs.run_task(
        cluster=cluster,
        taskDefinition=task_def,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {"subnets": subnet_ids, "securityGroups": security_group_ids, "assignPublicIp": "DISABLED"}
        },
    )
    return resp["tasks"][0]["taskArn"]


def register_task_def(config, ecs=None):
    ecs = ecs or create_ecs_client(config)
    if ecs is None:
        raise RuntimeError("Explicit AWS credentials are not configured")
    resp = ecs.register_task_definition(
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
                "command": ["--workspace", config.workspace, "--port", str(config.port)],
                "portMappings": [{"containerPort": config.port}],
            }
        ],
    )
    return resp["taskDefinition"]["taskDefinitionArn"]


def wait_for_fargate_task(task_arn, cluster, config):
    ecs = create_ecs_client(config)
    if ecs is None:
        raise RuntimeError("Explicit AWS credentials are not configured")
    waiter = ecs.get_waiter("tasks_running")
    waiter.wait(cluster=cluster, tasks=[task_arn])


def stop_fargate_task(task_arn, cluster, config):
    ecs = create_ecs_client(config)
    if ecs is None:
        raise RuntimeError("Explicit AWS credentials are not configured")
    ecs.stop_task(cluster=cluster, task=task_arn)


def get_fargate_endpoint(task_arn, cluster, port, config):
    ecs = create_ecs_client(config)
    if ecs is None:
        raise RuntimeError("Explicit AWS credentials are not configured")
    resp = ecs.describe_tasks(cluster=cluster, tasks=[task_arn])
    task = resp["tasks"][0]

    for att in task.get("attachments", []):
        if att.get("type") == "ElasticNetworkInterface":
            for d in att.get("details", []):
                if d.get("name") == "privateIPv4Address":
                    return f"http://{d['value']}:{port}"

    raise RuntimeError("Could not find task IP address")
