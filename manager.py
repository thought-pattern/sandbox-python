"""Container lifecycle management for Python sandbox."""

import subprocess

import boto3

try:
    ecs = boto3.client("ecs")
    ecs_init_error = None
except Exception as exc:  # region/profile/config resolution errors surface here
    ecs = None
    ecs_init_error = exc

task_def_cache = {}

DEFAULT_CONFIG = {
    "image": "python-sandbox:latest",
    "workspace_mount": "",
    "environment": {},
    "memory_limit": "2g",
    "cpu_limit": 1.0,
    "port": 8080,
}

CREATE_TIMEOUT = 120
RUNTIME_TIMEOUT = 60


def require_ecs():
    """Return the ECS client, or raise with the original initialization error.

    Returns:
        The boto3 ECS client.

    Raises:
        RuntimeError: If the client failed to initialize (missing region,
            credentials, or configuration).
    """
    if ecs is None:
        raise RuntimeError(f"AWS ECS client unavailable: {ecs_init_error}")
    return ecs


def parse_memory(mem):
    """Parse a memory string like '2g' or '512m' into a MB string.

    Args:
        mem: Memory size such as '2g' or '512m'.

    Returns:
        The size in megabytes as a string, e.g. '2048'.

    Raises:
        ValueError: If the format is not recognized.
    """
    mem = mem.lower()
    if mem.endswith("g"):
        mb = str(int(mem[:-1]) * 1024)
    elif mem.endswith("m"):
        mb = mem[:-1]
    else:
        raise ValueError(f"Invalid memory format: {mem}. Use '2g' or '512m'.")
    return mb


def create_container(config, runtime="docker"):
    """Create a sandbox container from a config dict.

    Args:
        config: Partial container config; missing keys fall back to
            DEFAULT_CONFIG.
        runtime: Container runtime executable ('docker' or 'podman').

    Returns:
        The created container id.

    Raises:
        RuntimeError: If the runtime fails to create the container.
    """
    config = {**DEFAULT_CONFIG, **config}
    port = config.get("port")
    cmd = [
        runtime,
        "create",
        "--publish",
        f"{port}:{port}",
        "--memory",
        config.get("memory_limit"),
        "--cpus",
        str(config.get("cpu_limit")),
        "--env",
        f"MCP_PORT={port}",
    ]
    for k, v in config.get("environment", {}).items():
        cmd.extend(["--env", f"{k}={v}"])
    workspace_mount = config.get("workspace_mount")
    if workspace_mount:
        # --mount separates source/destination by commas rather than colons,
        # so Windows source paths (e.g. C:\work) are not mis-split.
        cmd.extend(["--mount", f"type=bind,source={workspace_mount},destination=/workspace"])
    cmd.append(config.get("image"))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=CREATE_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create container: {result.stderr}")
    container_id = result.stdout.strip()
    return container_id


def start_container(container_id, runtime="docker"):
    """Start a previously created container.

    Args:
        container_id: Id returned by create_container.
        runtime: Container runtime executable.

    Raises:
        RuntimeError: If the runtime fails to start the container.
    """
    result = subprocess.run(
        [runtime, "start", container_id], capture_output=True, text=True, timeout=RUNTIME_TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container: {result.stderr}")


def stop_container(container_id, runtime="docker"):
    """Stop a running container.

    Args:
        container_id: Id of the container to stop.
        runtime: Container runtime executable.

    Raises:
        RuntimeError: If the runtime fails to stop the container.
    """
    result = subprocess.run(
        [runtime, "stop", "-t", "10", container_id], capture_output=True, text=True, timeout=RUNTIME_TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to stop container: {result.stderr}")


def destroy_container(container_id, runtime="docker"):
    """Remove a container, force-killing it if still running.

    Args:
        container_id: Id of the container to remove.
        runtime: Container runtime executable.

    Raises:
        RuntimeError: If the runtime fails to remove the container.
    """
    result = subprocess.run(
        [runtime, "rm", "-f", container_id], capture_output=True, text=True, timeout=RUNTIME_TIMEOUT
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to destroy container: {result.stderr}")


def create_fargate_task(config, cluster, subnet_ids, security_group_ids):
    """Run the sandbox as a Fargate task, reusing a cached task definition.

    Args:
        config: Partial container config; missing keys fall back to
            DEFAULT_CONFIG.
        cluster: ECS cluster name or ARN.
        subnet_ids: Non-empty list of subnet ids for the task ENI.
        security_group_ids: Non-empty list of security group ids.

    Returns:
        The running task ARN.

    Raises:
        ValueError: If subnet_ids or security_group_ids is empty.
        RuntimeError: If ECS is unavailable or task placement fails.
    """
    client = require_ecs()
    if not subnet_ids:
        raise ValueError("subnet_ids must not be empty")
    if not security_group_ids:
        raise ValueError("security_group_ids must not be empty")

    config = {**DEFAULT_CONFIG, **config}
    # Environment is baked into the task definition, so it must be part of the
    # cache key or two configs differing only in env would share a definition.
    env_key = ",".join(f"{k}={v}" for k, v in sorted(config.get("environment", {}).items()))
    key = f"{config.get('image')}:{config.get('memory_limit')}:{config.get('cpu_limit')}:{config.get('port')}:{env_key}"

    task_def = task_def_cache.get(key)
    if task_def is None:
        task_def = register_task_def(config)
        task_def_cache[key] = task_def

    resp = client.run_task(
        cluster=cluster,
        taskDefinition=task_def,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {"subnets": subnet_ids, "securityGroups": security_group_ids, "assignPublicIp": "DISABLED"}
        },
    )

    failures = resp.get("failures")
    if failures:
        raise RuntimeError(f"Failed to run Fargate task: {failures}")
    tasks = resp.get("tasks")
    if not tasks:
        raise RuntimeError("run_task returned no tasks and no failures")
    task_arn = tasks[0].get("taskArn")
    return task_arn


def register_task_def(config):
    """Register an ECS task definition for the sandbox container.

    Args:
        config: Partial container config; missing keys fall back to
            DEFAULT_CONFIG.

    Returns:
        The registered task definition ARN.

    Raises:
        RuntimeError: If ECS is unavailable.
    """
    client = require_ecs()

    config = {**DEFAULT_CONFIG, **config}
    cpu_limit = config.get("cpu_limit")
    if not isinstance(cpu_limit, (int, float)):
        raise ValueError(f"cpu_limit must be numeric, got {cpu_limit!r}")

    env = [{"name": "MCP_PORT", "value": str(config.get("port"))}]
    env.extend({"name": k, "value": v} for k, v in config.get("environment", {}).items())

    resp = client.register_task_definition(
        family="python-sandbox",
        networkMode="awsvpc",
        requiresCompatibilities=["FARGATE"],
        cpu=str(int(cpu_limit * 1024)),
        memory=parse_memory(config.get("memory_limit")),
        containerDefinitions=[
            {
                "name": "sandbox",
                "image": config.get("image"),
                "essential": True,
                "environment": env,
                "portMappings": [{"containerPort": config.get("port")}],
            }
        ],
    )
    task_def_arn = resp.get("taskDefinition", {}).get("taskDefinitionArn")
    return task_def_arn


def wait_for_fargate_task(task_arn, cluster):
    """Block until the given Fargate task reaches the RUNNING state.

    Args:
        task_arn: ARN of the task to wait on.
        cluster: ECS cluster name or ARN.

    Raises:
        RuntimeError: If ECS is unavailable.
    """
    client = require_ecs()
    waiter = client.get_waiter("tasks_running")
    waiter.wait(cluster=cluster, tasks=[task_arn])


def stop_fargate_task(task_arn, cluster):
    """Stop a running Fargate task.

    Args:
        task_arn: ARN of the task to stop.
        cluster: ECS cluster name or ARN.

    Raises:
        RuntimeError: If ECS is unavailable.
    """
    client = require_ecs()
    client.stop_task(cluster=cluster, task=task_arn)


def get_fargate_endpoint(task_arn, cluster, port):
    """Resolve the private HTTP endpoint of a running Fargate task.

    Args:
        task_arn: ARN of the running task.
        cluster: ECS cluster name or ARN.
        port: Container port to address.

    Returns:
        An 'http://IP:port' endpoint string.

    Raises:
        RuntimeError: If ECS is unavailable or the task IP cannot be found.
    """
    client = require_ecs()
    resp = client.describe_tasks(cluster=cluster, tasks=[task_arn])
    tasks = resp.get("tasks")
    if not tasks:
        raise RuntimeError("Task not found")
    task = tasks[0]

    for att in task.get("attachments", []):
        if att.get("type") == "ElasticNetworkInterface":
            for d in att.get("details", []):
                if d.get("name") == "privateIPv4Address":
                    ip = d.get("value")
                    if ip:
                        endpoint = f"http://{ip}:{port}"
                        return endpoint

    raise RuntimeError("Could not find task IP address")
