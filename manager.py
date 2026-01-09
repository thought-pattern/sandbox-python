"""Container lifecycle management for Python sandbox."""

import subprocess
from dataclasses import dataclass, field

import boto3

try:
    ecs = boto3.client("ecs")
except Exception:
    ecs = None

task_def_cache = {}


@dataclass
class ContainerConfig:
    image: str = "python-sandbox:latest"
    workspace_mount: str = None
    environment: dict = field(default_factory=dict)
    memory_limit: str = "2g"
    cpu_limit: float = 1.0
    port: int = 8080


def parse_memory(mem):
    """Parse memory string like '2g' or '512m' to MB string."""
    mem = mem.lower()
    if mem.endswith("g"):
        return str(int(mem[:-1]) * 1024)
    elif mem.endswith("m"):
        return mem[:-1]
    else:
        raise ValueError(f"Invalid memory format: {mem}. Use '2g' or '512m'.")


def create_container(config, runtime="docker"):
    cmd = [
        runtime, "create",
        "--publish", f"{config.port}:{config.port}",
        "--memory", config.memory_limit,
        "--cpus", str(config.cpu_limit),
        "--env", f"MCP_PORT={config.port}",
    ]
    for k, v in config.environment.items():
        cmd.extend(["--env", f"{k}={v}"])
    if config.workspace_mount:
        cmd.extend(["--volume", f"{config.workspace_mount}:/workspace"])
    cmd.append(config.image)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create container: {result.stderr}")
    return result.stdout.strip()


def start_container(container_id, runtime="docker"):
    result = subprocess.run([runtime, "start", container_id], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container: {result.stderr}")


def stop_container(container_id, runtime="docker"):
    subprocess.run([runtime, "stop", "-t", "10", container_id], capture_output=True, text=True)


def destroy_container(container_id, runtime="docker"):
    subprocess.run([runtime, "rm", "-f", container_id], capture_output=True, text=True)


def create_fargate_task(config, cluster, subnet_ids, security_group_ids):
    if ecs is None:
        raise RuntimeError("AWS credentials not configured")

    key = f"{config.image}:{config.memory_limit}:{config.cpu_limit}:{config.port}"
    if key in task_def_cache:
        task_def = task_def_cache[key]
    else:
        task_def = register_task_def(config)
        task_def_cache[key] = task_def

    resp = ecs.run_task(
        cluster=cluster,
        taskDefinition=task_def,
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": subnet_ids,
                "securityGroups": security_group_ids,
                "assignPublicIp": "DISABLED"
            }
        }
    )
    return resp["tasks"][0]["taskArn"]


def register_task_def(config):
    if ecs is None:
        raise RuntimeError("AWS credentials not configured")

    env = [{"name": "MCP_PORT", "value": str(config.port)}]
    env.extend({"name": k, "value": v} for k, v in config.environment.items())

    resp = ecs.register_task_definition(
        family="python-sandbox",
        networkMode="awsvpc",
        requiresCompatibilities=["FARGATE"],
        cpu=str(int(config.cpu_limit * 1024)),
        memory=parse_memory(config.memory_limit),
        containerDefinitions=[{
            "name": "sandbox",
            "image": config.image,
            "essential": True,
            "environment": env,
            "portMappings": [{"containerPort": config.port}]
        }]
    )
    return resp["taskDefinition"]["taskDefinitionArn"]


def wait_for_fargate_task(task_arn, cluster):
    if ecs is None:
        raise RuntimeError("AWS credentials not configured")
    waiter = ecs.get_waiter("tasks_running")
    waiter.wait(cluster=cluster, tasks=[task_arn])


def stop_fargate_task(task_arn, cluster):
    if ecs is None:
        raise RuntimeError("AWS credentials not configured")
    ecs.stop_task(cluster=cluster, task=task_arn)


def get_fargate_endpoint(task_arn, cluster, port):
    if ecs is None:
        raise RuntimeError("AWS credentials not configured")
    resp = ecs.describe_tasks(cluster=cluster, tasks=[task_arn])
    task = resp["tasks"][0]

    for att in task.get("attachments", []):
        if att.get("type") == "ElasticNetworkInterface":
            for d in att.get("details", []):
                if d.get("name") == "privateIPv4Address":
                    return f"http://{d['value']}:{port}"

    raise RuntimeError("Could not find task IP address")
