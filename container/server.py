"""HTTP tool server for the Python sandbox.

Exposes the workspace tools over a small plain-HTTP interface the consultant
reaches as a client, using only the standard library: GET /health for liveness,
GET /tools for the self-describing manifest, and POST /tools/<name> to invoke a
tool with a JSON object of arguments. The same interface serves development and
production; only the endpoint URL changes.
"""

import argparse
import inspect
import json
import re
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_WORKSPACE = Path("/workspace")
DEFAULT_PORT = 8080
WORKSPACE = DEFAULT_WORKSPACE

TOOLS = {}


def tool(fn):
    """Register a function as a callable tool surfaced in the manifest."""
    TOOLS[fn.__name__] = fn
    return fn


def resolve_path(path):
    """Resolve a path within the workspace, rejecting anything outside it."""
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError("Path must be within workspace")
    return target


@tool
def file_read(path):
    """Read contents of a file."""
    return resolve_path(path).read_text()


@tool
def file_write(path, content):
    """Write content to a file."""
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} bytes"


@tool
def file_patch(path, patches):
    """Apply find/replace patches to a file."""
    target = resolve_path(path)
    content = target.read_text()
    for patch in patches:
        old = patch.get("old", "")
        if old not in content:
            raise ValueError(f"Not found: {old[:50]}...")
        content = content.replace(old, patch.get("new", ""), 1)
    target.write_text(content)
    return f"Applied {len(patches)} patches"


@tool
def file_delete(path):
    """Delete a file."""
    resolve_path(path).unlink(missing_ok=False)
    return f"Deleted {path}"


@tool
def file_list(path=".", depth=2):
    """List files in a directory."""
    target = resolve_path(path)
    result = subprocess.run(
        ["find", str(target), "-maxdepth", str(depth), "-type", "f"], capture_output=True, text=True, cwd=WORKSPACE
    )
    lines = [l for l in result.stdout.strip().split("\n") if l]
    return [str(Path(f).relative_to(WORKSPACE)) for f in lines]


@tool
def file_search(pattern, path="."):
    """Search for a pattern in files using ripgrep."""
    target = resolve_path(path)
    result = subprocess.run(["rg", "--json", pattern, str(target)], capture_output=True, text=True, cwd=WORKSPACE)
    matches = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        if data.get("type") == "match":
            d = data.get("data", {})
            matches.append(
                {
                    "file": str(Path(d["path"]["text"]).relative_to(WORKSPACE)),
                    "line": d["line_number"],
                    "content": d["lines"]["text"].strip(),
                }
            )
    return matches


@tool
def pip_install(packages):
    """Install Python packages."""
    for pkg in packages:
        if not re.match(r"^[a-zA-Z0-9_.-]+([=<>!~\[\]][a-zA-Z0-9._,<>=!~\[\]]*)?$", pkg):
            raise ValueError(f"Invalid package: {pkg}")
    result = subprocess.run(["pip", "install", "--no-cache-dir"] + packages, capture_output=True, text=True, timeout=300)
    return {"stdout": result.stdout, "stderr": result.stderr, "exitCode": result.returncode}


@tool
def pip_uninstall(packages):
    """Uninstall Python packages."""
    for pkg in packages:
        if not re.match(r"^[a-zA-Z0-9_.-]+$", pkg):
            raise ValueError(f"Invalid package: {pkg}")
    result = subprocess.run(["pip", "uninstall", "-y"] + packages, capture_output=True, text=True, timeout=60)
    return {"stdout": result.stdout, "stderr": result.stderr, "exitCode": result.returncode}


@tool
def pip_list():
    """List installed packages."""
    result = subprocess.run(["pip", "list", "--format=json"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


@tool
def pip_freeze():
    """Get installed packages in requirements.txt format."""
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
    return result.stdout


@tool
def run_command(command, timeout=60):
    """Execute a shell command in the workspace."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=WORKSPACE, timeout=timeout)
        return {"stdout": result.stdout, "stderr": result.stderr, "exitCode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timed out after {timeout}s", "exitCode": -1}


@tool
def run_python(script, timeout=60):
    """Execute Python code."""
    try:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=WORKSPACE, timeout=timeout)
        return {"stdout": result.stdout, "stderr": result.stderr, "exitCode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timed out after {timeout}s", "exitCode": -1}


@tool
def git_clone(repo_url, branch="main"):
    """Clone a repository into the workspace."""
    for item in WORKSPACE.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    result = subprocess.run(["git", "clone", "--branch", branch, repo_url, "."], capture_output=True, text=True, cwd=WORKSPACE)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return f"Cloned {repo_url} (branch: {branch})"


@tool
def git_status():
    """Get the current git status."""
    branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, cwd=WORKSPACE)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=WORKSPACE)
    return {"branch": branch.stdout.strip(), "changes": [l for l in status.stdout.strip().split("\n") if l]}


@tool
def git_diff(staged=False):
    """Get the diff of changes."""
    cmd = ["git", "diff", "--staged"] if staged else ["git", "diff"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE)
    return result.stdout


@tool
def git_commit(message):
    """Stage all changes and commit."""
    add_result = subprocess.run(["git", "add", "-A"], capture_output=True, text=True, cwd=WORKSPACE)
    if add_result.returncode != 0:
        raise RuntimeError(add_result.stderr)
    result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True, cwd=WORKSPACE)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout


@tool
def git_push(remote="origin", branch=None):
    """Push commits to a remote."""
    if branch is None:
        result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, cwd=WORKSPACE)
        branch = result.stdout.strip()
    result = subprocess.run(["git", "push", remote, branch], capture_output=True, text=True, cwd=WORKSPACE)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return f"Pushed to {remote}/{branch}"


@tool
def python_version():
    """Get the Python version string."""
    return sys.version


def build_manifest():
    """Describe every registered tool: name, description, and parameters."""
    tools = []
    for name, fn in TOOLS.items():
        parameters = []
        for param_name, param in inspect.signature(fn).parameters.items():
            required = param.default is inspect.Parameter.empty
            parameters.append({"name": param_name, "required": required, "default": None if required else param.default})
        tools.append({"name": name, "description": (fn.__doc__ or "").strip(), "parameters": parameters})
    return {"tools": tools}


class ToolHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "healthy", "python": sys.version})
        elif self.path == "/tools":
            self.send_json(200, build_manifest())
        else:
            self.send_json(404, {"error": f"no such path: {self.path}"})

    def do_POST(self):
        if not self.path.startswith("/tools/"):
            self.send_json(404, {"error": f"no such path: {self.path}"})
            return
        name = self.path[len("/tools/") :]
        fn = TOOLS.get(name)
        if fn is None:
            self.send_json(404, {"error": f"no such tool: {name}"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            arguments = json.loads(raw) if raw else {}
        except ValueError as err:
            self.send_json(400, {"error": f"invalid JSON body: {err}"})
            return
        if not isinstance(arguments, dict):
            self.send_json(400, {"error": "request body must be a JSON object of arguments"})
            return
        try:
            result = fn(**arguments)
        except TypeError as err:
            self.send_json(400, {"error": f"bad arguments for {name}: {err}"})
            return
        except Exception as err:
            self.send_json(500, {"error": str(err)})
            return
        self.send_json(200, {"result": result})

    def log_message(self, format, *args):
        return


def parse_args(argv=None):
    """Parse explicit server settings supplied by the container manager."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args(argv)


def main(argv=None):
    global WORKSPACE
    args = parse_args(argv)
    WORKSPACE = args.workspace.resolve()
    ThreadingHTTPServer(("0.0.0.0", args.port), ToolHandler).serve_forever()


if __name__ == "__main__":
    main()
