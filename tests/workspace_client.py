# Copyright 2025-2026 Jason E. Robinson.
# SPDX-License-Identifier: Apache-2.0

"""Standard-library HTTP client that drives the tool server in tests."""

from copy import deepcopy
from http.client import HTTPConnection
from json import dumps as json_dumps
from json import loads as json_loads
from urllib.parse import urlsplit

DEFAULT_ARGUMENT_DICT = {}
DEFAULT_TIMEOUT = 120.0


class WorkspaceToolError(RuntimeError):
    """A structured failure retaining the HTTP status, error code and process result."""

    def __init__(self, message, *, code="tool_error", status_code=0, result=DEFAULT_ARGUMENT_DICT):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.result = deepcopy(result)


class WorkspaceClient:
    """Authenticated client bound to one server URL and bearer token.

    http.client never consults proxy environment variables, so every request
    reaches the server under test directly.
    """

    def __init__(self, base_url: str, auth_token: str, timeout: float = DEFAULT_TIMEOUT):
        parts = urlsplit(base_url)
        if parts.scheme != "http" or not parts.hostname or not parts.port:
            raise ValueError("base_url must be http://host:port")
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout
        self.host = parts.hostname
        self.port = parts.port

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        return None

    def request(self, method: str, path: str, arguments: dict = DEFAULT_ARGUMENT_DICT, timeout: float = 0.0):
        """Perform one authenticated request and return its status and decoded payload."""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        body = None
        if method == "POST":
            body = json_dumps(arguments).encode("utf-8")
            headers["Content-Type"] = "application/json"
        connection = HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            status_code = response.status
            raw = response.read()
        except OSError as err:
            raise WorkspaceToolError(f"{method} {path} transport failed: {err}", code="transport_error") from err
        finally:
            connection.close()
        try:
            payload = json_loads(raw)
        except ValueError as err:
            raise WorkspaceToolError(
                f"{method} {path} returned a non-JSON response", code="invalid_response", status_code=status_code
            ) from err
        if not isinstance(payload, dict):
            raise WorkspaceToolError(
                f"{method} {path} returned a non-object response", code="invalid_response", status_code=status_code
            )
        if status_code >= 400 or "error" in payload:
            error = payload.get("error", {})
            if not isinstance(error, dict):
                error = {}
            raise WorkspaceToolError(
                error.get("message", f"{method} {path} failed with status {status_code}"),
                code=error.get("code", "tool_error"),
                status_code=status_code,
            )
        return status_code, payload

    def health(self, timeout: float = 0.0) -> dict:
        status_code, payload = self.request("GET", "/health", timeout=timeout)
        return payload

    def fetch_manifest(self, timeout: float = 0.0) -> dict:
        status_code, payload = self.request("GET", "/tools", timeout=timeout)
        return payload

    def call_tool(self, name: str, arguments: dict = DEFAULT_ARGUMENT_DICT, timeout: float = 0.0):
        """Return a tool result, raising when the request or the tool's process failed."""
        status_code, payload = self.request("POST", f"/tools/{name}", arguments, timeout)
        if "result" not in payload:
            raise WorkspaceToolError("tool response has no result", code="invalid_response", status_code=status_code)
        result = payload.get("result", {})
        if isinstance(result, dict) and result.get("ok", True) is False:
            if result.get("timedOut", False):
                code = "timeout"
            elif result.get("outputLimited", False):
                code = "output_limit"
            else:
                code = "process_failed"
            message = result.get("stderr", "") or result.get("stdout", "") or f"tool {name} failed"
            raise WorkspaceToolError(message.strip(), code=code, status_code=status_code, result=result)
        return result
