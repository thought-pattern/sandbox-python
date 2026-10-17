# Copyright 2025-2026 Thought Pattern AI, LLC
# SPDX-License-Identifier: Apache-2.0

"""Retained component harness client using the actual Workspace protobuf interface."""

from copy import deepcopy
from uuid import uuid4

from grpc import RpcError, StatusCode, insecure_channel

from tapestry.interfaces.envelope import decode_envelope, encode_envelope
from tapestry.interfaces.workspace.constants import DEFAULT_MAX_REQUEST_BYTES, DEFAULT_MAX_RESPONSE_BYTES
from tapestry.interfaces.workspace.workspace_pb2 import WorkspaceEnvelope
from tapestry.interfaces.workspace.workspace_pb2_grpc import WorkspaceServiceStub

DEFAULT_ARGUMENT_DICT = {}
DEFAULT_TIMEOUT = 120.0
DEFAULT_ROOT_TASK_ID = "019cdc3d-a855-74f7-b720-f730ead81438"


class WorkspaceToolError(RuntimeError):
    """A tool observation or RPC failure retained by the component harness."""

    def __init__(self, message, *, code="tool_error", status_code=0, result=DEFAULT_ARGUMENT_DICT):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.result = deepcopy(result)


class WorkspaceClient:
    """Own a harness RPC connection and one default root Task."""

    def __init__(self, address: str, auth_token: str, timeout: float = DEFAULT_TIMEOUT, root_task_id: str = DEFAULT_ROOT_TASK_ID):
        self.address = address
        self.auth_token = auth_token
        self.timeout = timeout
        self.root_task_id = root_task_id
        self.channel = insecure_channel(address)
        self.stub = WorkspaceServiceStub(self.channel)
        self.health_address = ()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.channel.close()

    def rpc(self, operation: str, payload: dict, timeout: float = 0.0) -> dict:
        request = WorkspaceEnvelope(payload=encode_envelope(payload, DEFAULT_MAX_REQUEST_BYTES, "Workspace"))
        metadata = () if operation == "CheckHealth" else (("authorization", f"Bearer {self.auth_token}"),)
        try:
            response = getattr(self.stub, operation)(request, timeout=timeout or self.timeout, metadata=metadata)
        except RpcError as err:
            unauthorized = err.code() == StatusCode.UNAUTHENTICATED
            raise WorkspaceToolError(
                "Workspace RPC refused",
                code="unauthorized" if unauthorized else "transport_error",
                status_code=401 if unauthorized else 0,
            ) from err
        result = decode_envelope(response.payload, DEFAULT_MAX_RESPONSE_BYTES, "Workspace")
        return result

    def health(self, timeout: float = 0.0) -> dict:
        payload = self.rpc("CheckHealth", {}, timeout)
        if payload.get("status", "") != "healthy":
            error = payload.get("error", {})
            raise WorkspaceToolError(
                error.get("message", "Workspace is not ready"), code="unhealthy", status_code=503, result=payload
            )
        return payload

    def describe(self, timeout: float = 0.0) -> dict:
        payload = self.rpc("Describe", {"request_id": uuid4().hex}, timeout)
        return payload

    def invoke_tool(
        self, name: str, arguments: dict = DEFAULT_ARGUMENT_DICT, timeout: float = 0.0, *, root_task_id: str = ""
    ) -> dict:
        payload = self.rpc(
            "InvokeTool", {"name": name, "arguments": arguments, "request_id": uuid4().hex, "root_task_id": root_task_id}, timeout
        )
        status = payload.get("status_code", 0)
        result = payload.get("result", {})
        if status >= 400:
            error = result.get("error", {})
            raise WorkspaceToolError(error.get("message", "Tool failed"), code=error.get("code", "tool_error"), status_code=status)
        return payload

    def call_tool(self, name: str, arguments: dict = DEFAULT_ARGUMENT_DICT, timeout: float = 0.0):
        payload = self.invoke_tool(name, arguments, timeout, root_task_id=self.root_task_id)
        result = payload.get("result", {})
        if isinstance(result, dict) and result.get("ok", True) is False:
            code = (
                "timeout"
                if result.get("timedOut", False)
                else "output_limit" if result.get("outputLimited", False) else "process_failed"
            )
            raise WorkspaceToolError("Tool process failed", code=code, status_code=payload.get("status_code", 0), result=result)
        return result
