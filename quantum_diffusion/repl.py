"""Asynchronous Jupyter kernel lifecycle and execution-channel handling."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from jupyter_client import AsyncKernelManager
except ImportError as exc:  # pragma: no cover - exercised when optional deps are absent
    raise ImportError(
        "Jupyter kernel support requires jupyter-client and ipykernel"
    ) from exc


@dataclass
class SandboxProfile:
    """Configure a kernel's writable execution directory and optional bubblewrap."""

    workspace: Path | str | None = None
    use_bubblewrap: bool = True
    _owned_workspace: bool = field(default=False, init=False, repr=False)

    def prepare(self) -> Path:
        if self.workspace is None:
            self.workspace = Path(tempfile.mkdtemp(prefix="quantum-repl-"))
            self._owned_workspace = True
        else:
            self.workspace = Path(self.workspace).expanduser().resolve()
            self.workspace.mkdir(parents=True, exist_ok=True)
        if self.use_bubblewrap and shutil.which("bwrap") is None:
            self.cleanup()
            raise RuntimeError(
                "bubblewrap is required for filesystem sandboxing; install bwrap "
                "or explicitly opt out with use_bubblewrap=False"
            )
        return Path(self.workspace)

    def wrap_command(self, command: list[str], connection_file: str) -> list[str]:
        if not self.use_bubblewrap:
            return command
        workspace = Path(self.workspace).resolve()
        return [
            shutil.which("bwrap") or "bwrap",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/",
            "/",
            "--bind",
            str(workspace),
            str(workspace),
            "--chdir",
            str(workspace),
            *command,
        ]

    def cleanup(self) -> None:
        if self._owned_workspace and self.workspace is not None:
            shutil.rmtree(self.workspace, ignore_errors=True)
            self.workspace = None
            self._owned_workspace = False


class _SandboxedKernelManager(AsyncKernelManager):
    def __init__(self, *args: Any, profile: SandboxProfile, **kwargs: Any) -> None:
        self.profile = profile
        super().__init__(*args, **kwargs)

    def format_kernel_cmd(self, extra_arguments: list[str] | None = None) -> list[str]:
        command = super().format_kernel_cmd(extra_arguments)
        connection_file = str(self.connection_file)
        return self.profile.wrap_command(command, connection_file)


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    outputs: tuple[dict[str, Any], ...] = ()
    execution_count: int | None = None
    status: str = "ok"
    error: dict[str, Any] | None = None


class KernelSession:
    """Own one persistent IPython kernel and consume its async ZMQ channels."""

    def __init__(
        self,
        profile: SandboxProfile | None = None,
        *,
        kernel_name: str = "python3",
        startup_timeout: float = 30.0,
    ) -> None:
        self.profile = profile or SandboxProfile()
        self.kernel_name = kernel_name
        self.startup_timeout = startup_timeout
        self.manager: _SandboxedKernelManager | None = None
        self.client: Any | None = None
        self._execute_lock = asyncio.Lock()

    async def start(self) -> None:
        if self.manager is not None:
            return
        workspace = self.profile.prepare()
        connection_file = workspace / ".kernel-connection.json"
        self.manager = _SandboxedKernelManager(
            profile=self.profile,
            kernel_name=self.kernel_name,
            connection_file=str(connection_file),
        )
        try:
            inherited_env = {
                key: os.environ[key]
                for key in ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "WINDIR")
                if key in os.environ
            }
            await self.manager.start_kernel(
                cwd=str(workspace),
                env={
                    **inherited_env,
                    "PATH": os.defpath,
                    "TMPDIR": str(workspace),
                    "JUPYTER_RUNTIME_DIR": str(workspace),
                    "JUPYTER_CONFIG_DIR": str(workspace / ".jupyter"),
                    "JUPYTER_DATA_DIR": str(workspace / ".local" / "share" / "jupyter"),
                    "IPYTHONDIR": str(workspace / ".ipython"),
                    "XDG_CACHE_HOME": str(workspace / ".cache"),
                },
            )
            self.client = self.manager.client()
            self.client.start_channels()
            await self.client.wait_for_ready(timeout=self.startup_timeout)
        except BaseException:
            await self.close()
            raise

    async def execute(self, code: str, *, timeout: float = 60.0) -> ExecutionResult:
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        if self.manager is None or self.client is None:
            raise RuntimeError("kernel session is not started")
        async with self._execute_lock:
            message_id = self.client.execute(
                code, stop_on_error=False, allow_stdin=False
            )
            started = time.monotonic()
            shell_task = asyncio.create_task(self.client.get_shell_msg())
            iopub_task = asyncio.create_task(self.client.get_iopub_msg())
            stdout: list[str] = []
            stderr: list[str] = []
            outputs: list[dict[str, Any]] = []
            error: dict[str, Any] | None = None
            execution_count: int | None = None
            shell_reply: dict[str, Any] | None = None
            idle_seen = False
            try:
                while shell_reply is None or not idle_seen:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError(f"kernel execution exceeded {timeout} seconds")
                    done, _ = await asyncio.wait(
                        {task for task in (shell_task, iopub_task) if not task.done()},
                        timeout=remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        raise TimeoutError(f"kernel execution exceeded {timeout} seconds")
                    if iopub_task in done:
                        message = iopub_task.result()
                        parent_id = message.get("parent_header", {}).get("msg_id")
                        if parent_id == message_id:
                            message_type = message.get("header", {}).get("msg_type")
                            content = message.get("content", {})
                            if message_type == "stream":
                                target = stderr if content.get("name") == "stderr" else stdout
                                target.append(content.get("text", ""))
                            elif message_type == "error":
                                error = content
                                outputs.append({"output_type": "error", **content})
                            elif message_type == "execute_result":
                                execution_count = content.get("execution_count")
                                outputs.append(
                                    {
                                        "output_type": "execute_result",
                                        "data": content.get("data", {}),
                                        "metadata": content.get("metadata", {}),
                                        "execution_count": execution_count,
                                    }
                                )
                            elif message_type == "display_data":
                                outputs.append(
                                    {
                                        "output_type": "display_data",
                                        "data": content.get("data", {}),
                                        "metadata": content.get("metadata", {}),
                                    }
                                )
                            elif message_type == "status" and content.get("execution_state") == "idle":
                                idle_seen = True
                        if not idle_seen:
                            iopub_task = asyncio.create_task(self.client.get_iopub_msg())
                    if shell_task in done:
                        candidate = shell_task.result()
                        if candidate.get("parent_header", {}).get("msg_id") == message_id:
                            shell_reply = candidate.get("content", {})
                            execution_count = shell_reply.get("execution_count", execution_count)
                            if shell_reply.get("status") == "error" and error is None:
                                error = shell_reply
                        else:
                            shell_task = asyncio.create_task(self.client.get_shell_msg())
                return ExecutionResult(
                    stdout="".join(stdout),
                    stderr="".join(stderr),
                    outputs=tuple(outputs),
                    execution_count=execution_count,
                    status=(shell_reply or {}).get("status", "ok"),
                    error=error,
                )
            except TimeoutError:
                await self.manager.interrupt_kernel()
                raise
            finally:
                for task in (shell_task, iopub_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(shell_task, iopub_task, return_exceptions=True)

    async def close(self) -> None:
        manager, client = self.manager, self.client
        self.manager = None
        self.client = None
        if client is not None:
            client.stop_channels()
        if manager is not None:
            await manager.shutdown_kernel(now=True)
        self.profile.cleanup()

    async def __aenter__(self) -> KernelSession:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()
