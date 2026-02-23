# agentd/code_execution_engine.py
"""
Code Execution Engine - Function tool-based code execution.

Exposes an `execute_code` function tool that works with OpenAI's standard
function calling API (chat completions or responses), routing calls to a
pluggable execution backend (SubprocessExecutor, MicrosandboxCLIExecutor,
SandboxRuntimeExecutor, or any custom executor implementing execute_bash /
execute_python).

Supports parallel and sequential execution of multiple code blocks without
requiring the PTC guidance prompt.

Usage::

    from agentd import CodeExecutionEngine
    from openai import OpenAI

    engine = CodeExecutionEngine()
    client = OpenAI()

    # 1. Pass the tool definition to the model
    tools = [engine.tool_definition]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Calculate 2+2 in Python"}],
        tools=tools,
    )

    # 2. Route tool calls back to the engine
    if response.choices[0].message.tool_calls:
        tool_messages = engine.process_tool_calls(
            response.choices[0].message.tool_calls
        )
        # Append assistant message + tool results for the next turn
        messages.append(response.choices[0].message)
        messages.extend(tool_messages)
"""

import asyncio
import functools
import json
import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# Tool Definition
# =============================================================================

#: OpenAI function tool definition for ``execute_code``.
#: Pass this (or ``engine.tool_definition``) in the ``tools`` list when calling
#: ``client.chat.completions.create()``.
EXECUTE_CODE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Execute code in a sandboxed environment. "
            "Supports multiple languages via a pluggable execution backend."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The source code to execute.",
                },
                "language": {
                    "type": "string",
                    "description": (
                        "Programming language of the code. "
                        "Common values: 'python', 'bash', 'javascript', 'ruby', 'go'. "
                        "Defaults to 'python'."
                    ),
                },
                "command": {
                    "type": "string",
                    "description": (
                        "Optional shell command used to run the code. "
                        "When provided it overrides the default language runner. "
                        "The code is written to a temp file and this command is "
                        "executed. Use '{file}' as a placeholder for the temp "
                        "file path (e.g. 'node {file}', 'ruby {file}')."
                    ),
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExecuteCodeRequest:
    """A single code execution request."""

    code: str
    language: str = "python"
    command: str | None = None
    tool_call_id: str | None = None
    pythonpath: "Path | None" = None


@dataclass
class ExecuteCodeResult:
    """Result of a single code execution."""

    output: str
    exit_code: int
    tool_call_id: str | None = None

    @property
    def success(self) -> bool:
        """``True`` when ``exit_code == 0``."""
        return self.exit_code == 0

    def to_tool_message(self) -> dict:
        """Return an OpenAI-compatible tool result message dict."""
        content = json.dumps(
            {
                "output": self.output,
                "exit_code": self.exit_code,
                "success": self.success,
            }
        )
        msg: dict[str, Any] = {"role": "tool", "content": content}
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


# =============================================================================
# Language Runner Registry
# =============================================================================

#: Maps language name → (shell runner, file extension).
#: ``runner=None`` means fall back to executing via ``execute_bash``.
_LANGUAGE_RUNNERS: dict[str, tuple[str | None, str]] = {
    "python": ("python", ".py"),
    "py": ("python", ".py"),
    "bash": (None, ".sh"),
    "shell": (None, ".sh"),
    "sh": (None, ".sh"),
    "javascript": ("node", ".js"),
    "js": ("node", ".js"),
    "typescript": ("ts-node", ".ts"),
    "ts": ("ts-node", ".ts"),
    "ruby": ("ruby", ".rb"),
    "rb": ("ruby", ".rb"),
    "go": ("go run", ".go"),
}


def _file_suffix_from_command(command: str) -> str:
    """Infer a reasonable file suffix from a shell command string."""
    cmd = command.lower()
    for lang, (_, ext) in _LANGUAGE_RUNNERS.items():
        if lang in cmd:
            return ext
    return ".tmp"


# =============================================================================
# Code Execution Engine
# =============================================================================

class CodeExecutionEngine:
    """
    A pluggable code execution engine that exposes an ``execute_code``
    function tool compatible with OpenAI's chat completions API.

    The engine wraps any executor that implements ``execute_bash`` and
    ``execute_python`` (the :class:`~agentd.ptc.Executor` protocol), such as:

    * :class:`~agentd.ptc.SubprocessExecutor` (default — no extra deps)
    * :class:`~agentd.microsandbox_cli_executor.MicrosandboxCLIExecutor`
    * :class:`~agentd.microsandbox_executor.MicrosandboxExecutor`
    * :class:`~agentd.sandbox_runtime_executor.SandboxRuntimeExecutor`
    * Any custom object with ``execute_bash(cmd, cwd)`` and
      ``execute_python(code, cwd)`` methods.

    Parallel execution uses ``asyncio`` and delegates to async variants of
    executor methods when available (``execute_bash_async`` /
    ``execute_python_async``), falling back to thread-pool offloading for
    sync-only executors.
    """

    def __init__(
        self,
        executor: Any = None,
        cwd: "Path | str | None" = None,
        default_language: str = "python",
        timeout: int = 60,
    ) -> None:
        """
        Args:
            executor: Execution backend.  Defaults to
                :class:`~agentd.ptc.SubprocessExecutor`.
            cwd: Working directory for code execution.  A temporary
                directory is created when *None*.
            default_language: Language used when none is specified in a
                request.  Defaults to ``'python'``.
            timeout: Per-execution timeout in seconds.  Defaults to ``60``.
        """
        if executor is None:
            from agentd.ptc import SubprocessExecutor
            executor = SubprocessExecutor(timeout=timeout)

        self.executor = executor
        self.default_language = default_language
        self.timeout = timeout

        if cwd is None:
            self._tmp_dir: str | None = tempfile.mkdtemp(prefix="agentd_exec_")
            self.cwd = Path(self._tmp_dir)
        else:
            self._tmp_dir = None
            self.cwd = Path(cwd)

        self.cwd.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Tool definition
    # ------------------------------------------------------------------

    @property
    def tool_definition(self) -> dict:
        """The OpenAI function tool definition for ``execute_code``."""
        return EXECUTE_CODE_TOOL

    # ------------------------------------------------------------------
    # Single-request execution (sync)
    # ------------------------------------------------------------------

    def execute_code(
        self,
        code: str,
        language: str | None = None,
        command: str | None = None,
        pythonpath: "Path | None" = None,
    ) -> ExecuteCodeResult:
        """
        Execute a single piece of code synchronously.

        Args:
            code: Source code to run.
            language: Programming language.  Falls back to
                ``self.default_language`` when *None*.
            command: Optional shell command that overrides the language runner.
                Use ``{file}`` as a placeholder for the written temp-file path
                (e.g. ``'node {file}'``, ``'ruby {file}'``).
            pythonpath: Optional path added to ``PYTHONPATH`` when running
                Python code.  Used to expose a ``skills/lib`` directory so
                ``from lib.tools import ...`` imports work.

        Returns:
            :class:`ExecuteCodeResult` with ``output``, ``exit_code``, and
            ``success``.
        """
        lang = (language or self.default_language).lower()
        try:
            if command:
                output, exit_code = self._run_with_command(code, command)
            elif lang in ("bash", "shell", "sh"):
                output, exit_code = self.executor.execute_bash(code, self.cwd)
            elif lang in ("python", "py"):
                output, exit_code = self.executor.execute_python(code, self.cwd, pythonpath=pythonpath)
            else:
                output, exit_code = self._run_language(code, lang)
        except Exception as exc:
            output = f"Execution error: {exc}"
            exit_code = 1

        return ExecuteCodeResult(output=output, exit_code=exit_code)

    def _run_with_command(self, code: str, command: str) -> tuple[str, int]:
        """Write *code* to a temp file and run *command* against it."""
        suffix = _file_suffix_from_command(command)
        fname = f"_exec_{uuid.uuid4().hex[:8]}{suffix}"
        fpath = self.cwd / fname
        fpath.write_text(code)
        try:
            if "{file}" in command:
                bash_cmd = command.replace("{file}", str(fpath))
            else:
                bash_cmd = f"{command} {fpath}"
            return self.executor.execute_bash(bash_cmd, self.cwd)
        finally:
            fpath.unlink(missing_ok=True)

    def _run_language(self, code: str, language: str) -> tuple[str, int]:
        """Dispatch execution for languages that aren't bash or python."""
        runner, ext = _LANGUAGE_RUNNERS.get(language, (None, f".{language}"))
        if runner is None:
            # Unknown language — treat the code as a bash script
            return self.executor.execute_bash(code, self.cwd)
        return self._run_with_command(code, f"{runner} {{file}}")

    # ------------------------------------------------------------------
    # Sequential execution
    # ------------------------------------------------------------------

    def execute_sequential(
        self, requests: list[ExecuteCodeRequest]
    ) -> list[ExecuteCodeResult]:
        """
        Execute *requests* one after another, in order.

        Args:
            requests: List of :class:`ExecuteCodeRequest` objects.

        Returns:
            List of :class:`ExecuteCodeResult` in the same order.
        """
        results: list[ExecuteCodeResult] = []
        for req in requests:
            result = self.execute_code(
                code=req.code,
                language=req.language,
                command=req.command,
                pythonpath=req.pythonpath,
            )
            result.tool_call_id = req.tool_call_id
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    def execute_parallel(
        self, requests: list[ExecuteCodeRequest]
    ) -> list[ExecuteCodeResult]:
        """
        Execute *requests* concurrently via asyncio.

        Results are returned in the **same order** as *requests*, regardless
        of completion order.  Executors with ``execute_bash_async`` /
        ``execute_python_async`` methods are used directly; otherwise
        execution is offloaded to a thread pool.

        Args:
            requests: List of :class:`ExecuteCodeRequest` objects.

        Returns:
            List of :class:`ExecuteCodeResult` in the same order.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — run in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self._gather_parallel(requests)
                )
                return future.result()
        else:
            return asyncio.run(self._gather_parallel(requests))

    async def _gather_parallel(
        self, requests: list[ExecuteCodeRequest]
    ) -> list[ExecuteCodeResult]:
        tasks = [self._execute_async(req) for req in requests]
        return list(await asyncio.gather(*tasks))

    async def _execute_async(self, req: ExecuteCodeRequest) -> ExecuteCodeResult:
        """Async execution of a single :class:`ExecuteCodeRequest`."""
        lang = (req.language or self.default_language).lower()

        try:
            if req.command:
                output, exit_code = await asyncio.get_event_loop().run_in_executor(
                    None,
                    functools.partial(self._run_with_command, req.code, req.command),
                )
            elif lang in ("bash", "shell", "sh"):
                if hasattr(self.executor, "execute_bash_async"):
                    output, exit_code = await self.executor.execute_bash_async(
                        req.code, self.cwd
                    )
                else:
                    output, exit_code = await asyncio.get_event_loop().run_in_executor(
                        None,
                        functools.partial(self.executor.execute_bash, req.code, self.cwd),
                    )
            elif lang in ("python", "py"):
                if hasattr(self.executor, "execute_python_async"):
                    output, exit_code = await self.executor.execute_python_async(
                        req.code, self.cwd, pythonpath=req.pythonpath
                    )
                else:
                    output, exit_code = await asyncio.get_event_loop().run_in_executor(
                        None,
                        functools.partial(self.executor.execute_python, req.code, self.cwd, pythonpath=req.pythonpath),
                    )
            else:
                output, exit_code = await asyncio.get_event_loop().run_in_executor(
                    None, functools.partial(self._run_language, req.code, lang)
                )
        except Exception as exc:
            output = f"Execution error: {exc}"
            exit_code = 1

        return ExecuteCodeResult(
            output=output, exit_code=exit_code, tool_call_id=req.tool_call_id
        )

    # ------------------------------------------------------------------
    # Tool-call processing (OpenAI integration)
    # ------------------------------------------------------------------

    def process_tool_calls(
        self,
        tool_calls: list,
        parallel: bool = True,
    ) -> list[dict]:
        """
        Process OpenAI ``tool_calls`` and return ready-to-use tool messages.

        Extracts all ``execute_code`` calls, runs them (parallel or
        sequential), and returns a list of ``{"role": "tool", ...}`` dicts
        suitable for appending to the conversation history.

        Args:
            tool_calls: ``response.choices[0].message.tool_calls`` from a
                chat-completions response.
            parallel: When ``True`` (default) all calls are executed in
                parallel.  Set to ``False`` for sequential execution.

        Returns:
            List of tool-result message dicts.

        Example::

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=[engine.tool_definition],
            )
            if response.choices[0].message.tool_calls:
                messages.append(response.choices[0].message)
                messages.extend(
                    engine.process_tool_calls(
                        response.choices[0].message.tool_calls
                    )
                )
        """
        requests: list[ExecuteCodeRequest] = []
        for call in tool_calls:
            if self._get_tool_name(call) == "execute_code":
                args = self._parse_tool_args(call)
                requests.append(
                    ExecuteCodeRequest(
                        code=args.get("code", ""),
                        language=args.get("language", self.default_language),
                        command=args.get("command"),
                        tool_call_id=self._get_tool_call_id(call),
                    )
                )

        if not requests:
            return []

        if parallel and len(requests) > 1:
            results = self.execute_parallel(requests)
        else:
            results = self.execute_sequential(requests)

        return [r.to_tool_message() for r in results]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tool_name(call: Any) -> str:
        if hasattr(call, "function"):
            return call.function.name or ""
        if isinstance(call, dict):
            return call.get("function", {}).get("name", "")
        return ""

    @staticmethod
    def _get_tool_call_id(call: Any) -> str | None:
        if hasattr(call, "id"):
            return call.id
        if isinstance(call, dict):
            return call.get("id")
        return None

    @staticmethod
    def _parse_tool_args(call: Any) -> dict:
        """Return parsed argument dict from a tool call object."""
        # Pydantic parsed_arguments (openai SDK auto-parsing)
        if hasattr(call, "function"):
            fn = call.function
            if getattr(fn, "parsed_arguments", None) is not None:
                parsed = fn.parsed_arguments
                return {
                    k: getattr(parsed, k)
                    for k in vars(parsed)
                    if not k.startswith("_")
                }
            args = fn.arguments
        elif isinstance(call, dict):
            args = call.get("function", {}).get("arguments", "{}")
        else:
            return {}

        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                return {}
        if isinstance(args, dict):
            return args
        return {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources held by the executor."""
        if hasattr(self.executor, "close"):
            self.executor.close()

    def __enter__(self) -> "CodeExecutionEngine":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# =============================================================================
# Factory
# =============================================================================

def create_code_execution_engine(
    executor: Any = None,
    cwd: "Path | str | None" = None,
    default_language: str = "python",
    timeout: int = 60,
) -> CodeExecutionEngine:
    """
    Create a :class:`CodeExecutionEngine` with a pluggable executor backend.

    Args:
        executor: Backend executor (any object implementing ``execute_bash``
            and ``execute_python``).  Defaults to
            :class:`~agentd.ptc.SubprocessExecutor`.
        cwd: Working directory.  A temp directory is used when *None*.
        default_language: Language used when none is specified.
            Defaults to ``'python'``.
        timeout: Per-execution timeout in seconds.  Defaults to ``60``.

    Returns:
        A configured :class:`CodeExecutionEngine`.

    Example — use with microsandbox::

        from agentd import create_microsandbox_cli_executor
        from agentd import create_code_execution_engine
        from openai import OpenAI

        executor = create_microsandbox_cli_executor()
        engine   = create_code_execution_engine(executor=executor)
        client   = OpenAI()

        tools    = [engine.tool_definition]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Calculate 2+2"}],
            tools=tools,
        )
        if response.choices[0].message.tool_calls:
            tool_msgs = engine.process_tool_calls(
                response.choices[0].message.tool_calls
            )
    """
    return CodeExecutionEngine(
        executor=executor,
        cwd=cwd,
        default_language=default_language,
        timeout=timeout,
    )
