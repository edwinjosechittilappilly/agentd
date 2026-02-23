# agentd/agents.py
"""
OpenAI Agents SDK integration with PTC, code execution, and skills support.

Provides :func:`create_ptc_agent` — a factory that returns an
``agents.Agent`` pre-configured with:

* An ``execute_code`` :class:`~agents.FunctionTool` backed by
  :class:`~agentd.code_execution_engine.CodeExecutionEngine` (pluggable executor).
* A system-prompt fragment that tells the agent how to use the tool and
  discover skills (PTC + code-execution guidance).

:class:`AgentRunner` re-exports :class:`agents.Runner` unchanged so callers
have a single import point.

Quick start::

    from agentd.agents import create_ptc_agent, AgentRunner

    agent = create_ptc_agent(
        name="Coder",
        instructions="You are a helpful coding assistant.",
    )
    result = AgentRunner.run_sync(agent, "Write and run a Python script that prints 2 + 2.")
    print(result.final_output)

With a custom executor backend::

    from agentd import create_microsandbox_cli_executor
    from agentd.agents import create_ptc_agent, AgentRunner

    executor = create_microsandbox_cli_executor()
    agent = create_ptc_agent(
        name="SecureCoder",
        instructions="You are a secure coding agent.",
        executor=executor,
    )
    result = AgentRunner.run_sync(agent, "List files in /tmp and count them.")
    print(result.final_output)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# System-prompt fragment
# =============================================================================

DEFAULT_CODE_EXECUTION_INSTRUCTIONS: str = (
    "You have access to an execute_code tool that runs code in a sandboxed environment.\n\n"
    "Guidelines:\n"
    "- Use execute_code whenever you need to compute, verify, or produce data.\n"
    "- Prefer Python for data processing; use bash for file/system operations.\n"
    "- Always run code to validate results rather than guessing.\n"
    "- When a task requires multiple steps, chain several execute_code calls.\n"
    "- Skills are available via the skills/ directory (bash: skills list).\n"
)


# =============================================================================
# AgentRunner
# =============================================================================

class AgentRunner:
    """
    Thin wrapper around :class:`agents.Runner`.

    Re-exports ``run_sync``, ``run``, and ``run_streamed`` so callers have
    a single agentd import point.  All keyword arguments are forwarded
    verbatim to the underlying :class:`agents.Runner`.
    """

    @staticmethod
    def run_sync(
        starting_agent: Any,
        input: str | list,  # noqa: A002
        **kwargs: Any,
    ) -> Any:
        """
        Synchronous run.  Mirrors :meth:`agents.Runner.run_sync`.

        Args:
            starting_agent: The agent to run (created by :func:`create_ptc_agent`
                or :class:`agents.Agent`).
            input: User message string or list of response input items.
            **kwargs: Forwarded to :meth:`agents.Runner.run_sync`
                (``context``, ``max_turns``, ``run_config``, …).

        Returns:
            :class:`agents.RunResult` with ``.final_output``.
        """
        from agents import Runner
        return Runner.run_sync(starting_agent, input, **kwargs)

    @staticmethod
    async def run(
        starting_agent: Any,
        input: str | list,  # noqa: A002
        **kwargs: Any,
    ) -> Any:
        """
        Async run.  Mirrors :meth:`agents.Runner.run`.

        Returns:
            :class:`agents.RunResult` with ``.final_output``.
        """
        from agents import Runner
        return await Runner.run(starting_agent, input, **kwargs)

    @staticmethod
    def run_streamed(
        starting_agent: Any,
        input: str | list,  # noqa: A002
        **kwargs: Any,
    ) -> Any:
        """
        Streaming run.  Mirrors :meth:`agents.Runner.run_streamed`.

        Returns:
            :class:`agents.RunResultStreaming` for iterating over events.
        """
        from agents import Runner
        return Runner.run_streamed(starting_agent, input, **kwargs)


# =============================================================================
# execute_code FunctionTool builder
# =============================================================================

def _build_execute_code_tool(engine: Any) -> Any:
    """
    Build an :class:`agents.FunctionTool` that routes ``execute_code`` calls
    to *engine*.

    Args:
        engine: A :class:`~agentd.code_execution_engine.CodeExecutionEngine`
            instance.

    Returns:
        :class:`agents.FunctionTool`
    """
    from agents import FunctionTool

    params_schema: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The source code to execute.",
            },
            "language": {
                "type": "string",
                "description": (
                    "Programming language ('python', 'bash', 'javascript', …). "
                    "Defaults to 'python'."
                ),
            },
            "command": {
                "type": "string",
                "description": (
                    "Optional shell command to run the code. "
                    "Use '{file}' as placeholder for the temp file path."
                ),
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    async def on_invoke(ctx: Any, args_json: str) -> str:
        """Execute code and return the output as a JSON string."""
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON arguments", "success": False})

        code: str = args.get("code", "")
        language: str = args.get("language", engine.default_language)
        command: str | None = args.get("command")

        # Run in thread executor so we don't block the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine.execute_code(code=code, language=language, command=command),
        )

        return json.dumps(
            {
                "output": result.output,
                "exit_code": result.exit_code,
                "success": result.success,
            }
        )

    return FunctionTool(
        name="execute_code",
        description=(
            "Execute code in a sandboxed environment. "
            "Supports Python, bash, JavaScript and other languages via a "
            "pluggable execution backend."
        ),
        params_json_schema=params_schema,
        on_invoke_tool=on_invoke,
        strict_json_schema=False,
    )


# =============================================================================
# PTCAgent factory
# =============================================================================

def create_ptc_agent(
    name: str = "PTCAgent",
    instructions: str = "",
    *,
    executor: Any = None,
    cwd: "Path | str | None" = None,
    default_language: str = "python",
    timeout: int = 60,
    extra_tools: list | None = None,
    model: str | None = None,
    include_code_instructions: bool = True,
    **agent_kwargs: Any,
) -> Any:
    """
    Create an :class:`agents.Agent` pre-configured for code execution via PTC.

    The returned agent has an ``execute_code`` :class:`~agents.FunctionTool`
    backed by a :class:`~agentd.code_execution_engine.CodeExecutionEngine`.
    Any additional tools are merged after it.

    Args:
        name: Agent name.  Defaults to ``"PTCAgent"``.
        instructions: Agent system instructions.  Code-execution guidance is
            appended automatically (disable with *include_code_instructions*).
        executor: Executor backend (defaults to
            :class:`~agentd.ptc.SubprocessExecutor`).  Pass any object that
            implements ``execute_bash`` and ``execute_python``, e.g.:

            * :func:`~agentd.create_microsandbox_cli_executor`
            * :func:`~agentd.create_sandbox_runtime_executor`
        cwd: Working directory for code execution.  A temp directory is used
            when *None*.
        default_language: Language assumed when none is specified.
            Defaults to ``"python"``.
        timeout: Per-execution timeout in seconds.  Defaults to ``60``.
        extra_tools: Additional :class:`~agents.FunctionTool` objects to
            include alongside ``execute_code``.
        model: Model name to pass to :class:`agents.Agent`.  Falls back to
            the agents SDK default (``OPENAI_MODEL`` env var or ``gpt-4o``).
        include_code_instructions: When ``True`` (default),
            :data:`DEFAULT_CODE_EXECUTION_INSTRUCTIONS` is appended to
            *instructions*.
        **agent_kwargs: Forwarded verbatim to :class:`agents.Agent`.

    Returns:
        A fully configured :class:`agents.Agent`.

    Example::

        from agentd.agents import create_ptc_agent, AgentRunner

        agent = create_ptc_agent(
            name="Assistant",
            instructions="You are a helpful assistant.",
        )
        result = AgentRunner.run_sync(agent, "Calculate factorial(10) in Python.")
        print(result.final_output)
    """
    from agents import Agent
    from agentd.code_execution_engine import CodeExecutionEngine

    # Build executor and engine
    engine = CodeExecutionEngine(
        executor=executor,
        cwd=cwd,
        default_language=default_language,
        timeout=timeout,
    )

    # Build execute_code tool
    execute_code_tool = _build_execute_code_tool(engine)

    # Merge tools
    tools: list = [execute_code_tool]
    if extra_tools:
        tools.extend(extra_tools)

    # Build final instructions
    final_instructions = instructions
    if include_code_instructions:
        if final_instructions:
            final_instructions = final_instructions.rstrip() + "\n\n" + DEFAULT_CODE_EXECUTION_INSTRUCTIONS.strip()
        else:
            final_instructions = DEFAULT_CODE_EXECUTION_INSTRUCTIONS.strip()

    # Build kwargs for Agent
    kwargs: dict[str, Any] = {
        "name": name,
        "instructions": final_instructions,
        "tools": tools,
    }
    if model is not None:
        kwargs["model"] = model
    kwargs.update(agent_kwargs)

    return Agent(**kwargs)
