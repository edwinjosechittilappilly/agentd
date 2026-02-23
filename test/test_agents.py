"""
Unit tests for agentd.agents module.

Run with:  uv run pytest test/test_agents.py -v
"""
import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agentd.agents import (
    AgentRunner,
    DEFAULT_CODE_EXECUTION_INSTRUCTIONS,
    SKILLS_CODE_EXECUTION_INSTRUCTIONS,
    _build_execute_code_tool,
    create_ptc_agent,
)


# =============================================================================
# DEFAULT_CODE_EXECUTION_INSTRUCTIONS
# =============================================================================

def test_default_instructions_is_non_empty_string():
    assert isinstance(DEFAULT_CODE_EXECUTION_INSTRUCTIONS, str)
    assert len(DEFAULT_CODE_EXECUTION_INSTRUCTIONS.strip()) > 0


def test_default_instructions_mentions_execute_code():
    assert "execute_code" in DEFAULT_CODE_EXECUTION_INSTRUCTIONS


# =============================================================================
# SKILLS_CODE_EXECUTION_INSTRUCTIONS
# =============================================================================

def test_skills_instructions_is_non_empty_string():
    assert isinstance(SKILLS_CODE_EXECUTION_INSTRUCTIONS, str)
    assert len(SKILLS_CODE_EXECUTION_INSTRUCTIONS.strip()) > 0


def test_skills_instructions_mentions_skills_list():
    assert "skills list" in SKILLS_CODE_EXECUTION_INSTRUCTIONS


def test_skills_instructions_mentions_lib_tools():
    assert "lib.tools" in SKILLS_CODE_EXECUTION_INSTRUCTIONS


# =============================================================================
# _build_execute_code_tool
# =============================================================================

def _make_engine(tmp_path: Path):
    from agentd.code_execution_engine import CodeExecutionEngine
    return CodeExecutionEngine(cwd=str(tmp_path))


def test_build_execute_code_tool_name(tmp_path):
    engine = _make_engine(tmp_path)
    tool = _build_execute_code_tool(engine)
    assert tool.name == "execute_code"


def test_build_execute_code_tool_has_description(tmp_path):
    engine = _make_engine(tmp_path)
    tool = _build_execute_code_tool(engine)
    assert tool.description
    assert "execute" in tool.description.lower()


def test_build_execute_code_tool_schema_has_code(tmp_path):
    engine = _make_engine(tmp_path)
    tool = _build_execute_code_tool(engine)
    props = tool.params_json_schema.get("properties", {})
    assert "code" in props
    assert "language" in props
    assert "command" in props


def test_build_execute_code_tool_invoke_python(tmp_path):
    """on_invoke_tool should execute Python code and return JSON."""
    engine = _make_engine(tmp_path)
    tool = _build_execute_code_tool(engine)

    args = json.dumps({"code": "print(2 + 2)", "language": "python"})
    result_json = asyncio.run(tool.on_invoke_tool(None, args))
    data = json.loads(result_json)
    assert data["success"] is True
    assert data["output"] == "4"
    assert data["exit_code"] == 0


def test_build_execute_code_tool_invoke_bash(tmp_path):
    engine = _make_engine(tmp_path)
    tool = _build_execute_code_tool(engine)

    args = json.dumps({"code": "echo hello", "language": "bash"})
    result_json = asyncio.run(tool.on_invoke_tool(None, args))
    data = json.loads(result_json)
    assert data["success"] is True
    assert "hello" in data["output"]


def test_build_execute_code_tool_invoke_returns_error_info_on_bad_code(tmp_path):
    """A failing execution sets success=False, not raises an exception."""
    engine = _make_engine(tmp_path)
    tool = _build_execute_code_tool(engine)

    args = json.dumps({"code": "raise ValueError('boom')", "language": "python"})
    result_json = asyncio.run(tool.on_invoke_tool(None, args))
    data = json.loads(result_json)
    assert data["success"] is False


def test_build_execute_code_tool_invoke_bad_json(tmp_path):
    """Malformed JSON arguments return an error dict, not an exception."""
    engine = _make_engine(tmp_path)
    tool = _build_execute_code_tool(engine)

    result_json = asyncio.run(tool.on_invoke_tool(None, "not-json"))
    data = json.loads(result_json)
    assert "error" in data
    assert data["success"] is False


# =============================================================================
# create_ptc_agent
# =============================================================================

def test_create_ptc_agent_returns_agent(tmp_path):
    from agents import Agent
    agent = create_ptc_agent(name="Test", cwd=str(tmp_path))
    assert isinstance(agent, Agent)


def test_create_ptc_agent_name(tmp_path):
    agent = create_ptc_agent(name="MyCoder", cwd=str(tmp_path))
    assert agent.name == "MyCoder"


def test_create_ptc_agent_has_execute_code_tool(tmp_path):
    agent = create_ptc_agent(name="T", cwd=str(tmp_path))
    tool_names = [t.name for t in agent.tools]
    assert "execute_code" in tool_names


def test_create_ptc_agent_instructions_includes_guidance(tmp_path):
    agent = create_ptc_agent(
        name="T",
        instructions="You are helpful.",
        cwd=str(tmp_path),
    )
    assert "execute_code" in agent.instructions
    assert "You are helpful." in agent.instructions


def test_create_ptc_agent_instructions_without_guidance(tmp_path):
    agent = create_ptc_agent(
        name="T",
        instructions="You are helpful.",
        include_code_instructions=False,
        cwd=str(tmp_path),
    )
    # Should only have the provided instructions, no extra guidance
    assert agent.instructions == "You are helpful."


def test_create_ptc_agent_no_instructions(tmp_path):
    agent = create_ptc_agent(name="T", cwd=str(tmp_path))
    assert agent.instructions  # Non-empty (has default guidance)
    assert "execute_code" in agent.instructions


def test_create_ptc_agent_extra_tools(tmp_path):
    from agents import FunctionTool

    async def dummy(ctx, args):
        return "dummy"

    extra = FunctionTool(
        name="my_tool",
        description="A test tool.",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=dummy,
        strict_json_schema=False,
    )
    agent = create_ptc_agent(name="T", cwd=str(tmp_path), extra_tools=[extra])
    tool_names = [t.name for t in agent.tools]
    assert "execute_code" in tool_names
    assert "my_tool" in tool_names


def test_create_ptc_agent_model_kwarg(tmp_path):
    agent = create_ptc_agent(name="T", cwd=str(tmp_path), model="gpt-4o-mini")
    assert agent.model == "gpt-4o-mini"


def test_create_ptc_agent_custom_executor(tmp_path):
    """A custom executor is accepted and wired into the execute_code tool."""
    from unittest.mock import MagicMock
    mock_executor = MagicMock()
    mock_executor.execute_python.return_value = ("42", 0)

    agent = create_ptc_agent(name="T", executor=mock_executor, cwd=str(tmp_path))
    tool_names = [t.name for t in agent.tools]
    assert "execute_code" in tool_names

    # Invoke the tool to confirm the mock executor is used
    tool = next(t for t in agent.tools if t.name == "execute_code")
    args = json.dumps({"code": "x = 42", "language": "python"})
    result_json = asyncio.run(tool.on_invoke_tool(None, args))
    data = json.loads(result_json)
    mock_executor.execute_python.assert_called_once()
    assert data["output"] == "42"


# =============================================================================
# AgentRunner
# =============================================================================

def test_agent_runner_has_run_sync():
    assert callable(AgentRunner.run_sync)


def test_agent_runner_has_run():
    assert callable(AgentRunner.run)


def test_agent_runner_has_run_streamed():
    assert callable(AgentRunner.run_streamed)


def test_agent_runner_run_sync_delegates(tmp_path):
    """AgentRunner.run_sync should call agents.Runner.run_sync."""
    agent = create_ptc_agent(name="T", cwd=str(tmp_path))

    mock_result = MagicMock()
    mock_result.final_output = "hello"

    with patch("agents.Runner.run_sync", return_value=mock_result) as mock_run:
        result = AgentRunner.run_sync(agent, "test input")
        mock_run.assert_called_once_with(agent, "test input")
        assert result.final_output == "hello"


def test_agent_runner_run_delegates(tmp_path):
    """AgentRunner.run should await agents.Runner.run."""
    agent = create_ptc_agent(name="T", cwd=str(tmp_path))

    mock_result = MagicMock()
    mock_result.final_output = "async result"

    async def _test():
        with patch("agents.Runner.run", new=AsyncMock(return_value=mock_result)) as mock_run:
            result = await AgentRunner.run(agent, "test input")
            mock_run.assert_called_once_with(agent, "test input")
            assert result.final_output == "async result"

    asyncio.run(_test())


# =============================================================================
# Package-level exports
# =============================================================================

def test_package_exports():
    import agentd
    assert hasattr(agentd, "create_ptc_agent")
    assert hasattr(agentd, "AgentRunner")
    assert hasattr(agentd, "DEFAULT_CODE_EXECUTION_INSTRUCTIONS")
    assert hasattr(agentd, "SKILLS_CODE_EXECUTION_INSTRUCTIONS")


# =============================================================================
# create_ptc_agent – mcp_servers parameter
# =============================================================================

def test_create_ptc_agent_accepts_mcp_servers_none(tmp_path):
    agent = create_ptc_agent(name="T", mcp_servers=None, cwd=str(tmp_path))
    assert agent.name == "T"


def test_create_ptc_agent_with_mcp_servers_uses_skills_instructions(tmp_path):
    mock_server = MagicMock()
    agent = create_ptc_agent(
        name="T",
        instructions="You are helpful.",
        mcp_servers=[mock_server],
        cwd=str(tmp_path),
    )
    assert "lib.tools" in agent.instructions
    assert "You are helpful." in agent.instructions


def test_create_ptc_agent_without_mcp_servers_uses_default_instructions(tmp_path):
    agent = create_ptc_agent(name="T", cwd=str(tmp_path))
    assert "execute_code" in agent.instructions


def test_create_ptc_agent_accepts_skills_dir(tmp_path):
    custom_skills = tmp_path / "my_skills"
    agent = create_ptc_agent(name="T", skills_dir=str(custom_skills), cwd=str(tmp_path))
    assert agent.name == "T"


# =============================================================================
# Skills setup – lazy initialization
# =============================================================================

def test_execute_code_tool_no_skills_setup_without_servers(tmp_path):
    engine_cwd = tmp_path / "workspace"
    engine_cwd.mkdir()
    from agentd.code_execution_engine import CodeExecutionEngine
    engine = CodeExecutionEngine(cwd=str(engine_cwd))
    with patch("agentd.ptc.setup_skills_directory", new=AsyncMock()) as mock_setup:
        # Patch the module where SCHEMA_REGISTRY is actually read inside _ensure_skills_ready
        with patch("agentd.tool_decorator.SCHEMA_REGISTRY", {}):
            tool = _build_execute_code_tool(engine, mcp_servers=None, skills_dir=None)
            args = json.dumps({"code": "print('hi')", "language": "python"})
            asyncio.run(tool.on_invoke_tool(None, args))
            mock_setup.assert_not_called()


def test_execute_code_tool_pythonpath_passed_for_python(tmp_path):
    from agentd.code_execution_engine import CodeExecutionEngine
    mock_exec = MagicMock()
    mock_exec.execute_python.return_value = ("ok", 0)
    engine = CodeExecutionEngine(executor=mock_exec, cwd=str(tmp_path))
    skills = tmp_path / "skills"
    skills.mkdir()
    tool = _build_execute_code_tool(engine, mcp_servers=None, skills_dir=skills)
    with patch("agentd.tool_decorator.SCHEMA_REGISTRY", {}):
        args = json.dumps({"code": "print(1)", "language": "python"})
        asyncio.run(tool.on_invoke_tool(None, args))
    mock_exec.execute_python.assert_called_once()
    call_kwargs = mock_exec.execute_python.call_args
    # pythonpath is passed as a keyword argument
    assert call_kwargs[1].get("pythonpath") == skills


def test_execute_code_tool_no_pythonpath_for_bash(tmp_path):
    from agentd.code_execution_engine import CodeExecutionEngine
    mock_exec = MagicMock()
    mock_exec.execute_bash.return_value = ("ok", 0)
    engine = CodeExecutionEngine(executor=mock_exec, cwd=str(tmp_path))
    skills = tmp_path / "skills"
    skills.mkdir()
    tool = _build_execute_code_tool(engine, mcp_servers=None, skills_dir=skills)
    with patch("agentd.tool_decorator.SCHEMA_REGISTRY", {}):
        args = json.dumps({"code": "echo hi", "language": "bash"})
        asyncio.run(tool.on_invoke_tool(None, args))
    mock_exec.execute_bash.assert_called_once()


# =============================================================================
# CodeExecutionEngine – pythonpath parameter
# =============================================================================

def test_engine_execute_code_passes_pythonpath(tmp_path):
    from agentd.code_execution_engine import CodeExecutionEngine
    mock_exec = MagicMock()
    mock_exec.execute_python.return_value = ("42", 0)
    engine = CodeExecutionEngine(executor=mock_exec, cwd=str(tmp_path))
    pythonpath = tmp_path / "skills"
    engine.execute_code(code="print(42)", language="python", pythonpath=pythonpath)
    mock_exec.execute_python.assert_called_once_with(
        "print(42)", engine.cwd, pythonpath=pythonpath
    )


def test_engine_execute_code_bash_ignores_pythonpath(tmp_path):
    from agentd.code_execution_engine import CodeExecutionEngine
    mock_exec = MagicMock()
    mock_exec.execute_bash.return_value = ("ok", 0)
    engine = CodeExecutionEngine(executor=mock_exec, cwd=str(tmp_path))
    pythonpath = tmp_path / "skills"
    engine.execute_code(code="echo hi", language="bash", pythonpath=pythonpath)
    mock_exec.execute_bash.assert_called_once()
    mock_exec.execute_python.assert_not_called()


def test_execute_code_request_has_pythonpath_field():
    from agentd.code_execution_engine import ExecuteCodeRequest
    req = ExecuteCodeRequest(code="pass", pythonpath=Path("/tmp/skills"))
    assert req.pythonpath == Path("/tmp/skills")
