"""
Unit tests for agentd.code_execution_engine.

Run with:  uv run pytest test/test_code_execution_engine.py -v
"""
import json
import types
from pathlib import Path
from unittest.mock import MagicMock

from agentd.code_execution_engine import (
    EXECUTE_CODE_TOOL,
    CodeExecutionEngine,
    ExecuteCodeRequest,
    ExecuteCodeResult,
    create_code_execution_engine,
)


# =============================================================================
# Tool definition
# =============================================================================

def test_execute_code_tool_structure():
    """EXECUTE_CODE_TOOL has the expected OpenAI function-tool shape."""
    assert EXECUTE_CODE_TOOL["type"] == "function"
    fn = EXECUTE_CODE_TOOL["function"]
    assert fn["name"] == "execute_code"
    params = fn["parameters"]
    assert "code" in params["properties"]
    assert "language" in params["properties"]
    assert "command" in params["properties"]
    assert params["required"] == ["code"]
    assert params["additionalProperties"] is False


# =============================================================================
# ExecuteCodeRequest
# =============================================================================

def test_execute_code_request_defaults():
    req = ExecuteCodeRequest(code="print('hi')")
    assert req.language == "python"
    assert req.command is None
    assert req.tool_call_id is None


def test_execute_code_request_fields():
    req = ExecuteCodeRequest(
        code="echo hi",
        language="bash",
        command="bash {file}",
        tool_call_id="call_abc123",
    )
    assert req.language == "bash"
    assert req.command == "bash {file}"
    assert req.tool_call_id == "call_abc123"


# =============================================================================
# ExecuteCodeResult
# =============================================================================

def test_execute_code_result_success():
    result = ExecuteCodeResult(output="hello", exit_code=0)
    assert result.success is True


def test_execute_code_result_failure():
    result = ExecuteCodeResult(output="error", exit_code=1)
    assert result.success is False


def test_execute_code_result_to_tool_message_no_id():
    result = ExecuteCodeResult(output="42", exit_code=0)
    msg = result.to_tool_message()
    assert msg["role"] == "tool"
    assert "tool_call_id" not in msg
    data = json.loads(msg["content"])
    assert data["output"] == "42"
    assert data["exit_code"] == 0
    assert data["success"] is True


def test_execute_code_result_to_tool_message_with_id():
    result = ExecuteCodeResult(output="hi", exit_code=0, tool_call_id="call_xyz")
    msg = result.to_tool_message()
    assert msg["tool_call_id"] == "call_xyz"


# =============================================================================
# CodeExecutionEngine – execute_code (sync)
# =============================================================================

def _make_engine(tmp_path: Path) -> CodeExecutionEngine:
    """Create an engine backed by the default SubprocessExecutor."""
    return CodeExecutionEngine(cwd=str(tmp_path))


def test_execute_python_code(tmp_path):
    engine = _make_engine(tmp_path)
    result = engine.execute_code(code="print(2 + 2)")
    assert result.output == "4"
    assert result.success


def test_execute_python_code_explicit_language(tmp_path):
    engine = _make_engine(tmp_path)
    result = engine.execute_code(code="print('hello')", language="python")
    assert result.output == "hello"
    assert result.success


def test_execute_bash_code(tmp_path):
    engine = _make_engine(tmp_path)
    result = engine.execute_code(code="echo 'hello bash'", language="bash")
    assert "hello bash" in result.output
    assert result.success


def test_execute_bash_alias_shell(tmp_path):
    engine = _make_engine(tmp_path)
    result = engine.execute_code(code="echo shell", language="shell")
    assert "shell" in result.output
    assert result.success


def test_execute_code_with_custom_command(tmp_path):
    """The 'command' parameter overrides the language runner."""
    engine = _make_engine(tmp_path)
    result = engine.execute_code(
        code="print('via command')",
        command="python {file}",
    )
    assert "via command" in result.output
    assert result.success


def test_execute_code_exit_code_on_error(tmp_path):
    engine = _make_engine(tmp_path)
    result = engine.execute_code(code="raise ValueError('boom')", language="python")
    assert result.exit_code != 0
    assert not result.success


def test_execute_code_default_language(tmp_path):
    """When language is omitted the engine uses default_language."""
    engine = CodeExecutionEngine(cwd=str(tmp_path), default_language="python")
    result = engine.execute_code(code="print('default')")
    assert result.output == "default"
    assert result.success


# =============================================================================
# Sequential execution
# =============================================================================

def test_execute_sequential(tmp_path):
    engine = _make_engine(tmp_path)
    requests = [
        ExecuteCodeRequest(code="print(1)", language="python", tool_call_id="id1"),
        ExecuteCodeRequest(code="print(2)", language="python", tool_call_id="id2"),
    ]
    results = engine.execute_sequential(requests)
    assert len(results) == 2
    assert results[0].output == "1"
    assert results[0].tool_call_id == "id1"
    assert results[1].output == "2"
    assert results[1].tool_call_id == "id2"


def test_execute_sequential_preserves_order(tmp_path):
    engine = _make_engine(tmp_path)
    requests = [
        ExecuteCodeRequest(code="print('a')", language="python"),
        ExecuteCodeRequest(code="print('b')", language="python"),
        ExecuteCodeRequest(code="print('c')", language="python"),
    ]
    results = engine.execute_sequential(requests)
    assert [r.output for r in results] == ["a", "b", "c"]


# =============================================================================
# Parallel execution
# =============================================================================

def test_execute_parallel(tmp_path):
    engine = _make_engine(tmp_path)
    requests = [
        ExecuteCodeRequest(code="print(10)", language="python", tool_call_id="p1"),
        ExecuteCodeRequest(code="print(20)", language="python", tool_call_id="p2"),
    ]
    results = engine.execute_parallel(requests)
    assert len(results) == 2
    outputs = {r.tool_call_id: r.output for r in results}
    assert outputs["p1"] == "10"
    assert outputs["p2"] == "20"


def test_execute_parallel_single_request(tmp_path):
    """A single-request parallel call should still work."""
    engine = _make_engine(tmp_path)
    requests = [ExecuteCodeRequest(code="print('solo')", language="python")]
    results = engine.execute_parallel(requests)
    assert len(results) == 1
    assert results[0].output == "solo"


# =============================================================================
# process_tool_calls
# =============================================================================

def _make_tool_call(call_id: str, args: dict):
    """Create a mock OpenAI tool-call object."""
    call = MagicMock()
    call.id = call_id
    call.function.name = "execute_code"
    call.function.arguments = json.dumps(args)
    call.function.parsed_arguments = None
    return call


def test_process_tool_calls_single(tmp_path):
    engine = _make_engine(tmp_path)
    tool_calls = [_make_tool_call("call_1", {"code": "print(99)", "language": "python"})]
    messages = engine.process_tool_calls(tool_calls)
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    data = json.loads(msg["content"])
    assert data["output"] == "99"
    assert data["success"] is True


def test_process_tool_calls_parallel(tmp_path):
    engine = _make_engine(tmp_path)
    tool_calls = [
        _make_tool_call("call_a", {"code": "print('A')", "language": "python"}),
        _make_tool_call("call_b", {"code": "print('B')", "language": "python"}),
    ]
    messages = engine.process_tool_calls(tool_calls, parallel=True)
    assert len(messages) == 2
    ids = {m["tool_call_id"] for m in messages}
    assert ids == {"call_a", "call_b"}


def test_process_tool_calls_sequential(tmp_path):
    engine = _make_engine(tmp_path)
    tool_calls = [
        _make_tool_call("call_x", {"code": "print('X')", "language": "python"}),
        _make_tool_call("call_y", {"code": "print('Y')", "language": "python"}),
    ]
    messages = engine.process_tool_calls(tool_calls, parallel=False)
    assert len(messages) == 2
    assert messages[0]["tool_call_id"] == "call_x"
    assert messages[1]["tool_call_id"] == "call_y"


def test_process_tool_calls_ignores_non_execute_code(tmp_path):
    """Tool calls for other functions should be silently ignored."""
    engine = _make_engine(tmp_path)
    other_call = MagicMock()
    other_call.id = "call_other"
    other_call.function.name = "get_weather"
    other_call.function.arguments = json.dumps({"city": "London"})
    other_call.function.parsed_arguments = None

    messages = engine.process_tool_calls([other_call])
    assert messages == []


def test_process_tool_calls_with_command(tmp_path):
    engine = _make_engine(tmp_path)
    tool_calls = [
        _make_tool_call(
            "call_cmd",
            {"code": "print('cmd')", "command": "python {file}"},
        )
    ]
    messages = engine.process_tool_calls(tool_calls)
    assert len(messages) == 1
    data = json.loads(messages[0]["content"])
    assert "cmd" in data["output"]


# =============================================================================
# tool_definition property
# =============================================================================

def test_engine_tool_definition_is_execute_code_tool(tmp_path):
    engine = _make_engine(tmp_path)
    assert engine.tool_definition is EXECUTE_CODE_TOOL


# =============================================================================
# Pluggable executor
# =============================================================================

def test_pluggable_executor(tmp_path):
    """Engine should delegate to the provided executor."""
    mock_executor = MagicMock()
    mock_executor.execute_python.return_value = ("mocked output", 0)
    mock_executor.execute_bash.return_value = ("mocked bash", 0)

    engine = CodeExecutionEngine(executor=mock_executor, cwd=str(tmp_path))
    result = engine.execute_code(code="x = 1", language="python")

    mock_executor.execute_python.assert_called_once()
    assert result.output == "mocked output"
    assert result.success


# =============================================================================
# Factory
# =============================================================================

def test_create_code_execution_engine(tmp_path):
    engine = create_code_execution_engine(cwd=str(tmp_path))
    assert isinstance(engine, CodeExecutionEngine)
    result = engine.execute_code(code="print('factory')")
    assert result.output == "factory"


def test_create_code_execution_engine_with_custom_executor(tmp_path):
    mock_executor = MagicMock()
    mock_executor.execute_python.return_value = ("custom", 0)
    engine = create_code_execution_engine(
        executor=mock_executor, cwd=str(tmp_path)
    )
    result = engine.execute_code(code="pass", language="python")
    assert result.output == "custom"


# =============================================================================
# Context manager
# =============================================================================

def test_engine_context_manager(tmp_path):
    with CodeExecutionEngine(cwd=str(tmp_path)) as engine:
        result = engine.execute_code(code="print('ctx')")
    assert result.output == "ctx"


# =============================================================================
# __init__.py exports
# =============================================================================

def test_exports_from_agentd_package():
    import agentd
    assert hasattr(agentd, "CodeExecutionEngine")
    assert hasattr(agentd, "ExecuteCodeRequest")
    assert hasattr(agentd, "ExecuteCodeResult")
    assert hasattr(agentd, "EXECUTE_CODE_TOOL")
    assert hasattr(agentd, "create_code_execution_engine")
