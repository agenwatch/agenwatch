import pytest
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.mock_provider import MockLLMProvider
from agenwatch._kernel.errors import RecoverableToolError
from agenwatch._kernel.tools.function_tool import FunctionTool
from agenwatch._kernel.tools.registry import ToolRegistry

@pytest.mark.asyncio
async def test_agent_retries_schema_valid_tool_and_succeeds():
    call_count = {"count": 0}

    async def flaky_tool(x: int):
        call_count["count"] += 1
        if call_count["count"] < 3:
            raise RecoverableToolError("temporary failure")
        return {"result": "ok"}

    provider = MockLLMProvider()
    provider.responses = [
        {
            "text": "I need to call the flaky tool to get the result",
            "tool_calls": [{"name": "flaky", "arguments": {"x": 1}}],
        },
        {
            "text": "<final>Task complete. The flaky tool returned: ok</final>",
            "tool_calls": None,
        },
    ] * 2

    # In kernel Agent, 'tools' argument should be a list of tool objects
    tool = FunctionTool(name="flaky", fn=flaky_tool)
    
    agent = Agent(
        user_id="test-user",
        llm_provider=provider,
        tools=[tool],
        max_iterations=5,
    )
    agent.execution_manager.policy.enable_fingerprinting = False
    agent.execution_manager.reset_cache()

    result = await agent.run("run flaky tool")

    assert call_count["count"] == 3
    assert result.success is True
