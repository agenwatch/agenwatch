import pytest
import asyncio
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.registry import ToolRegistry
from agenwatch._kernel.mock_provider import MockLLMProvider

@pytest.mark.skip(reason="Requires ToolRegistry.register() API - not yet implemented")
@pytest.mark.asyncio
async def test_tool_cancelled_error():
    calls = 0

    async def tool(_):
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError()

    tools = ToolRegistry()
    tools.register("t", tool)

    llm = MockLLMProvider([
        {"tool_calls": [{"name": "t", "args": {}}]},
    ])

    agent = Agent(llm_provider=llm, tools=tools)

    with pytest.raises(asyncio.CancelledError):
        await agent.run("interrupt")

    assert calls == 1


