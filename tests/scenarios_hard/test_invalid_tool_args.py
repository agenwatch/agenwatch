import pytest
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.registry import ToolRegistry
from agenwatch._kernel.mock_provider import MockLLMProvider

@pytest.mark.skip(reason="Requires ToolRegistry.register() API - not yet implemented")
@pytest.mark.asyncio
async def test_invalid_tool_arguments():
    executed = False

    async def tool(args):
        nonlocal executed
        executed = True
        return {"ok": True}

    tools = ToolRegistry()
    tools.register("t", tool)

    llm = MockLLMProvider([
        {"tool_calls": [{"name": "t", "args": {"bad": 1}}]},
    ])

    agent = Agent(
        llm_provider=llm,
        tools=tools,
        max_retries=1,
    )

    result = await agent.run("bad")

    assert executed is False
    assert result.success is False





