import pytest
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.registry import ToolRegistry
from agenwatch._kernel.mock_provider import MockLLMProvider

@pytest.mark.skip(reason="Requires ToolRegistry.register() API - not yet implemented")
@pytest.mark.asyncio
async def test_invalid_schema_then_recovery():
    calls = 0

    async def tool(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"wrong": "shape"}
        return {"value": 1}

    tools = ToolRegistry()
    tools.register("t", tool)

    llm = MockLLMProvider([
        {"tool_calls": [{"name": "t", "args": {}}]},
        {"final": "ok"},
    ])

    agent = Agent(
        llm_provider=llm,
        tools=tools,
        max_retries=1,
    )

    result = await agent.run("schema")

    assert calls == 2
    assert result.success is True




