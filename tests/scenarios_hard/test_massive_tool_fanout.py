import pytest
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.registry import ToolRegistry
from agenwatch._kernel.mock_provider import MockLLMProvider

@pytest.mark.skip(reason="Requires ToolRegistry.register() API - not yet implemented")
@pytest.mark.asyncio
async def test_massive_tool_fanout():
    calls = []

    async def tool(args):
        calls.append(args["i"])
        return {"ok": True}

    tools = ToolRegistry()
    for i in range(20):
        tools.register(f"t{i}", tool)

    llm = MockLLMProvider([
        {"tool_calls": [
            {"name": f"t{i}", "args": {"i": i}} for i in range(20)
        ]},
        {"final": "done"},
    ])

    agent = Agent(llm_provider=llm, tools=tools)

    result = await agent.run("fanout")

    assert calls == list(range(20))
    assert result.success is True




