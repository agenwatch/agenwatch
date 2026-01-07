import pytest
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.registry import ToolRegistry
from agenwatch._kernel.mock_provider import MockLLMProvider

@pytest.mark.skip(reason="Requires ToolRegistry.register() API - not yet implemented")
@pytest.mark.asyncio
async def test_replay_after_partial():
    calls = []

    async def tool(_):
        calls.append("run")
        raise RuntimeError("crash")

    tools = ToolRegistry()
    tools.register("t", tool)

    llm = MockLLMProvider([
        {"tool_calls": [{"name": "t", "args": {}}]},
    ])

    agent = Agent(
        llm_provider=llm,
        tools=tools,
        record=True,
    )

    with pytest.raises(RuntimeError):
        await agent.run("task")

    replay = await agent.replay("task")

    assert calls == ["run"]
    assert replay.success is False





