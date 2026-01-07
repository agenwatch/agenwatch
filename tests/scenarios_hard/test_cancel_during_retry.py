import pytest
import asyncio
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.registry import ToolRegistry
from agenwatch._kernel.mock_provider import MockLLMProvider

@pytest.mark.skip(reason="Requires ToolRegistry.register() API - not yet implemented")
@pytest.mark.asyncio
async def test_cancel_during_retry():
    calls = 0

    async def fail(_):
        nonlocal calls
        calls += 1
        raise RuntimeError("fail")

    tools = ToolRegistry()
    tools.register("f", fail)

    llm = MockLLMProvider([
        {"tool_calls": [{"name": "f", "args": {}}]},
    ])

    agent = Agent(
        llm_provider=llm,
        tools=tools,
        max_retries=3,
    )

    task = asyncio.create_task(agent.run("cancel"))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == 1


