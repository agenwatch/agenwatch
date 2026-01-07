import pytest
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.registry import ToolRegistry
from agenwatch._kernel.mock_provider import MockLLMProvider

@pytest.mark.skip(reason="Requires ToolRegistry.register() API - not yet implemented")
@pytest.mark.asyncio
async def test_failure_fingerprint_not_reused():
    calls = 0

    async def flaky(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("fail")
        return {"ok": True}

    tools = ToolRegistry()
    tools.register("f", flaky)

    llm = MockLLMProvider([
        {"tool_calls": [{"name": "f", "args": {}}]},
        {"final": "done"},
    ])

    agent = Agent(llm_provider=llm, tools=tools, max_retries=1)

    await agent.run("task")
    await agent.run("task")

    assert calls == 3




