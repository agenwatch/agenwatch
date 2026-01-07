import pytest
import asyncio
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.registry import ToolRegistry
from agenwatch._kernel.mock_provider import MockLLMProvider
from agenwatch._kernel.safety.budget_manager import BudgetManager

@pytest.mark.skip(reason="Requires ToolRegistry.register() API and BudgetManager API updates - not yet implemented")
@pytest.mark.asyncio
async def test_budget_retry_timeout():
    calls = 0

    async def tool(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(2)
        return {"ok": True}

    tools = ToolRegistry()
    tools.register("t", tool)

    llm = MockLLMProvider([
        {"tool_calls": [{"name": "t", "args": {}}]},
        {"final": "done"},
    ])

    agent = Agent(
        llm_provider=llm,
        tools=tools,
        budget=BudgetManager(limit=1),
        tool_timeout=0.5,
        max_retries=1,
    )

    result = await agent.run("task")

    assert calls == 2
    assert result.success is True



