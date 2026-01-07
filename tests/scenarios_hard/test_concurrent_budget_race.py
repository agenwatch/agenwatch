import pytest
import asyncio
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.registry import ToolRegistry
from agenwatch._kernel.mock_provider import MockLLMProvider
from agenwatch._kernel.safety.budget_manager import BudgetManager

@pytest.mark.skip(reason="Requires BudgetManager API updates - not yet implemented")
@pytest.mark.asyncio
async def test_concurrent_budget_race():
    budget = BudgetManager(limit=1)
    calls = 0

    async def tool(_):
        nonlocal calls
        calls += 1
        return {"ok": True}

    tools = ToolRegistry()
    tools.register("t", tool)

    llm1 = MockLLMProvider([{"tool_calls": [{"name": "t", "args": {}}]}])
    llm2 = MockLLMProvider([{"tool_calls": [{"name": "t", "args": {}}]}])

    a1 = Agent(llm_provider=llm1, tools=tools, budget=budget)
    a2 = Agent(llm_provider=llm2, tools=tools, budget=budget)

    await asyncio.gather(
        a1.run("a"),
        a2.run("b"),
        return_exceptions=True,
    )

    assert calls == 1
    assert budget.remaining >= 0





