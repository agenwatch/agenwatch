import asyncio
import pytest

from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.function_tool import FunctionTool
from agenwatch._kernel.mock_provider import MockLLMProvider
from agenwatch._kernel.safety.budget_manager import BudgetManager


# ---------------------------------------------------------
# 1. Budget exhaustion must be terminal (no zombie loop)
# ---------------------------------------------------------

@pytest.mark.skip(reason="Requires updated Agent API and MockLLMProvider API - not yet implemented")
@pytest.mark.asyncio
async def test_budget_exhaustion_is_terminal():
    calls = {"count": 0}

    async def tool(_):
        calls["count"] += 1
        return {"ok": True}

    agent = Agent(
        llm_provider=MockLLMProvider(always_call="tool"),
        tools=[FunctionTool(name="tool", fn=tool)],
        max_iterations=10,
    )
    agent.execution_manager.budget_manager = BudgetManager(max_budget=1)

    result = await agent.run("work")

    assert calls["count"] == 1
    assert result["success"] is False


# ---------------------------------------------------------
# 2. Retry must not double-charge budget
# ---------------------------------------------------------

@pytest.mark.skip(reason="Requires updated Agent API and MockLLMProvider API - not yet implemented")
@pytest.mark.asyncio
async def test_retry_does_not_double_charge_budget():
    attempts = 0

    async def flaky(_):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")
        return {"ok": True}

    agent = Agent(
        llm_provider=MockLLMProvider(always_call="flaky"),
        tools=[FunctionTool(name="flaky", fn=flaky)],
        max_iterations=5,
    )
    agent.execution_manager.budget_manager = BudgetManager(max_budget=2)

    result = await agent.run("retry")

    assert attempts == 2
    assert result["success"] is True


# ---------------------------------------------------------
# 3. Cancellation during tool execution
# ---------------------------------------------------------

@pytest.mark.skip(reason="Requires updated Agent API and MockLLMProvider API - not yet implemented")
@pytest.mark.asyncio
async def test_cancellation_mid_tool_execution():
    started = asyncio.Event()

    async def long_tool(_):
        started.set()
        await asyncio.sleep(5)
        return {"ok": True}

    agent = Agent(
        llm_provider=MockLLMProvider(always_call="long"),
        tools=[FunctionTool(name="long", fn=long_tool)],
        max_iterations=3,
    )

    task = asyncio.create_task(agent.run("cancel"))

    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------
# 4. Deterministic replay must have zero side effects
# ---------------------------------------------------------

@pytest.mark.skip(reason="Requires updated Agent API and MockLLMProvider API - not yet implemented")
@pytest.mark.asyncio
async def test_replay_has_no_side_effects():
    calls = {"count": 0}

    async def tool(_):
        calls["count"] += 1
        return {"x": 1}

    agent = Agent(
        llm_provider=MockLLMProvider(always_call="tool"),
        tools=[FunctionTool(name="tool", fn=tool)],
        max_iterations=3,
        record=True,
    )

    first = await agent.run("task")
    replay = await agent.replay("task")

    assert calls["count"] == 1
    assert first["result"] == replay["result"]


# ---------------------------------------------------------
# 5. Infinite reasoning loop must stop early
# ---------------------------------------------------------

@pytest.mark.skip(reason="Requires updated Agent API and MockLLMProvider API - not yet implemented")
@pytest.mark.asyncio
async def test_infinite_reasoning_loop_stops():
    llm = MockLLMProvider(always_text="thinking...")

    agent = Agent(
        llm_provider=llm,
        tools=[],
        max_iterations=5,
    )

    result = await agent.run("think")

    assert result["success"] is False





