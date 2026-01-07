# tests/test_step2_infinite_loop_protection.py

import pytest
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.function_tool import FunctionTool
from agenwatch._kernel.mock_provider import MockLLMProvider

pytestmark = pytest.mark.xfail(
    reason="Kernel contract: successful tool execution is terminal"
)

@pytest.mark.asyncio
async def test_agent_detects_infinite_tool_loop_and_stops_early():
    """
    STEP 2 — HARD LOOP TEST
    """

    async def echo_tool(**kwargs):
        return {"result": "ok"}

    provider = MockLLMProvider()
    provider.responses = [
        {"text": "I need to call echo tool", "tool_calls": [{"name": "echo", "arguments": {"text": "hello"}}]}
    ] * 20

    agent = Agent(
        user_id="test-user",
        llm_provider=provider,
        max_iterations=10
    )

    # Register tool properly
    agent.tool_registry.register_tool(FunctionTool(name="echo", fn=echo_tool))
    
    # DISABLE DAG PLANNER
    agent.dag_planner = None

    result = await agent.run("loop test")
    
    assert result.success is False
    assert (
        (result.error_type and ("CIRCUIT_BREAKER" in result.error_type or "LOOP" in result.error_type or "GOVERNANCE" in result.error_type))
        or (result.terminal_reason and ("CIRCUIT_BREAKER" in result.terminal_reason or "LOOP" in result.terminal_reason or "GOVERNANCE" in result.terminal_reason))
    )


