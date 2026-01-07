"""
Test ExecutionManager Integration
Save as: tests/test_execution_manager_integration.py
"""

import pytest
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.mock_provider import MockLLMProvider
from agenwatch._kernel.tools.function_tool import FunctionTool
from agenwatch._kernel.errors import RecoverableToolError


@pytest.mark.asyncio
async def test_execution_manager_wired_correctly():
    """Verify ExecutionManager is initialized and working"""
    
    provider = MockLLMProvider()
    agent = Agent(
        user_id="test-user",
        llm_provider=provider,
        max_iterations=5,
    )
    
    # Verify ExecutionManager exists
    assert hasattr(agent, "execution_manager"), "ExecutionManager not initialized"
    assert agent.execution_manager is not None
    
    # Verify it has the right components
    assert agent.execution_manager.registry is not None
    assert agent.execution_manager.circuit_breaker is not None
    assert agent.execution_manager.fail_fast_engine is not None
    
    print("✅ ExecutionManager wired correctly")


@pytest.mark.asyncio
async def test_simple_tool_execution():
    """Test basic tool execution through ExecutionManager"""
    
    call_count = {"count": 0}
    
    async def simple_tool(x: int):
        call_count["count"] += 1
        return {"result": x * 2}
    
    provider = MockLLMProvider()
    provider.responses = [
        {
            "text": "Call simple tool",
            "tool_calls": [{"name": "simple", "arguments": {"x": 5}}],
        }
    ]
    
    agent = Agent(
        user_id="test-user",
        llm_provider=provider,
        max_iterations=5,
    )
    
    # Register tool
    tool = FunctionTool(name="simple", fn=simple_tool)
    agent.tool_registry.register_tool(tool)
    
    # Run agent
    result = await agent.run("test simple tool")
    
    # Verify tool was called
    assert call_count["count"] == 1, f"Expected 1 call, got {call_count['count']}"
    print(f"✅ Tool executed successfully, call_count={call_count['count']}")


@pytest.mark.asyncio
async def test_retry_with_recoverable_error():
    """Test retry logic for RecoverableToolError"""
    
    call_count = {"count": 0}
    
    async def flaky_tool(x: int):
        call_count["count"] += 1
        if call_count["count"] < 3:
            raise RecoverableToolError("temporary failure")
        return {"result": "success"}
    
    provider = MockLLMProvider()
    provider.responses = [
        {
            "text": "Call flaky tool",
            "tool_calls": [{"name": "flaky", "arguments": {"x": 1}}],
        },
        {
            "text": "<final>Task complete, tool succeeded</final>",
            "tool_calls": None,
        }
    ]
    
    agent = Agent(
        user_id="test-user",
        llm_provider=provider,
        max_iterations=5,
    )
    
    # Register tool
    tool = FunctionTool(name="flaky", fn=flaky_tool)
    agent.tool_registry.register_tool(tool)
    
    # Run agent
    result = await agent.run("test retry")
    
    print(f"Call count: {call_count['count']}")
    print(f"Result: {result}")
    
    # Should retry 3 times
    assert call_count["count"] == 3, f"Expected 3 calls, got {call_count['count']}"
    assert result.success, "Tool should eventually succeed"
    
    print(f"✅ Retry logic working correctly")


if __name__ == "__main__":
    import asyncio
    
    print("Running tests...\n")
    
    print("Test 1: ExecutionManager Wiring")
    asyncio.run(test_execution_manager_wired_correctly())
    
    print("\nTest 2: Simple Tool Execution")
    asyncio.run(test_simple_tool_execution())
    
    print("\nTest 3: Retry Logic")
    asyncio.run(test_retry_with_recoverable_error())
    
    print("\n✅ All tests passed!")




