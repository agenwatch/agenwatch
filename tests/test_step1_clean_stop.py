import pytest
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.event_types import EventType
from agenwatch._kernel.mock_provider import MockLLMProvider

@pytest.mark.asyncio
async def test_agent_stops_after_single_tool_call():
    async def echo_tool(args):
        return {"result": "hello"}

    provider = MockLLMProvider()
    provider.responses = [
        {
            "tool_calls": [{
                "name": "echo",
                "arguments": {}
            }],
            "text": "Calling echo tool"
        },
        {
            "tool_calls": None,
            "text": "Final answer: hello"
        }
    ]
    agent = Agent(
        user_id="test-user",
        llm_provider=provider,
        max_iterations=3
    )
    
    # Register tool using the tool_registry (new agent architecture)
    from agenwatch._kernel.tools.function_tool import FunctionTool
    agent.tool_registry.register_tool(FunctionTool(name="echo", fn=echo_tool))
    result = await agent.run("Echo the word hello")
    
    # Main assertions
    assert result is not None
    assert result.success is True



