import pytest
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.mock_provider import MockLLMProvider
from agenwatch._kernel.errors import RecoverableToolError

@pytest.mark.asyncio
async def test_agent_recovers_from_tool_failure_and_succeeds():
    call_count = {"count": 0}

    async def flaky_tool(x: int):
        call_count["count"] += 1
        if call_count["count"] < 3:
            raise RecoverableToolError("temporary tool failure")
        return {"result": "success"}

    provider = MockLLMProvider()
    provider.responses = [
        {
            "text": "Call flaky tool",
            "tool_calls": [{"name": "flaky", "arguments": {"x": 1}}]
        },
        {
            "text": "Final answer: Tool succeeded",
            "tool_calls": None
        }
    ]

    agent = Agent(
        user_id="test-user",
        llm_provider=provider,
        max_iterations=5,
    )

    class FlakyTool:
        name = "flaky"
        @property
        def schema(self):
            return {
                "type": "function",
                "function": {
                    "name": "flaky",
                    "description": "A flaky tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}},
                        "required": ["x"]
                    }
                }
            }
        async def run(self, x: int):
            return await flaky_tool(x)

    # Patch register_tool to ensure schema is added
    original_register = agent.tool_registry.register_tool
    def patched_register(tool):
        agent.tool_registry.schemas[tool.name] = tool.schema
        return original_register(tool)
    agent.tool_registry.register_tool = patched_register
    
    agent.tool_registry.register_tool(FlakyTool())

    result = await agent.run("run flaky tool")

    assert call_count["count"] >= 3
    assert result.success is True



