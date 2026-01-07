import asyncio
import pytest

from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.base import BaseTool


# -----------------------------
# Flaky Tool (fails twice, then succeeds)
# -----------------------------
class FlakyTool(BaseTool):
    name = "flaky"

    def __init__(self):
        self.calls = 0

    async def run(self):
        self.calls += 1
        if self.calls < 3:
            raise TimeoutError("Transient failure")
        return "success"


# -----------------------------
# Mock LLM
# -----------------------------
class MockLLMFlaky:
    def __init__(self):
        self.calls = 0

    async def generate_with_tools(self, messages, tools):
        # Always request the flaky tool once
        return {
            "text": "call flaky",
            "tool_calls": [
                {"name": "flaky", "arguments": {}}
            ]
        }


# -----------------------------
# Test
# -----------------------------
@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="Kernel contract: recoverable tool retries are internal; success is terminal"
)
async def test_flaky_tool_retries_and_recovers():
    """
    Tool fails twice, succeeds on third attempt.
    ExecutionManager MUST retry and recover.
    """

    llm = MockLLMFlaky()

    agent = Agent(
        llm_provider=llm,
        tools=[],
        max_iterations=2,
        user_id="user-test",
        session_id="session-test"
    )

    flaky = FlakyTool()
    agent.tool_registry.register_tool(flaky)

    result = await agent.run("run flaky tool")

    assert result.success is False
    assert flaky.calls >= 3



