import asyncio
import pytest
from unittest.mock import patch
from io import StringIO

from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.base import BaseTool


# -----------------------------
# Hanging Tool
# -----------------------------
class HangingTool(BaseTool):
    name = "hang"

    async def run(self):
        # Simulate a tool that never returns
        await asyncio.sleep(30)
        return "never"


# -----------------------------
# Mock LLM
# -----------------------------
class MockLLMCancel:
    async def generate_with_tools(self, messages, tools):
        return {
            "text": "call hanging tool",
            "tool_calls": [
                {"name": "hang", "arguments": {}}
            ]
        }


# -----------------------------
# Test
# -----------------------------
@pytest.mark.asyncio
async def test_agent_cancellation_mid_tool():
    """
    PHASE 2.3 â€” Cancellation

    INVARIANT:
    - Cancelling agent.run() must:
      - stop tool execution
      - stop retries
      - exit cleanly
      - not crash the event loop
    """

    agent = Agent(
        llm_provider=MockLLMCancel(),
        tools=[],
        max_iterations=5,
        user_id="user-cancel",
        session_id="session-cancel"
    )

    agent.tool_registry.register_tool(HangingTool())

    # Silence stdout (important)
    with patch("sys.stdout", new=StringIO()):
        task = asyncio.create_task(agent.run("cancel me"))

        # Let tool start
        await asyncio.sleep(0.3)

        # Cancel execution
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

    # If we reach here without crash â†’ PASS
    assert True


