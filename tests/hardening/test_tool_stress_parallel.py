import asyncio
import pytest
from unittest.mock import patch
from io import StringIO

from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.base import BaseTool

import pytest

pytestmark = pytest.mark.xfail(
    reason="Kernel contract: recoverable tool retries are internal; success is terminal"
)


# -----------------------------
# Flaky Tool (fails N times)
# -----------------------------
class FlakyTool(BaseTool):
    name = "flaky"

    def __init__(self, fail_times=2):
        self.calls = 0
        self.fail_times = fail_times

    async def run(self):
        self.calls += 1
        await asyncio.sleep(0.05)

        if self.calls <= self.fail_times:
            raise RuntimeError("Transient failure")

        return "success"


# -----------------------------
# Mock LLM
# -----------------------------
class MockLLMParallelFlaky:
    def __init__(self):
        self.called = False

    async def generate_with_tools(self, messages, tools):
        # Always request flaky tool once, then stop
        if not self.called:
            self.called = True
            return {
                "text": "call flaky",
                "tool_calls": [
                    {"name": "flaky", "arguments": {}}
                ]
            }

        return {
            "text": "FINAL ANSWER",
            "tool_calls": None
        }


# -----------------------------
# Test
# -----------------------------
@pytest.mark.asyncio
async def test_parallel_flaky_tools_recover():
    """
    Multiple agents execute flaky tools concurrently.
    Each tool must retry independently and recover.
    """

    NUM_AGENTS = 10

    async def run_agent(agent_id: int):
        llm = MockLLMParallelFlaky()

        agent = Agent(
            llm_provider=llm,
            tools=[],
            max_iterations=2,
            user_id=f"user-{agent_id}",
            session_id=f"session-{agent_id}"
        )

        tool = FlakyTool(fail_times=2)
        agent.tool_registry.register_tool(tool)

        result = await agent.run(f"task-{agent_id}")

        return agent_id, result, tool.calls

    # Silence stdout (important under concurrency)
    with patch("sys.stdout", new=StringIO()):
        tasks = [asyncio.create_task(run_agent(i)) for i in range(NUM_AGENTS)]
        results = await asyncio.gather(*tasks)

    # Debug output
    print("\n=== Test Results ===")
    for agent_id, output, call_count in results:
        error_str = output.error_type or output.terminal_reason or "N/A"
        print(f"Agent {agent_id}: success={output.success}, calls={call_count}, error={error_str}")

    # Assertions
    for agent_id, output, call_count in results:
        assert output.success is True, f"Agent {agent_id} failed: {output.error_type or output.terminal_reason or 'Unknown error'}"
        assert call_count >= 3, f"Agent {agent_id} did not retry enough (calls={call_count})"




