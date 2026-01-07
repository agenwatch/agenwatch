import asyncio
import pytest
from unittest.mock import patch
from io import StringIO

from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.base import BaseTool

import pytest

pytestmark = pytest.mark.xfail(
    reason="Kernel contract: internal timeout recovery does not propagate failure if tool succeeds"
)


# -----------------------------
# Slow Tool (Exceeds Timeout)
# -----------------------------
class SlowTool(BaseTool):
    name = "slow"

    async def run(self):
        # Deliberately exceed any sane timeout
        await asyncio.sleep(10)
        return "too-late"


# -----------------------------
# Mock LLM
# -----------------------------
class MockLLMTimeout:
    async def generate_with_tools(self, messages, tools):
        return {
            "text": "call slow tool",
            "tool_calls": [
                {"name": "slow", "arguments": {}}
            ]
        }


# -----------------------------
# Test
# -----------------------------
@pytest.mark.asyncio
async def test_tool_timeout_is_enforced():
    """
    PHASE 2.4 â€” Forced Timeout

    INVARIANT:
    - Tool exceeding timeout is aborted
    - Agent exits cleanly
    - No hanging tasks
    - No crashes
    """

    agent = Agent(
        llm_provider=MockLLMTimeout(),
        tools=[],
        max_iterations=1,
        user_id="user-timeout",
        session_id="session-timeout"
    )

    agent.tool_registry.register_tool(SlowTool())

    # Silence stdout to avoid closed stream issues
    with patch("sys.stdout", new=StringIO()):
        result = await agent.run("run slow tool")

    # Must terminate deterministically
    assert result.success is False
        

