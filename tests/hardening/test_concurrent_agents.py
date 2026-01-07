import asyncio
import pytest
from unittest.mock import patch
from io import StringIO

from agenwatch._kernel.agent import Agent
from agenwatch._kernel.tools.registry import ToolRegistry
from agenwatch._kernel.tools.base import BaseTool
from agenwatch._kernel.timeline_logger import ExecutionTimelineLogger


# -----------------------------
# Tool Implementations
# -----------------------------
class IdentityTool(BaseTool):
    name = "identity"

    async def run(self, input_val):
        await asyncio.sleep(0.05)
        return f"processed:id-{input_val}"


class PingTool(BaseTool):
    name = "ping"

    async def run(self):
        return "pong"


# -----------------------------
# Mock LLM Provider
# -----------------------------
class MockLLMProviderConcurrent:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    async def generate(self, messages, tools=None):
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return {"text": "done", "tool_calls": None}


# -----------------------------
# Tests
# -----------------------------
@pytest.mark.asyncio
async def test_high_concurrency_agent_isolation():
    NUM_AGENTS = 50

    async def run_agent(agent_id: int):
        llm_provider = MockLLMProviderConcurrent([
            {
                "text": "Calling identity tool",
                "tool_calls": [
                    {
                        "name": "identity",
                        "arguments": {"input_val": agent_id}
                    }
                ]
            },
            {
                "text": "Task complete",
                "tool_calls": None
            }
        ])

        agent = Agent(
            llm_provider=llm_provider,
            tools=[],
            max_iterations=3,
            user_id=f"user-{agent_id}",
            session_id=f"session-{agent_id}"
        )

        tool = IdentityTool()
        agent.tool_registry.register_tool(tool)

        result = await agent.run(f"task-{agent_id}")
        return agent_id, result

    with patch("sys.stdout", new=StringIO()):
        tasks = [asyncio.create_task(run_agent(i)) for i in range(NUM_AGENTS)]
        results = await asyncio.gather(*tasks)

        for agent_id, output in results:
            assert output.success is True


@pytest.mark.asyncio
async def test_shared_registry_read_safety():
    registry = ToolRegistry()
    registry.register_tool(PingTool())

    async def lookup_loop():
        for _ in range(200):
            tool = registry.get_tool("ping")
            assert tool is not None
            await asyncio.sleep(0)

    await asyncio.gather(*(lookup_loop() for _ in range(100)))


@pytest.mark.asyncio
async def test_session_log_isolation():
    """
    Logs must be isolated per session.
    Only EVENT lines are required to contain the session id.
    Metadata lines inherit context and are not asserted directly.
    """
    NUM_AGENTS = 10

    async def run_agent_with_sink(agent_id: int):
        logs = []

        def session_sink(line: str):
            logs.append(line)

        timeline = ExecutionTimelineLogger(
            user_id=f"user-{agent_id}",
            sink=session_sink
        )

        llm_provider = MockLLMProviderConcurrent([
            {
                "text": "Calling ping tool",
                "tool_calls": [
                    {
                        "name": "ping",
                        "arguments": {}
                    }
                ]
            },
            {
                "text": "Task complete",
                "tool_calls": None
            }
        ])

        agent = Agent(
            llm_provider=llm_provider,
            tools=[],
            max_iterations=3,
            user_id=f"user-{agent_id}",
            session_id=f"session-{agent_id}"
        )

        agent.timeline = timeline

        tool = PingTool()
        agent.tool_registry.register_tool(tool)

        result = await agent.run(f"task-{agent_id}")

        return agent_id, result, logs, f"session-{agent_id}"

    with patch("sys.stdout", new=StringIO()):
        tasks = [asyncio.create_task(run_agent_with_sink(i)) for i in range(NUM_AGENTS)]
        results = await asyncio.gather(*tasks)

        for agent_id, output, logs, session_id in results:
            assert output.success is True, f"Agent {agent_id} failed"

        # Assert ONLY event lines carry session id
        for log_line in logs:
            if "[Session:" in log_line:
                assert session_id in log_line, (
                    f"Session {session_id} has foreign event log: {log_line}"
                )

        # Ensure no other session ids appear in event lines
        for other_agent_id in range(NUM_AGENTS):
            if other_agent_id == agent_id:
                continue
            other_session_id = f"session-{other_agent_id}"
            for log_line in logs:
                if "[Session:" in log_line:
                    assert other_session_id not in log_line, (
                        f"Session {session_id} leaked logs from {other_session_id}: {log_line}"
                    )





