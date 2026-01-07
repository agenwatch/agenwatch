import pytest
import asyncio
from agenwatch._kernel.agent import Agent, EventType
from agenwatch._kernel.mock_provider import MockLLMProvider

@pytest.mark.asyncio
async def test_event_sink_receives_events():
    events = []

    def sink(et, data):
        events.append({"event": et, "data": data})

    # Mock LLM that just returns a final answer
    llm = MockLLMProvider(responses=[{"final": "done"}])
    
    agent = Agent(
        llm_provider=llm,
        event_sink=sink
    )

    result = await agent.run("hello")

    assert result.success is True
    assert len(events) > 0
    # Check for session start and agent done events
    event_names = [e["event"] for e in events]
    # In kernel, these might be strings or EventType enums if they haven't been converted to strings yet
    # Actually, _emit_event passes (event_type.value, data) or just the event name.
    
    # Let's normalize event names for checking
    normalized_names = [str(name).split('.')[-1].lower() for name in event_names]
    
    assert "session_start" in normalized_names
    assert "agent_done" in normalized_names

@pytest.mark.asyncio
async def test_event_sink_does_not_crash_execution():
    def exploding_sink(et, data):
        raise RuntimeError("BOOM")

    llm = MockLLMProvider(responses=[{"final": "done"}])
    
    agent = Agent(
        llm_provider=llm,
        event_sink=exploding_sink
    )

    # Execution should still succeed even if sink crashes
    result = await agent.run("hello")
    assert result.success is True



