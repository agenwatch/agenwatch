from agenwatch import Agent, tool
from agenwatch.llm_provider import MockLLMProvider
import pytest
import asyncio

def test_sdk_stream_emits_events_sync():
    @tool
    def add(args):
        """Add two numbers"""
        return {"sum": args["a"] + args["b"]}

    llm = MockLLMProvider(
        responses=[
            {
                "tool_calls": [{"name": "add", "arguments": {"a": 1, "b": 2}}]
            },
            {
                "text": "<final>The sum is 3</final>"
            }
        ]
    )

    agent = Agent(tools=[add], llm=llm)

    events = list(agent.stream("calculate 1+2"))

    event_types = [e.type for e in events]
    print(f"\nSync Captured event types: {event_types}")
    
    assert "session_start" in event_types
    assert "agent_done" in event_types

@pytest.mark.asyncio
async def test_sdk_stream_emits_events_async():
    @tool
    def add(args):
        """Add two numbers"""
        return {"sum": args["a"] + args["b"]}

    llm = MockLLMProvider(
        responses=[
            {
                "tool_calls": [{"name": "add", "arguments": {"a": 1, "b": 2}}]
            },
            {
                "text": "<final>The sum is 3</final>"
            }
        ]
    )

    agent = Agent(tools=[add], llm=llm)

    events = []
    async for e in agent.astream("calculate 1+2"):
        events.append(e)

    event_types = [e.type for e in events]
    print(f"\nAsync Captured event types: {event_types}")
    
    assert "session_start" in event_types
    assert "agent_done" in event_types
