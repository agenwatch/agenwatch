import pytest
from agenwatch._kernel.agent import Agent
from agenwatch._kernel.mock_provider import MockLLMProvider

@pytest.mark.skip(reason="Requires Response object type handling in agent.py - not yet implemented")
@pytest.mark.asyncio
async def test_infinite_reasoning_stops():
    llm = MockLLMProvider([{"thought": "thinking"}] * 10)

    agent = Agent(
        llm_provider=llm,
        max_iterations=3,
    )

    result = await agent.run("think")

    assert result.iterations == 3
    assert result.success is False





