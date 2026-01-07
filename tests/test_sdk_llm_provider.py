import pytest
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from agenwatch import Agent, AgentConfig
from agenwatch.llm_provider import LLMProvider, LLMMessage, LLMResponse, MockLLMProvider

class CustomProvider:
    """A fake provider implementing LLMProvider protocol"""
    def __init__(self):
        self.received_messages = []
        self.model = "custom-model"

    async def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        self.received_messages = messages
        return LLMResponse(
            text="Custom response",
            model=self.model,
            tokens_used=50
        )

    @property
    def model_name(self) -> str:
        return self.model

class SyncProvider:
    """A provider that incorrectly implements sync generate_with_tools"""
    def generate(self, messages, tools=None, **kwargs):
        return "Not a coroutine"
    
    @property
    def model_name(self) -> str:
        return "sync-model"

@pytest.mark.asyncio
async def test_custom_provider_works():
    provider = CustomProvider()
    agent = Agent(llm=provider)
    
    result = await agent.arun("Hello")
    
    if not result.success:
        print(f"DEBUG: result fail reason: {result.error}")
    assert result.success is True
    # The kernel might wrap the output or return it as is.
    # LLMResponse.content becomes AgentResult.output
    assert result.output == "Custom response"
    assert len(provider.received_messages) > 0
    assert agent.llm == provider

@pytest.mark.asyncio
async def test_no_llm_uses_mock():
    agent = Agent()
    result = await agent.arun("Hello")
    
    assert result.success is True
    assert result.output == "Mock response"

@pytest.mark.asyncio
async def test_kernel_receives_normalized_messages():
    provider = CustomProvider()
    agent = Agent(llm=provider)
    
    await agent.arun("Test task")
    
    # Check that messages passed to provider are LLMMessage objects
    for msg in provider.received_messages:
        # The adapter might convert them to LLMMessage
        assert hasattr(msg, "role")
        assert hasattr(msg, "content")

@pytest.mark.asyncio
async def test_sync_provider_raises_error():
    provider = SyncProvider()
    agent = Agent(llm=provider)
    
    result = await agent.arun("Hello")
    
    assert result.success is False
    # The kernel catches the TypeError and returns an error result
    assert result.error == "AGENT_ERROR"


