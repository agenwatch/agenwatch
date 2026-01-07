"""
LLM Provider Protocol (v0.1.x)
SDK-level interface for pluggable LLM backends
"""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class LLMMessage:
    """Standard message format across all providers"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class LLMResponse:
    """Standard response format from any LLM"""
    text: str
    model: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None  # Provider-specific, not standardized
    
    # Metadata
    tokens_used: int = 0


class LLMProvider(Protocol):
    """
    Protocol for LLM providers
    
    Any class implementing these methods can be used as an LLM backend.
    The kernel never sees this - it's SDK-only.
    
    IMPORTANT: Tool schema format is provider-specific.
    The SDK passes tools through unchanged to the provider.
    """
    
    async def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM with tool support
        """
        ...
    
    @property
    def model_name(self) -> str:
        """Return the model identifier"""
        ...


# =============================================================================
# REFERENCE IMPLEMENTATION: Anthropic Claude
# =============================================================================

class AnthropicProvider:
    """
    Reference implementation for Anthropic's Claude
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        
        # Lazy import - only load if user actually uses Anthropic
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "Anthropic SDK not installed. "
                "Install with: pip install anthropic"
            )
    
    async def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API"""
        
        # Convert to Anthropic format
        anthropic_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
            if msg.role != "system"
        ]
        
        # Extract system message
        system_messages = [msg.content for msg in messages if msg.role == "system"]
        system = system_messages[0] if system_messages else None
        
        # Call API (sync client, but wrapped in async)
        # Note: In production, use anthropic's async client
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=anthropic_messages,
            tools=tools or [],
            **kwargs
        )
        
        # Convert response
        content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input
                })
        
        return LLMResponse(
            text=content,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            model=response.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens
        )
    
    @property
    def model_name(self) -> str:
        return self.model


# =============================================================================
# REFERENCE IMPLEMENTATION: OpenAI
# =============================================================================

class OpenAIProvider:
    """
    Reference implementation for OpenAI models
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo",
        max_tokens: int = 4096
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI SDK not installed. "
                "Install with: pip install openai"
            )
    
    async def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API"""
        
        # Convert to OpenAI format
        openai_messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
        
        # Call API (async)
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=openai_messages,
            tools=tools or [],
            **kwargs
        )
        
        message = response.choices[0].message
        
        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                })
        
        return LLMResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            stop_reason=response.choices[0].finish_reason,
            model=response.model,
            tokens_used=response.usage.total_tokens
        )
    
    @property
    def model_name(self) -> str:
        return self.model


# =============================================================================
# MOCK PROVIDER (for testing)
# =============================================================================

class MockLLMProvider:
    """
    Mock provider for testing.
    Supports single response or a list of responses.
    """
    
    def __init__(
        self,
        response: Optional[str] = None,
        tool_to_call: Optional[str] = None,
        responses: Optional[List[Dict[str, Any]]] = None
    ):
        self._static_response = response or "Mock response"
        self._static_tool = tool_to_call
        self.responses = responses
        self._index = 0
    
    async def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Return next mock response or static one"""
        
        if self.responses and self._index < len(self.responses):
            data = self.responses[self._index]
            self._index += 1
            
            # Handle both dict formats
            text = data.get("text") or data.get("final") or ""
            tool_calls = data.get("tool_calls") or []
        elif self.responses and self._index >= len(self.responses):
            # Exhausted scripted responses - return explicit terminal response
            return LLMResponse(
                text="done",
                model="mock-model",
                tool_calls=[],
                stop_reason="stop",
                tokens_used=0
            )
        else:
            text = self._static_response
            tool_calls = []
            if self._static_tool:
                tool_calls.append({
                    "id": f"call_{self._index}",
                    "name": self._static_tool,
                    "arguments": {}
                })
        
        return LLMResponse(
            text=text,
            model="mock-model",
            tool_calls=tool_calls,
            stop_reason="end_turn",
            tokens_used=100
        )
    
    @property
    def model_name(self) -> str:
        return "mock-model"


