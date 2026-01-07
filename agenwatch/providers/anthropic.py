"""
Anthropic Claude Provider
Official provider for Claude models via Anthropic API
"""

from typing import List, Dict, Any, Optional
from agenwatch.llm_provider import LLMProvider, LLMMessage, LLMResponse


class AnthropicProvider:
    """
    Official Anthropic Claude provider
    
    Supports all Claude models via the Anthropic API.
    
    Installation:
        pip install anthropic
    
    Usage:
        import os
        from agenwatch.providers import AnthropicProvider
        
        provider = AnthropicProvider(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-sonnet-4-20250514"
        )
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 1.0,
        timeout: float = 60.0
    ):
        """
        Initialize Anthropic provider
        
        Args:
            api_key: Anthropic API key (get from console.anthropic.com)
            model: Model name (claude-3-5-sonnet-20241022, claude-sonnet-4-20250514, etc.)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Lazy import
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(
                api_key=api_key,
                timeout=timeout
            )
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
        """
        Generate response using Anthropic API
        
        Args:
            messages: Conversation history
            tools: Tool definitions in Anthropic format
            **kwargs: Additional API parameters
        
        Returns:
            LLMResponse with content and tool calls
        """
        
        # Separate system messages from conversation
        system_content = None
        conversation = []
        
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                # Convert to Anthropic format
                anthropic_msg = {"role": msg.role, "content": msg.content}
                
                # Add tool results if present
                if msg.role == "tool" and msg.tool_call_id:
                    anthropic_msg["tool_use_id"] = msg.tool_call_id
                
                conversation.append(anthropic_msg)
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": conversation,
            **kwargs
        }
        
        if system_content:
            request_params["system"] = system_content
        
        if tools:
            request_params["tools"] = tools
        
        # Call API
        response = await self.client.messages.create(**request_params)
        
        # Parse response
        content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input
                })
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            model=response.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens
        )
    
    @property
    def model_name(self) -> str:
        """Return model identifier"""
        return self.model


