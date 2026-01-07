"""
OpenAI Provider
Official provider for GPT models via OpenAI API
"""

import json
from typing import List, Dict, Any, Optional
from agenwatch.llm_provider import LLMProvider, LLMMessage, LLMResponse


class OpenAIProvider:
    """
    Official OpenAI provider
    
    Supports GPT-4, GPT-3.5, and other OpenAI models.
    
    Installation:
        pip install openai
    
    Usage:
        import os
        from agenwatch.providers import OpenAIProvider
        
        provider = OpenAIProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo"
        )
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 60.0
    ):
        """
        Initialize OpenAI provider
        
        Args:
            api_key: OpenAI API key (get from platform.openai.com)
            model: Model name (gpt-4-turbo, gpt-4o, gpt-3.5-turbo, etc.)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Lazy import
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=api_key,
                timeout=timeout
            )
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
        """
        Generate response using OpenAI API
        
        Args:
            messages: Conversation history
            tools: Tool definitions in OpenAI format
            **kwargs: Additional API parameters
        
        Returns:
            LLMResponse with content and tool calls
        """
        
        # Convert messages to OpenAI format
        openai_messages = []
        
        for msg in messages:
            openai_msg = {
                "role": msg.role,
                "content": msg.content
            }
            
            # Add tool call info if present
            if msg.tool_calls:
                openai_msg["tool_calls"] = msg.tool_calls
            
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            
            openai_messages.append(openai_msg)
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": openai_messages,
            **kwargs
        }
        
        if tools:
            request_params.pop("tools", None)
        
        # Call API
        response = await self.client.chat.completions.create(**request_params)
        
        message = response.choices[0].message
        
        # Parse tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments (OpenAI returns JSON string)
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": arguments
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
        """Return model identifier"""
        return self.model

