"""
Groq Provider
Fast inference provider using Groq's infrastructure
"""

import json
from typing import List, Dict, Any, Optional
from agenwatch.llm_provider import LLMProvider, LLMMessage, LLMResponse


class GroqProvider:
    """
    Official Groq provider
    
    Supports fast inference for open models like Llama, Mixtral, etc.
    
    Installation:
        pip install groq
    
    Usage:
        import os
        from agenwatch.providers import GroqProvider
        
        provider = GroqProvider(
            api_key=os.getenv("GROQ_API_KEY"),
            model="mixtral-8x7b-32768"
        )
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",  # Updated default
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 60.0
    ):
        """
        Initialize Groq provider
        
        Args:
            api_key: Groq API key (get from console.groq.com)
            model: Model name (llama-3.3-70b-versatile, llama-3.1-70b-versatile, gemma2-9b-it, etc.)
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
            from groq import AsyncGroq
            self.client = AsyncGroq(
                api_key=api_key,
                timeout=timeout
            )
        except ImportError:
            raise ImportError(
                "Groq SDK not installed. "
                "Install with: pip install groq"
            )
    
    async def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using Groq API
        
        Args:
            messages: Conversation history
            tools: Tool definitions in OpenAI-compatible format
            **kwargs: Additional API parameters
        
        Returns:
            LLMResponse with content and tool calls
        """
        
        # Convert messages to Groq format (OpenAI-compatible)
        groq_messages = []
        
        for msg in messages:
            groq_msg = {
                "role": msg.role,
                "content": msg.content
            }
            
            # Add tool call info if present
            if msg.tool_calls:
                groq_msg["tool_calls"] = msg.tool_calls
            
            if msg.tool_call_id:
                groq_msg["tool_call_id"] = msg.tool_call_id
            
            groq_messages.append(groq_msg)
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": groq_messages,
            **kwargs
        }
        
        request_params.pop("tools", None)

        # Call API
        response = await self.client.chat.completions.create(**request_params)
        
        message = response.choices[0].message
        
        # Parse tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments (Groq returns JSON string like OpenAI)
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