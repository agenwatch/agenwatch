"""
AgenWatch Official Providers

This module contains official LLM provider implementations.

Available providers:
- AnthropicProvider: Claude models via Anthropic API
- OpenAIProvider: GPT models via OpenAI API
- GroqProvider: Fast open models via Groq infrastructure

Each provider requires its respective SDK to be installed:
    pip install anthropic  # For AnthropicProvider
    pip install openai     # For OpenAIProvider
    pip install groq       # For GroqProvider
"""

from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .groq import GroqProvider

__all__ = [
    "AnthropicProvider",
    "OpenAIProvider",
    "GroqProvider",
]

