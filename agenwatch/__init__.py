"""
AgenWatch SDK – Public API

Anything not exported here is internal and unstable.
"""

from .sdk import Agent, tool
from .types import AgentConfig, ExecutionResult

__version__ = "0.1.2"

__all__ = [
    "Agent",
    "tool",
    "AgentConfig",
    "ExecutionResult",
]



