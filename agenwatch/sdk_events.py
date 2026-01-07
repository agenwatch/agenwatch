from dataclasses import dataclass
from typing import Any, Literal

EventType = Literal[
    "start",
    "llm_call",
    "tool_call",
    "tool_result",
    "iteration",
    "error",
    "final",
]

@dataclass(frozen=True)
class StreamEvent:
    type: EventType
    payload: Any



