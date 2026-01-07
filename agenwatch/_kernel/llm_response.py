from dataclasses import dataclass, field
from typing import Any, List, Optional

@dataclass
class LLMResponse:
    text: Optional[str] = None
    tool_calls: Optional[list] = None
    output: Optional[str] = None
    content: Optional[str] = None
    raw: Any = None
    instrumentation: Any = field(default_factory=dict)

__INTERNAL__ = True


