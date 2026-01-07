from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from .llm_provider import LLMProvider


@dataclass(frozen=True)
class AgentConfig:
    max_iterations: int = 10
    verbose: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    output: Any | None
    error: str | None
    iterations: int
    tool_calls: List[str] = field(default_factory=list)
    cost: float = 0.0  # Total budget spent



@dataclass(frozen=True)
class StreamEvent:
    type: str                 # "session_start", "llm_call", "tool_call", "done"
    payload: Dict[str, Any]   # normalized data


