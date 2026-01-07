from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AgentResult:
    """Terminal result object for agent execution."""
    success: bool
    output: Any = None
    error_type: Optional[str] = None
    iterations: int = 0
    terminal_reason: Optional[str] = None
    cost: float = 0.0  # Total budget spent

    @classmethod
    def success(cls, output: Any = None, terminal_reason: str = None, iterations: int = 0, cost: float = 0.0) -> "AgentResult":
        return cls(
            success=True,
            output=output,
            terminal_reason=terminal_reason,
            iterations=iterations,
            cost=cost,
        )

    @classmethod
    def failure(cls, error_type: str, terminal_reason: str = None, iterations: int = 0, cost: float = 0.0) -> "AgentResult":
        return cls(
            success=False,
            output=None,
            error_type=error_type,
            terminal_reason=terminal_reason,
            iterations=iterations,
            cost=cost,
        )


__INTERNAL__ = True



