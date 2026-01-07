from agenwatch.types import ExecutionResult


def adapt_agent_result(kernel_result) -> ExecutionResult:
    """
    Convert kernel AgentResult → SDK ExecutionResult.
    Kernel types must never escape.
    """
    return ExecutionResult(
        success=bool(getattr(kernel_result, "success", False)),
        output=getattr(kernel_result, "output", None),
        error=getattr(kernel_result, "error_type", None),
        iterations=getattr(kernel_result, "iterations", 0),
        tool_calls=list(getattr(kernel_result, "tool_calls", [])),
        cost=getattr(kernel_result, "cost", 0.0),
    )




