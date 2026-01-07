from agenwatch._kernel.sandbox.tool_sandbox import ToolSandbox
import asyncio
import logging
from typing import Any, Dict, Optional

from agenwatch._kernel.timeline_logger import timeline_logger
from agenwatch._kernel.safety.safety_guard import SafetyGuard
from agenwatch._kernel.repair_memory import RepairMemory
from agenwatch._kernel.retry_manager import ToolRetryManager, RetryPolicy
from agenwatch._kernel.safety.circuit_breaker import CircuitBreaker
from agenwatch._kernel.timeline_logger import ExecutionTimelineLogger as TimelineLogger
timeline_logger = TimelineLogger()


logger = logging.getLogger("agenwatch.tools")


class ToolExecutionEngine:
    """
    Executes tools in a controlled sandbox:

    - Timeout enforcement
    - CPU/memory monitoring
    - stdout/stderr capture
    - exception isolation
    - supports async + sync tools
    """

    def __init__(self, registry, default_timeout: float = 10.0):
        self.registry = registry
        self.default_timeout = float(default_timeout)

        # Circuit breaker for repeated tool failures
        self.circuit_breaker = CircuitBreaker(max_failures=3)

        # Fully wired RetryManager
        self.retry_manager = ToolRetryManager(
            name="tool_execution",
            retry_policy=RetryPolicy(max_attempts=3),
            circuit_breaker=self.circuit_breaker,
            timeline=timeline_logger,
            safety_guard=SafetyGuard,
            repair_memory=RepairMemory,
        )

        # Shared sandbox environment
        self.sandbox = ToolSandbox(
            max_execution_time=self.default_timeout,
            max_memory_mb=512.0,
            max_cpu_percent=90.0,
            enable_stream_capture=True,
            enable_metrics=True,
        )

        logger.info("[ToolExecutionEngine] initialized with sandbox")

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:

        arguments = arguments or {}
        timeout = float(timeout) if timeout else self.default_timeout

        # Retrieve tool instance
        tool = self.registry.get(tool_name)
        if tool is None:
            return {
                "error": f"Unknown tool: {tool_name}",
                "was_repaired": False
            }

        logger.info(f"[ToolExecutionEngine] (sandboxed) Running {tool_name} args={arguments}")

        # Wrap tool.run() for sandbox execution
        async def wrapped():
            result = tool.run(**arguments)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        async def sandbox_call():
            return await self.sandbox.execute(
                tool_func=wrapped,
                args={},
                policies=[],
                tool_name=tool_name
            )

        # Run with retry manager
        retry_res = await self.retry_manager.execute_async(sandbox_call)

        # FAILURE PATH
        if not retry_res.success:
            self.circuit_breaker.check_tool_failure(tool_name, str(retry_res.last_exception))

            return {
                "result": None,
                "error": str(retry_res.last_exception),
                "was_repaired": retry_res.repaired,
                "fallback_used": retry_res.fallback_used
            }

        # SUCCESS PATH → reset breaker
        self.circuit_breaker.reset_consecutive_failures()

        sandbox_out = retry_res.result
        result = sandbox_out

        # Convert SandboxResult → API format
        if not result.success:
            return {
                "result": None,
                "error": result.error,
                "violations": result.violations,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "metrics": result.metrics.to_dict() if result.metrics else None,
                "was_repaired": False
            }

        return {
            "result": result.output,
            "error": None,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "metrics": result.metrics.to_dict() if result.metrics else None,
            "was_repaired": False
        }

__INTERNAL__ = True



