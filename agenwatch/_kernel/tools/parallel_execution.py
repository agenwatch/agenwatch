"""
AgenWatch Parallel Tool Execution Engine
==========================================
Runs multiple tools in parallel with:
- Per-tool timeout
- Error isolation (one failure does not stop others)
- Success = (error is None)
- Performance tracking
- Circuit breaker for failing tools
- Result merging for context aggregation
- Fail-fast mode for strict execution
- DAG planner for dependency-based execution
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum

from agenwatch._kernel.tools.execution import ToolExecutionEngine
from agenwatch._kernel.tools.merger import ToolResultMerger
from agenwatch._kernel.sandbox.tool_sandbox import ToolSandbox, SafetyPolicy
from agenwatch._kernel.tools.registry import ToolRegistry


# =====================================================================
# ENUMS
# =====================================================================
class ExecutionMode(Enum):
    """Execution mode for parallel engine."""
    PARALLEL = "parallel"      # All tools run in parallel
    FAIL_FAST = "fail_fast"    # Stop on first failure
    DAG = "dag"                # Run based on dependency graph


# =====================================================================
# DATACLASSES
# =====================================================================
@dataclass
class ToolExecutionResult:
    """Result of a single tool execution."""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timed_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out
        }


@dataclass
class DAGNode:
    """Node in dependency graph."""
    tool_name: str
    args: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)  # tool names this depends on
    result: Optional[ToolExecutionResult] = None
    executed: bool = False


@dataclass
class MergedResult:
    """Merged results from multiple tools."""
    individual_results: Dict[str, ToolExecutionResult]
    merged_context: Dict[str, Any]
    success: bool
    total_duration_ms: float
    tool_count: int
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    final_message: str = ""
    timestamp: str = ""


# =====================================================================
# PARALLEL TOOL EXECUTION ENGINE
# =====================================================================
class ParallelToolExecutionEngine:
    """
    Executes multiple tools in parallel with advanced features:
    - Parallel execution (all tools concurrent)
    - Fail-fast mode (stop on first failure)
    - DAG execution (respect dependencies)
    - Result merging (aggregate results)
    - Circuit breaker (protect failing tools)
    """

    def __init__(
        self,
        engine: ToolExecutionEngine,
        default_timeout: float = 20.0,
        max_parallel: int = 10,
        execution_mode: ExecutionMode = ExecutionMode.PARALLEL
    ):
        """
        Initialize parallel execution engine.
        
        Args:
            engine: ToolExecutionEngine instance
            default_timeout: Default timeout per tool in seconds
            max_parallel: Max tools to run in parallel
            execution_mode: Execution mode (parallel, fail_fast, dag)
        """
        self.engine = engine
        self.default_timeout = float(default_timeout)
        self.max_parallel = max_parallel
        self.execution_mode = execution_mode
        
        self.registry = getattr(engine, "registry", None)
        
        self.sandbox = ToolSandbox(
            max_execution_time=30.0,
            max_memory_mb=512,
            max_cpu_percent=90.0
        )
        
        # telemetry storage
        self.telemetry_events = []
        self.registry = getattr(engine, "registry", None)


        
        # circuit breaker state
        self.failure_counts = {}       # {tool_name: number_of_failures}
        self.failure_threshold = 3     # trip breaker after 3 failures

    
    def _is_circuit_open(self, tool_name: str) -> bool:
        """Check if circuit breaker is open for this tool."""
        return self.failure_counts.get(tool_name, 0) >= self.failure_threshold

    
    def _emit_telemetry(self, tool_name: str, r: ToolExecutionResult):
        """Record telemetry event for tool execution."""
        event = {
            "tool_name": tool_name,
            "success": r.success,
            "error": r.error,
            "duration_ms": r.duration_ms,
            "timed_out": r.timed_out,
        }
        self.telemetry_events.append(event)
        print(f"[Telemetry] {event}")


    # =====================================================================
    # 1️⃣ RESULT MERGING
    # =====================================================================


    async def _merge_results(
        self,
        results: Dict[str, ToolExecutionResult],
        dependencies: Dict[str, list] | None = None,
        merge_context: bool = True
    ) -> MergedResult:
        """
        Merge individual tool results using the new ToolResultMerger.

        Args:
            results: {tool_name: ToolExecutionResult}
            dependencies: Optional {tool_name: [deps]}
            merge_context: Whether context merging is enabled

        Returns:
            MergeResult (from merger.py)
        """

        # Convert ToolExecutionResult → plain dict expected by merger
        indiv = {}

        for tool_name, r in results.items():
            if r.success:
                indiv[tool_name] = {"result": r.result}
            else:
                indiv[tool_name] = {
                    "error": r.error,
                    "timed_out": r.timed_out
                }

        # Use your new merger
        merger = ToolResultMerger()
        merge_result = merger.merge(individual_results=indiv, dependencies=dependencies)

        # Get summary stats
        summary_dict = merge_result.to_dict()["summary"]

        # Wrap back into MergedResult so parallel engine API stays the same
        return MergedResult(
            individual_results=results,               # original objects preserved
            merged_context=merge_result.merged,       # merged dict
            success=summary_dict["failed_tools"] == 0,  # Extract from summary dict
            total_duration_ms=sum(r.duration_ms for r in results.values()),
            tool_count=len(results),
            conflicts=[c.to_dict() for c in merge_result.conflicts],
            gaps=[g.to_dict() for g in merge_result.gaps],
            warnings=merge_result.warnings,
            final_message=merge_result.final_message,
            timestamp=merge_result.timestamp
        )



    # =====================================================================
    # 2️⃣ FAIL-FAST MODE
    # =====================================================================
    async def execute_fail_fast(
        self,
        calls: Dict[str, Dict[str, Any]],
        timeout: Optional[float] = None
    ) -> Dict[str, ToolExecutionResult]:
        """
        Execute tools in parallel but stop on first failure.
        
        Args:
            calls: {tool_name: args_dict}
            timeout: Override default timeout
        
        Returns:
            {tool_name: ToolExecutionResult} (stops early on failure)
        """
        timeout = float(timeout) if timeout else self.default_timeout
        results = {}
        
        # Create async tasks (SANDBOX)
        tasks = {
            name: asyncio.create_task(
                self._execute_tool(name, args)   # <— sandbox-powered
            )
            for name, args in calls.items()
        }


        # Execute with early termination
        for coro in asyncio.as_completed(tasks.values()):
            result = await coro
            results[result.tool_name] = result
            
            # FAIL-FAST: Stop if any tool fails
            if not result.success:
                print(f"[FailFast] Stopping execution - {result.tool_name} failed: {result.error}")
                
                # Cancel remaining tasks
                for task in tasks.values():
                    if not task.done():
                        task.cancel()
                
                break

        return results


    # =====================================================================
    # 3️⃣ DAG PLANNER
    # =====================================================================
    def _build_dag(
        self,
        calls: Dict[str, Dict[str, Any]],
        dependencies: Dict[str, List[str]]
    ) -> List[DAGNode]:
        """
        Build directed acyclic graph from dependencies.
        
        Args:
            calls: {tool_name: args}
            dependencies: {tool_name: [dependent_tool_names]}
        
        Returns:
            List of DAGNode objects
        """
        nodes = {}
        
        for tool_name, args in calls.items():
            deps = dependencies.get(tool_name, [])
            nodes[tool_name] = DAGNode(
                tool_name=tool_name,
                args=args,
                depends_on=deps
            )
        
        return list(nodes.values())

    async def execute_dag(
        self,
        calls: Dict[str, Dict[str, Any]],
        dependencies: Dict[str, List[str]],
        timeout: Optional[float] = None
    ) -> Dict[str, ToolExecutionResult]:
        """
        Execute tools respecting dependency graph.
        
        Args:
            calls: {tool_name: args_dict}
            dependencies: {tool_name: [tools_it_depends_on]}
            timeout: Override default timeout
        
        Returns:
            {tool_name: ToolExecutionResult}
        """
        timeout = float(timeout) if timeout else self.default_timeout
        nodes = self._build_dag(calls, dependencies)
        results = {}
        executed_set: Set[str] = set()

        # Topological execution
        while len(executed_set) < len(nodes):
            # Find executable nodes (all dependencies met)
            ready_nodes = [
                n for n in nodes
                if not n.executed and all(dep in executed_set for dep in n.depends_on)
            ]

            if not ready_nodes:
                raise ValueError("Circular dependency detected in DAG")

            # Execute ready nodes in parallel (SANDBOX)
            tasks = {
                node.tool_name: asyncio.create_task(
                    self._execute_tool(
                        node.tool_name,
                        node.args
                    )
                )
                for node in ready_nodes
            }


            gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)

            for node, result in zip(ready_nodes, gathered):
                if isinstance(result, ToolExecutionResult):
                    results[node.tool_name] = result
                    node.result = result
                    node.executed = True
                    executed_set.add(node.tool_name)
                else:
                    result = ToolExecutionResult(
                        tool_name=node.tool_name,
                        success=False,
                        error=str(result),
                        duration_ms=0.0
                    )
                    results[node.tool_name] = result
                    node.result = result
                    node.executed = True
                    executed_set.add(node.tool_name)

                # FAIL-FAST in DAG: Stop if critical tool fails
                if not result.success:
                    print(f"[DAG] Tool {node.tool_name} failed: {result.error}")

        return results


    # =====================================================================
    # PUBLIC: MAIN EXECUTION DISPATCHER
    # =====================================================================
    async def execute_parallel(
        self,
        calls: Dict[str, Dict[str, Any]],
        timeout: Optional[float] = None,
        mode: Optional[ExecutionMode] = None,
        dependencies: Optional[Dict[str, List[str]]] = None,
        merge_results: bool = False,
        tools: Optional[Dict[str, Any]] = None
    ) -> Any:

        """
        Execute tools using configured or specified mode.
        
        Args:
            calls: {tool_name: args_dict}
            timeout: Override default timeout
            mode: Override execution mode (parallel/fail_fast/dag)
            dependencies: For DAG mode, tool dependencies
            merge_results: Whether to merge results
        
        Returns:
            Dict[str, ToolExecutionResult] or MergedResult
        """
        if not calls:
            return {}

        exec_mode = mode or self.execution_mode
        timeout = float(timeout) if timeout else self.default_timeout

        if tools:
            if not self.registry:
                self.registry = ToolRegistry()
            for name, fn in tools.items():
                if not self.registry.get(name):
                    self.registry.register(name, fn)

        # Execute based on mode
        if exec_mode == ExecutionMode.FAIL_FAST:
            results = await self.execute_fail_fast(calls, timeout)
        elif exec_mode == ExecutionMode.DAG:
            if not dependencies:
                raise ValueError("DAG mode requires dependencies parameter")
            results = await self.execute_dag(calls, dependencies, timeout)
        else:  # PARALLEL (default)
            results = await self._execute_parallel_standard(calls, timeout)

        # Optionally merge results
        if merge_results:
            # pass dependencies to merger only in DAG mode
            deps = dependencies if exec_mode == ExecutionMode.DAG else None
            return await self._merge_results(results, dependencies=deps)


        return results


    async def _execute_parallel_standard(
        self,
        calls: Dict[str, Dict[str, Any]],
        timeout: float
    ) -> Dict[str, ToolExecutionResult]:
        """Standard parallel execution."""
        if len(calls) > self.max_parallel:
            return await self._execute_batched(calls, timeout)

        tasks = {
            name: asyncio.create_task(
                self._execute_tool(name, args)
            )
            for name, args in calls.items()
        }


        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        results = {}

        for (tool_name, _), item in zip(tasks.items(), gathered):
            if isinstance(item, ToolExecutionResult):
                results[tool_name] = item
            elif isinstance(item, Exception):
                results[tool_name] = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(item),
                    duration_ms=0.0
                )
            else:
                results[tool_name] = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error="Unexpected result type",
                    duration_ms=0.0
                )

        return results


    # =====================================================================
    # INTERNAL: Execute single tool
    # =====================================================================
    async def _execute_single_with_timeout(
        self,
        tool_name: str,
        args: Dict[str, Any],
        timeout: float
    ) -> ToolExecutionResult:
        """Execute a single tool with timeout and retry logic."""

        start = time.time()
        retries = 1
        attempt = 0

        # Circuit breaker check
        if self._is_circuit_open(tool_name):
            r = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"circuit_open: tool '{tool_name}' temporarily disabled",
                duration_ms=0.0,
                timed_out=False
            )
            self._emit_telemetry(tool_name, r)
            return r

        while attempt <= retries:
            try:
                result = await asyncio.wait_for(
                    self.engine.execute_tool(tool_name, args),
                    timeout=timeout
                )

                duration = (time.time() - start) * 1000
                error = result.get("error")
                success = error is None

                r = ToolExecutionResult(
                    tool_name=tool_name,
                    success=success,
                    result=result.get("result") if success else None,
                    error=error,
                    duration_ms=duration,
                    timed_out=False
                )
                self.failure_counts[tool_name] = 0
                self._emit_telemetry(tool_name, r)
                return r

            except asyncio.TimeoutError:
                duration = (time.time() - start) * 1000
                r = ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"Tool '{tool_name}' timed out after {timeout}s",
                    duration_ms=duration,
                    timed_out=True
                )
                self._emit_telemetry(tool_name, r)
                return r

            except Exception as e:
                self.failure_counts[tool_name] = self.failure_counts.get(tool_name, 0) + 1

                attempt += 1
                if attempt > retries:
                    duration = (time.time() - start) * 1000
                    r = ToolExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        error=str(e),
                        duration_ms=duration,
                        timed_out=False
                    )
                    self._emit_telemetry(tool_name, r)
                    return r

                await asyncio.sleep(0.1)
                
                
    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolExecutionResult:
        start = time.time()

        try:
            tool = self.registry.get(tool_name)
            if tool is None:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=f"Unknown tool '{tool_name}'",
                    duration_ms=0.0
                )

            sandbox_result = await self.sandbox.execute(
                tool_func=tool.run,
                args=args,
                policies=[SafetyPolicy.NO_SUBPROCESS]
            )

            duration = (time.time() - start) * 1000

            if sandbox_result.error:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=str(sandbox_result.error),
                    duration_ms=duration
                )

            return ToolExecutionResult(
                tool_name=tool_name,
                success=True,
                result=sandbox_result.output,
                error=None,
                duration_ms=duration
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                duration_ms=duration
            )



        # =====================================================================
        # BATCH EXECUTION
        # =====================================================================
        async def _execute_batched(
            self,
            calls: Dict[str, Dict[str, Any]],
            timeout: float
        ) -> Dict[str, ToolExecutionResult]:
            """Execute tools in batches."""
            items = list(calls.items())
            all_results = {}

            for i in range(0, len(items), self.max_parallel):
                batch = dict(items[i:i + self.max_parallel])
                batch_results = await self._execute_parallel_standard(batch, timeout)
                all_results.update(batch_results)

            return all_results


    # =====================================================================
    # UTILITIES
    # =====================================================================
    def get_summary(self, results: Dict[str, ToolExecutionResult]) -> Dict[str, Any]:
        """Get execution summary statistics."""
        total = len(results)
        success_count = sum(1 for r in results.values() if r.success)
        timeout_count = sum(1 for r in results.values() if r.timed_out)

        return {
            "total_tools": total,
            "successful": success_count,
            "failed": total - success_count,
            "timed_out": timeout_count,
            "success_rate": success_count / total if total > 0 else 0.0,
            "avg_duration_ms": (
                sum(r.duration_ms for r in results.values()) / total if total else 0.0
            ),
            "results": {name: r.to_dict() for name, r in results.items()}
        }

    def get_breaker_status(self) -> Dict[str, int]:
        """Get circuit breaker status for all tools."""
        return self.failure_counts.copy()

    def reset_breaker(self, tool_name: Optional[str] = None):
        """Reset circuit breaker for a tool or all tools."""
        if tool_name:
            self.failure_counts[tool_name] = 0
        else:
            self.failure_counts.clear()

__INTERNAL__ = True



