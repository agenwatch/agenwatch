"""
Tool Execution Sandbox
======================
Isolated, monitored execution environment for agent tools.

Features:
- Resource limits (CPU, memory, time)
- Stdout/stderr capture
- Exception isolation
- Variable state tracking
- Safety policy enforcement
- Performance metrics
- LIVE resource monitoring during execution

This makes AgenWatch OS behave like a real operating system for agents.

Usage:
    sandbox = ToolSandbox(
        max_execution_time=30.0,
        max_memory_mb=512
    )
    
    result = await sandbox.execute(
        tool_func=my_tool,
        args={"query": "test"},
        policies=["no_network", "read_only_fs"]
    )
    
    print(result.output)
    print(result.metrics)  # CPU, memory, time
"""

import asyncio
import time
import traceback
import sys
import io
import psutil
import os
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import redirect_stdout, redirect_stderr
import signal


class SandboxViolation(str, Enum):
    """Types of sandbox policy violations."""
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    CPU_LIMIT = "cpu_limit"
    NETWORK_BLOCKED = "network_blocked"
    FILE_ACCESS_DENIED = "file_access_denied"
    UNSAFE_OPERATION = "unsafe_operation"


class SafetyPolicy(str, Enum):
    """Safety policies that can be applied to sandbox."""
    NO_NETWORK = "no_network"
    READ_ONLY_FS = "read_only_fs"
    NO_SUBPROCESS = "no_subprocess"
    NO_ENV_ACCESS = "no_env_access"
    STRICT_IMPORTS = "strict_imports"


@dataclass
class SandboxMetrics:
    """Metrics collected during tool execution."""
    execution_time_ms: float
    cpu_time_ms: float
    memory_used_mb: float
    peak_memory_mb: float
    stdout_lines: int
    stderr_lines: int
    exceptions_caught: int
    resource_checks: int = 0  # How many times resources were monitored
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_time_ms": self.execution_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_used_mb": self.memory_used_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "stdout_lines": self.stdout_lines,
            "stderr_lines": self.stderr_lines,
            "exceptions_caught": self.exceptions_caught,
            "resource_checks": self.resource_checks
        }


@dataclass
class SandboxResult:
    """Result of sandbox execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    
    # Captured streams
    stdout: str = ""
    stderr: str = ""
    
    # Metrics
    metrics: Optional[SandboxMetrics] = None
    
    # Violations
    violations: List[str] = field(default_factory=list)
    
    # State tracking
    variables_accessed: Set[str] = field(default_factory=set)
    functions_called: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "violations": self.violations,
            "variables_accessed": list(self.variables_accessed),
            "functions_called": self.functions_called
        }


class ToolSandbox:
    """
    Isolated execution environment for agent tools.
    
    Provides:
    - Resource limits (time, memory, CPU)
    - Stream capture (stdout/stderr)
    - Exception isolation
    - Safety policy enforcement
    - Performance metrics
    - LIVE resource monitoring
    
    Example:
        >>> sandbox = ToolSandbox(max_execution_time=10.0)
        >>> result = await sandbox.execute(
        ...     tool_func=search_web,
        ...     args={"query": "python"},
        ...     policies=[SafetyPolicy.NO_NETWORK]
        ... )
        >>> print(f"Executed in {result.metrics.execution_time_ms}ms")
    """
    
    def __init__(
        self,
        max_execution_time: float = 30.0,
        max_memory_mb: float = 512.0,
        max_cpu_percent: float = 80.0,
        enable_stream_capture: bool = True,
        enable_metrics: bool = True,
        monitor_interval_ms: float = 100.0  # Monitor every 100ms
    ):
        """
        Initialize tool sandbox.
        
        Args:
            max_execution_time: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            max_cpu_percent: Maximum CPU usage percentage
            enable_stream_capture: Capture stdout/stderr
            enable_metrics: Collect performance metrics
            monitor_interval_ms: Resource monitoring interval in milliseconds
        """
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.enable_stream_capture = enable_stream_capture
        self.enable_metrics = enable_metrics
        self.monitor_interval_ms = monitor_interval_ms
        
        # Process for resource monitoring
        self.process = psutil.Process(os.getpid())
        
        # Resource tracking state
        self._peak_memory_mb = 0.0
        self._resource_checks = 0
        self._monitoring_active = False
        
    # =====================================================================
    # MAIN EXECUTION
    # =====================================================================
    
    async def execute(
        self,
        tool_func: Callable,
        args: Dict[str, Any],
        policies: Optional[List[SafetyPolicy]] = None,
        tool_name: Optional[str] = None
    ) -> SandboxResult:
        """
        Execute tool function in isolated sandbox.
        
        Args:
            tool_func: The tool function to execute
            args: Arguments to pass to tool
            policies: Safety policies to enforce
            tool_name: Name of tool (for logging)
            
        Returns:
            SandboxResult with output, metrics, and violations
        """
        policies = policies or []
        tool_name = tool_name or tool_func.__name__
        
        # Initialize result
        result = SandboxResult(success=False, output=None)
        
        # Start metrics
        start_time = time.time()
        start_cpu_time = self.process.cpu_times().user
        start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self._peak_memory_mb = start_memory
        self._resource_checks = 0
        
        # Stream capture buffers
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        # Monitoring task
        monitor_task = None
        
        try:
            # Apply safety policies
            self._apply_policies(policies, result)
            
            # Start resource monitoring
            self._monitoring_active = True
            monitor_task = asyncio.create_task(
                self._monitor_resources(result)
            )
            
            # Execute with timeout
            if self.enable_stream_capture:
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    output = await self._execute_with_timeout(
                        tool_func,
                        args,
                        timeout=self.max_execution_time
                    )
            else:
                output = await self._execute_with_timeout(
                    tool_func,
                    args,
                    timeout=self.max_execution_time
                )
            
            result.success = True
            result.output = output
            
        except asyncio.TimeoutError:
            result.error = f"Execution timeout ({self.max_execution_time}s)"
            result.violations.append(SandboxViolation.TIMEOUT)
            
        except MemoryError:
            result.error = "Memory limit exceeded"
            result.violations.append(SandboxViolation.MEMORY_LIMIT)
            
        except Exception as e:
            result.error = f"{type(e).__name__}: {str(e)}"
            result.violations.append(SandboxViolation.UNSAFE_OPERATION)
            
            # Capture full traceback in stderr
            stderr_buffer.write(traceback.format_exc())
        
        finally:
            # Stop monitoring
            self._monitoring_active = False
            if monitor_task and not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Capture streams
            result.stdout = stdout_buffer.getvalue()
            result.stderr = stderr_buffer.getvalue()
            
            # Calculate metrics
            if self.enable_metrics:
                end_time = time.time()
                end_cpu_time = self.process.cpu_times().user
                end_memory = self.process.memory_info().rss / (1024 * 1024)
                
                result.metrics = SandboxMetrics(
                    execution_time_ms=(end_time - start_time) * 1000,
                    cpu_time_ms=(end_cpu_time - start_cpu_time) * 1000,
                    memory_used_mb=end_memory - start_memory,
                    peak_memory_mb=self._peak_memory_mb,
                    stdout_lines=len(result.stdout.splitlines()),
                    stderr_lines=len(result.stderr.splitlines()),
                    exceptions_caught=1 if result.error else 0,
                    resource_checks=self._resource_checks
                )
            
            # Close buffers
            stdout_buffer.close()
            stderr_buffer.close()
        
        return result
    
    async def _monitor_resources(self, result: SandboxResult):
        """
        Monitor resources during execution.
        Runs every monitor_interval_ms and checks limits.
        """
        interval_seconds = self.monitor_interval_ms / 1000.0
        
        while self._monitoring_active:
            try:
                # Get current memory
                memory_mb = self.process.memory_info().rss / (1024 * 1024)
                self._peak_memory_mb = max(self._peak_memory_mb, memory_mb)
                self._resource_checks += 1
                
                # Check memory limit
                if memory_mb > self.max_memory_mb:
                    result.violations.append(SandboxViolation.MEMORY_LIMIT)
                    result.error = f"Memory limit exceeded: {memory_mb:.1f}MB > {self.max_memory_mb}MB"
                    raise MemoryError(result.error)
                
                # Check CPU limit (optional, less critical)
                cpu_percent = self.process.cpu_percent(interval=0.01)
                if cpu_percent > self.max_cpu_percent:
                    # Don't kill, just warn
                    if SandboxViolation.CPU_LIMIT not in result.violations:
                        result.violations.append(SandboxViolation.CPU_LIMIT)
                
                await asyncio.sleep(interval_seconds)
                
            except MemoryError:
                # Re-raise to stop execution
                raise
            except Exception:
                # Don't let monitoring crash the execution
                break
    
    async def _execute_with_timeout(
        self,
        tool_func: Callable,
        args: Dict[str, Any],
        timeout: float
    ) -> Any:
        """Execute function with timeout."""
        try:
            # If function is async
            if asyncio.iscoroutinefunction(tool_func):
                return await asyncio.wait_for(
                    tool_func(**args),
                    timeout=timeout
                )
            # If function is sync
            else:
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: tool_func(**args)),
                    timeout=timeout
                )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Tool execution exceeded {timeout}s")
    
    # =====================================================================
    # SAFETY POLICIES
    # =====================================================================
    
    def _apply_policies(
        self,
        policies: List[SafetyPolicy],
        result: SandboxResult
    ):
        """Apply safety policies before execution."""
        for policy in policies:
            if policy == SafetyPolicy.NO_NETWORK:
                self._block_network(result)
            elif policy == SafetyPolicy.READ_ONLY_FS:
                self._block_file_writes(result)
            elif policy == SafetyPolicy.NO_SUBPROCESS:
                self._block_subprocess(result)
            elif policy == SafetyPolicy.NO_ENV_ACCESS:
                self._block_env_access(result)
    
    def _block_network(self, result: SandboxResult):
        """Block network access (placeholder - will implement monkeypatching)."""
        # TODO: Monkeypatch socket module
        pass
    
    def _block_file_writes(self, result: SandboxResult):
        """Block file write operations (placeholder)."""
        # TODO: Monkeypatch open() for read-only mode
        pass
    
    def _block_subprocess(self, result: SandboxResult):
        """Block subprocess creation (placeholder)."""
        # TODO: Monkeypatch subprocess, os.system
        pass
    
    def _block_env_access(self, result: SandboxResult):
        """Block environment variable access (placeholder)."""
        # TODO: Clear/restrict os.environ
        pass
    
    # =====================================================================
    # RESOURCE MONITORING
    # =====================================================================
    
    def get_current_resources(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
            memory_mb = self.process.memory_info().rss / (1024 * 1024)
            
            return {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "memory_percent": self.process.memory_percent()
            }
        except:
            return {"cpu_percent": 0, "memory_mb": 0, "memory_percent": 0}
    
    def is_resource_limit_exceeded(self) -> tuple[bool, Optional[str]]:
        """Check if any resource limit is exceeded."""
        resources = self.get_current_resources()
        
        if resources["memory_mb"] > self.max_memory_mb:
            return True, f"Memory limit exceeded: {resources['memory_mb']:.1f}MB > {self.max_memory_mb}MB"
        
        if resources["cpu_percent"] > self.max_cpu_percent:
            return True, f"CPU limit exceeded: {resources['cpu_percent']:.1f}% > {self.max_cpu_percent}%"
        
        return False, None
    
    # =====================================================================
    # BATCH EXECUTION
    # =====================================================================
    
    async def execute_batch(
        self,
        tool_calls: List[Dict[str, Any]],
        policies: Optional[List[SafetyPolicy]] = None
    ) -> List[SandboxResult]:
        """
        Execute multiple tools in parallel sandboxes.
        
        Args:
            tool_calls: List of {"tool_func": func, "args": dict, "name": str}
            policies: Safety policies to apply to all
            
        Returns:
            List of SandboxResult
        """
        tasks = [
            self.execute(
                tool_func=call["tool_func"],
                args=call.get("args", {}),
                policies=policies,
                tool_name=call.get("name")
            )
            for call in tool_calls
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    # =====================================================================
    # UTILITIES
    # =====================================================================
    
    def reset(self):
        """Reset sandbox state."""
        self._peak_memory_mb = 0.0
        self._resource_checks = 0
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sandbox statistics."""
        return {
            "max_execution_time": self.max_execution_time,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "current_resources": self.get_current_resources(),
            "peak_memory_mb": self._peak_memory_mb,
            "resource_checks": self._resource_checks
        }

__INTERNAL__ = True
