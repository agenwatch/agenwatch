"""
Fail-Fast Rule Engine
=====================
Global circuit breaker for agent stability.
Stops agent execution when:
- Too many consecutive tool failures
- Too many repair attempts
- Invalid tool calls pattern detected
- Agent enters bad state (loops, oscillations)

Mode: BALANCED (stops after 2-3 consecutive failures)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime, timedelta


class FailureType(Enum):
    """Types of failures that trigger circuit breaker."""
    TOOL_EXECUTION = "tool_execution"      # Tool crashed/timed out
    INVALID_SCHEMA = "invalid_schema"      # Tool args invalid
    REPAIR_FAILED = "repair_failed"        # Auto-repair couldn't fix it
    LOOP_DETECTED = "loop_detected"        # Agent oscillating
    DEPENDENCY_MISSING = "dependency_missing"  # Required dependency failed
    TIMEOUT = "timeout"                    # Tool took too long
    MEMORY_ERROR = "memory_error"          # Out of memory
    QUOTA_EXCEEDED = "quota_exceeded"      # API quota hit


@dataclass
class FailureRecord:
    """Record of a single failure."""
    failure_type: FailureType
    tool_name: str
    error_message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "tool_name": self.tool_name,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }


@dataclass
class CircuitBreakerState:
    """State of the circuit breaker."""
    is_open: bool = False              # True = stop execution
    consecutive_failures: int = 0       # Count of failures in a row
    last_failure_time: Optional[datetime] = None
    total_failures: int = 0
    failure_history: List[FailureRecord] = field(default_factory=list)
    reset_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_open": self.is_open,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "failure_count": len(self.failure_history),
            "reset_reason": self.reset_reason
        }


class FailFastRuleEngine:
    """
    Balanced fail-fast circuit breaker.
    
    Stops agent when:
    - 3+ consecutive tool failures
    - 5+ total repair attempts
    - Loop/oscillation detected
    - Dependency chain broken (critical tool fails)
    """
    
    def __init__(
        self,
        max_consecutive_failures: int = 3,
        max_total_failures: int = 10,
        max_repair_attempts: int = 5,
        failure_window_minutes: int = 5
    ):
        """
        Initialize fail-fast engine.
        
        Args:
            max_consecutive_failures: Stop after N failures in a row (balanced: 3)
            max_total_failures: Stop if total failures exceed this (balanced: 10)
            max_repair_attempts: Stop if repair attempts exceed this (balanced: 5)
            failure_window_minutes: Time window for tracking failures
        """
        self.max_consecutive_failures = max_consecutive_failures
        self.max_total_failures = max_total_failures
        self.max_repair_attempts = max_repair_attempts
        self.failure_window = timedelta(minutes=failure_window_minutes)
        
        self.state = CircuitBreakerState()
        self.repair_attempts = 0
    
    def check_tool_failure(
        self,
        tool_name: str,
        error: str,
        failure_type: FailureType = FailureType.TOOL_EXECUTION,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Check if tool failure should trigger circuit breaker.
        
        Args:
            tool_name: Name of failed tool
            error: Error message
            failure_type: Type of failure
            context: Additional context
        
        Returns:
            (should_stop: bool, reason: str or None)
        """
        context = context or {}
        
        # Record the failure
        failure = FailureRecord(
            failure_type=failure_type,
            tool_name=tool_name,
            error_message=error,
            context=context
        )
        
        self.state.failure_history.append(failure)
        self.state.consecutive_failures += 1
        self.state.total_failures += 1
        self.state.last_failure_time = datetime.utcnow()
        
        # =========================================================
        # BALANCED MODE: Check circuit breaker thresholds
        # =========================================================
        
        # Rule 1: Too many consecutive failures
        if self.state.consecutive_failures >= self.max_consecutive_failures:
            self.state.is_open = True
            reason = f"Circuit breaker: {self.state.consecutive_failures} consecutive failures"
            return True, reason
        
        # Rule 2: Too many total failures
        if self.state.total_failures >= self.max_total_failures:
            self.state.is_open = True
            reason = f"Circuit breaker: {self.state.total_failures} total failures exceeded"
            return True, reason
        
        # Rule 3: Critical failures (dependency, quota)
        if failure_type in [
            FailureType.DEPENDENCY_MISSING,
            FailureType.QUOTA_EXCEEDED,
            FailureType.MEMORY_ERROR
        ]:
            self.state.is_open = True
            reason = f"Circuit breaker: Critical failure ({failure_type.value})"
            return True, reason
        
        # Rule 4: Repair attempts exceeded
        if self.repair_attempts >= self.max_repair_attempts:
            self.state.is_open = True
            reason = f"Circuit breaker: {self.repair_attempts} repair attempts exceeded"
            return True, reason
        
        return False, None
    
    def check_loop_detected(self, loop_pattern: str) -> tuple[bool, Optional[str]]:
        """
        Check if loop/oscillation should trigger circuit breaker.
        
        Args:
            loop_pattern: Description of detected loop (e.g., "A→B→A→B")
        
        Returns:
            (should_stop: bool, reason: str or None)
        """
        self.state.is_open = True
        reason = f"Circuit breaker: Oscillation detected ({loop_pattern})"
        
        failure = FailureRecord(
            failure_type=FailureType.LOOP_DETECTED,
            tool_name="agent",
            error_message=f"Oscillation: {loop_pattern}",
            context={"pattern": loop_pattern}
        )
        self.state.failure_history.append(failure)
        
        return True, reason
    
    def check_invalid_schema_pattern(self, count: int) -> tuple[bool, Optional[str]]:
        """
        Check if repeated schema errors trigger circuit breaker.
        
        Args:
            count: Number of consecutive schema errors
        
        Returns:
            (should_stop: bool, reason: str or None)
        """
        # If 3+ schema errors in a row, stop
        if count >= 3:
            self.state.is_open = True
            reason = f"Circuit breaker: {count} consecutive schema errors"
            
            failure = FailureRecord(
                failure_type=FailureType.INVALID_SCHEMA,
                tool_name="agent",
                error_message=f"{count} consecutive schema validation failures",
                context={"schema_error_count": count}
            )
            self.state.failure_history.append(failure)
            
            return True, reason
        
        return False, None
    
    def record_repair_attempt(self) -> None:
        """Record that auto-repair was attempted."""
        self.repair_attempts += 1
    
    def reset_consecutive_failures(self, reason: str = "Tool succeeded") -> None:
        """
        Reset consecutive failure counter (tool succeeded).
        
        Args:
            reason: Why we're resetting
        """
        self.state.consecutive_failures = 0
        self.state.reset_reason = reason
    
    def reset_circuit(self, reason: str = "Manual reset") -> None:
        """
        Manually reset the circuit breaker.
        
        Args:
            reason: Why we're resetting
        """
        self.state.is_open = False
        self.state.consecutive_failures = 0
        self.repair_attempts = 0
        self.state.reset_reason = reason
        self.state.last_failure_time = None
    
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (stop execution)."""
        return self.state.is_open
    
    def should_continue_iteration(self) -> tuple[bool, Optional[str]]:
        """
        Check if agent should continue to next iteration.
        
        Returns:
            (should_continue: bool, reason: str or None)
        """
        if self.state.is_open:
            reason = "Circuit breaker is open - stopping execution"
            return False, reason
        
        return True, None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "circuit_open": self.state.is_open,
            "consecutive_failures": self.state.consecutive_failures,
            "total_failures": self.state.total_failures,
            "repair_attempts": self.repair_attempts,
            "max_consecutive": self.max_consecutive_failures,
            "max_total": self.max_total_failures,
            "max_repairs": self.max_repair_attempts,
            "last_failure": self.state.last_failure_time.isoformat() if self.state.last_failure_time else None,
            "failure_history": [f.to_dict() for f in self.state.failure_history[-5:]]  # Last 5
        }
    
    def get_failure_summary(self) -> str:
        """Get human-readable failure summary."""
        if not self.state.failure_history:
            return "No failures recorded."
        
        lines = [
            f"Failure Report ({len(self.state.failure_history)} total):",
            f"- Circuit: {'OPEN (stopped)' if self.state.is_open else 'CLOSED (running)'}",
            f"- Consecutive: {self.state.consecutive_failures}/{self.max_consecutive_failures}",
            f"- Total: {self.state.total_failures}/{self.max_total_failures}",
            f"- Repairs: {self.repair_attempts}/{self.max_repair_attempts}",
            ""
        ]
        
        # Last few failures
        lines.append("Recent failures:")
        for failure in self.state.failure_history[-5:]:
            lines.append(
                f"  - {failure.tool_name} ({failure.failure_type.value}): {failure.error_message[:50]}"
            )
        
        return "\n".join(lines)


# =====================================================================
# Integration helper
# =====================================================================

def create_balanced_circuit_breaker() -> FailFastRuleEngine:
    """
    Create a balanced fail-fast circuit breaker.
    
    Settings:
    - Max 3 consecutive failures
    - Max 10 total failures
    - Max 5 repair attempts
    - 5-minute failure window
    """
    return FailFastRuleEngine(
        max_consecutive_failures=3,
        max_total_failures=10,
        max_repair_attempts=5,
        failure_window_minutes=5
    )


def create_strict_circuit_breaker() -> FailFastRuleEngine:
    """Create strict circuit breaker (1 failure stops everything)."""
    return FailFastRuleEngine(
        max_consecutive_failures=1,
        max_total_failures=3,
        max_repair_attempts=2,
        failure_window_minutes=10
    )


def create_soft_circuit_breaker() -> FailFastRuleEngine:
    """Create lenient circuit breaker (many failures allowed)."""
    return FailFastRuleEngine(
        max_consecutive_failures=10,
        max_total_failures=30,
        max_repair_attempts=10,
        failure_window_minutes=3
    )

__INTERNAL__ = True



