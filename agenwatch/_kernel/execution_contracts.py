"""
AgenWatch Execution Contracts
===============================

This module defines the LAWS of tool execution in AgenWatch OS.
These contracts are the "Constitution" that governs how tools are executed,
how failures are handled, and what guarantees the system provides.

Layer: Execution Governance (Layer 2)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, List, Dict
import time


# ============================================================
# FAILURE TAXONOMY (The 4 Universal Laws)
# ============================================================

class FailureType(Enum):
    """
    Universal classification of tool failures.
    Every failure MUST map to one of these categories.
    """
    RECOVERABLE = "recoverable"      # Retry with backoff
    SCHEMA_ERROR = "schema_error"    # Trigger repair, then retry
    CIRCUIT_OPEN = "circuit_open"    # Tool blacklisted, skip
    POLICY_VIOLATION = "policy_violation"  # FailFast blocked, immediate stop
    FATAL = "fatal"                  # Terminate session immediately
    BUDGET_EXCEEDED = "budget_exceeded" 

class ExecutionAction(Enum):
    """
    Actions the governance layer can take in response to failures.
    """
    RETRY = "retry"                  # Try again with backoff
    REPAIR_AND_RETRY = "repair_retry" # Fix args, then retry
    SKIP = "skip"                    # Don't execute, return error
    TERMINATE = "terminate"          # Kill session immediately
    CONTINUE = "continue"            # Success, move forward


# ============================================================
# EXECUTION RESULT (The Passport)
# ============================================================

@dataclass
class ExecutionResult:
    """
    The unified return type for ALL tool executions.
    
    This is the "Passport" that Layer 2 hands back to Layer 1.
    It contains everything the Agent needs to decide what to do next.
    
    Guarantees:
    - Always present, even on failure
    - Contains complete timeline of execution
    - Classifies failure type (never "unknown error")
    - Provides actionable metadata
    """
    
    # Core result
    success: bool
    output: Any = None
    
    # Failure classification
    error_type: Optional[FailureType] = None
    error_message: Optional[str] = None
    
    # Execution metadata
    tool_name: str = ""
    attempts: int = 1
    duration_ms: float = 0.0
    
    # Governance metadata
    was_repaired: bool = False
    repair_confidence: float = 0.0
    circuit_tripped: bool = False
    
    # Observability
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    
    # Original arguments (for debugging)
    original_args: Dict[str, Any] = field(default_factory=dict)
    final_args: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/debugging"""
        return {
            "success": self.success,
            "output": self.output,
            "error_type": self.error_type.value if self.error_type else None,
            "error_message": self.error_message,
            "tool_name": self.tool_name,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "was_repaired": self.was_repaired,
            "repair_confidence": self.repair_confidence,
            "circuit_tripped": self.circuit_tripped,
            "timeline": self.timeline,
        }
    
    @classmethod
    def success_result(cls, tool_name: str, output: Any, duration_ms: float, 
                       attempts: int = 1, timeline: List = None) -> 'ExecutionResult':
        """Factory for successful executions"""
        return cls(
            success=True,
            output=output,
            tool_name=tool_name,
            attempts=attempts,
            duration_ms=duration_ms,
            timeline=timeline or []
        )
    
    @classmethod
    def failure_result(cls, tool_name: str, error_type: FailureType, 
                       error_message: str, attempts: int = 1, 
                       duration_ms: float = 0.0, timeline: List = None) -> 'ExecutionResult':
        """Factory for failed executions"""
        return cls(
            success=False,
            error_type=error_type,
            error_message=error_message,
            tool_name=tool_name,
            attempts=attempts,
            duration_ms=duration_ms,
            timeline=timeline or []
        )


# ============================================================
# EXECUTION POLICY (Governance Rules)
# ============================================================

@dataclass
class ExecutionPolicy:
    """
    Configurable rules for tool execution governance.
    
    This defines WHAT the system is allowed to do, not HOW it does it.
    Agent configures policy, ExecutionManager enforces it.
    """
    
    # Retry configuration
    max_retries: int = 3
    retry_backoff_base: float = 0.5  # seconds
    retry_backoff_multiplier: float = 2.0
    retry_backoff_max: float = 10.0
    
    # Repair configuration
    enable_auto_repair: bool = True
    min_repair_confidence: float = 0.7
    
    # Circuit breaker configuration
    enable_circuit_breaker: bool = True
    circuit_failure_threshold: int = 3
    circuit_reset_timeout: float = 60.0
    
    # Sandbox configuration
    enable_sandbox: bool = True
    sandbox_timeout_seconds: float = 30.0
    sandbox_max_memory_mb: int = 512
    
    # Observability
    enable_timeline_logging: bool = True
    enable_fingerprinting: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy for logging"""
        return {
            "max_retries": self.max_retries,
            "retry_backoff_base": self.retry_backoff_base,
            "enable_auto_repair": self.enable_auto_repair,
            "min_repair_confidence": self.min_repair_confidence,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "circuit_failure_threshold": self.circuit_failure_threshold,
            "enable_sandbox": self.enable_sandbox,
            "sandbox_timeout_seconds": self.sandbox_timeout_seconds,
        }


# ============================================================
# EXECUTION CONTEXT (Runtime State)
# ============================================================

@dataclass
class ExecutionContext:
    """
    Runtime context passed through the execution pipeline.
    
    This is the "state bag" that flows through:
    Agent → ExecutionManager → RepairSystem → RuntimeEngine
    
    It accumulates metadata as it moves through the layers.
    """
    
    # Identity
    session_id: str
    user_id: str
    tool_name: str
    
    # Arguments (mutable through pipeline)
    original_args: Dict[str, Any]
    current_args: Dict[str, Any]
    
    execution_id: str
    is_replay: bool = False
    
    # Execution state
    attempt_number: int = 1
    start_time: float = field(default_factory=time.time)
    
    # Governance state
    circuit_open: bool = False
    repair_attempted: bool = False
    repair_confidence: float = 0.0
    
    # Timeline (accumulated events)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_event(self, event_type: str, **metadata):
        """Add event to timeline"""
        self.events.append({
            "type": event_type,
            "timestamp": time.time(),
            "attempt": self.attempt_number,
            **metadata
        })
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        return (time.time() - self.start_time) * 1000


# ============================================================
# FAILURE CLASSIFICATION RULES (The Decision Tree)
# ============================================================

class FailureClassifier:
    """
    Classifies exceptions into FailureType categories.
    
    This is the "Triage Nurse" that looks at an error and decides:
    - Is this recoverable?
    - Does it need repair?
    - Should we circuit break?
    - Is this fatal?
    """
    
    @staticmethod
    def classify(error: Exception, context: ExecutionContext) -> FailureType:
        """
        Map exception to FailureType.
        
        This is the implementation of the "4 Universal Laws" table.
        """
        from agenwatch._kernel.errors import RecoverableToolError
        
        error_msg = str(error).lower()
        
        # Law 1: Explicit recoverable errors
        if isinstance(error, RecoverableToolError):
            return FailureType.RECOVERABLE
        
        # Law 2: Schema/validation errors
        if any(keyword in error_msg for keyword in [
            "schema", "validation", "missing required", 
            "invalid argument", "type mismatch"
        ]):
            return FailureType.SCHEMA_ERROR
        
        # Law 3: Circuit breaker (based on context)
        if context.circuit_open:
            return FailureType.CIRCUIT_OPEN
        
        # Law 4: Fatal errors (no retry)
        if any(keyword in error_msg for keyword in [
            "permission denied", "authentication failed",
            "not authorized", "access denied"
        ]):
            return FailureType.FATAL
        
        # Default: treat as recoverable (conservative)
        return FailureType.RECOVERABLE
    
    @staticmethod
    def get_action(failure_type: FailureType, attempts: int, max_retries: int) -> ExecutionAction:
        """
        Decide what action to take based on failure type.
        
        This is the "Treatment Protocol" that maps diagnosis → action.
        """
        if failure_type == FailureType.RECOVERABLE:
            if attempts < max_retries:
                return ExecutionAction.RETRY
            else:
                return ExecutionAction.SKIP
        
        elif failure_type == FailureType.SCHEMA_ERROR:
            if attempts < max_retries:
                return ExecutionAction.REPAIR_AND_RETRY
            else:
                return ExecutionAction.SKIP
        
        elif failure_type == FailureType.CIRCUIT_OPEN:
            return ExecutionAction.SKIP
        
        elif failure_type == FailureType.FATAL:
            return ExecutionAction.TERMINATE
        
        return ExecutionAction.SKIP


# ============================================================
# EXECUTION GUARANTEES (The Contract)
# ============================================================

class ExecutionGuarantees:
    """
    Documents what the Execution Layer promises to deliver.
    
    This is not code - it's a CONTRACT that tests verify.
    """
    
    GUARANTEES = {
        "result_always_returned": (
            "ExecutionResult is always returned, even on catastrophic failure"
        ),
        "failure_always_classified": (
            "Every failure maps to a FailureType, never 'unknown error'"
        ),
        "timeline_always_present": (
            "Timeline contains all execution events, in order"
        ),
        "idempotency": (
            "Same tool + same args = same result (with fingerprinting)"
        ),
        "retry_transparency": (
            "Number of attempts is always logged in result"
        ),
        "governance_enforcement": (
            "Circuit breaker, retries, and repair follow policy exactly"
        ),
    }
    
    NON_GUARANTEES = {
        "tool_success": (
            "Tools may fail even after all retries"
        ),
        "schema_perfection": (
            "Repair may not fix all schema issues"
        ),
        "deterministic_timing": (
            "Execution time varies based on retries and backoff"
        ),
    }

__INTERNAL__ = True



