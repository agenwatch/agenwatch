"""
AgenWatch Observability System
================================
Complete logging, tracing, and metrics for production agents.
"""

import time
import json
from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


# ============================================================================
# EVENT TYPES
# ============================================================================
class EventType(str, Enum):
    """Standard event types for agent operations."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"
    
    MEMORY_RECALL = "memory_recall"
    MEMORY_STORE = "memory_store"
    MEMORY_EXTRACT = "memory_extract"
    
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    
    TOOL_CALL = "tool_call"
    TOOL_SUCCESS = "tool_success"
    TOOL_ERROR = "tool_error"
    TOOL_REPAIR = "tool_repair"
    
    VALIDATION_PASS = "validation_pass"
    VALIDATION_FAIL = "validation_fail"
    
    AGENT_ERROR = "agent_error"
    AGENT_DONE = "agent_done"
    
    CONTEXT_TRIM = "context_trim" 


# ============================================================================
# LOGGER
# ============================================================================
class AgentLogger:
    """
    Lightweight structured logger for AgenWatch OS.
    
    Features:
    - Pretty console output
    - Structured JSON logs
    - Event categorization
    - Metadata attachment
    
    Example:
        >>> logger = AgentLogger(pretty=True)
        >>> logger.log("tool_call", session_id="abc", tool="search", args={"q": "test"})
    """
    
    def __init__(
        self,
        pretty: bool = True,
        json_logs: bool = False,
        log_file: Optional[str] = None
    ):
        """
        Initialize logger.
        
        Args:
            pretty: Enable pretty console output
            json_logs: Enable JSON structured logging
            log_file: Optional file path for persistent logs
        """
        self.pretty = pretty
        self.json_logs = json_logs
        self.log_file = log_file
        self._log_buffer: List[Dict[str, Any]] = []
    
    def _timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def log(
        self,
        event: str,
        session_id: Optional[str] = None,
        level: str = "INFO",
        **data
    ) -> Dict[str, Any]:
        """
        Log an event with metadata.
        
        Args:
            event: Event type or name
            session_id: Optional session identifier
            level: Log level (INFO, WARN, ERROR, DEBUG)
            **data: Additional event data
        
        Returns:
            The log entry dictionary
        """
        log_entry = {
            "timestamp": self._timestamp(),
            "event": event,
            "level": level,
            "session_id": session_id,
            "data": data
        }
        
        # Store in buffer
        self._log_buffer.append(log_entry)
        
        # Pretty console output
        if self.pretty:
            self._print_pretty(log_entry)
        
        # JSON output
        if self.json_logs:
            print(json.dumps(log_entry))
        
        # File output
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        return log_entry
    
    def _print_pretty(self, entry: Dict[str, Any]):
        """Pretty print log entry."""
        level = entry.get("level", "INFO")
        event = entry.get("event", "")
        session = entry.get("session_id", "")

        colors = {
            "INFO": "\033[36m",
            "WARN": "\033[33m",
            "ERROR": "\033[31m",
            "DEBUG": "\033[90m",
            "SUCCESS": "\033[32m"
        }
        reset = "\033[0m"

        color = colors.get(level, "")
        prefix = f"{color}[{entry['timestamp']}] [{event}]"
        if session:
            prefix += f" [Session: {session[:8]}]"
        prefix += reset

        try:
            print(prefix)

            data = entry.get("data", {})
            if data:
                for key, val in data.items():
                    if isinstance(val, (dict, list)):
                        val_str = json.dumps(val, indent=2)
                    else:
                        val_str = str(val)

                    if len(val_str) > 100:
                        val_str = val_str[:97] + "..."

                    print(f"   └─ {key}: {val_str}")

        except ValueError:
            # stdout closed during concurrent shutdown — safe to ignore
            return

    
    def get_logs(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logs with optional filtering.
        
        Args:
            session_id: Filter by session
            event_type: Filter by event type
        
        Returns:
            List of matching log entries
        """
        logs = self._log_buffer
        
        if session_id:
            logs = [l for l in logs if l.get("session_id") == session_id]
        
        if event_type:
            logs = [l for l in logs if l.get("event") == event_type]
        
        return logs
    
    def clear_logs(self):
        """Clear log buffer."""
        self._log_buffer = []


# ============================================================================
# SESSION TRACER
# ============================================================================
@dataclass
class SessionMetrics:
    """Metrics for a single agent session."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    
    iterations: int = 0
    tool_calls: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    tool_repairs: int = 0
    
    memory_recalls: int = 0
    memory_stores: int = 0
    
    llm_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    
    errors: int = 0
    
    def duration(self) -> float:
        """Get session duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def tool_success_rate(self) -> float:
        """Calculate tool success rate."""
        total = self.tool_successes + self.tool_failures
        if total == 0:
            return 0.0
        return self.tool_successes / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class SessionTracer:
    """
    Track and analyze agent session metrics.
    
    Features:
    - Real-time metrics tracking
    - Session summaries
    - Performance analysis
    - Cost tracking
    
    Example:
        >>> tracer = SessionTracer()
        >>> tracer.start_session("abc123")
        >>> tracer.record_tool_call("abc123", success=True)
        >>> metrics = tracer.get_metrics("abc123")
    """
    
    def __init__(self):
        """Initialize session tracer."""
        self._sessions: Dict[str, SessionMetrics] = {}
    
    def start_session(self, session_id: str) -> SessionMetrics:
        """
        Start tracking a new session.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            SessionMetrics object
        """
        metrics = SessionMetrics(
            session_id=session_id,
            start_time=time.time()
        )
        self._sessions[session_id] = metrics
        return metrics
    
    def end_session(self, session_id: str):
        """Mark session as ended."""
        if session_id in self._sessions:
            self._sessions[session_id].end_time = time.time()
    
    def record_iteration(self, session_id: str):
        """Record an agent iteration."""
        if session_id in self._sessions:
            self._sessions[session_id].iterations += 1
    
    def record_tool_call(
        self,
        session_id: str,
        success: bool = True,
        repaired: bool = False
    ):
        """Record a tool call."""
        if session_id in self._sessions:
            metrics = self._sessions[session_id]
            metrics.tool_calls += 1
            if success:
                metrics.tool_successes += 1
            else:
                metrics.tool_failures += 1
            if repaired:
                metrics.tool_repairs += 1
    
    def record_memory_operation(
        self,
        session_id: str,
        operation: str  # "recall" or "store"
    ):
        """Record memory operation."""
        if session_id in self._sessions:
            metrics = self._sessions[session_id]
            if operation == "recall":
                metrics.memory_recalls += 1
            elif operation == "store":
                metrics.memory_stores += 1
    
    def record_llm_call(
        self,
        session_id: str,
        tokens: int = 0,
        cost: float = 0.0
    ):
        """Record LLM call with token usage."""
        if session_id in self._sessions:
            metrics = self._sessions[session_id]
            metrics.llm_calls += 1
            metrics.total_tokens += tokens
            metrics.total_cost += cost
    
    def record_error(self, session_id: str):
        """Record an error."""
        if session_id in self._sessions:
            self._sessions[session_id].errors += 1
    
    def get_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """Get metrics for a session."""
        return self._sessions.get(session_id)
    
    def print_summary(self, session_id: str):
        """Print a formatted summary of session metrics."""
        metrics = self.get_metrics(session_id)
        if not metrics:
            print(f"No metrics found for session: {session_id}")
            return
        
        print("\n" + "=" * 60)
        print(f"SESSION SUMMARY: {session_id[:12]}...")
        print("=" * 60)
        print(f"Duration: {metrics.duration():.2f}s")
        print(f"Iterations: {metrics.iterations}")
        print(f"\n📊 Tool Calls:")
        print(f"   Total: {metrics.tool_calls}")
        print(f"   Success: {metrics.tool_successes}")
        print(f"   Failed: {metrics.tool_failures}")
        print(f"   Repaired: {metrics.tool_repairs}")
        print(f"   Success Rate: {metrics.tool_success_rate():.1%}")
        print(f"\n🧠 Memory:")
        print(f"   Recalls: {metrics.memory_recalls}")
        print(f"   Stores: {metrics.memory_stores}")
        print(f"\n💬 LLM:")
        print(f"   Calls: {metrics.llm_calls}")
        print(f"   Tokens: {metrics.total_tokens}")
        print(f"   Cost: ${metrics.total_cost:.4f}")
        print(f"\n❌ Errors: {metrics.errors}")
        print("=" * 60 + "\n")
    
    def get_all_sessions(self) -> List[SessionMetrics]:
        """Get all tracked sessions."""
        return list(self._sessions.values())


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Initialize
    logger = AgentLogger(pretty=True, json_logs=False)
    tracer = SessionTracer()
    
    # Start session
    session_id = "test_session_123"
    tracer.start_session(session_id)
    
    logger.log(
        EventType.SESSION_START,
        session_id=session_id,
        user_id="alice"
    )
    
    # Simulate agent operations
    for i in range(3):
        tracer.record_iteration(session_id)
        
        logger.log(
            EventType.ITERATION_START,
            session_id=session_id,
            iteration=i+1
        )
        
        # Memory recall
        tracer.record_memory_operation(session_id, "recall")
        logger.log(
            EventType.MEMORY_RECALL,
            session_id=session_id,
            memories_found=3,
            latency_ms=142
        )
        
        # LLM call
        tracer.record_llm_call(session_id, tokens=450, cost=0.0009)
        logger.log(
            EventType.LLM_CALL,
            session_id=session_id,
            model="llama-3.3-70b",
            tokens=450
        )
        
        # Tool call
        if i == 0:
            # Failed tool call
            tracer.record_tool_call(session_id, success=False)
            logger.log(
                EventType.TOOL_ERROR,
                session_id=session_id,
                level="ERROR",
                tool="calculator",
                error="Invalid arguments: 'expression' is a required property"
            )
            
            # Repair
            tracer.record_tool_call(session_id, success=True, repaired=True)
            logger.log(
                EventType.TOOL_REPAIR,
                session_id=session_id,
                tool="calculator",
                fixed="expresion → expression"
            )
        else:
            # Successful tool call
            tracer.record_tool_call(session_id, success=True)
            logger.log(
                EventType.TOOL_SUCCESS,
                session_id=session_id,
                level="SUCCESS",
                tool="calculator",
                result=5,
                latency_ms=23
            )
    
    # End session
    tracer.end_session(session_id)
    logger.log(
        EventType.SESSION_END,
        session_id=session_id,
        level="SUCCESS"
    )
    
    # Print summary
    tracer.print_summary(session_id)

__INTERNAL__ = True


