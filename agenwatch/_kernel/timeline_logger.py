from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import time
import json
from datetime import datetime
import asyncio
from agenwatch._kernel.observability import EventType


def _safe_print(*args, **kwargs):
    """Wrapper for print() that silently fails if stdout is closed"""
    try:
        print(*args, **kwargs)
    except ValueError:
        # stdout is closed, silently skip
        pass


class EventType(Enum):
    """Event types in execution timeline."""
    
    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    
    # Agent iteration
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"
    
    # LLM operations
    LLM_CALL = "llm_call"
    MEMORY_RECALL = "memory_recall"
    
    # Tool lifecycle
    TOOL_CALL = "tool_call"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_RETRY = "tool_retry"
    
    # Dependencies
    DEPENDENCY_WAIT = "dependency_wait"
    
    # Parallel execution
    PARALLEL_BATCH_START = "parallel_batch_start"
    PARALLEL_BATCH_END = "parallel_batch_end"
    
    # Agent operations
    AGENT_THINK = "agent_think"
    AGENT_RESPONSE = "agent_response"
    ERROR = "error"


# Define the class with the name that other modules expect
class ExecutionTimelineLogger:
    """Logs execution timeline events."""
    
    def __init__(self, user_id: str = "system", sink: Optional[Callable[[str], None]] = None):
        """
        Initialize the timeline logger.
        
        Args:
            user_id: User identifier for this logger
            sink: Optional callable to receive log lines instead of printing.
                  If provided, log lines are sent to sink(line) instead of print().
                  Default: None (uses print())
        """
        self.user_id = user_id
        self.session_id = None
        self.events = []  # Store events for tests
        self.parallel_batches = {}
        self.sink = sink  # Optional sink for log output
    
    def parallel_batch_start(self, tools):
        """
        Start a parallel execution batch.
        
        Args:
            tools: List of tool names to execute in parallel
            
        Returns:
            batch_id: Unique identifier for this batch
        """
        batch_id = f"batch_{len(self.parallel_batches) + 1}"
        self.parallel_batches[batch_id] = {
            "tools": tools,
            "started_at": datetime.utcnow().isoformat(),
            "results": None,
            "ended_at": None
        }
        return batch_id

    def parallel_batch_end(self, batch_id, results):
        """
        End a parallel execution batch.
        
        Args:
            batch_id: ID returned by parallel_batch_start
            results: List of tool execution results
        """
        if batch_id in self.parallel_batches:
            self.parallel_batches[batch_id]["results"] = results
            self.parallel_batches[batch_id]["ended_at"] = datetime.utcnow().isoformat()

    def to_json(self):
        """Convert timeline to JSON format."""
        return {
            "events": [str(event) for event in self.events] if hasattr(self, 'events') else []
        }

    def get_summary(self):
        """Get timeline summary."""
        class TimelineSummary:
            def to_dict(self):
                return {
                    "event_count": len(self.events) if hasattr(self, 'events') else 0,
                    "session_id": self.session_id if hasattr(self, 'session_id') else None
                }
        
        summary = TimelineSummary()
        summary.events = self.events if hasattr(self, 'events') else []
        summary.session_id = self.session_id if hasattr(self, 'session_id') else None
        return summary

    def _log_event(self, event_type: EventType, metadata: dict = None):
        """Internal method to log an event."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Build log line
        log_line = f"[{timestamp}] [{event_type.value}] [Session: {self.session_id or 'unknown'}]"
        
        # Write to sink or print
        if self.sink:
            self.sink(log_line)
        else:
            _safe_print(log_line)
        
        # Print metadata if exists
        if metadata:
            for key, value in metadata.items():
                meta_line = f"   +- {key}: {value}"
                if self.sink:
                    self.sink(meta_line)
                else:
                    _safe_print(meta_line)
        
        # Store event for tests
        if hasattr(self, 'events'):
            self.events.append(type('Event', (), {
                'event_type': event_type,
                'metadata': metadata or {},
                'timestamp': timestamp
            })())
    
    def session_start(self, session_id: str, metadata: dict = None):
        """Log the start of a session."""
        self.session_id = session_id
        self._log_event(
            event_type=EventType.SESSION_START,
            metadata=metadata
        )
    
    def session_end(self, metadata: dict = None):
        """Log the end of a session."""
        self._log_event(
            event_type=EventType.SESSION_END,
            metadata=metadata
        )
    
    def tool_start(self, tool_name: str, metadata: dict = None):
        """Log the start of a tool execution."""
        self._log_event(
            event_type=EventType.TOOL_START,
            metadata={
                "tool": tool_name,
                **(metadata or {})
            }
        )
    
    def tool_end(self, tool_name: str, result: Any, metadata: dict = None):
        """Log the end of a tool execution."""
        self._log_event(
            event_type=EventType.TOOL_END,
            metadata={
                "tool": tool_name,
                "result": result,
                **(metadata or {})
            }
        )


# Create a global instance with a different name to avoid conflicts
timeline_logger_instance = ExecutionTimelineLogger("system")

# For backward compatibility, also export as timeline_logger
timeline_logger = timeline_logger_instance

# Expose for external use
__all__ = ["timeline_logger", "ExecutionTimelineLogger", "EventType"]

__INTERNAL__ = True



