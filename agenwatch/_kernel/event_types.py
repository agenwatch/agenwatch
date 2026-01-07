from enum import Enum


class EventType(Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"

    MEMORY_RECALL = "memory_recall"
    LLM_CALL = "llm_call"

    TOOL_CALL = "tool_call"
    TOOL_START = "tool_start"
    TOOL_SUCCESS = "tool_success"
    TOOL_ERROR = "tool_error"
    TOOL_END = "tool_end"

    PARALLEL_BATCH_START = "parallel_batch_start"
    PARALLEL_BATCH_END = "parallel_batch_end"

    DAG_PLAN_DETECTED = "dag_plan_detected"

    AGENT_DONE = "agent_done"
    AGENT_ERROR = "agent_error"
    
    CONTEXT_TRIM = "context_trim"

__INTERNAL__ = True
