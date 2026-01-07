import logging
import asyncio
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional
import json
import re
from agenwatch._kernel.tools import ToolRegistry
from agenwatch._kernel.agent_result import AgentResult
from agenwatch._kernel.observability import AgentLogger, SessionTracer
from agenwatch._kernel.observability import EventType
from agenwatch._kernel.repair_memory import RepairMemoryManager
from agenwatch._kernel.fail_fast import FailFastRuleEngine
from agenwatch._kernel.timeline_logger import ExecutionTimelineLogger
from agenwatch._kernel.sandbox.tool_sandbox import ToolSandbox
from agenwatch._kernel.safety.circuit_breaker import CircuitBreaker
from agenwatch._kernel.tools.function_tool import FunctionTool
from agenwatch._kernel.execution_manager import ToolExecutionManager, FailureType
from agenwatch._kernel.loop_breaker import LoopBreaker, LoopBreakerConfig
from agenwatch._kernel.context_manager import ContextManager, ContextConfig, Message, MessageRole
from agenwatch._kernel.observability import EventType
from agenwatch._kernel.execution_contracts import ExecutionPolicy, ExecutionResult, FailureType
from pathlib import Path
from agenwatch._kernel.execution_manager import ExecutionMode
from agenwatch._kernel.replay import ExecutionLog, ReplayEngine
from agenwatch._kernel.safety.budget_manager import BudgetExceededError

logger = logging.getLogger("agenwatch.agent")


def _safe_print(*args, **kwargs):
    """Wrapper for print() that silently fails if stdout is closed"""
    try:
        print(*args, **kwargs)
    except ValueError:
        # stdout is closed, silently skip   
        pass


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    DONE = "done"
    ERROR = "error"
    REPLAYING = "replaying"


@dataclass
class AgentMessage:
    role: str
    content: str
    tool_name: Optional[str] = None
    tool_result: Optional[Any] = None
    timestamp: float = time.time()


@dataclass
class ToolCall:
    """Pure container - no logic"""
    tool_name: str
    raw_args: dict
    repaired_args: Optional[dict] = None
    schema_repair: Optional[dict] = None
    schema_confidence: Optional[float] = None
    final_args: Optional[dict] = None

    def __post_init__(self):
        self.final_args = self.final_args or self.repaired_args or self.raw_args

class ReplayLLMProvider:
    model_name = "replay"
    max_context_tokens = 999999

    async def generate(self, *args, **kwargs):
        raise RuntimeError("LLM must not be called in REPLAY mode")


class Agent:
    """Production Agent - Kernel/Orchestrator only"""

    MAX_SAME_TOOL_CALLS = 3

    def __init__(
        self,
        client=None,
        llm_provider=None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 10,
        system_prompt: Optional[str] = None,
        memory_system=None,
        user_id: str = "default",
        execution_mode: str = "normal",
        replay_log_path: Optional[Path] = None,
        session_id: Optional[str] = None,
        event_sink=None,
        budget_manager=None,
    ):
        self._budget_manager = budget_manager
        if isinstance(execution_mode, str):
            execution_mode = ExecutionMode(execution_mode)
        self.execution_mode = execution_mode
        
        # Basic setup
        self.client = client or self._create_default_client()
        if self.execution_mode == ExecutionMode.REPLAY:
            self.llm_provider = ReplayLLMProvider()
        else:
            self.llm_provider = llm_provider or self._create_default_llm()

        self.user_id = user_id
        self.session_id = session_id
        
        
        # Observability
        self.logger = AgentLogger(pretty=True, json_logs=False)
        self.tracer = SessionTracer()
        self.timeline = ExecutionTimelineLogger()
        self._event_sink = event_sink
        
        # State management (Kernel owns this)
        self.state = AgentState.IDLE
        self.conversation_history: List[AgentMessage] = []
        self.current_iteration = 0
        self.max_iterations = max_iterations
        
        
        # ---- SAFE LLM METADATA EXTRACTION ----
        model_name = getattr(self.llm_provider, "model_name", "unknown-llm")
        max_context_tokens = getattr(
            self.llm_provider,
            "max_context_tokens",
            8192  # sane default
        )

        self.context_manager = ContextManager(
            ContextConfig(
                model_name=model_name,
                max_context_tokens=max_context_tokens
            )
        )
        
        self.execution_mode = execution_mode
        self.replay_engine = None

        if self.execution_mode == ExecutionMode.REPLAY:
            if not replay_log_path:
                raise ValueError("replay_log_path is required in REPLAY mode")

            log = ExecutionLog.load(replay_log_path)
            self.replay_engine = ReplayEngine(log)

        # Tool system (delegated to ExecutionManager)
        self.tool_registry = ToolRegistry()
        self._register_tools(tools or [])
        
        # Execution Manager (Layer 2 - does all execution)
        self.execution_manager = ToolExecutionManager(
            tool_registry=self.tool_registry,
            repair_memory_manager=RepairMemoryManager(
                memory_system=memory_system,
                user_id=user_id,
                session_id=self.session_id or f"session_{int(time.time())}"
            ),
            circuit_breaker=CircuitBreaker(max_failures=3),
            fail_fast_engine=FailFastRuleEngine(),
            sandbox=ToolSandbox(max_execution_time=30, max_memory_mb=512),
            retry_manager=None,
            policy=ExecutionPolicy(
                max_retries=3,
                enable_auto_repair=True,
                enable_circuit_breaker=True,
                enable_sandbox=True,
            ),
            budget_manager=self._budget_manager,
        )
        
        
        self.execution_manager.execution_mode = self.execution_mode
        self.execution_manager.replay_engine = self.replay_engine
        
        # Loop control (Kernel owns stop decisions)
        self.loop_breaker = LoopBreaker(LoopBreakerConfig(max_same_fingerprint=3))
        
        # System prompt
        self.system_prompt = system_prompt or self._default_system_prompt()
            
        logger.info(f"[Agent] Initialized (max_iterations={max_iterations})")

    def _emit_event(self, event_type, **data):
        """Centralized event emission with optional sink mirror"""
        # Prepare session ID
        sid = data.get("session_id") or getattr(self, "session_id", None)
        
        # Existing behavior: log to local logger
        log_data = {k: v for k, v in data.items() if k != "session_id"}
        self.logger.log(event_type, session_id=sid, **log_data)
        
        # New behavior: optional mirror to external sink
        if hasattr(self, "_event_sink") and self._event_sink:
            try:
                # Mirror to external sink with two arguments
                # Ensure event_type is a string for the SDK
                et = event_type.value if hasattr(event_type, "value") else str(event_type)
                self._event_sink(et, data)
            except Exception:
                # NEVER let observability crash execution
                pass

    def set_event_sink(self, sink):
        """Allow dynamic injection of event sink"""
        self._event_sink = sink

    def _register_tools(self, tools: List[Any]):
        """Register tools safely - convert dicts to FunctionTool if needed"""
        for t in tools:
            if isinstance(t, FunctionTool):
                self.tool_registry.register_tool(t)
            elif isinstance(t, dict):
                # Handle dict tool definition
                name = t.get("name")
                fn = t.get("function") or t.get("fn")
                if name and fn:
                    self.tool_registry.register_tool(FunctionTool(name=name, fn=fn))
            else:
                # Unknown type, log warning
                logger.warning(f"[Agent] Skipping unknown tool type: {type(t)}")


    def _parse_tool_call_text(self, text: str):
        """
        Parse <function="name">{json}</function>
        """
        match = re.search(
            r'<function="(?P<name>[^"]+)">\s*(?P<args>\{.*?\})\s*</function>',
            text,
            re.DOTALL
        )

        if not match:
            raise RuntimeError(f"Invalid tool call format:\n{text}")

        tool_name = match.group("name")
        args = json.loads(match.group("args"))

        return tool_name, args

    
    def _create_default_client(self):
        """Default mock client"""
        class _DefaultClient:
            async def start_session(self, meta=None): 
                return "test-session"
            async def end_session(self, session_id): 
                pass
            async def log_event(self, *args, **kwargs): 
                pass
        return _DefaultClient()

    def _create_default_llm(self):
        """Default mock LLM"""
        class _DefaultLLMProvider:
            async def generate(self, messages, tools):
                raise RuntimeError("LLM provider not configured")
        return _DefaultLLMProvider()

    def _get_cost(self) -> float:
        """Get current budget spent (0.0 if no budget manager)"""
        if self._budget_manager is not None:
            return self._budget_manager.spent
        return 0.0


    async def run(self, task: str, session_id: Optional[str] = None):
        """MAIN EXECUTION LOOP - Production Kernel"""

        owns_session = False

        if not hasattr(self, 'session_id') or self.session_id is None:
            if session_id is None:
                self.session_id = await self.client.start_session(meta={"agent": True})
                owns_session = True
            else:
                self.session_id = session_id
                owns_session = False
        else:
            if session_id is not None:
                self.session_id = session_id
            owns_session = False

        self.timeline.session_id = self.session_id
        self.tracer.start_session(self.session_id)
        self._emit_event(EventType.SESSION_START, session_id=self.session_id, task=task)
        self.timeline.tool_start("agent", {"task": task, "session_id": self.session_id})

        # Initialize per-run replay set (MANDATORY for retries)
        self._executed_tools = set()
        try:
            self.state = AgentState.THINKING
            self.current_iteration = 0
            self.conversation_history.clear()
            self.conversation_history.append(
                AgentMessage(role="user", content=task)
            )

            skip_next_think = False
            while self.current_iteration < self.max_iterations:
                self.current_iteration += 1

                self._emit_event(
                    EventType.ITERATION_START,
                    session_id=self.session_id,
                    iteration=self.current_iteration
                )

                if not skip_next_think:
                    response = await self._think()
                skip_next_think = False

                # ===== HARD NORMALIZATION (REQUIRED) =====
                if isinstance(response, dict):
                    response = type(
                        "LLMResponse",
                        (),
                        {
                            "text": response.get("text"),
                            "tool_calls": response.get("tool_calls"),
                        },
                    )()

                # ===== NORMALIZE LLM RESPONSE (CRITICAL) =====
                tool_calls = getattr(response, "tool_calls", None)
                text = getattr(response, "text", None)

                # ===== TERMINAL COMPLETION CHECK (CANONICAL) =====
                is_terminal = (
                    text is not None and text.strip() != ""
                    and (tool_calls is None or len(tool_calls) == 0)
                )
                
                if is_terminal:
                    self.state = AgentState.DONE
                    return AgentResult(
                        success=True,
                        output=text,
                        iterations=self.current_iteration,
                        terminal_reason="NATURAL_COMPLETION",
                        cost=self._get_cost(),
                    )

                # ? Handle explicit terminal result from _think (Fix for infinite loop)
                if isinstance(response, AgentResult):
                    response.iterations = self.current_iteration
                    response.cost = self._get_cost()
                    self.state = AgentState.DONE if response.success else AgentState.ERROR
                    return response

                # ?? Check if _think() returned early due to budget exhaustion
                if isinstance(response, ExecutionResult):
                    if (
                        not response.success
                        and response.error_type == FailureType.BUDGET_EXCEEDED
                    ):
                        self.state = AgentState.ERROR
                        return AgentResult(
                            success=False,
                            error_type="BUDGET_EXCEEDED",
                            iterations=self.current_iteration,
                            terminal_reason="BUDGET_EXHAUSTED",
                            cost=self._get_cost(),
                        )

                # ??? Handle tool calls

                tool_calls = self._extract_tool_calls(response)

                if tool_calls:
                    self.state = AgentState.EXECUTING

                    self._emit_event(
                        EventType.TOOL_CALL,
                        session_id=self.session_id,
                        tools=[tc.tool_name for tc in tool_calls]
                    )

                    self.conversation_history.append(
                        AgentMessage(
                            role="assistant",
                            content=f"Calling tools: {[tc.tool_name for tc in tool_calls]}"
                        )
                    )

                    # Extract tool call FIRST
                    tc = tool_calls[0]

                    # Initialize per-run replay set (already done at start of run)

                    # Build fingerprint safely
                    args = tc.raw_args or {}
                    tool_fingerprint = (tc.tool_name, tuple(sorted(args.items())))

                    # Prevent infinite replay
                    if tool_fingerprint in self._executed_tools:
                        return AgentResult(
                            success=True,
                            output=None,
                            iterations=self.current_iteration,
                            terminal_reason="TOOL_ALREADY_EXECUTED",
                            cost=self._get_cost(),
                        )

                    self._executed_tools.add(tool_fingerprint)

                    # Now execute
                    tool_results = await self._execute_tool_calls(tool_calls)

                    # HARD STOP ON BUDGET EXHAUSTION
                    for tc, result in zip(tool_calls, tool_results):
                        if (
                            isinstance(result, dict)
                            and not result.get("success", True)
                            and result.get("error_type") == FailureType.BUDGET_EXCEEDED
                        ):
                            self.state = AgentState.ERROR
                            return AgentResult(
                                success=False,
                                error_type="BUDGET_EXCEEDED",
                                iterations=self.current_iteration,
                                terminal_reason="BUDGET_EXHAUSTED",
                                cost=self._get_cost(),
                            )

                        # Append tool result to conversation history
                        self.conversation_history.append(
                            AgentMessage(
                                role="tool",
                                content=str(result),
                                tool_name=tc.tool_name,
                                tool_result=result
                            )
                        )

                    # DO NOT call _think() again here.
                    # Tool results are already appended.
                    # Let the main loop fetch the next response.
                    self.state = AgentState.THINKING
                    skip_next_think = False  # Changed: Reset to False to think next iteration
                    continue

                # 🔔 If we get here with no tool calls and no terminal text, continue thinking
                self.state = AgentState.THINKING
                skip_next_think = False

        # 💰 BUDGET EXHAUSTION — TERMINAL (catch before general Exception)
        except BudgetExceededError as e:
            self.state = AgentState.ERROR
            self._emit_event(
                EventType.AGENT_ERROR,
                session_id=self.session_id,
                error=f"Budget exceeded: {str(e)}"
            )
            return AgentResult.failure(
                error_type="budget_exceeded",
                terminal_reason=str(e),
                iterations=self.current_iteration,
                cost=self._get_cost(),
            )

        except Exception as e:
            self.state = AgentState.ERROR
            from agenwatch._kernel.utils.safe_traceback import safe_print_exc
            safe_print_exc()
            self._emit_event(
                EventType.AGENT_ERROR,
                session_id=self.session_id,
                error=str(e)
            )
            return AgentResult(
                success=False,
                error_type="AGENT_ERROR",
                iterations=self.current_iteration,
                terminal_reason=f"ERROR: {str(e)}",
                cost=self._get_cost(),
            )

        finally:
            await self._cleanup(owns_session)

        # ===== ABSOLUTE SAFETY RETURN (NON-NEGOTIABLE) =====
        return AgentResult(
            success=False,
            output=None,
            iterations=self.current_iteration,
            error_type="MAX_ITERATIONS",
            terminal_reason="MAX_ITERATIONS_REACHED",
            cost=self._get_cost(),
        )


    async def _think(self) -> Any:
        """Call LLM - context-governed, deterministic"""
        
        if self.execution_mode == ExecutionMode.REPLAY:
            return self.replay_engine.get_llm_response()

        
        # 1. Build ContextManager messages
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=self.system_prompt
            )
        ]

        for m in self.conversation_history:
            if m.role == "tool":
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=f"Tool result ({m.tool_name}): {m.content}"
                    )
                )
            else:
                messages.append(
                    Message(
                        role=MessageRole(m.role),
                        content=m.content
                    )
                )

        # 2. PREPARE CONTEXT (AUTHORITATIVE)
        trim_result = self.context_manager.prepare_context(messages)

        # 3. HARD FAIL if still unsafe
        if trim_result.tokens_after > self.context_manager.config.max_context_tokens:
            raise RuntimeError(
                f"Context overflow AFTER trim: "
                f"{trim_result.tokens_after} > {self.context_manager.config.max_context_tokens}"
            )

        # 4. OBSERVABILITY (MANDATORY)
        self._emit_event(
            EventType.CONTEXT_TRIM.value,
            session_id=self.session_id,
            tokens_before=trim_result.tokens_before,
            tokens_after=trim_result.tokens_after,
            dropped=trim_result.messages_dropped,
            reason=trim_result.trim_reason
        )

        # 5. Convert to LLM format
        llm_messages = [
            m.to_llm_dict() if hasattr(m, "to_llm_dict") else m.to_dict()
            for m in trim_result.messages
        ]

        # 6. CALL LLM (ONLY HERE)
        if self.replay_engine:
            response = await self.replay_engine.call_llm(
                messages=llm_messages,
                tools=self.tool_registry.list_tool_schemas()
            )
        else:
            coro = self.llm_provider.generate(
                messages=llm_messages,
                tools=self.tool_registry.list_tool_schemas()
            )
            if not asyncio.iscoroutine(coro):
                raise TypeError(f"LLMProvider.generate must be a coroutine, got {type(coro)}")
            response = await coro
            
        # 7. Log LLM call
        self._emit_event(
            EventType.LLM_CALL,
            session_id=self.session_id,
            preview=self._extract_response_text(response)[:200]
        )

        # 8. HANDLE LLM RESPONSE (AUTHORITATIVE STOP CONTRACT)
        if isinstance(response, dict):
            text = (response.get("text") or "").strip()
            tool_calls = response.get("tool_calls")
            final = response.get("final")
        else:
            text = (getattr(response, "text", "") or "").strip()
            tool_calls = getattr(response, "tool_calls", None)
            final = getattr(response, "final", None)

        # 1. Explicit final -> HARD STOP
        if final:
            self._emit_event(
                EventType.AGENT_DONE,
                session_id=self.session_id,
                success=True,
                reason="final_answer"
            )
            return AgentResult.success(output=final)

        # 2. Extract tool calls (if not already done)
        if tool_calls is None:
            tool_calls = self._extract_tool_calls(response)

        # 0. <final> in text always terminates
        if text and "<final>" in text:
            final_text = text.replace("<final>", "").replace("</final>", "").strip()
            self._emit_event(
                EventType.AGENT_DONE,
                session_id=self.session_id,
                success=True,
                reason="LLM_FINAL"
            )
            return AgentResult.success(
                output=final_text,
                terminal_reason="LLM_FINAL"
            )

        # 3. Tool calls ALWAYS win
        if tool_calls:
            # FIXED: Return the response object, not tool execution result
            # Let run() handle the tool execution
            return response

        # 4. Explicit final (redundant but safe)
        if final:
            self._emit_event(
                EventType.AGENT_DONE,
                session_id=self.session_id,
                success=True,
                reason="FINAL_ANSWER"
            )
            return AgentResult.success(
                output=final,
                terminal_reason="FINAL_ANSWER"
            )

        # 5. Natural language completion ONLY if no tools are registered
        if text and not self.tool_registry.list_tool_names():
            self._emit_event(
                EventType.AGENT_DONE,
                session_id=self.session_id,
                success=True,
                reason="NATURAL_COMPLETION"
            )
            return AgentResult.success(
                output=text,
                terminal_reason="NATURAL_COMPLETION"
            )
        
        # 6. CRITICAL FIX: If we have text but no tool calls and tools ARE registered
        # This is the normal completion case that was missing
        if text and tool_calls is None:
            self._emit_event(
                EventType.AGENT_DONE,
                session_id=self.session_id,
                success=True,
                reason="NATURAL_COMPLETION_WITH_TOOLS"
            )
            return AgentResult.success(
                output=text,
                terminal_reason="NATURAL_COMPLETION_WITH_TOOLS"
            )

        # 7. Silence is failure
        if not text:
            self._emit_event(
                EventType.AGENT_DONE,
                session_id=self.session_id,
                success=False,
                reason="EMPTY_LLM_OUTPUT"
            )
            return AgentResult.failure(
                error_type="EMPTY_LLM_OUTPUT",
                terminal_reason="EMPTY_LLM_OUTPUT"
            )
        
        # 8. Default: return the response for run() to handle
        return response


    def _extract_tool_calls(self, response) -> List[ToolCall]:
        """Parse intent only - pure function, no side effects"""
        tool_calls = []

        # Handle dict response (MockLLMProvider)
        if isinstance(response, dict):
            raw_calls = response.get("tool_calls") or []   # NULL GUARD
            if isinstance(raw_calls, list):
                for tc in raw_calls:
                    if isinstance(tc, dict):
                        tool_name = tc.get("name") or tc.get("tool_name")
                        if tool_name:
                            tool_calls.append(
                                ToolCall(
                                    tool_name=tool_name,
                                    raw_args=tc.get("arguments") or tc.get("args") or {}
                                )
                            )

        # Handle SDK-style response objects
        elif hasattr(response, "tool_calls"):
            raw_calls = response.tool_calls or []          # NULL GUARD
            for tc in raw_calls:
                if isinstance(tc, dict):
                    tool_name = tc.get("name") or tc.get("tool_name")
                    if tool_name:
                        tool_calls.append(
                            ToolCall(
                                tool_name=tool_name,
                                raw_args=tc.get("arguments") or tc.get("args") or {}
                            )
                        )
                else:
                    tool_name = getattr(tc, "name", None) or getattr(tc, "tool_name", None)
                    if tool_name:
                        tool_calls.append(
                            ToolCall(
                                tool_name=tool_name,
                                raw_args=getattr(tc, "arguments", None)
                                        or getattr(tc, "args", {})
                                        or {}
                            )
                        )

        # Also parse <function="tool">...</function> markup in text field
        text = None
        if isinstance(response, dict):
            text = response.get("text", "")
        elif hasattr(response, "text"):
            text = getattr(response, "text", "")
        if text:
            import re, json
            pattern = r'<function="([^"]+)">\s*(\{.*?\})\s*</function>'
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                tool_name = match.group(1)
                try:
                    args = json.loads(match.group(2))
                except Exception:
                    args = {}
                tool_calls.append(ToolCall(tool_name=tool_name, raw_args=args))

        return tool_calls

    async def _execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
        """Delegate execution to ExecutionManager (NEVER execute directly)"""

        _safe_print(f"?? [_execute_tool_calls] Called with {len(tool_calls)} tool calls")
        _safe_print(f"?? [_execute_tool_calls] Tool names: {[tc.tool_name for tc in tool_calls]}")

        results: List[Dict[str, Any]] = []

        if not tool_calls:
            return results

        # Kernel contract: one tool call at a time
        tc = tool_calls[0]

        _safe_print(f"?? [_execute_tool_calls] Processing tool: {tc.tool_name}")
        _safe_print(f"?? [_execute_tool_calls] Tool args: {tc.raw_args}")

        tool = self.tool_registry.get_tool(tc.tool_name)
        _safe_print(f"?? [_execute_tool_calls] Tool found in registry: {tool is not None}")

        exec_result = await self.execution_manager.execute_with_governance(
            tool_call=tc,
            session_id=self.session_id,
            user_id=self.user_id,
        )

        _safe_print(f"?? [_execute_tool_calls] Execution result: success={exec_result.success}")
        _safe_print(f"?? [_execute_tool_calls] Execution error: {exec_result.error_message}")

        # ? FAILURE PATH
        if not exec_result.success:
            # Budget exhaustion must be terminal
            if "budget" in str(exec_result.error_message).lower():
                return [{
                    "success": False,
                    "error_type": "BUDGET_EXCEEDED",
                    "error_message": exec_result.error_message,
                }]

            # Non-terminal failure ? return structured failure
            return [{
                "success": False,
                "error_type": exec_result.error_type,
                "error_message": exec_result.error_message,
            }]

        # ? SUCCESS PATH
        results.append({
            "success": True,
            "output": exec_result.output,
        })

        # HARD budget guard (post-execution)
        budget = getattr(self.execution_manager, "budget_manager", None)
        if (
            budget is not None
            and hasattr(budget, "spent")
            and hasattr(budget, "max_budget")
            and budget.spent >= budget.max_budget
        ):
            return [{
                "success": False,
                "error_type": "BUDGET_EXCEEDED",
                "error_message": "Budget exhausted",
            }]

        return results


    def _apply_governance(self, tool_calls: List[ToolCall]) -> bool:
        """Apply governance decisions (circuit breaker, loop breaker)"""
        # Loop breaker: check tool fingerprints from recent calls
        recent_tool_names = [
            m.tool_name for m in self.conversation_history 
            if m.role == "tool" and m.tool_name
        ]
        
        # Simple loop detection: same tool called too many times
        current_tool_names = [tc.tool_name for tc in tool_calls]
        for tool_name in current_tool_names:
            if recent_tool_names.count(tool_name) > self.MAX_SAME_TOOL_CALLS:
                logger.warning(f"[Agent] Loop breaker: tool '{tool_name}' called too many times")
                return False
        
        return True

    def _extract_response_text(self, response) -> str:
        """Extract text from LLM response"""
        if response is None:
            return ""
        if isinstance(response, dict):
            return response.get("text", "") or ""
        elif hasattr(response, "text") and response.text:
            return response.text
        else:
            return ""
        
    def _parse_tool_call(self, text: str):
        """
        Parse tool call of form:
        <function="tool_name">
        {...json...}
        </function>
        """
        import json
        import re

        match = re.search(
            r'<function="(?P<name>[^"]+)">\s*(?P<args>\{.*?\})\s*</function>',
            text,
            re.DOTALL,
        )

        if not match:
            raise ValueError(f"Invalid tool call format: {text}")

        tool_name = match.group("name")
        args = json.loads(match.group("args"))

        return tool_name, args

    def _is_final_answer(self, text: str) -> bool:
        """Check if text indicates a final answer (not thinking/planning)"""
        if not text:
            return False
            
        text_lower = text.lower().strip()
        
        # Explicit final answer markers
        final_markers = [
            "<final>",
            "</final>",
            "final answer",
            "final answer:",
            "answer:",
            "result:",
            "here is",
            "here's",
            "task complete",
            "done:",
            "done",  # TEMP: SDK smoke test compatibility - remove in v0.2
            "complete",
            "finished",
            "here you go",
            "the result is",
        ]
        
        # Check for markers
        for marker in final_markers:
            if marker in text_lower:
                return True
        
        # Check for meaningless/thinking text
        thinking_phrases = [
            "call ",
            "i'll ",
            "i will ",
            "let me ",
            "i need to ",
            "i should ",
            "maybe ",
            "i think ",
            "i believe ",
        ]
        
        for phrase in thinking_phrases:
            if text_lower.startswith(phrase):
                return False  # This is thinking, not final answer
        
        # If text is very short and not a final marker, it's probably thinking
        if len(text_lower) < 20:
            return False
        
        # Otherwise, assume it's a final answer
        return False
        
    def _default_system_prompt(self):
        """Default system prompt"""
        return """You are AgenWatch.

Your job:
- Provide helpful answers directly.
- Call tools when needed to complete the task.
- When done, provide final answer clearly.

Format:
- Use tools when necessary
- Provide final answer directly"""

    async def _cleanup(self, owns_session: bool):
        """Guaranteed cleanup"""
        self.logger.log(EventType.SESSION_END, session_id=self.session_id)
        self.timeline.tool_end("agent", {"success": self.state == AgentState.DONE})
        self.tracer.end_session(self.session_id)
        
        # Only end session if we own it
        if owns_session:
            await self.client.end_session(self.session_id)

    def get_stats(self):
        """Get agent statistics"""
        return {
            "state": self.state.value,
            "iterations": self.current_iteration,
            "messages": len(self.conversation_history),
        }

    def get_conversation(self):
        """Get conversation history"""
        return [asdict(m) for m in self.conversation_history]

    def replay(self) -> AgentResult:
        """
        Replay recorded steps from the ExecutionLog without executing tools or modifying state.
        """
        if not self.replay_engine:
            raise RuntimeError("Replay engine is not initialized. Ensure execution_mode is REPLAY.")

        self.logger.log(EventType.REPLAY_START, session_id=self.session_id)
        self.state = AgentState.REPLAYING

        try:
            replayed_steps = self.replay_engine.replay_steps()
            self.conversation_history = replayed_steps

            # Extract the final step to determine the result
            final_step = replayed_steps[-1] if replayed_steps else None
            if final_step and final_step.role == "assistant":
                self.state = AgentState.DONE
                return AgentResult(
                    success=True,
                    output=final_step.content,
                    iterations=len(replayed_steps),
                    terminal_reason="REPLAY_COMPLETED"
                )
            else:
                self.state = AgentState.ERROR
                return AgentResult(
                    success=False,
                    error_type="REPLAY_INCOMPLETE",
                    iterations=len(replayed_steps),
                    terminal_reason="REPLAY_INCOMPLETE"
                )

        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.log(EventType.REPLAY_ERROR, error=str(e))
            return AgentResult(
                success=False,
                error_type="REPLAY_ERROR",
                iterations=0,
                terminal_reason="REPLAY_ERROR"
            )

__INTERNAL__ = True



