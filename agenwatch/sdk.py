"""
AgenWatch Public SDK
Clean, stable, minimal surface API
"""

from __future__ import annotations
from typing import (
    Callable, Dict, Any, Optional, List,
    Union, AsyncIterator, Iterator
)
from dataclasses import dataclass, field
from enum import Enum
import inspect
import asyncio
import logging
import os
from agenwatch._kernel.tools.registry import ToolRegistry
from agenwatch._kernel.execution_manager import ToolExecutionManager
from agenwatch.types import AgentConfig, ExecutionResult, StreamEvent
# =============================================================================
# PUBLIC TYPES
# =============================================================================
import inspect
from typing import Callable, Any
from dataclasses import replace
from agenwatch._kernel.tools.function_tool import FunctionTool

def tool(_fn=None, *, description: str | None = None):
    def decorator(fn):
        async def wrapper(*args, **kwargs):
            # Accepts either a single dict arg or kwargs, normalizes to dict
            if args and isinstance(args[0], dict) and not kwargs:
                call_args = args[0]
            else:
                call_args = kwargs
            result = fn(call_args)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = description or fn.__doc__
        wrapper._is_tool = True
        wrapper.__AgenWatch_tool__ = {
            "name": fn.__name__,
            "description": wrapper.__doc__,
            "callable": wrapper,
        }
        return wrapper

    # Case 1: @tool
    if callable(_fn):
        return decorator(_fn)

    # Case 2: @tool("desc")
    if isinstance(_fn, str):
        return lambda fn: decorator(fn)

    # Case 3: @tool(description="desc")
    return decorator

class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    AZURE = "azure"
    BEDROCK = "bedrock"
    GROQ = "groq" 

class CostUnit(str, Enum):
    USD = "usd"
    TOKENS = "tokens"
    REQUESTS = "requests"

@dataclass
class ToolCost:
    tool_name: str
    cost_per_call: float
    cost_unit: CostUnit = CostUnit.USD

@dataclass
class BudgetPolicy:
    max_total_cost: float
    on_exceed: str = "stop"  # stop | warn | continue

# ExecutionResult is now imported from .types

# StreamEvent is now imported from .types

@dataclass
class Param:
    description: str
    default: Any = inspect.Parameter.empty
    required: bool = True
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None

# =============================================================================
# ERRORS
# =============================================================================

class AgentError(Exception): ...
class BudgetExceededError(AgentError): ...
class MaxIterationsError(AgentError): ...
class ToolExecutionError(AgentError): ...


# =============================================================================
# AGENT CONFIG
# =============================================================================

# AgentConfig is now imported from .types

# =============================================================================
# AGENT (PUBLIC ENTRY)
# =============================================================================

class Agent:
    """
    The ONLY class users touch
    """

    def __init__(
        self,
        tools: Optional[List[Any]] = None,
        llm: Optional["LLMProvider"] = None,
        config: Optional[AgentConfig] = None,
        budget: Optional[float] = None,
        **overrides
    ):
        # 1. Capture LLM (precedence: override > explicit arg > None)
        self.llm = llm or overrides.pop("llm", None)
        
        # 2. Capture budget
        self._budget = budget

        # 3. Build config
        config = config or AgentConfig()
        if overrides:
            config = replace(config, **overrides)
        self.config = config

        self._logger = self._init_logger()
        self._tool_defs = self._register_tools(tools or [])

        self._init_core()

    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[ExecutionResult, Iterator[StreamEvent]]:
        if stream:
            return self._stream(task, context or {})
        
        # Check if we're already in an event loop (e.g., in pytest.mark.asyncio)
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, return the coroutine for the caller to await
            return self._arun(task, context or {})
        except RuntimeError:
            # No running event loop, create one
            return asyncio.run(self._arun(task, context or {}))

    async def arun(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[ExecutionResult, AsyncIterator[StreamEvent]]:
        if stream:
            return self._astream(task, context or {})
        return await self._arun(task, context or {})

    # ------------------------------------------------------------------

    def replay(self, log_path: str) -> ExecutionResult:
        return self._executor.replay_execution(log_path)

    # =============================================================================
    # INTERNAL
    # =============================================================================

    def _init_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"agenwatch.sdk.{id(self)}")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(h)
            logger.setLevel(logging.DEBUG if self.config.verbose else logging.WARNING)
        return logger

    def _register_tools(self, tools):
        wrapped = []
        for fn in tools:
            meta = getattr(fn, "__AgenWatch_tool__", None)
            if not meta:
                raise ValueError(f"{fn.__name__} is missing @tool decorator")

            wrapped.append(
                FunctionTool(
                    name=meta["name"],
                    fn=meta["callable"],
                )
            )
        return wrapped

    # def _create_llm_provider(self):
    #     """Create LLM provider based on config"""
    #     from agenwatch._kernel.universal_llm import UniversalLLM
        
    #     return UniversalLLM(
    #         api_key=self.config.api_key,
    #         model=self.config.model,
    #     )

    def _init_core(self):
        from agenwatch._kernel.agent import Agent as CoreAgent
        from agenwatch.llm_provider import MockLLMProvider
        from agenwatch._kernel.safety.budget_manager import BudgetManager

        # 1. Resolve LLM (STRICT, SAFE)
        llm = self.llm or MockLLMProvider()
        
        # 2. Create BudgetManager if budget specified
        self._budget_manager = BudgetManager(self._budget) if self._budget else None

        # 3. Kernel/SDK Isolation Wrapper
        class KernelLLMAdapter:
            def __init__(self, provider):
                self.provider = provider
                self.model_name = getattr(provider, "model_name", "unknown")

            async def generate(self, messages, tools=None):
                from agenwatch.llm_provider import LLMMessage
                # Ensure kernel dicts are normalized to LLMMessage
                norm_messages = [
                    LLMMessage(role=m["role"], content=m["content"])
                    if isinstance(m, dict) else m
                    for m in messages
                ]
                return await self.provider.generate(norm_messages, tools)

        # 4. Create core agent
        self._core_agent = CoreAgent(
            llm_provider=KernelLLMAdapter(llm),
            tools=self._tool_defs,
            max_iterations=self.config.max_iterations,
            user_id="sdk_user",
            system_prompt=getattr(self.config, "system_prompt", None),
            budget_manager=self._budget_manager,
        )

    async def _arun(self, task: str, context: Dict[str, Any]) -> ExecutionResult:
        from agenwatch.sdk_adapter import adapt_agent_result
        kernel_result = await self._core_agent.run(task)
        return adapt_agent_result(kernel_result)


    def stream(self, task: str):
        """
        Stream execution events for a task.
        """
        from agenwatch.sdk_event_sink import SDKEventSink

        sink = SDKEventSink()

        # Attach sink to kernel
        self._core_agent.set_event_sink(sink)

        # Run agent in background
        import threading

        def run_agent():
            try:
                # We call the synchronous run method
                # It will handle its own event loop in the thread
                self.run(task)
            finally:
                sink.queue.put(None)  # sentinel

        threading.Thread(target=run_agent, daemon=True).start()

        # Yield events
        while True:
            event = sink.next()
            if event is None:
                break
            yield event

    async def astream(self, task: str) -> AsyncIterator[StreamEvent]:
        """
        Async version of stream().
        """
        from agenwatch.sdk_event_sink import SDKEventSink
        
        sink = SDKEventSink()
        self._core_agent.set_event_sink(sink)
        
        # Run in background task
        loop = asyncio.get_running_loop()
        agent_task = asyncio.create_task(self._arun(task, {}))
        
        def on_done(_):
            sink.queue.put(None)
            
        agent_task.add_done_callback(on_done)
        
        while True:
            # We need to not block the loop, but Queue.get() is blocking.
            # In a real async implementation we'd use asyncio.Queue.
            # For now, following the user's "v0.1 friendly" synchronous queue approach
            # but providing a stub or simple wrapper if possible.
            # Actually, to keep it simple and consistent with user's specific "stream" request:
            event = await loop.run_in_executor(None, sink.next)
            if event is None:
                break
            yield event

    def _stream(self, task: str, context: Dict[str, Any]) -> Iterator[StreamEvent]:
        return self.stream(task)

    async def _astream(self, task: str, context: Dict[str, Any]) -> AsyncIterator[StreamEvent]:
        return self.astream(task)

# =============================================================================
# HELPERS
# =============================================================================

def _python_to_json_type(tp) -> str:
    if hasattr(tp, "__origin__"):
        if tp.__origin__ in (list, List):
            return "array"
        if tp.__origin__ in (dict, Dict):
            return "object"
    return {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }.get(tp, "string")




