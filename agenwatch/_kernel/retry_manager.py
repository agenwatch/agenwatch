# AgenWatch/retry_manager.py
"""
AgenWatch Retry + Recovery Manager
- Integrates with CircuitBreaker, SafetyGuard, RepairMemory, Timeline and SandboxExecutionResult.
- Supports sync/async tool callables (auto-detected).
- Configurable retry/backoff strategies and repair/fallback hooks that accept full context.
"""

from __future__ import annotations
import time
import asyncio
import random
import logging
import inspect
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Optional, Tuple, Dict, Type, Union, Awaitable, List
)
from datetime import datetime

logger = logging.getLogger("agenwatch.retry")
logger.addHandler(logging.NullHandler())

# --- Types ---------------------------------------------------------------
# A tool callable may be sync (returns result or raises) or async (coroutine).
ToolCallable = Union[Callable[..., Any], Callable[..., Awaitable[Any]]]
ToolArgs = Tuple[Any, ...]
ToolKwargs = Dict[str, Any]

# Expected minimal interface of a Sandbox execution result (duck-typed)
class SandboxExecutionResult:
    """
    Minimal adapter class to explain expected attributes.
    In your real repo, your SandboxExecutionResult will have these attributes.
    This class is only for type-hinting and should not be imported at runtime
    if your real class is already present in the codebase.
    """
    success: bool
    result: Any
    metrics: Dict[str, Any]
    error: Optional[BaseException]

# ----------------------------------------------------------------------------
# Retry/backoff strategies
# ----------------------------------------------------------------------------
def _exponential_backoff(initial: float, multiplier: float, max_backoff: float, attempt: int, jitter_frac: float) -> float:
    base = initial * (multiplier ** (attempt - 1))
    base = min(base, max_backoff)
    jitter_amount = base * jitter_frac
    return max(0.0, base + random.uniform(-jitter_amount, jitter_amount))

def _full_jitter_backoff(initial: float, multiplier: float, max_backoff: float, attempt: int, jitter_frac: float) -> float:
    # base exponential cap, then uniform(0,cap)
    cap = initial * (multiplier ** (attempt - 1))
    cap = min(cap, max_backoff)
    return random.uniform(0, cap)

def _fixed_backoff(initial: float, *_args, **_kwargs) -> float:
    return initial

def _linear_backoff(initial: float, increment: float, max_backoff: float, attempt: int, jitter_frac: float) -> float:
    val = initial + increment * (attempt - 1)
    return min(val, max_backoff)

BACKOFF_STRATEGIES = {
    "exponential": _exponential_backoff,
    "full_jitter": _full_jitter_backoff,
    "fixed": _fixed_backoff,
    "linear": _linear_backoff,
}

# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------
@dataclass
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff: float = 0.2
    max_backoff: float = 5.0
    multiplier: float = 2.0
    jitter: float = 0.1              # fraction of backoff to jitter +/- unless strategy does full jitter
    increment: float = 0.5           # used by linear strategy
    backoff_strategy: str = "exponential"  # one of BACKOFF_STRATEGIES keys
    retryable_exceptions: Tuple[Type[BaseException], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[BaseException], ...] = tuple()
    timeout_per_attempt: Optional[float] = None

@dataclass
class RetryResult:
    success: bool
    result: Optional[Any] = None
    attempts: int = 0
    last_exception: Optional[BaseException] = None
    fallback_used: bool = False
    repaired: bool = False
    events: List[Dict[str, Any]] = field(default_factory=list)

# ----------------------------------------------------------------------------
# Retry manager
# ----------------------------------------------------------------------------
class ToolRetryManager:
    """
    ToolRetryManager wraps any tool callable (sync/async) and:
    - retries based on policy
    - runs fallback tool (sync/async)
    - runs repair_hook (sync/async) with full context
    - logs timeline events via injected timeline (timeline.record(event_dict))
    - checks circuit breaker via injected circuit_breaker.is_open(tool_name)
    - consults safety_guard.analyze_failure(context) before final retries/fallback
    - records failures into repair_memory.record_failure(context)
    """

    def __init__(
        self,
        *,
        retry_policy: Optional[RetryPolicy] = None,
        fallback_tool: Optional[ToolCallable] = None,
        repair_hook: Optional[Callable[[Dict[str, Any]], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]] = None,
        timeline: Optional[Any] = None,            # expected: timeline.record(event: dict)
        circuit_breaker: Optional[Any] = None,     # expected: circuit_breaker.is_open(tool_name) -> bool
        safety_guard: Optional[Any] = None,        # expected: safety_guard.analyze_failure(context)->bool (True = allow retry)
        repair_memory: Optional[Any] = None,       # expected: repair_memory.record_failure(context)
        name: str = "tool",
    ):
        self.retry_policy = retry_policy or RetryPolicy()
        self.fallback_tool = fallback_tool
        self.repair_hook = repair_hook
        self.timeline = timeline
        self.circuit_breaker = circuit_breaker
        self.safety_guard = safety_guard
        self.repair_memory = repair_memory
        self.name = name

    # --------------------------
    # Utilities
    # --------------------------
    def _now_ts(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _timeline_emit(self, typ: str, details: Dict[str, Any]):
        event = {"ts": self._now_ts(), "type": typ, "tool": self.name, "details": details}
        try:
            if self.timeline and hasattr(self.timeline, "record"):
                self.timeline.record(event)
            else:
                # fallback to logging
                logger.debug("timeline event: %s", event)
        except Exception:
            logger.exception("timeline.record failed")

    def _is_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, self.retry_policy.non_retryable_exceptions):
            return False
        return isinstance(exc, self.retry_policy.retryable_exceptions)

    def _compute_backoff(self, attempt: int) -> float:
        strat = self.retry_policy.backoff_strategy
        fn = BACKOFF_STRATEGIES.get(strat)
        if fn is None:
            fn = _exponential_backoff
        # call with attempt and configured params
        return float(fn(
            self.retry_policy.initial_backoff,
            self.retry_policy.multiplier,
            self.retry_policy.max_backoff,
            attempt,
            self.retry_policy.jitter
        )) if strat in ("exponential", "full_jitter") else float(fn(
            self.retry_policy.initial_backoff,
            self.retry_policy.increment,
            self.retry_policy.max_backoff,
            attempt,
            self.retry_policy.jitter
        ))

    async def _maybe_await(self, val):
        if inspect.isawaitable(val):
            return await val
        return val

    def _handle_sandbox_result(self, res: Any) -> Tuple[bool, Any, Dict[str, Any]]:
        """
        Detect and normalize SandboxExecutionResult-like objects.
        Returns (success, payload, metrics)
        """
        # duck-typing: check for attributes commonly present
        metrics = {}
        if res is None:
            return True, None, metrics
        # If result is an object with .success attribute (our sandbox)
        if hasattr(res, "success") and hasattr(res, "metrics"):
            try:
                success = bool(getattr(res, "success"))
                payload = getattr(res, "result", None)
                metrics = getattr(res, "metrics", {}) or {}
                return success, payload, dict(metrics)
            except Exception:
                # fallback to treat as normal result
                return True, res, {}
        # otherwise treat as normal result
        return True, res, {}

    # --------------------------
    # Public API - unified execute
    # --------------------------
    def execute(self, tool_callable: ToolCallable, *args: Any, **kwargs: Any) -> RetryResult:
        """
        Synchronous entrypoint. Auto-detects if tool_callable is async and runs accordingly
        (runs the coroutine via asyncio.run if needed).
        """
        if inspect.iscoroutinefunction(tool_callable) or inspect.isawaitable(tool_callable):
            # run the async path in current event loop if present; otherwise use asyncio.run
            try:
                loop = asyncio.get_running_loop()
                # we are inside an event loop -> run async via create_task and await is impossible here in sync function.
                # This situation is dangerous; prefer execute_async when in async context.
                raise RuntimeError("execute called from running event loop; use execute_async instead")
            except RuntimeError as e:
                # differentiate "no running loop" vs "running loop present"
                if "no running event loop" in str(e) or "There is no current event loop" in str(e):
                    # safe to run top-level
                    return asyncio.run(self.execute_async(tool_callable, *args, **kwargs))
                # running loop present -> raise explicit error
                raise

        # sync path
        return asyncio.run(self._execute_sync_internal(tool_callable, args, kwargs))

    async def execute_async(self, tool_callable: ToolCallable, *args: Any, **kwargs: Any) -> RetryResult:
        """
        Async entrypoint: runs the retry loop in async style. Accepts sync or async callables.
        """
        return await self._execute_sync_internal(tool_callable, args, kwargs, async_mode=True)

    async def _execute_sync_internal(self, tool_callable: ToolCallable, args: ToolArgs, kwargs: ToolKwargs, async_mode: bool = False) -> RetryResult:
        """
        Core engine used by both sync and async entrypoints.
        If async_mode == False and tool_callable is sync, this function will be run via asyncio.run.
        """
        rp = self.retry_policy

        # Circuit breaker integration
        if self.circuit_breaker and hasattr(self.circuit_breaker, "is_open"):
            try:
                if self.circuit_breaker.is_open(self.name):
                    exc = RuntimeError("circuit-breaker-open")
                    self._timeline_emit("circuit_breaker_open", {"reason": "breaker open"})
                    return RetryResult(success=False, attempts=0, last_exception=exc, events=[{"type": "circuit_breaker_open"}])
            except Exception:
                logger.exception("circuit_breaker.is_open raised; proceeding with attempts")

        attempts = 0
        last_exc: Optional[BaseException] = None
        events: List[Dict[str, Any]] = []

        # main retry loop
        for attempt in range(1, rp.max_attempts + 1):
            attempts = attempt
            self._timeline_emit("attempt_start", {"attempt": attempt, "policy": rp.__dict__})
            try:
                # call tool (sync -> run in thread if in async mode; async -> await)
                if inspect.iscoroutinefunction(tool_callable):
                    res = await tool_callable(*args, **kwargs)
                else:
                    # sync callable: run in thread executor when in async_mode to avoid blocking event loop
                    if async_mode:
                        loop = asyncio.get_running_loop()
                        res = await loop.run_in_executor(None, lambda: tool_callable(*args, **kwargs))
                    else:
                        # running in sync context via asyncio.run wrapper; call directly
                        res = tool_callable(*args, **kwargs)

                # Normalize sandbox result if present
                success, payload, metrics = self._handle_sandbox_result(res)
                if success:
                    self._timeline_emit("attempt_success", {"attempt": attempt, "metrics": metrics})
                    events.append({"attempt": attempt, "status": "success", "metrics": metrics})
                    return RetryResult(success=True, result=payload, attempts=attempts, events=events)
                else:
                    # sandbox reported failure (treat similar to exception)
                    err = getattr(res, "error", RuntimeError("sandbox_failure"))
                    raise err

            except BaseException as exc:
                last_exc = exc
                # emit attempt_error with exception and any sandbox metrics if available
                events.append({"attempt": attempt, "status": "error", "error": repr(exc)})
                self._timeline_emit("attempt_error", {"attempt": attempt, "error": str(exc)})

                # Record failure in repair_memory (best-effort)
                try:
                    if self.repair_memory and hasattr(self.repair_memory, "record_failure"):
                        ctx_snapshot = {
                            "tool": self.name,
                            "args": args,
                            "kwargs": kwargs,
                            "attempt": attempt,
                            "exception": repr(exc),
                            "ts": self._now_ts(),
                        }
                        self.repair_memory.record_failure(ctx_snapshot)
                except Exception:
                    logger.exception("repair_memory.record_failure failed")

                # Decide whether exception is retryable
                if not self._is_retryable(exc):
                    self._timeline_emit("non_retryable", {"attempt": attempt, "error": str(exc)})
                    return RetryResult(success=False, attempts=attempts, last_exception=exc, events=events)

                # Ask SafetyGuard whether we may continue retries
                try:
                    if self.safety_guard and hasattr(self.safety_guard, "analyze_failure"):
                        allow = self.safety_guard.analyze_failure({
                            "tool": self.name,
                            "exception": exc,
                            "attempt": attempt,
                            "kwargs": kwargs,
                        })
                        # safety guard might be async
                        allow = await self._maybe_await(allow)
                        if allow is False:
                            self._timeline_emit("safety_guard_block", {"reason": "safety_guard recommended stop"})
                            return RetryResult(success=False, attempts=attempts, last_exception=exc, events=events)
                except Exception:
                    logger.exception("safety_guard.analyze_failure raised; continuing with retry decision")

                # If there are remaining attempts -> backoff then retry
                if attempt < rp.max_attempts:
                    backoff = self._compute_backoff(attempt)
                    self._timeline_emit("backoff", {"attempt": attempt, "backoff": backoff})
                    # sleep appropriately for async_mode
                    if async_mode:
                        await asyncio.sleep(backoff)
                    else:
                        time.sleep(backoff)
                    continue
                # exhausted attempts -> fallthrough to fallback/repair
                break

        # exhausted attempts
        # Attempt fallback if configured
        fallback_used = False
        repaired = False

        if self.fallback_tool is not None:
            self._timeline_emit("fallback_start", {"attempts": attempts})
            try:
                if inspect.iscoroutinefunction(self.fallback_tool):
                    fb_res = await self.fallback_tool(*args, **kwargs)
                else:
                    if async_mode:
                        loop = asyncio.get_running_loop()
                        fb_res = await loop.run_in_executor(None, lambda: self.fallback_tool(*args, **kwargs))
                    else:
                        fb_res = self.fallback_tool(*args, **kwargs)
                success, payload, metrics = self._handle_sandbox_result(fb_res)
                if success:
                    events.append({"fallback": "success"})
                    self._timeline_emit("fallback_success", {"metrics": metrics})
                    return RetryResult(success=True, result=payload, attempts=attempts, last_exception=last_exc, fallback_used=True, events=events)
                else:
                    raise getattr(fb_res, "error", RuntimeError("fallback_failed"))
            except BaseException as exc:
                events.append({"fallback": "error", "error": repr(exc)})
                self._timeline_emit("fallback_error", {"error": str(exc)})
                # proceed to repair hook

        # Attempt repair hook if provided
        if self.repair_hook is not None:
            self._timeline_emit("repair_start", {"attempts": attempts, "last_error": str(last_exc)})
            try:
                # Build full context for repair hook
                context = {
                    "tool": self.name,
                    "args": args,
                    "kwargs": kwargs,
                    "last_exception": last_exc,
                    "attempts": attempts,
                    "ts": self._now_ts(),
                }
                # repair may be async or sync
                new_context = self.repair_hook(context)
                new_context = await self._maybe_await(new_context)
                if not isinstance(new_context, dict):
                    raise RuntimeError("repair_hook must return a dict with keys 'args' and/or 'kwargs'")
                # The hook may return replacement 'args' and/or 'kwargs'
                new_args = tuple(new_context.get("args", args))
                new_kwargs = dict(new_context.get("kwargs", kwargs))
                repaired = True
                # Try a final attempt using repaired inputs
                self._timeline_emit("repair_applied", {"attempts": attempts, "repaired_args": bool(new_args), "repaired_kwargs": bool(new_kwargs)})
                if inspect.iscoroutinefunction(tool_callable):
                    final_res = await tool_callable(*new_args, **new_kwargs)
                else:
                    if async_mode:
                        loop = asyncio.get_running_loop()
                        final_res = await loop.run_in_executor(None, lambda: tool_callable(*new_args, **new_kwargs))
                    else:
                        final_res = tool_callable(*new_args, **new_kwargs)
                success, payload, metrics = self._handle_sandbox_result(final_res)
                if success:
                    events.append({"repair": "success"})
                    self._timeline_emit("repair_success", {"metrics": metrics})
                    return RetryResult(success=True, result=payload, attempts=attempts + 1, last_exception=None, fallback_used=False, repaired=True, events=events)
                else:
                    raise getattr(final_res, "error", RuntimeError("repair_final_attempt_failed"))
            except BaseException as exc:
                events.append({"repair": "error", "error": repr(exc)})
                self._timeline_emit("repair_error", {"error": str(exc)})
                # record the final failure in repair_memory too
                try:
                    if self.repair_memory and hasattr(self.repair_memory, "record_failure"):
                        ctx_snapshot = {
                            "tool": self.name,
                            "args": args,
                            "kwargs": kwargs,
                            "attempts": attempts,
                            "final_exception": repr(exc),
                            "ts": self._now_ts(),
                        }
                        self.repair_memory.record_failure(ctx_snapshot)
                except Exception:
                    logger.exception("repair_memory.record_failure failed (final)")
                return RetryResult(success=False, attempts=attempts + 1, last_exception=exc, fallback_used=fallback_used, repaired=repaired, events=events)

        # nothing helped -> return failure
        self._timeline_emit("final_failure", {"attempts": attempts, "error": repr(last_exc)})
        return RetryResult(success=False, attempts=attempts, last_exception=last_exc, fallback_used=fallback_used, repaired=repaired, events=events)


# ------------------------------------------------------------------
# Backward-compatibility shim for existing Agent logic
# ------------------------------------------------------------------

class RetryManager(ToolRetryManager):
    """
    Compatibility wrapper so existing agent code does NOT break.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._consecutive_failures = 0

    def reset_consecutive_failures(self):
        self._consecutive_failures = 0

    def check_tool_failure(self, tool_name: str, error: str):
        """
        Matches old API:
        returns (should_retry: bool, reason: str)
        """
        self._consecutive_failures += 1

        if self._consecutive_failures >= self.retry_policy.max_attempts:
            return True, f"Max retries exceeded for tool '{tool_name}'"

        return False, "retry_allowed"

__INTERNAL__ = True



