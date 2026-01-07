"""
ToolExecutionManager - The Execution Governance Layer
======================================================

This is Layer 2 of the AgenWatch OS architecture.

Responsibilities:
- Owns ALL tool execution governance
- Enforces execution policy
- Handles repair → retry → circuit breaking
- Returns ExecutionResult contracts
- Agent never calls tools directly

Does NOT:
- Make agent-level decisions (that's Layer 1)
- Implement tool logic (that's Layer 3)
- Know about LLM providers
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any
import uuid

from agenwatch._kernel.execution_contracts import (
    ExecutionResult,
    ExecutionPolicy,
    ExecutionContext,
    FailureType,
    ExecutionAction,
    FailureClassifier,
)
from agenwatch._kernel.errors import RecoverableToolError
from agenwatch._kernel.safety.budget_manager import BudgetManager, BudgetExceededError

from enum import Enum

class ExecutionMode(Enum):
    NORMAL = "normal"
    RECORD = "record"
    REPLAY = "replay"


logger = logging.getLogger("agenwatch.execution_manager")


def _safe_print(*args, **kwargs):
    """Wrapper for print() that silently fails if stdout is closed"""
    try:
        print(*args, **kwargs)
    except ValueError:
        pass


class ToolExecutionManager:
    def __init__(
        self,
        tool_registry,
        repair_memory_manager=None,
        circuit_breaker=None,
        fail_fast_engine=None,
        sandbox=None,
        retry_manager=None,
        policy: Optional[ExecutionPolicy] = None,
        *,
        execution_mode: ExecutionMode = ExecutionMode.NORMAL,
        replay_engine=None,
        budget_manager=None,
        recorder=None,
        session_id: str = None,
    ):  
        self.session_id = session_id
        self.budget_manager = budget_manager
        self.registry = tool_registry
        self.repair_memory = repair_memory_manager
        self.circuit_breaker = circuit_breaker
        self.fail_fast_engine = fail_fast_engine
        self.sandbox = sandbox
        self.retry_manager = retry_manager
        self.policy = policy or ExecutionPolicy()
        # NOTE: budget_manager passed in, not overwritten

        self.execution_mode = execution_mode
        self.replay_engine = replay_engine
        self.recorder = recorder

        self.fingerprint_cache: Dict[str, ExecutionResult] = {}

        if self.execution_mode == ExecutionMode.REPLAY and not self.replay_engine:
            raise ValueError("REPLAY mode requires replay_engine")

        if self.execution_mode == ExecutionMode.RECORD and not self.recorder:
            raise ValueError("RECORD mode requires recorder")

        logger.info(
            "[ExecutionManager] Initialized | mode=%s | policy=%s",
            self.execution_mode.value,
            self.policy.to_dict()
        )

    
    async def execute_with_governance(
        self,
        tool_call,
        session_id: str,
        user_id: str,
        
    ) -> ExecutionResult:
        """
        SINGLE ENTRY POINT for all tool execution.
        """
        # 🔐 KERNEL PRE-FLIGHT (NON-BYPASSABLE)
        # check() MUST be called BEFORE execution — raises BudgetExceededError
        if self.budget_manager is not None:
            cost = getattr(tool_call, "cost", 1.0)  # v0.1 placeholder: cost=1.0
            self.budget_manager.check(cost)  # Raises BudgetExceededError if over budget

        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] ENTER")
        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Tool: {tool_call.tool_name}")
        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Args: {tool_call.raw_args}")
        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Tool call type: {type(tool_call)}")

        start_time = time.time()

        from uuid import uuid4

        # ALWAYS define first
        execution_id = f"{self.session_id}:{tool_call.tool_name}:{uuid4().hex}"
        
        # Create execution context
        context = ExecutionContext(
            execution_id=execution_id,
            session_id=session_id,
            user_id=user_id,
            tool_name=tool_call.tool_name,
            original_args=tool_call.raw_args or {},
            current_args=tool_call.raw_args or {},
        )

        context.add_event("execution_start", tool=tool_call.tool_name)

        # Step 1: FailFast guard
        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Step 1: FailFast check")
        if self.fail_fast_engine and self.fail_fast_engine.is_circuit_open():
            _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Circuit breaker OPEN - blocking execution")
            return ExecutionResult.failure_result(
                tool_name=tool_call.tool_name,
                error_type=FailureType.POLICY_VIOLATION,
                error_message="Circuit breaker is open - execution blocked",
                duration_ms=(time.time() - start_time) * 1000,
                timeline=context.events,
            )

        # Step 2: Validate tool exists
        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Step 2: Tool validation")
        tool = self.registry.get_tool(tool_call.tool_name)
        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Tool found: {tool is not None}")
        if not tool:
            return ExecutionResult.failure_result(
                tool_name=tool_call.tool_name,
                error_type=FailureType.FATAL,
                error_message=f"Tool not found: {tool_call.tool_name}",
                duration_ms=(time.time() - start_time) * 1000,
                timeline=context.events,
            )
        # 🔐 STEP 2.5: Budget pre-flight check — use canonical API
        if self.budget_manager:
            try:
                self.budget_manager.check(1.0)  # v0.1 placeholder cost
            except BudgetExceededError as e:
                context.add_event("budget_blocked", reason=str(e))
                return ExecutionResult.failure_result(
                    tool_name=tool_call.tool_name,
                    error_type=FailureType.BUDGET_EXCEEDED,
                    error_message=f"Budget blocked execution: {e}",
                    duration_ms=(time.time() - start_time) * 1000,
                    timeline=context.events,
                )
        
        # Step 3: Compute fingerprint ONLY (NO CACHE READ HERE)
        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Step 3: Fingerprint compute")
        fingerprint = None
        if self.policy.enable_fingerprinting:
            fingerprint = self._compute_fingerprint(
                tool_call.tool_name, context.current_args
            )
            _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Fingerprint: {fingerprint[:16]}")
        else:
            _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Fingerprinting disabled")

        # 🔐 STEP 3.5: Budget execution pre-check (cost reservation)
        # Note: This step is removed - budget recording happens after successful execution
        
        # Step 4: Execute with retry (THIS MUST ALWAYS RUN)
        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Step 4: Executing with retry")
        result = await self._execute_with_retry(tool, context)

        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Result success: {result.success}")
        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Result error: {result.error_message}")
        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Result output: {result.output}")
            
        # Step 5: Circuit breakers
        _safe_print(f"🔍 [ExecutionManager.execute_with_governance] Step 5: Update circuit breakers")
        
        if result.success:
            # 💰 STEP 5.1: charge() AFTER successful execution with semantic fingerprint
            # Explicit replay check: context.is_replay is False (not just falsy)
            if self.budget_manager and context.is_replay is False:
                fingerprint = BudgetManager.compute_fingerprint(
                    kind="tool",
                    name=tool_call.tool_name,
                    args=context.current_args or {}
                )
                # Return value is internal idempotency signal — do NOT branch on it
                self.budget_manager.charge(cost=1.0, fingerprint=fingerprint)
                context.add_event(
                    "budget_charged",
                    tool=tool_call.tool_name,
                    cost=1.0,
                    fingerprint=fingerprint[:32]
                )
            if self.circuit_breaker:
                self.circuit_breaker.reset_consecutive_failures("Tool succeeded")
            if self.fail_fast_engine:
                self.fail_fast_engine.reset_consecutive_failures("Tool succeeded")
        else:
            if self.circuit_breaker:
                should_stop, reason = self.circuit_breaker.check_tool_failure(
                    tool_name=tool_call.tool_name,
                    error=result.error_message or "Unknown error",
                )
                if should_stop:
                    result.error_message = reason
                    result.error_type = FailureType.CIRCUIT_OPEN
                    result.circuit_tripped = True

            if self.fail_fast_engine:
                from agenwatch._kernel.fail_fast import FailureType as FFFailureType
                ff_type = FFFailureType.TOOL_EXECUTION
                should_stop, reason = self.fail_fast_engine.check_tool_failure(
                    tool_name=tool_call.tool_name,
                    error=result.error_message or "Unknown error",
                    failure_type=ff_type,
                )
                if should_stop:
                    result.error_message = reason
                    result.error_type = FailureType.POLICY_VIOLATION

        # Step 6: CACHE ONLY IF NO BUDGET MANAGER
        _safe_print("🔍 [ExecutionManager.execute_with_governance] Step 6: Cache result")

        if (
            self.policy.enable_fingerprinting
            and result.success
            and fingerprint
            and self.budget_manager is None
        ):
            _safe_print("🔍 [ExecutionManager.execute_with_governance] Caching successful result")
            self.fingerprint_cache[fingerprint] = result
        else:
            _safe_print("🔍 [ExecutionManager.execute_with_governance] Not caching (budget-controlled execution)")

        _safe_print("🔍 [ExecutionManager.execute_with_governance] EXIT")
        return result


    async def _execute_with_retry(
    self,
    tool,
    context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute tool with retry logic.

        Budget rules:
        - NEVER charge during retries
        - Charge ONCE on final success
        - Charge ONCE on final failure
        - NEVER charge during replay
        """

        last_error = None
        backoff = self.policy.retry_backoff_base

        while context.attempt_number <= self.policy.max_retries:
            try:
                context.add_event(
                    "attempt_start",
                    attempt=context.attempt_number + 1,
                    args=context.current_args
                )

                # Repair on retry (schema errors)
                if context.repair_attempted and self.policy.enable_auto_repair:
                    context.current_args = await self._attempt_repair(
                        tool.name,
                        context.current_args,
                        context
                    )
                # 🔒 BUDGET PREFLIGHT — check() BEFORE each attempt
                # NOTE: Retries do NOT double-charge (charge only on final success)
                if self.budget_manager and context.is_replay is False:
                    # check() raises BudgetExceededError if over budget
                    try:
                        self.budget_manager.check(cost=1.0)  # v0.1 placeholder
                    except BudgetExceededError as e:
                        return ExecutionResult.failure_result(
                            tool_name=tool.name,
                            error_type=FailureType.BUDGET_EXCEEDED,
                            error_message=str(e),
                            attempts=0,
                            duration_ms=0,
                            timeline=context.events,
                        )

                # Execute tool (REAL or REPLAY happens below this layer)
                result = await self._execute_tool(tool, context)

                return ExecutionResult.success_result(
                    tool_name=tool.name,
                    output=result,
                    duration_ms=context.elapsed_ms(),
                    attempts=context.attempt_number + 1,
                    timeline=context.events
                )

            except Exception as e:
                last_error = e

                failure_type = FailureClassifier.classify(e, context)

                context.add_event(
                    "attempt_failed",
                    attempt=context.attempt_number + 1,
                    error=str(e),
                    failure_type=failure_type.value
                )

                action = FailureClassifier.get_action(
                    failure_type,
                    context.attempt_number,
                    self.policy.max_retries
                )

                # ❌ TERMINATE — FINAL FAILURE
                # NOTE: Per canonical spec, budget is ONLY charged on success
                # Failures do NOT consume budget (retry-safe)
                if action == ExecutionAction.TERMINATE:
                    return ExecutionResult.failure_result(
                        tool_name=tool.name,
                        error_type=failure_type,
                        error_message=str(e),
                        attempts=context.attempt_number + 1,
                        duration_ms=context.elapsed_ms(),
                        timeline=context.events
                    )

                # ❌ SKIP — FINAL FAILURE (no budget charge)
                elif action == ExecutionAction.SKIP:
                    return ExecutionResult.failure_result(
                        tool_name=tool.name,
                        error_type=failure_type,
                        error_message=f"Max retries reached: {str(e)}",
                        attempts=context.attempt_number + 1,
                        duration_ms=context.elapsed_ms(),
                        timeline=context.events
                    )

                # 🔁 REPAIR + RETRY (NO BUDGET)
                elif action == ExecutionAction.REPAIR_AND_RETRY:
                    context.repair_attempted = True
                    context.attempt_number += 1
                    await asyncio.sleep(backoff)
                    backoff = min(
                        backoff * self.policy.retry_backoff_multiplier,
                        self.policy.retry_backoff_max
                    )
                    continue

                # 🔁 RETRY (NO BUDGET)
                elif action == ExecutionAction.RETRY:
                    context.attempt_number += 1
                    await asyncio.sleep(backoff)
                    backoff = min(
                        backoff * self.policy.retry_backoff_multiplier,
                        self.policy.retry_backoff_max
                    )
                    continue

        # ❌ MAX RETRIES EXHAUSTED — FINAL FAILURE (no budget charge)
        return ExecutionResult.failure_result(
            tool_name=tool.name,
            error_type=FailureType.RECOVERABLE,
            error_message=f"Max retries exhausted: {str(last_error)}",
            attempts=context.attempt_number - 1,
            duration_ms=context.elapsed_ms(),
            timeline=context.events
        )

    
    async def _execute_tool(self, tool, context: ExecutionContext) -> Any:
        """
        Layer-2 → Layer-3 boundary.

        In REPLAY mode:
        - NO real tool execution
        - NO sandbox
        - Result comes from ReplayEngine

        In RECORD mode:
        - Execute real tool
        - Record input/output deterministically
        """

        _safe_print(f"🔍 [ExecutionManager._execute_tool] ENTER")
        _safe_print(f"🔍 [ExecutionManager._execute_tool] Mode: {self.execution_mode.value}")
        _safe_print(f"🔍 [ExecutionManager._execute_tool] Tool: {tool.name}")
        _safe_print(f"🔍 [ExecutionManager._execute_tool] Args: {context.current_args}")

        # ==========================================================
        # 🔁 REPLAY MODE — ABSOLUTE OVERRIDE
        # ==========================================================
        if self.execution_mode == ExecutionMode.REPLAY:
            _safe_print(f"▶️ [REPLAY] Returning recorded tool result")

            return self.replay_engine.get_tool_result(
                tool_name=tool.name,
                arguments=context.current_args
            )

        # ==========================================================
        # 📹 RECORD MODE — WRAP REAL EXECUTION
        # ==========================================================
        call_event_id = None
        if self.execution_mode == ExecutionMode.RECORD:
            call_event_id = self.recorder.record_tool_call(
                tool_name=tool.name,
                arguments=context.current_args
            )

        try:
            # ------------------------------------------------------
            # NORMAL EXECUTION PATH (UNCHANGED)
            # ------------------------------------------------------

            if hasattr(tool, "fn"):
                if asyncio.iscoroutinefunction(tool.fn):
                    result = await tool.fn(**context.current_args)
                else:
                    result = tool.fn(**context.current_args)

            elif self.policy.enable_sandbox and self.sandbox:
                sandbox_result = await self.sandbox.execute(
                    tool_func=tool.run,
                    args=context.current_args,
                    policies=[],
                    tool_name=tool.name
                )

                if sandbox_result.error:
                    raise RuntimeError(sandbox_result.error)

                result = sandbox_result.output

            elif hasattr(tool, "run"):
                result = await tool.run(**context.current_args)

            else:
                raise RuntimeError(
                    f"Tool {tool.name} has no executable method"
                )

            # ------------------------------------------------------
            # RECORD SUCCESS
            # ------------------------------------------------------
            if self.execution_mode == ExecutionMode.RECORD:
                self.recorder.record_tool_result(
                    call_event_id,
                    result,
                    success=True
                )
            return result

        except Exception as e:
            # ------------------------------------------------------
            # RECORD FAILURE
            # ------------------------------------------------------
            if self.execution_mode == ExecutionMode.RECORD:
                self.recorder.record_tool_result(
                    call_event_id,
                    None,
                    success=False,
                    error=str(e)
                )
            raise

    
    async def _attempt_repair(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Attempt to repair arguments using repair memory.
        """
        if not self.repair_memory:
            return args
        
        try:
            repair_result = await self.repair_memory.match_and_fix(
                tool_name=tool_name,
                args=args
            )
            
            if repair_result["was_repaired"]:
                context.add_event(
                    "repair_applied",
                    confidence=repair_result.get("confidence", 0.0),
                    pattern=repair_result.get("pattern", "unknown")
                )
                
                context.repair_confidence = repair_result.get("confidence", 0.0)
                
                logger.info(
                    f"[ExecutionManager] Repaired args for {tool_name} "
                    f"(confidence: {context.repair_confidence:.2f})"
                )
                
                return repair_result["fixed_args"]
        
        except Exception as e:
            logger.warning(f"[ExecutionManager] Repair failed: {e}")
        
        return args
    
    def _compute_fingerprint(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Compute stable fingerprint for deduplication.
        """
        import json
        import hashlib
        
        data = json.dumps({
            "tool": tool_name,
            "args": args
        }, sort_keys=True)
        
        return hashlib.sha256(data.encode()).hexdigest()
    
    def reset_cache(self):
        """Clear fingerprint cache (useful for testing)"""
        self.fingerprint_cache.clear()
        logger.debug("[ExecutionManager] Cache cleared")

__INTERNAL__ = True



