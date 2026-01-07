# AgenWatch/client.py

import logging
import asyncio
from typing import Dict, Any, Optional, Callable

from sdk_legacy_experiment.dispatcher import Dispatcher
from sdk_legacy_experiment.persistence import PersistenceManager
from sdk_legacy_experiment.session_manager import SessionManager
from sdk_legacy_experiment.middleware.session_injector import SessionInjectorMiddleware
from sdk_legacy_experiment.tools.validator import ToolValidator
from sdk_legacy_experiment.self_healing.engine import SelfHealingEngine
from sdk_legacy_experiment.tools.executor import ToolExecutor

logger = logging.getLogger("agenwatch.client")


class AgenWatchClient:
    """
    Main AgenWatch SDK client.

    Orchestrates:
        - Dispatcher
        - SessionManager
        - Middleware (session auto-inject)
        - ToolValidator (schema validation + healing)
        - SelfHealingEngine (AI corrections)
        - ToolExecutor (run internal/external tools)
    """

    def __init__(
        self,
        *,
        debugger: Optional[Any] = None,
        enable_persistence: bool = True,
        correction_timeout: float = 30.0,
        max_healing_attempts: int = 2,
        healing_wait_timeout: float = 10.0,
        executor_max_workers: int = 5,
    ):
        # ----------------------------
        # Basic validation
        # ----------------------------
        if correction_timeout <= 0:
            raise ValueError("correction_timeout must be > 0")
        if max_healing_attempts <= 0:
            raise ValueError("max_healing_attempts must be > 0")
        if healing_wait_timeout <= 0:
            raise ValueError("healing_wait_timeout must be > 0")

        self._is_shutdown = False
        self._debugger = debugger

        logger.info("[AgenWatchClient] Initializing...")

        # ----------------------------
        # Core: Dispatcher + SessionManager
        # ----------------------------
        self.dispatcher = Dispatcher()
        self.persistence = PersistenceManager() if enable_persistence else None
        self.session_manager = SessionManager(self.dispatcher)

        # ----------------------------
        # Middleware: session auto inject
        # ----------------------------
        session_injector = SessionInjectorMiddleware(self.session_manager)
        self.dispatcher.add_middleware(session_injector)

        # ----------------------------
        # Healing Engine (must be created BEFORE ToolValidator)
        # ----------------------------
        self.healing_engine = SelfHealingEngine(
            dispatcher=self.dispatcher,
            debugger=debugger,
            persistence=self.persistence,
            max_attempts_per_call=max_healing_attempts,
            correction_timeout_sec=correction_timeout,
            executor_max_workers=executor_max_workers,
        )

        # ----------------------------
        # Validator (must be created AFTER healing engine)
        # ----------------------------
        self.validator = ToolValidator(
            dispatcher=self.dispatcher,
            healing_timeout_sec=healing_wait_timeout,
        )

        # ----------------------------
        # Tool Executor
        # ----------------------------
        self.executor = ToolExecutor(self.dispatcher, self.session_manager)

        # ----------------------------
        # Schema registry
        # ----------------------------
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "[AgenWatchClient] Initialized (Debugger: %s, Persistence: %s)",
            "attached" if debugger else "NOT ATTACHED",
            "enabled" if enable_persistence else "disabled",
        )

    # ============================================================
    # Context Manager
    # ============================================================

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
        return False

    # ============================================================
    # Session Management
    # ============================================================

    async def start_session(self, meta: Optional[Dict[str, Any]] = None) -> str:
        self._check_ready()
        session_id = await self.session_manager.start_session_async(meta)
        logger.info("[AgenWatchClient] Session started: %s", session_id)
        return session_id

    async def end_session(self, session_id: Optional[str] = None):
        self._check_ready()
        self.session_manager.end_session(session_id)
        logger.info("[AgenWatchClient] Session ended: %s", session_id or "current")

    # ============================================================
    # Tool Registration
    # ============================================================

    def register_tool(self, name: str, func: Callable, schema: Dict[str, Any]):
        self._check_ready()

        if name in self._tool_schemas:
            raise ValueError(f"Tool '{name}' already registered")

        self.executor.register_tool(name, func)
        self._tool_schemas[name] = schema

        logger.info("[AgenWatchClient] Registered tool: %s", name)

    def register_external_executor(self, func: Callable):
        self._check_ready()
        self.executor.register_external_executor(func)
        logger.info("[AgenWatchClient] External executor registered")

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        return self._tool_schemas.get(name)

    # ============================================================
    # Debugger Management
    # ============================================================

    def attach_debugger(self, debugger: Any):
        self._check_ready()

        if not hasattr(debugger, "get_corrected_params"):
            raise ValueError("Debugger must implement get_corrected_params()")

        self._debugger = debugger
        self.healing_engine.debugger = debugger

        logger.info("[AgenWatchClient] Debugger attached")

    def has_debugger(self) -> bool:
        return self._debugger is not None

    # ============================================================
    # Main Action: call_tool()
    # ============================================================

    async def call_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Any:
        self._check_ready()
        self._check_active_session()

        # Find schema if not given
        if schema is None:
            schema = self.get_tool_schema(tool_name)
            if schema is None:
                raise ValueError(f"No schema found for tool '{tool_name}'")

        session_id = self.session_manager.current_session_id
        original_args = dict(args)

        logger.info("[AgenWatchClient] Calling tool '%s' (session=%s)", tool_name, session_id)

        # STEP 1 — Validate (and trigger healing)
        valid, corrected_args = await self.validator.validate(
            tool_name=tool_name,
            args=args,
            schema=schema,
            session_id=session_id,
        )

        if not valid:
            raise ValueError(
                f"Validation failed for tool '{tool_name}'. "
                f"Could not auto-correct. Args: {original_args}"
            )

        was_corrected = corrected_args != original_args

        if was_corrected:
            logger.warning(
                "[AgenWatchClient] Auto-correct applied: %s → %s",
                original_args,
                corrected_args,
            )

        # STEP 2 — Execute
        result, meta = await self.executor.execute_tool(
            tool_name=tool_name,
            args=corrected_args,
            is_fixed=was_corrected,   # FIXED ARGUMENT NAME
        )

        logger.info(
            "[AgenWatchClient] Tool '%s' executed in %.1fms",
            tool_name,
            meta.get("duration_ms", 0),
        )

        return result

    # ============================================================
    # Lifecycle / Utility
    # ============================================================

    async def shutdown(self):
        if self._is_shutdown:
            return

        logger.info("[AgenWatchClient] Shutdown started...")

        try:
            await self.healing_engine.shutdown()

            # end active sessions
            if self.session_manager.current_session_id:
                self.session_manager.end_session()

        except Exception as e:
            logger.exception("[AgenWatchClient] Error during shutdown: %s", e)

        self._is_shutdown = True
        logger.info("[AgenWatchClient] Shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_shutdown": self._is_shutdown,
            "debugger_attached": self.has_debugger(),
            "tools_registered": len(self._tool_schemas),
            "active_session": self.session_manager.current_session_id,
            "persistence_enabled": self.persistence is not None,
        }

    # ============================================================
    # Internal Checks
    # ============================================================

    def _check_ready(self):
        if self._is_shutdown:
            raise RuntimeError("agenwatchClient is shutdown")

    def _check_active_session(self):
        if self.session_manager.current_session_id is None:
            raise RuntimeError("No active session. Call start_session() first.")





