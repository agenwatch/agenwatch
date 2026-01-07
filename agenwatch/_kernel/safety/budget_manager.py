"""
BudgetManager — Canonical Implementation
=========================================

Transaction ledger with a kill switch.

MUST DO:
- Enforce hard upper limit (spent > max → STOP)
- Track actual spend (charge AFTER execution)
- Retry-safe via fingerprinting (no double charge)
- Replay-safe (explicit is_replay check by caller)
- Thread-safe via lock

MUST NOT DO:
- Estimate future cost
- Provide soft warnings
- Allow refunds (v0.1)
- Depend on time

v0.1 Note: cost=1.0 is a placeholder. Real cost attribution comes later.
"""

import threading
import hashlib
import json
from typing import Set, Dict, Any


class BudgetExceededError(Exception):
    """
    Budget exceeded — execution must stop.
    
    This error is KERNEL-ONLY. Do NOT re-export to SDK.
    SDK sees `AgentResult.failure(error_type='budget_exceeded')`.
    """
    pass


class BudgetManager:
    """
    Transaction ledger with a kill switch.
    
    Responsibilities:
    - Enforce hard budget limit
    - Track actual spend (not estimates)
    - Idempotent charging via fingerprints (retry-safe)
    
    Does NOT:
    - Decide execution flow
    - Estimate future costs
    - Provide soft warnings
    - Know tool semantics
    """
    
    def __init__(self, max_budget: float):
        """
        Initialize budget manager.
        
        Args:
            max_budget: Hard upper limit. Once exceeded, execution stops.
        """
        if max_budget <= 0:
            raise ValueError("max_budget must be positive")
        
        self.max_budget = max_budget
        self._spent = 0.0
        self._lock = threading.Lock()
        self._charged_fingerprints: Set[str] = set()
    
    # =========================================================
    # READ-ONLY TELEMETRY
    # =========================================================
    
    @property
    def spent(self) -> float:
        """Total cost spent so far."""
        with self._lock:
            return self._spent
    
    @property
    def remaining(self) -> float:
        """Remaining budget."""
        with self._lock:
            return self.max_budget - self._spent
    
    # =========================================================
    # ENFORCEMENT API
    # =========================================================
    
    def check(self, cost: float) -> None:
        """
        Pre-flight check — MUST be called BEFORE execution.
        
        Raises:
            BudgetExceededError: If spending `cost` would exceed budget.
        
        Note:
            This does NOT charge. Use `charge()` after successful execution.
        """
        with self._lock:
            if self._spent + cost > self.max_budget:
                raise BudgetExceededError(
                    f"Budget exceeded: {self._spent:.4f} + {cost:.4f} > {self.max_budget:.4f}"
                )
    
    def charge(self, cost: float, fingerprint: str) -> bool:
        """
        Record cost — MUST be called AFTER successful execution.
        
        Args:
            cost: Actual cost of the execution.
            fingerprint: Semantic identity of the execution (for idempotency).
                         Format: "{kind}:{name}:{args_hash}"
        
        Returns:
            True if charged, False if fingerprint already seen (idempotent).
        
        IMPORTANT:
            Callers MUST NOT branch on the return value.
            It is an internal signal for idempotency.
        """
        with self._lock:
            if fingerprint in self._charged_fingerprints:
                # Already charged — idempotent no-op
                return False
            
            self._charged_fingerprints.add(fingerprint)
            self._spent += cost
            return True
    
    # =========================================================
    # FINGERPRINT UTILITIES
    # =========================================================
    
    @staticmethod
    def compute_fingerprint(kind: str, name: str, args: Dict[str, Any]) -> str:
        """
        Compute stable fingerprint for deduplication.
        
        Args:
            kind: Execution kind ('tool' or 'llm')
            name: Tool or provider name
            args: Arguments (will be JSON-serialized)
        
        Returns:
            Fingerprint string: "{kind}:{name}:{hash}"
        """
        try:
            args_str = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            args_str = str(args)
        
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]
        return f"{kind}:{name}:{args_hash}"


__INTERNAL__ = True
