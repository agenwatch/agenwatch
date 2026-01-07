"""
AgenWatch Loop Breaker
=======================
Detects and prevents infinite LLM reasoning loops.

How it works:
1. Hash each tool call (tool_name + arguments) → fingerprint
2. Track fingerprints in session history
3. If same fingerprint appears N times → loop detected
4. Break the loop by forcing agent to stop or retry differently

This prevents the agent from getting stuck in:
- Repeated failed tool calls with same args
- Oscillating between same tools
- Infinite reasoning cycles
"""

import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from agenwatch._kernel.errors import RecoverableToolError

logger = logging.getLogger("agenwatch.loop_breaker")


@dataclass
class LoopBreakerConfig:
    """Configuration for loop breaker."""
    max_same_fingerprint: int = 3      # How many times before breaking
    check_enabled: bool = True
    log_fingerprints: bool = True


@dataclass
class FingerprintRecord:
    """Record of a single fingerprint occurrence."""
    fingerprint: str
    tool_name: str
    args_hash: str
    iteration: int
    timestamp: float


class LoopBreaker:
    """
    Detects infinite loops in agent reasoning.
    
    Tracks tool call fingerprints and flags when:
    - Same tool called with same args repeatedly
    - Oscillating patterns detected
    - Maximum threshold exceeded
    """

    def __init__(self, config: Optional[LoopBreakerConfig] = None):
        """
        Initialize loop breaker.
        
        Args:
            config: LoopBreakerConfig instance
        """
        self.config = config or LoopBreakerConfig()
        
        # Session state
        self.fingerprint_history: List[FingerprintRecord] = []
        self.fingerprint_counts: Dict[str, int] = defaultdict(int)
        self.loop_detected = False
        self.loop_detected_at_iteration = -1
        self.loop_breaking_tool = None
        
    
    def _compute_fingerprint(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Compute fingerprint for a tool call.
        
        Returns:
            (full_fingerprint, args_hash)
        """
        # Hash arguments (order-independent)
        args_str = str(sorted(arguments.items()))
        args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
        
        # Full fingerprint: tool_name + args_hash
        full_fp = f"{tool_name}:{args_hash}"
        
        return full_fp, args_hash
    
    
    def check_and_record(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        iteration: int,
        timestamp: float,
        error: Optional[Exception] = None   # ← ADD THIS
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if this tool call creates a loop.
        
        Args:
            tool_name: Name of tool being called
            arguments: Tool arguments
            iteration: Current iteration number
            timestamp: Current timestamp
        
        Returns:
            (is_loop_detected, reason_or_none)
        """
        if not self.config.check_enabled:
            return False, None
        
        fp, args_hash = self._compute_fingerprint(tool_name, arguments)
        
        # Record this fingerprint
        record = FingerprintRecord(
            fingerprint=fp,
            tool_name=tool_name,
            args_hash=args_hash,
            iteration=iteration,
            timestamp=timestamp
        )
        self.fingerprint_history.append(record)

        # Do NOT count recoverable failures as loops
        from agenwatch._kernel.errors import RecoverableToolError

        if iteration > 1 and not isinstance(error, RecoverableToolError):
            self.fingerprint_counts[fp] += 1


        count = self.fingerprint_counts[fp]

        # Only break on pure repetition, not recoverable errors
        if (
    count >= self.config.max_same_fingerprint
    and iteration > 1
    and not isinstance(error, RecoverableToolError)
):
            self.loop_detected = True
            self.loop_detected_at_iteration = iteration
            self.loop_breaking_tool = tool_name
            
            reason = (
                f"Loop detected: tool '{tool_name}' called with same args "
                f"{count} times (threshold: {self.config.max_same_fingerprint})"
            )
            
            logger.warning(
                "[LoopBreaker] %s at iteration %d",
                reason,
                iteration
            )
            
            return True, reason
        
        # Log fingerprint if enabled
        if self.config.log_fingerprints:
            logger.debug(
                "[LoopBreaker] Iteration %d: %s (count: %d/%d)",
                iteration,
                fp,
                count,
                self.config.max_same_fingerprint
            )
        
        return False, None
    
    
    def detect_oscillation(self, window_size: int = 5) -> Tuple[bool, Optional[str]]:
        """
        Detect oscillating pattern (A → B → A → B).
        
        Args:
            window_size: Number of recent calls to check
        
        Returns:
            (is_oscillating, reason_or_none)
        """
        if len(self.fingerprint_history) < window_size:
            return False, None
        
        recent = self.fingerprint_history[-window_size:]
        fps = [r.fingerprint for r in recent]
        
        # Check for alternating pattern
        if len(set(fps)) == 2:  # Only 2 unique fingerprints
            # Check if they alternate
            alternating = True
            for i in range(len(fps) - 1):
                if fps[i] == fps[i + 1]:
                    alternating = False
                    break
            
            if alternating:
                reason = (
                    f"Oscillation detected: alternating between "
                    f"{fps[0]} and {fps[1]} (window: {window_size})"
                )
                logger.warning("[LoopBreaker] %s", reason)
                return True, reason
        
        return False, None
    
    
    def get_fingerprint_stats(self) -> Dict[str, Any]:
        """Get fingerprint statistics."""
        return {
            "total_calls": len(self.fingerprint_history),
            "unique_fingerprints": len(self.fingerprint_counts),
            "fingerprint_counts": dict(self.fingerprint_counts),
            "loop_detected": self.loop_detected,
            "loop_at_iteration": self.loop_detected_at_iteration,
            "loop_breaking_tool": self.loop_breaking_tool,
            "most_common": (
                max(self.fingerprint_counts.items(), key=lambda x: x[1])[0]
                if self.fingerprint_counts else None
            )
        }
    
    
    def get_recent_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent call history."""
        return [
            {
                "fingerprint": r.fingerprint,
                "tool": r.tool_name,
                "iteration": r.iteration,
                "timestamp": r.timestamp
            }
            for r in self.fingerprint_history[-n:]
        ]
    
    
    def reset(self):
        """Reset loop breaker state for new session."""
        self.fingerprint_history = []
        self.fingerprint_counts = defaultdict(int)
        self.loop_detected = False
        self.loop_detected_at_iteration = -1
        self.loop_breaking_tool = None
        logger.info("[LoopBreaker] Reset")
    
    
    def should_force_stop(self) -> bool:
        """Should agent be forced to stop?"""
        return self.loop_detected
    
    
    def should_retry_differently(self) -> bool:
        """Should agent retry with different approach?"""
        oscillating, _ = self.detect_oscillation()
        return oscillating or self.loop_detected

__INTERNAL__ = True


