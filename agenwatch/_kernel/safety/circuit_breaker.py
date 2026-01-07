class CircuitBreaker:
    """
    Minimal circuit breaker used by AgenWatch.

    Tracks:
    - consecutive tool failures
    - loop detection triggers
    - last failure reason

    Does NOT include timeout windows or half-open logic.
    """

    def __init__(self, max_failures: int = 3):
        self.max_failures = max_failures
        self.consecutive_failures = 0
        self.last_reason = None

    # -------------------------------------------------
    # TOOL FAILURE BREAKER
    # -------------------------------------------------
    def check_tool_failure(self, tool_name: str, error: str):
        """
        Called whenever a tool returns an error.
        Returns (should_stop, reason).
        """

        # 🔑 DO NOT COUNT TOOL_REPAIR AS FAILURE
        if error == "TOOL_REPAIR":
            return False, None

        self.consecutive_failures += 1
        self.last_reason = f"Tool '{tool_name}' failed: {error}"

        if self.consecutive_failures >= self.max_failures:
            return True, f"Circuit breaker triggered: {self.last_reason}"

        return False, None


    def reset_consecutive_failures(self, reason: str = None):
        """Reset counter after a successful tool call."""
        self.consecutive_failures = 0
        if reason:
            self.last_reason = reason

    # -------------------------------------------------
    # LOOP DETECTION BREAKER
    # -------------------------------------------------
    def check_loop_detected(self, loop_reason: str):
        """Always triggers breaker when loops/oscillation detected."""
        self.last_reason = loop_reason
        return True, loop_reason

    # -------------------------------------------------
    # STATUS
    # -------------------------------------------------
    def get_status(self):
        return {
            "failures": self.consecutive_failures,
            "max_failures": self.max_failures,
            "reason": self.last_reason,
        }

__INTERNAL__ = True


