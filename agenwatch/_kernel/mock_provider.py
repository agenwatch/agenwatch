from agenwatch._kernel.llm_response import LLMResponse
class MockLLMProvider:
    """Flexible mock LLM provider that uses configured responses."""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.response_index = 0
        self._exhausted = False
    
    def add_response(self, text=None, tool_calls=None):
        """Convenient way to build responses sequentially."""
        self.responses.append({"text": text, "tool_calls": tool_calls})
    
    def reset(self):
        """Reset index for reuse in multiple tests."""
        self.response_index = 0
        self._exhausted = False
    
    async def generate(self, messages, tools=None):
        """Return the next configured response or repeat final response."""
        # Once exhausted, keep returning the last response
        if self._exhausted and self.responses:
            data = self.responses[-1]  # Return last response again
        elif self.response_index < len(self.responses):
            data = self.responses[self.response_index]
            self.response_index += 1
            # Mark as exhausted if this was the last response
            if self.response_index >= len(self.responses):
                self._exhausted = True
        else:
            # No responses configured, return empty
            data = {"text": None, "tool_calls": None}

        # Handle both dict and object responses
        if isinstance(data, dict):
            text = data.get("text")
            tool_calls = data.get("tool_calls")
            # TEST COMPATIBILITY SHIM: Support "final" key as alias for "text"
            # This is test-only glue, not a feature. Do not expand further.
            if text is None and "final" in data:
                text = data.get("final")
        else:
            text = getattr(data, "text", None)
            tool_calls = getattr(data, "tool_calls", None)

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            raw=data,
            instrumentation={}
        )

__INTERNAL__ = True


