# example_app.py

import asyncio
import logging

from agenwatch.client import AgenWatchClient

logging.basicConfig(level=logging.INFO)


# ============================================================
# Fake AI Debugger (Replace with your Groq/OpenAI later)
# ============================================================

class FakeDebugger:
    """
    Minimal AI debugger mock that "fixes" errors.
    Replace this with your universal LLM debugger later.
    """

    def get_corrected_params(self, original_tool_call, validation_error, tool_schema):
        args = original_tool_call["arguments"]

        # Example: Auto-fix wrong input types
        corrected = dict(args)

        # If "limit" must be int but user sends string — fix it
        if "limit" in args and isinstance(args["limit"], str):
            corrected["limit"] = int(args["limit"])

        # If "query" is missing — fill a default
        if "query" in tool_schema.get("required", []) and "query" not in args:
            corrected["query"] = "default"

        return corrected


# ============================================================
# Define Sample Tool
# ============================================================

def search_tool(query: str, limit: int = 5):
    """Simple internal tool."""
    return {
        "query": query,
        "results": [f"item_{i}" for i in range(limit)]
    }


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50}
    },
    "required": ["query"]
}


# ============================================================
# RUN EXAMPLE
# ============================================================

async def main():
    client = AgenWatchClient(
        debugger=FakeDebugger(),         # Attach fake debugger
        enable_persistence=False,        # Turn on later
        correction_timeout=10.0,
        healing_wait_timeout=5.0
    )

    # Start session
    session_id = await client.start_session({"user": "demo"})
    print("SESSION:", session_id)

    # Register internal tool
    client.register_tool("search", search_tool, SEARCH_SCHEMA)

    print("\n=== VALID CALL ===")
    result1 = await client.call_tool("search", {"query": "books", "limit": 3})
    print("RESULT:", result1)

    print("\n=== INVALID CALL (auto-healing triggered) ===")
    # limit should be int, user passes string => healing fixes it
    result2 = await client.call_tool("search", {"query": "games", "limit": "10"})
    print("RESULT:", result2)

    print("\n=== INVALID CALL MISSING REQUIRED FIELD ===")
    # Missing "query" — debugger fills "default"
    result3 = await client.call_tool("search", {"limit": 7})
    print("RESULT:", result3)

    await client.shutdown()
    print("\nClient shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())





