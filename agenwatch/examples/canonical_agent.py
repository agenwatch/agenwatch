"""
Canonical agenwatch Example

This file demonstrates the *correct* and *minimal* way to use agenwatch.
It intentionally avoids presets, helpers, or shortcuts.

If this file breaks, the release is broken.
"""

import asyncio

from agenwatch.sdk import Agent, tool
from agenwatch._kernel.testing.mock_llm import MockLLMProvider


# -------------------------------------------------------------------
# 1. Define tools (pure functions, no side effects unless intentional)
# -------------------------------------------------------------------

@tool
async def add(args: dict) -> dict:
    """
    Add two numbers.

    Args:
        a (int): first number
        b (int): second number
    """
    return {"result": args["a"] + args["b"]}


# -------------------------------------------------------------------
# 2. Configure the LLM (using mock for this example)
# -------------------------------------------------------------------

mock_llm = MockLLMProvider(
    responses=[
        {
            "tool_calls": [
                {
                    "name": "add",
                    "args": {"a": 2, "b": 3},
                }
            ]
        },
        {
            "final": "The sum of 2 and 3 is 5"
        },
    ]
)


# -------------------------------------------------------------------
# 3. Create the agent
# -------------------------------------------------------------------

agent = Agent(
    llm=mock_llm,
    tools=[add],
)


# -------------------------------------------------------------------
# 4. Run the agent
# -------------------------------------------------------------------

async def main():
    result = await agent.run("Add 2 and 3")

    print("=== EXECUTION RESULT ===")
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Iterations: {result.iterations}")
    print(f"Cost: ${result.cost}")
    print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())


