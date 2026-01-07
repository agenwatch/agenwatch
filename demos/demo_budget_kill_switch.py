"""
DEMO: Budget-Enforced Kill Switch (Impossible in LangSmith)

🎯 What this demo proves:
The agent is physically prevented from making Call #2.
Not logged. Not warned. BLOCKED.
"""

from agenwatch import Agent, tool
from agenwatch._kernel.errors import RecoverableToolError
from agenwatch.llm_provider import MockLLMProvider


# ---- Tool definition ----
@tool("Flaky tool that fails once")
def flaky_tool(args: dict) -> dict:
    """
    First call succeeds.
    Second call fails (recoverable).
    Retries should NOT double-charge.
    """
    x = args.get("x", 0)
    if not hasattr(flaky_tool, "called"):
        flaky_tool.called = True
        print("🟢 TOOL CALL #1 EXECUTED")
        return {"result": x + 1}

    print("🔴 TOOL FAILURE (recoverable)")
    raise RecoverableToolError("temporary failure")


# ---- Agent setup ----
# Mock provider that always tries to call the tool
mock_llm = MockLLMProvider(
    responses=[
        {"tool_calls": [{"id": "call_1", "name": "flaky_tool", "arguments": {"x": 1}}]},
        {"tool_calls": [{"id": "call_2", "name": "flaky_tool", "arguments": {"x": 2}}]},
        {"tool_calls": [{"id": "call_3", "name": "flaky_tool", "arguments": {"x": 3}}]},
    ]
)

agent = Agent(
    tools=[flaky_tool],
    llm=mock_llm,
    budget=1.0,          # 🔥 ONLY 1 unit allowed
    max_iterations=5,
)


# ---- Run agent ----
print("\n=== START DEMO ===\n")

result = agent.run("Call the flaky tool twice")

print("\n=== FINAL RESULT ===")
print("Success:", result.success)
print("Error:", result.error)
print("Iterations:", result.iterations)
print("Tool calls:", result.tool_calls)
print("Cost spent:", result.cost)

print("\n=== DEMO END ===")
