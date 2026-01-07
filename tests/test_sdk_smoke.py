import pytest

from agenwatch import Agent, tool
from agenwatch.llm_provider import MockLLMProvider


@pytest.mark.asyncio
async def test_sdk_smoke_single_tool_success():
    # --- define tool via SDK decorator ---
    @tool
    async def add(args):
        return {"sum": args["a"] + args["b"]}

    # --- mock LLM that calls the tool once, then finishes ---
    llm = MockLLMProvider(
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
                "final": "done"
            },
        ]
    )

    # --- create agent via SDK ---
    agent = Agent(
        llm=llm,
        tools=[add],
    )

    # --- run ---
    result = await agent.run("add two numbers")

    # --- assertions ---
    assert result.success is True
    assert result.output == "done"
    assert result.iterations == 2


def test_kernel_is_private():
    """
    Kernel modules must not be used by SDK users.

    Import may succeed due to Python mechanics,
    but access is unsupported and explicitly guarded.
    """
    import agenwatch._kernel.execution_manager as em

    # Kernel must NOT be part of public surface
    import agenwatch
    public_all = getattr(agenwatch, "__all__", [])
    assert "execution_manager" not in public_all
    assert "_kernel" not in public_all

    # Kernel modules must carry an internal marker
    assert getattr(em, "__INTERNAL__", False) is True


