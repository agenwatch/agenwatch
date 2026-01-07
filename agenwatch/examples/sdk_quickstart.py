"""
AgenWatch SDK – Canonical Quickstart

This example uses ONLY the public SDK surface.
Anything else is internal and unsupported.
"""

from agenwatch import Agent, tool, AgentConfig


@tool("Add two numbers")
def add(x: int, y: int) -> dict:
    return {"sum": x + y}


def main():
    agent = Agent(
        tools=[add],
        config=AgentConfig(
            max_iterations=5,
        ),
    )

    result = agent.run("Add 2 and 3")

    print("Success:", result.success)
    print("Output:", result.output)
    print("Iterations:", result.iterations)
    print("Tool calls:", result.tool_calls)


if __name__ == "__main__":
    main()



