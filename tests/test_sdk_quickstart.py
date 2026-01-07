def test_sdk_quickstart_runs():
    from agenwatch import Agent, tool, AgentConfig
    from agenwatch.llm_provider import MockLLMProvider

    @tool
    def add(x: int, y: int):
        return {"sum": x + y}

    llm = MockLLMProvider(
        responses=[
            {"tool_calls": [{"name": "add", "args": {"x": 1, "y": 2}}]},
            {"text": "<final>The sum is 3</final>"}
        ]
    )

    agent = Agent(
        tools=[add],
        llm=llm,
        config=AgentConfig(max_iterations=3),
    )

    result = agent.run("Add 1 and 2")

    assert result.success is True
    assert result.output is not None
    assert "sum" in str(result.output)
