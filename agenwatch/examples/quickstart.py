from agenwatch import Agent, tool

@tool
def echo(text: str) -> str:
    return f"echo: {text}"

agent = Agent(tools=[echo], max_iterations=2)

result = agent.run("say hello using echo")

print(result)





