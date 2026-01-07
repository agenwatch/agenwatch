from agenwatch import Agent, tool

@tool("Echo input")
def echo(text: str):
    return text

agent = Agent(tools=[echo])
print(agent.run("Say hello"))

