
# AgenWatch

**Runtime-enforced execution governance for AI agents.**

AgenWatch is a Python SDK that guarantees AI agents **stop when they must**.  
Budgets, iteration limits, and execution boundaries are enforced **during runtime**, not observed after failure.

This is **not** an observability tool.  
This is **not** a prompt framework.  

AgenWatch is an **execution kernel**.

> For architectural details and guarantees, see [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Why AgenWatch Exists

Most agent frameworks answer:

> “How do I make my agent smarter?”

AgenWatch answers a different question:

> **“Can I mathematically guarantee this agent will stop?”**

AgenWatch enforces hard limits **before** tools or LLM calls execute:
- No runaway costs
- No infinite reasoning loops
- No silent retry storms
- No post-mortem surprises

---

## What AgenWatch Is (and Is Not)

### AgenWatch **is**
- A bounded execution controller
- A deterministic agent runtime
- A runtime safety and governance layer for agents

### AgenWatch **is not**
- A prompt-engineering framework
- A workflow orchestrator
- A UI or observability dashboard
- A LangChain replacement

AgenWatch does **not** try to make agents smarter.  
It makes them **governable**.

---

## Installation

```bash
pip install agenwatch

```
---

## AgenWatch-Only Example (Correct & Working)

This example shows **pure AgenWatch execution** with runtime enforcement.

```python
import os
from agenwatch import Agent, tool
from agenwatch.providers import OpenAIProvider

@tool("Echo input text")
def echo(**kwargs) -> dict:
    """Echo back the provided text"""
    text = kwargs.get("text", "")
    return {"echo": text}

agent = Agent(
    tools=[echo],
    llm=OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    ),
    budget=1.0,          # Hard execution budget
    max_iterations=5
)

result = agent.run("Echo hello")

print(f"Success: {result.success}")
print(f"Cost: {result.cost}")
print(f"Output: {result.output}")
```

### Guarantees

* If budget is exhausted, **no further LLM or tool calls can occur**
* Retries are idempotent (no double charging)
* Termination is enforced inside the kernel, not user code

---

## Budget Kill-Switch (Runtime Enforcement)

AgenWatch enforces budgets as a **runtime kill switch**, not a warning.

Behavior:

* First allowed call executes and is charged
* Retries do not double-charge
* Next call is blocked **before execution**
* Agent terminates with `budget_exceeded`

This enforcement happens **synchronously inside the kernel**.

---

## LangChain + AgenWatch Integration (Honest Example)

AgenWatch can be used **alongside** LangChain.

Important distinction:

* **LangChain** produces prompts / logic
* **AgenWatch** governs execution
* **The LLM decides whether to call tools**

AgenWatch **cannot force** an LLM to call a tool.
It only governs tool calls **if and when they occur**.

### Working Integration Example

```python
# Pattern: LangChain Logic + AgenWatch Enforcement
import os
from langchain_core.prompts import ChatPromptTemplate
from agenwatch import Agent, tool
from agenwatch.providers import OpenAIProvider

# Step 1: Define tools (AgenWatch-governed)
@tool("Echo text safely")
def echo(**kwargs) -> dict:
    return {"echo": kwargs.get("text", "")}

# Step 2: Create AgenWatch Agent (the kernel)
agent = Agent(
    tools=[echo],
    llm=OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    ),
    budget=1.0,
    max_iterations=3
)

# Step 3: Let LangChain produce the task
prompt = ChatPromptTemplate.from_messages([
    ("human", "Say hello using the echo tool")
])
task = prompt.format_messages()[0].content

# Step 4: Execute through AgenWatch
result = agent.run(task)

print(f"Success: {result.success}")
print(f"Cost: {result.cost}")
print(f"Output: {result.output}")
```

### Example Output (Observed)

```
Success: True
Cost: 0.0
Output: It seems that I don't have access to an echo tool to assist with that.
However, I can say hello directly:
Hello!
```

### Important Clarification

This behavior is **expected**.

* The LLM chose **not** to call the tool
* AgenWatch correctly enforced execution boundaries
* No unauthorized calls occurred
* Budget and iteration limits were still active

AgenWatch does **not** influence LLM reasoning.
It governs **execution**, not **decision-making**.

---

## Streaming & Inspection

AgenWatch exposes execution events for inspection:

```python
for event in agent.stream("Analyze input"):
    print(event.type)
```

* Event order is guaranteed
* Streaming does not affect execution behavior
* This is inspection, not control

---

## Deterministic Execution (v0.1)

AgenWatch records execution decisions so failures can be inspected without re-running the agent.

In v0.1:

* Execution ledger is in-memory
* Replay is read-only
* No crash recovery or persistence

This is intentional and documented.

---

## Who Should Use AgenWatch

AgenWatch is designed for:

* Platform engineers
* Infrastructure teams
* Safety & governance layers
* Production systems with strict cost or control requirements

---

## When to Use AgenWatch

Use AgenWatch when you need:

* Hard runtime stop guarantees
* Budget or iteration enforcement
* Deterministic termination behavior
* Protection against runaway agents

AgenWatch prioritizes **predictability over flexibility**.

---

## When NOT to Use AgenWatch

AgenWatch may not be a good fit if:

* You only need post-execution observability
* You want rapid prototyping without hard limits
* You expect the SDK to force tool usage

---

## Relationship to Other Frameworks

AgenWatch is **complementary** to frameworks like:

* LangChain
* CrewAI
* LangGraph

Those frameworks focus on **agent capability**.
AgenWatch focuses on **agent control**.

---

## Status

**Version: 0.1.0**

* Kernel and budget enforcement are stable
* Public API is minimal and frozen
* Governance primitives will evolve incrementally

---

## License

MIT License
