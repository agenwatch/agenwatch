# AgenWatch

**Runtime-enforced execution governance for AI agents.**

AgenWatch is a Python SDK that guarantees AI agents **stop when they must**.
Budgets, iteration limits, and execution boundaries are enforced **during runtime**, not observed after failure.

This is not an observability tool.
This is an execution kernel.

> **Deep Dive:** See [ARCHITECTURE.md](ARCHITECTURE.md) for design philosophy and guarantees.

---

## Why AgenWatch Exists

Most agent frameworks answer:
> "How do I make my agent smarter?"

AgenWatch answers a different question:
> **"Can I mathematically guarantee this agent will stop?"**

AgenWatch enforces hard limits **before** tools or LLM calls execute:
- No runaway costs
- No infinite loops
- No silent retries
- No post-mortem surprises

---

## What AgenWatch Is (and Is Not)

**AgenWatch is:**
- A bounded execution controller
- A deterministic agent runtime
- A safety and governance layer for agents

**AgenWatch is not:**
- A prompt engineering framework
- A UI or observability dashboard
- A workflow orchestration system
- A LangChain replacement

---

## Installation

```bash
pip install agenwatch
```

Optional providers:

```bash
pip install agenwatch[openai]
pip install agenwatch[anthropic]
pip install agenwatch[groq]
```

---

## Quick Example

```python
import os
from agenwatch import Agent, tool
from agenwatch.providers import AnthropicProvider

@tool("Echo input")
def echo(text: str) -> dict:
    return {"echo": text}

agent = Agent(
    tools=[echo],
    llm=AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY")),
    budget=1.0,          # Hard execution budget
    max_iterations=5
)

result = agent.run("Echo hello")

print(result.success)
print(result.cost)
```

**Guarantee:**
If the budget is exhausted, the agent cannot make another tool or LLM call.

---

## Budget Kill-Switch (Proof)

AgenWatch enforces budgets as a runtime kill switch, not a warning.

**Behavior:**

- First call executes and is charged
- Retries are idempotent (no double charge)
- Next call is blocked before execution
- Agent terminates with `budget_exceeded`

This is enforced inside the kernel, not in user code.

> Note: AgenWatch enforces limits at runtime. If a limit is exceeded, execution is terminated immediately.

---

## Streaming & Observability

AgenWatch exposes execution events for live inspection:

```python
for event in agent.stream("Analyze input"):
    print(event.type)
```

Event order is guaranteed.
Event payloads are inspectable JSON.

Streaming does not affect execution behavior.

---

## Deterministic Execution

AgenWatch records execution decisions so failures can be replayed without re-calling the LLM.

In v0.1:

- Replay is backend-controlled
- Results are inspectable
- Programmatic replay APIs will arrive later

---

## Who Should Use AgenWatch

AgenWatch is designed for:

- Platform engineers
- Infrastructure teams
- AI safety & governance layers
- Production systems with cost or compliance constraints

It is not designed for:

- Chatbot demos
- Prompt experimentation
- No-code workflows

---

## When Should You Use AgenWatch?

Use AgenWatch if you need **hard execution guarantees** for AI agents.

Typical use cases include:
- Preventing runaway tool or LLM calls
- Enforcing strict budget or iteration caps
- Debugging failures with deterministic replay
- Running agents in cost- or safety-sensitive environments

AgenWatch is designed for engineers who care about **governance and correctness**
as much as model quality.

---

## When NOT to Use AgenWatch

AgenWatch may not be the right fit if:
- You only need observability or tracing after execution
- You want rapid prototyping without hard limits
- You are looking for a high-level agent framework with many abstractions

AgenWatch intentionally trades flexibility for **predictability and control**.

---

## Relationship to Other Frameworks

AgenWatch is complementary to frameworks like LangChain or CrewAI.

Those frameworks focus on agent capability.
AgenWatch focuses on agent control.

AgenWatch can act as a runtime enforcement layer beneath other frameworks.

---

## Status

**Version: 0.1.0**

- Kernel, SDK, and budget enforcement are stable
- Public API is minimal and frozen
- Actively evolving toward stronger governance primitives

---

## License

MIT License



