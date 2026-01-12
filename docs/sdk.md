
````markdown
# AgenWatch SDK (v0.1.x)

This document defines the **public, supported SDK surface** of AgenWatch.

Anything **not documented here is internal**, unstable, and **not part of the public API contract**.

---

## 🎯 Purpose

The AgenWatch SDK provides a **minimal, intentional interface** for building and running agents.

It deliberately hides:
- Execution internals
- Budget enforcement mechanics
- Retry logic
- Safety systems
- Replay and governance layers

These remain **kernel-private**.

---

## 📦 Public API Surface (Non-Negotiable)

These are the **only supported imports**:

```python
from agenwatch import Agent, tool, AgentConfig, ExecutionResult
````

That’s it.

Importing anything else is **unsupported** and may break without notice.

---

## 🧠 Core Concepts

### Agent

`Agent` is the sole user-facing execution unit.

Responsibilities:

* Orchestrates reasoning
* Calls tools safely
* Enforces limits and budgets
* Produces a final result

Users do **not** control:

* Execution managers
* Budget managers
* Retry systems
* Circuit breakers
* Tool registries
* Session handling

These are internal and non-overridable.

---

### AgentConfig

Configuration is intentionally minimal.

```python
AgentConfig(
    max_iterations=10,
    budget=None,
    verbose=False,
)
```

Supported options:

* `max_iterations` — hard upper bound on reasoning steps
* `budget` — optional execution budget
* `verbose` — enable diagnostic logging

No other configuration knobs are supported in v0.1.x.

---

### ExecutionResult

`ExecutionResult` is the **SDK-level result type** returned by `Agent.run()`.

```python
@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    output: Any | None
    error: str | None
    iterations: int
    tool_calls: list[str]
```

Notes:

* Kernel result types are never exposed
* Errors are normalized into user-readable strings
* `tool_calls` lists the names of tools actually executed
* Internal failure modes remain hidden

---

## 🛠 Defining Tools

Tools are declared using the `@tool` decorator.

```python
from agenwatch import tool

@tool("Add two numbers")
def add(x: int, y: int) -> dict:
    return {"sum": x + y}
```

### Tool requirements

* Must have type hints
* Must return a JSON-serializable `dict`
* Description is optional (docstring used if omitted)

Tools **may**:

* Call external APIs
* Read or write files
* Perform network requests
* Have side effects

The kernel governs safety, retries, and budgets.

---

## ▶ Running an Agent

```python
agent = Agent(
    tools=[add],
    max_iterations=3,
)

result = agent.run("Add 2 and 3")

if result.success:
    print(result.output)
else:
    print(result.error)
```

Behavior:

* Stops on natural completion
* Stops on budget exhaustion
* Stops on iteration limits
* Infinite loops are impossible

---

## 💰 Budget Behavior

Budgets are enforced **inside the kernel**.

If the budget is exhausted:

* Tool execution stops immediately
* The agent terminates
* An error result is returned

Budget enforcement **cannot be bypassed** from the SDK.

---

## 🔁 Retry Behavior

Retry logic is **automatic and internal**.

Users cannot:

* Force retries
* Configure retry counts
* Re-execute tools manually

This ensures deterministic and safe execution.

---

## 🔄 Async Execution (v0.1)

In v0.1.x:

* `Agent.run()` is the **only public execution method**
* It is **synchronous**
* Async internals are hidden

An async public API (`arun`) may be introduced in v0.2.

---

## 🚫 What Is Explicitly NOT Supported

The following are intentionally unsupported:

* Importing kernel internals
* Accessing execution managers or safety primitives
* Custom retry policies
* Forcing tool re-execution
* Session management
* Direct budget manipulation
* Escape hatches of any kind

If it is not documented here, it is **not supported**.

---

## 📚 Canonical Example

```python
from agenwatch import Agent, tool

@tool
def multiply(x: int, y: int):
    return {"product": x * y}

agent = Agent(
    tools=[multiply],
    max_iterations=3,
)

result = agent.run("Multiply 4 and 5")

assert result.success
assert result.output["product"] == 20
```

This example uses **only public API**.

---

## 🔒 Stability Guarantees

* This SDK follows **semantic versioning**
* All public APIs in this document are stable for v0.1.x
* Breaking changes require a major version bump

Anything outside this document:

* Has **no stability guarantee**
* May change or disappear without notice

---

## ✅ Summary

This SDK is intentionally:

* Minimal
* Opinionated
* Safe
* Deterministic

It exposes **only what users need** and nothing more.

> **Anything not documented here is internal and unstable.**




