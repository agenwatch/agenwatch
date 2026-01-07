# AgenWatch Architecture

AgenWatch is built around a single core idea:

> **Agent execution must be governable at runtime, not merely observable after the fact.**

This document describes the architectural decisions behind AgenWatch, the guarantees it provides, and the boundaries it intentionally enforces.

---

## Architectural Overview

At its core, AgenWatch is a **bounded execution kernel** that sits between:
- the agent's reasoning loop, and
- the external world (LLMs, tools, APIs).

All execution flows through this kernel.

Nothing executes unless the kernel allows it.

```
Agent (SDK)
    |
    v
Execution Kernel
    ├── Budget Manager
    ├── Iteration Controller
    ├── Retry & Idempotency Layer
    ├── Event Sink (Streaming / Audit)
    |
    v
LLMs / Tools / External APIs
```

The kernel is **authoritative**.  
The agent cannot bypass it.

---

## Deterministic Execution Model

Most agent frameworks rely on implicit LLM memory and conversational drift.  
AgenWatch does not.

AgenWatch enforces determinism through:

- **Explicit execution steps**
- **Recorded decisions**
- **Monotonic state transitions**

Once a step completes successfully, it is recorded and cannot be "hallucinated away" by later reasoning.

Replay does not re-invoke LLMs or tools.  
It replays recorded outcomes.

This guarantees:

- Same inputs → same execution path
- Reproducible debugging
- No hidden side effects during replay

---

## Runtime-Enforced Guardrails (Key Distinction)

Guardrails in AgenWatch are **synchronous and preventative**, not advisory.

### How enforcement works

Before *every* tool call or LLM invocation:

1. The kernel performs a **pre-flight check**:
   - Budget remaining
   - Iteration limit
   - Timeout window
   - Circuit state

2. If any guardrail would be violated:
   - Execution is **terminated immediately**
   - The call never reaches the external API

There is no "best effort" execution.
There is no post-hoc rollback pretending the call didn't happen.

If the kernel says no, the call does not occur.

---

## Budget Manager (v0.1)

The budget system is implemented as a **monotonic ledger with a hard kill switch**.

Properties:

- Charges occur **after successful execution**
- Charging is **idempotent** via operation fingerprinting
- Retries do not double-charge
- Replays do not charge
- No refunds in v0.1

When the budget is exhausted:
- Execution halts
- A terminal `budget_exceeded` state is returned
- No further calls are possible

This provides a **mathematical guarantee of bounded cost**.

---

## Partial State & Failure Philosophy

AgenWatch does **not** attempt automatic rollback of external side effects.

If a tool provisions a resource (e.g., creates a VPC) and execution halts before the next step:

- The system **freezes**
- The final state is recorded deterministically
- The execution is surfaced for inspection

This is intentional.

Automatic rollback of arbitrary external systems is:
- Non-deterministic
- Domain-specific
- Often unsafe

AgenWatch's philosophy is:

> **Freeze and alert, not guess and undo.**

Rollback orchestration belongs in higher-level, domain-aware tooling — not the kernel.

---

## Kernel vs SDK Boundary

AgenWatch strictly separates concerns.

### Kernel (Internal, Unstable)
- Execution loop
- Budget enforcement
- Retry logic
- Circuit breakers
- Deterministic replay machinery

### SDK (Public, Stable)
- `Agent`
- `@tool`
- `ExecutionResult`
- Streaming interface
- LLM provider adapters

The kernel is not exposed.
Users cannot bypass or modify enforcement logic.

This boundary is enforced in code and packaging.

---

## Streaming & Observability

Streaming events are emitted **as a side-channel**.

Important properties:
- Streaming does not affect execution
- Consumers can disconnect safely
- Event order is guaranteed
- Event payloads are informational, not authoritative

Observability exists to **explain what happened**, not to control what happens.

---

## Positioning

AgenWatch is not an agent framework.

It is a **runtime enforcement layer** that can sit underneath:
- LangChain
- CrewAI
- Custom orchestration systems

Think of AgenWatch as the **execution kernel**, not the application layer.

---

## Non-Goals (By Design)

AgenWatch does NOT aim to:
- Provide domain-specific tools
- Replace agent frameworks
- Automatically compensate or rollback side effects
- Optimize prompts or reasoning quality

Those belong elsewhere.

---

## Current Status

- Version: v0.1.0
- Guarantees: Budget, iteration, determinism, replay safety
- Scope: Single-agent, single-process execution
- Stability: Kernel semantics are stable; APIs are intentionally minimal

---

## Closing Principle

> *Agents fail in production not because they are dumb, but because they are ungoverned.*

AgenWatch exists to make agent execution **bounded, inspectable, and enforceable**.



