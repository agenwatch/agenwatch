# AgenWatch Architecture

AgenWatch is built on a single, non-negotiable principle:

> **Agent execution must be governed at runtime, not merely observed after failure.**

This document explains how AgenWatch enforces that principle, what guarantees it provides, and—equally important—what it intentionally does *not* attempt to solve.

---

## 1. Architectural Role

AgenWatch is a **runtime execution kernel** for AI agents.

It sits **between**:
- the agent’s reasoning loop, and
- the external world (LLMs, tools, APIs).

All execution passes through this kernel.  
Nothing executes unless the kernel explicitly allows it.

```
User / Framework Logic
         |
         v
  Agent (SDK surface)
         |
         v
AgenWatch Execution Kernel
  ├─ Budget Manager
  ├─ Iteration Controller
  ├─ Retry & Idempotency Layer
  ├─ Event & Audit Sink
         |
         v
LLMs / Tools / External APIs
```

The kernel is **authoritative**.  
The agent cannot bypass it.

---

## 2. Deterministic Execution Model

Most agent frameworks rely on conversational context and LLM memory, which naturally drift over time.

AgenWatch does not.

AgenWatch enforces determinism through:

- **Explicit execution steps**
- **Recorded decisions**
- **Monotonic state transitions**

Once a step completes, its outcome is recorded and becomes immutable for that execution.

An agent cannot:
- “forget” a previous step
- reason its way backward
- retroactively alter history

### Deterministic Replay (v0.1)

Replay in AgenWatch is **non-executing**:

- LLMs are not re-called
- Tools are not re-invoked
- Recorded outcomes are replayed for inspection

This enables:
- reproducible debugging
- post-mortem analysis
- exact reconstruction of failure states

Replay is **read-only** in v0.1.

---

## 3. Runtime-Enforced Guardrails (Core Differentiator)

Guardrails in AgenWatch are **synchronous and preventative**, not advisory.

They are enforced **before execution**, not after failure.

### Enforcement Flow

Before *every* LLM call or tool invocation:

1. The kernel performs a **pre-execution check**:
   - remaining budget
   - iteration count
   - timeout window
   - circuit state

2. If any guardrail would be violated:
   - execution is terminated immediately
   - the call never reaches the external system

There is:
- no “best effort” execution
- no warning-only mode
- no post-hoc rollback pretending the call didn’t happen

If the kernel denies the call, **the call does not occur**.

---

## 4. Budget Manager & Bounded Overrun (v0.1)

Budget enforcement in AgenWatch is implemented as a **monotonic execution ledger** with a hard stop.

### Properties

- Costs are charged **after successful execution**
- Charging is **atomic and idempotent**
- Retries do **not** double-charge
- Replay does **not** charge
- No refunds in v0.1

### Bounded Overrun Guarantee

Exact token cost is unknown until an LLM finishes responding.  
AgenWatch addresses this with a **bounded overrun policy**:

- The kernel blocks calls *between* steps
- At most **one in-flight call** can exceed the remaining budget
- Once charged, execution halts deterministically

This guarantees:
- no runaway loops
- no unbounded cost escalation
- mathematically bounded spending

---

## 5. Partial State & Failure Philosophy

AgenWatch does **not** attempt automatic rollback of external side effects.

If an execution halts midway through a multi-step task:

- the final state is frozen
- the execution outcome is recorded deterministically
- the system surfaces the halted state for inspection

This is intentional.

Automatic rollback of arbitrary external systems is:
- domain-specific
- non-deterministic
- often unsafe

AgenWatch’s philosophy is:

> **Freeze and alert — do not guess and undo.**

Rollback orchestration belongs in higher-level, domain-aware systems.

---

## 6. Kernel vs SDK Boundary

AgenWatch enforces a strict separation of concerns.

### Kernel (Internal, Unstable)
- execution loop
- budget enforcement
- retry and idempotency logic
- circuit breakers
- deterministic replay machinery

### SDK (Public, Stable)
- `Agent`
- `@tool`
- `ExecutionResult`
- streaming interface
- LLM provider adapters

The kernel is not exposed.  
Users cannot bypass or modify enforcement logic.

This boundary is enforced in code, packaging, and public exports.

---

## 7. Streaming & Inspection

AgenWatch emits execution events as a **side-channel**.

Important characteristics:
- streaming does not affect execution
- consumers can disconnect safely
- event order is guaranteed
- payloads are informational, not authoritative

Streaming exists to **explain what happened**, not to control what happens.

---

## 8. Positioning

AgenWatch is **not** an agent framework.

It is a **runtime enforcement layer** that can sit beneath:
- LangChain
- CrewAI
- LangGraph
- custom orchestration systems

Those systems handle *what* an agent should do.  
AgenWatch governs *whether* it is allowed to continue.

---

## 9. Non-Goals (By Design)

AgenWatch does not attempt to:
- provide domain-specific tools
- replace agent frameworks
- automatically roll back side effects
- sandbox the operating system
- optimize prompts or reasoning quality

These concerns belong outside the kernel.

---

## 10. Current Scope & Status

- Version: **v0.1.0**
- Scope: single-agent, single-process execution
- Guarantees:
  - runtime budget enforcement
  - iteration limits
  - deterministic termination
  - replay-safe inspection
- Stability:
  - kernel semantics are stable
  - public API is minimal and intentionally frozen

---

## Closing Principle

> *Agents fail in production not because they are unintelligent,  
> but because they are ungoverned.*

AgenWatch exists to make agent execution **bounded, inspectable, and enforceable**.

