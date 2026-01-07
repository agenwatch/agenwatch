# AgenWatch

**Deterministic, runtime-governed execution for AI agents.**

AgenWatch provides a low-level execution kernel that enforces hard limits
(budget, iterations, time) **during execution**, not after failure.

This answers a single production-critical question:

> *"Can I mathematically guarantee this agent will stop?"*

---

## Why AgenWatch Exists

Most agent frameworks focus on *reasoning quality* and *observability*.
AgenWatch focuses on **governance**.

We built AgenWatch for teams who need:
- Hard execution guarantees
- Deterministic, replayable behavior
- Protection against runaway agents in production

---

## What Makes AgenWatch Different

- **Runtime-enforced guardrails**  
  Budget, iteration, and timeout limits are enforced synchronously by the kernel.

- **Deterministic execution**  
  Same inputs → same execution path → same outcome.

- **Kernel-first architecture**  
  Governance is not optional and cannot be bypassed from the SDK layer.

AgenWatch is intentionally minimal.  
It is designed to be a **foundation**, not a framework zoo.

---

## Status

- PyPI: https://pypi.org/project/agenwatch/
- Language: Python 3.10+
- Maturity: v0.1.0 (stable core, focused scope)

---

## Repositories

- **agenwatch** → Core SDK and execution kernel  
  https://github.com/agenwatch/agenwatch

---

## Philosophy

AgenWatch is built for **production agents**, not demos.

If you need smarter agents, there are many tools.  
If you need **governable agents**, this is where the conversation starts.


