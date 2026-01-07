# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-01-07

### Added
- Initial public release
- Runtime-enforced budget limits with kernel-level kill switches
- Deterministic agent execution with replay support
- Self-healing tool execution with automatic retry and repair
- Provider support: OpenAI, Anthropic, Groq, LiteLLM (100+ providers)
- Clean SDK API: `Agent`, `tool`, `AgentConfig`, `ExecutionResult`
- Fingerprint-based deduplication (no double-charging on retries)
