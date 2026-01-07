"""
Context Window Management + Memory Tiering
Deterministic, observable, production-safe
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum
import tiktoken
from threading import Lock


# ============================================================
# ENUMS
# ============================================================

class MemoryTier(Enum):
    WORKING = "working"
    SESSION = "session"
    LONGTERM = "longterm"


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ============================================================
# MESSAGE MODEL
# ============================================================

@dataclass
class Message:
    role: MessageRole
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tier: MemoryTier = MemoryTier.WORKING

    def to_llm_dict(self) -> Dict[str, Any]:
        msg = {"role": self.role.value, "content": self.content}
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


# ============================================================
# TRIM RESULT
# ============================================================

@dataclass
class TrimResult:
    messages: List[Message]
    tokens_before: int
    tokens_after: int
    messages_dropped: int
    messages_summarized: int
    trim_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.trim_reason}: "
            f"{self.tokens_before} → {self.tokens_after} tokens | "
            f"dropped={self.messages_dropped}, summarized={self.messages_summarized}"
        )


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ContextConfig:
    model_name: str = "claude-sonnet-4-20250514"
    max_context_tokens: int = 180000

    preserve_system: bool = True
    preserve_last_n_user: int = 3
    preserve_tool_outputs: bool = True

    trim_strategy: Literal["sliding_window", "oldest_first"] = "sliding_window"
    summarize_threshold: int = 0  # 0 = disabled

    enable_session_memory: bool = True
    enable_longterm_memory: bool = False

    hard_token_limit: Optional[int] = None


# ============================================================
# TOKEN COUNTER
# ============================================================

class TokenCounter:
    def __init__(self, model_name: str):
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_message(self, m: Message) -> int:
        tokens = 4
        tokens += len(self.encoder.encode(m.content))
        if m.tool_calls:
            for tc in m.tool_calls:
                tokens += len(self.encoder.encode(str(tc)))
        return tokens

    def count_messages(self, messages: List[Message]) -> int:
        return sum(self.count_message(m) for m in messages)


# ============================================================
# MESSAGE TRIMMER
# ============================================================

class MessageTrimmer:
    def __init__(self, config: ContextConfig):
        self.config = config
        self.counter = TokenCounter(config.model_name)

    def trim(self, messages: List[Message]) -> TrimResult:
        tokens_before = self.counter.count_messages(messages)

        if self.config.hard_token_limit and tokens_before > self.config.hard_token_limit:
            raise ValueError("Hard token limit exceeded")

        if tokens_before <= self.config.max_context_tokens:
            return TrimResult(
                messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                messages_dropped=0,
                messages_summarized=0,
                trim_reason="no_trim"
            )

        if self.config.trim_strategy == "sliding_window":
            return self._sliding_window(messages, tokens_before)

        return self._oldest_first(messages, tokens_before)

    def _sliding_window(self, messages: List[Message], tokens_before: int) -> TrimResult:
        preserved = []
        preserved_ids = set()

        def keep(m):
            preserved.append(m)
            preserved_ids.add(id(m))

        if self.config.preserve_system:
            for m in messages:
                if m.role == MessageRole.SYSTEM:
                    keep(m)

        if self.config.preserve_tool_outputs:
            for m in messages:
                if m.role == MessageRole.TOOL:
                    keep(m)

        user_msgs = [m for m in messages if m.role == MessageRole.USER][-self.config.preserve_last_n_user:]

        for u in user_msgs:
            idx = messages.index(u)
            keep(u)
            if idx + 1 < len(messages) and messages[idx + 1].role == MessageRole.ASSISTANT:
                keep(messages[idx + 1])

        trimmed = preserved.copy()
        dropped = 0

        for m in reversed(messages):
            if id(m) in preserved_ids:
                continue
            test = [m] + trimmed
            if self.counter.count_messages(test) <= self.config.max_context_tokens:
                trimmed.insert(0, m)
            else:
                dropped += 1

        order = {id(m): i for i, m in enumerate(messages)}
        trimmed.sort(key=lambda m: order[id(m)])

        tokens_after = self.counter.count_messages(trimmed)

        return TrimResult(
            messages=trimmed,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            messages_dropped=dropped,
            messages_summarized=0,
            trim_reason="sliding_window"
        )

    def _oldest_first(self, messages: List[Message], tokens_before: int) -> TrimResult:
        trimmed = []
        dropped = 0

        for m in reversed(messages):
            test = [m] + trimmed
            if self.counter.count_messages(test) <= self.config.max_context_tokens:
                trimmed.insert(0, m)
            else:
                dropped += 1

        tokens_after = self.counter.count_messages(trimmed)

        return TrimResult(
            messages=trimmed,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            messages_dropped=dropped,
            messages_summarized=0,
            trim_reason="oldest_first"
        )


# ============================================================
# MEMORY TIERS
# ============================================================

class MemoryTierManager:
    def __init__(self, config: ContextConfig):
        self.config = config
        self.session_memory: List[Message] = []

    def archive(self, messages: List[Message], reason: str):
        if not self.config.enable_session_memory:
            return

        existing = {id(m) for m in self.session_memory}
        for m in messages:
            if id(m) in existing:
                continue
            self.session_memory.append(
                Message(
                    role=m.role,
                    content=m.content,
                    tool_calls=m.tool_calls,
                    metadata={**m.metadata, "archived_reason": reason},
                    tier=MemoryTier.SESSION
                )
            )


# ============================================================
# CONTEXT MANAGER (MAIN API)
# ============================================================

class ContextManager:
    def __init__(self, config: ContextConfig):
        self.config = config
        self.trimmer = MessageTrimmer(config)
        self.memory = MemoryTierManager(config)
        self.trim_history: List[TrimResult] = []
        self._lock = Lock()

    def prepare_context(self, messages: List[Message]) -> TrimResult:
        with self._lock:
            result = self.trimmer.trim(messages)
            dropped = [m for m in messages if m not in result.messages]
            self.memory.archive(dropped, reason=result.trim_reason)
            self.trim_history.append(result)
            return result

    def to_llm(self, messages: List[Message]) -> List[Dict[str, Any]]:
        return [m.to_llm_dict() for m in messages]

    def stats(self) -> Dict[str, Any]:
        if not self.trim_history:
            return {}
        return {
            "trims": len(self.trim_history),
            "tokens_saved": sum(r.tokens_before - r.tokens_after for r in self.trim_history),
            "last": self.trim_history[-1].summary()
        }

__INTERNAL__ = True



