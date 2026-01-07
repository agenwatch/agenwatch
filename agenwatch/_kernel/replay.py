"""
Deterministic Replay System (Production-Grade)
Exact execution recording and replay without LLM calls
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Set
from datetime import datetime
from enum import Enum
import json
import hashlib
import threading
from pathlib import Path
from agenwatch._kernel.observability import EventType

# =========================
# ENUMS
# =========================

class EventType(Enum):
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE_CHANGE = "state_change"
    DECISION = "decision"
    ERROR = "error"


class ReplayMode(Enum):
    STRICT = "strict"
    LENIENT = "lenient"
    DEBUG = "debug"


# =========================
# DATA MODELS
# =========================

@dataclass
class ExecutionEvent:
    event_id: str
    event_type: EventType
    timestamp: datetime
    step_number: int
    data: Dict[str, Any] = field(default_factory=dict)
    parent_event_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "step_number": self.step_number,
            "data": self.data,
            "parent_event_id": self.parent_event_id,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionEvent":
        return cls(
            event_id=d["event_id"],
            event_type=EventType(d["event_type"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            step_number=d["step_number"],
            data=d.get("data", {}),
            parent_event_id=d.get("parent_event_id"),
            correlation_id=d.get("correlation_id"),
        )


@dataclass
class ExecutionLog:
    execution_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    events: List[ExecutionEvent] = field(default_factory=list)
    agent_config: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def add_event(self, event: ExecutionEvent):
        self.events.append(event)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "events": [e.to_dict() for e in self.events],
            "agent_config": self.agent_config,
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def load(cls, path: Path) -> "ExecutionLog":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            execution_id=data["execution_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            events=[ExecutionEvent.from_dict(e) for e in data["events"]],
            agent_config=data.get("agent_config", {}),
            success=data.get("success", True),
            error_message=data.get("error_message"),
        )


# =========================
# EXECUTION RECORDER
# =========================

class ExecutionRecorder:
    def __init__(self, execution_id: str, agent_config: Dict[str, Any]):
        self.log = ExecutionLog(
            execution_id=execution_id,
            started_at=datetime.now(),
            agent_config=agent_config,
        )
        self._lock = threading.Lock()
        self.step = 0
        self.correlation_id: Optional[str] = None

    def start_correlation(self, correlation_id: str):
        self.correlation_id = correlation_id

    def _eid(self, prefix: str) -> str:
        raw = f"{prefix}:{self.step}:{datetime.now().isoformat()}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _hash(self, obj: Any) -> str:
        return hashlib.sha256(
            json.dumps(obj, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def record_llm_request(self, messages, model, tools=None) -> str:
        with self._lock:
            eid = self._eid("llm_req")
            self.log.add_event(
                ExecutionEvent(
                    event_id=eid,
                    event_type=EventType.LLM_REQUEST,
                    timestamp=datetime.now(),
                    step_number=self.step,
                    correlation_id=self.correlation_id,
                    data={
                        "messages": messages,
                        "model": model,
                        "tools": tools,
                        "fingerprint": self._hash({"m": messages, "mo": model, "t": tools}),
                    },
                )
            )
            return eid

    def record_llm_response(self, request_id, response):
        with self._lock:
            self.log.add_event(
                ExecutionEvent(
                    event_id=self._eid("llm_resp"),
                    event_type=EventType.LLM_RESPONSE,
                    timestamp=datetime.now(),
                    step_number=self.step,
                    parent_event_id=request_id,
                    correlation_id=self.correlation_id,
                    data={"response": response},
                )
            )
            self.step += 1

    def record_tool_call(self, tool_name, arguments) -> str:
        with self._lock:
            eid = self._eid("tool_call")
            self.log.add_event(
                ExecutionEvent(
                    event_id=eid,
                    event_type=EventType.TOOL_CALL,
                    timestamp=datetime.now(),
                    step_number=self.step,
                    correlation_id=self.correlation_id,
                    data={
                        "tool": tool_name,
                        "args": arguments,
                        "args_hash": self._hash(arguments),
                    },
                )
            )
            return eid

    def record_tool_result(self, call_id, result):
        with self._lock:
            self.log.add_event(
                ExecutionEvent(
                    event_id=self._eid("tool_result"),
                    event_type=EventType.TOOL_RESULT,
                    timestamp=datetime.now(),
                    step_number=self.step,
                    parent_event_id=call_id,
                    correlation_id=self.correlation_id,
                    data={"result": result},
                )
            )

    def record_state(self, state: Dict[str, Any]):
        with self._lock:
            self.log.add_event(
                ExecutionEvent(
                    event_id=self._eid("state"),
                    event_type=EventType.STATE_CHANGE,
                    timestamp=datetime.now(),
                    step_number=self.step,
                    correlation_id=self.correlation_id,
                    data={
                        "state": state,
                        "state_hash": self._hash(state),
                    },
                )
            )

    def finalize(self):
        self.log.ended_at = datetime.now()


# =========================
# REPLAY ENGINE
# =========================

class ReplayEngine:
    def __init__(self, log: ExecutionLog, mode: ReplayMode = ReplayMode.STRICT):
        self.log = log
        self.mode = mode
        self._lock = threading.Lock()
        self._used_llm: Set[str] = set()
        self._used_tools: Set[str] = set()

        self._llm_index: Dict[str, List[ExecutionEvent]] = {}
        self._tool_index: Dict[str, List[ExecutionEvent]] = {}
        self._build_indexes()

    def _build_indexes(self):
        for e in self.log.events:
            if e.event_type == EventType.LLM_REQUEST:
                fp = e.data["fingerprint"]
                self._llm_index.setdefault(fp, []).append(e)

            elif e.event_type == EventType.TOOL_CALL:
                key = f"{e.data['tool']}:{e.data['args_hash']}"
                self._tool_index.setdefault(key, []).append(e)

    def replay_llm(self, messages, model, tools=None) -> Dict[str, Any]:
        fp = hashlib.sha256(
            json.dumps({"m": messages, "mo": model, "t": tools}, sort_keys=True).encode()
        ).hexdigest()[:16]

        with self._lock:
            for req in self._llm_index.get(fp, []):
                if req.event_id not in self._used_llm:
                    self._used_llm.add(req.event_id)
                    for e in self.log.events:
                        if e.parent_event_id == req.event_id and e.event_type == EventType.LLM_RESPONSE:
                            return e.data["response"]

        raise ValueError("LLM replay mismatch")

    def get_llm_response(self):
        """
        Replay mode: return a deterministic no-op LLM response.
        Budget must NOT be mutated.
        """
        return {
            "text": "<final>\n(replay)\n</final>",
            "instrumentation": {
                "replay": True
            }
        }
    
    
    def replay_tool(self, tool_name, arguments):
        args_hash = hashlib.sha256(
            json.dumps(arguments, sort_keys=True).encode()
        ).hexdigest()[:16]
        key = f"{tool_name}:{args_hash}"

        with self._lock:
            for call in self._tool_index.get(key, []):
                if call.event_id not in self._used_tools:
                    self._used_tools.add(call.event_id)
                    for e in self.log.events:
                        if e.parent_event_id == call.event_id:
                            return e.data["result"]

        raise ValueError("Tool replay mismatch")

__INTERNAL__ = True



