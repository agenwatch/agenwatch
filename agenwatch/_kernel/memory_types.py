from enum import Enum

class MemoryType(str, Enum):
    EPISODIC = "episodic"
    DECLARATIVE = "declarative"
    LONG_TERM = "long_term"
    WORKING = "working"
    REPAIR = "repair"

__INTERNAL__ = True



