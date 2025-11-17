from enum import Enum, auto


class HPAgentEvent(Enum):
    NOOP = auto()
    PING = auto()
    BARRIER = auto()
    FAILURE = auto()
    UNKNOWN = auto()


class HPAgentResponse(Enum):
    OK = auto()
    PONG = auto()
    BARRIER = auto()
    FAILURE = auto()
    ERROR = auto()
    INVALID = auto()
    UNKNOWN = auto()
