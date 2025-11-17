import struct

HEADER_TYPE = "!I"
HEADER_SIZE = struct.calcsize(HEADER_TYPE)

# store prefix
HP_AGENT_BARRIER_PREFIX = "hyperpod/agent/barrier"
HP_AGENT_FAILURE_PREFIX = "hyperpod/agent/failure"
HP_AGENT_STORE_BARRIER_PREFIX = "hyperpod/agent/store/barrier"
