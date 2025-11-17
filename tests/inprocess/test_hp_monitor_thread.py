import queue
import tempfile
from pathlib import Path

import pytest

from hyperpod_checkpointless_training.inprocess.elastic.hp_agent_event import HPAgentEvent, HPAgentResponse
from hyperpod_checkpointless_training.inprocess.elastic.hp_agent_server_api import AgentServerAPI


class HPAgentServerAPI(AgentServerAPI):
    def __init__(self, test_signal):
        super().__init__()
        self.barrier_queue = queue.Queue()
        self.test_signal = test_signal

    def pop_barrier(self, timeout=1):
        try:
            return self.barrier_queue.get(timeout=timeout)
        except queue.Empty:
            pass

    async def start(self, server, loop):
        while not server.stop_event.is_set():
            await loop.run_in_executor(None, self.pop_barrier)
            await server.broadcast((HPAgentResponse.BARRIER, 1))
            await server.broadcast(self.test_signal)

    def handle_request(self, req):
        if not req:
            return HPAgentResponse.INVALID
        if not isinstance(req, dict):
            return
        event = req.get("event", None)
        if not event:
            return HPAgentResponse.UNKNOWN
        if event == HPAgentEvent.BARRIER:
            return self.handle_barrier(req)

    def handle_barrier(self, req):
        self.barrier_queue.put(req)
        return HPAgentResponse.OK


@pytest.fixture
def addr():
    return Path(tempfile.gettempdir()) / "hp.sock"
