import asyncio
import tempfile
from pathlib import Path
from typing import Any

import torch.distributed.elastic.utils.store as store_util
from torch.distributed import TCPStore
from torch.distributed.elastic.utils.logging import get_logger

from .hp_agent_event import HPAgentEvent, HPAgentResponse
from .hp_const import (
    HP_AGENT_BARRIER_PREFIX,
    HP_AGENT_FAILURE_PREFIX,
    HP_AGENT_STORE_BARRIER_PREFIX,
)

logger = get_logger(__name__)

DEFAULT_AGENT_ADDRESS = Path(tempfile.gettempdir()) / "hprun.sock"


def get_default_agent_addr():
    return DEFAULT_AGENT_ADDRESS


class AgentServerAPI:
    async def start(self, server, loop) -> None:
        raise NotImplementedError("start not implemented")

    def handle_request(self, req) -> Any:
        raise NotImplementedError("handle_request not implemented")


class HPAgentServerAPI(AgentServerAPI):
    def __init__(self):
        self.master_addr = None
        self.master_port = None
        self.group_rank = None
        self.global_world_size = None
        self.group_world_size = None
        self.local_world_size = None
        self.global_ranks = None
        self.store = None
        self.interval = 1.0
        self.seq = 0

    def setup(self):
        assert self.master_addr is not None
        assert self.master_port is not None
        assert self.group_rank is not None
        assert self.global_ranks is not None
        assert self.global_world_size is not None
        assert self.group_world_size is not None
        assert self.local_world_size is not None
        self.store = TCPStore(self.master_addr, self.master_port)
        keys = [HP_AGENT_FAILURE_PREFIX, HP_AGENT_BARRIER_PREFIX]
        vals = ["", ""]
        self.store.multi_set(keys, vals)

    async def start(self, server, loop) -> None:
        while not server.stop_event.is_set():
            keys = [HP_AGENT_FAILURE_PREFIX, HP_AGENT_BARRIER_PREFIX]
            data = self.store.multi_get(keys)
            await self.notify_failure(data[0], server)
            await self.notify_barrier(data[1], server, loop)
            await asyncio.sleep(self.interval)

    def reset_store(self, keys):
        if self.group_rank == 0:
            self.store.multi_set(keys, [""] * len(keys))

    def store_barrier(self):
        self.seq += 1
        key_prefix = f"{HP_AGENT_STORE_BARRIER_PREFIX}/{self.seq}"
        world_size = self.group_world_size
        store_util.barrier(self.store, world_size, key_prefix)

    async def notify_failure(self, data, server):
        try:
            if not data:
                return
            await server.broadcast((HPAgentResponse.FAILURE, self.seq))
        except Exception as e:
            logger.error(f"notify failure fail. {e}")

    async def notify_barrier(self, data, server, loop):
        try:
            data = data.decode().split(",")
            ranks = set([int(r) for r in data if r.isdigit()])
            if len(ranks) != self.global_world_size:
                return
            self.store_barrier()
            self.reset_store([HP_AGENT_BARRIER_PREFIX, HP_AGENT_FAILURE_PREFIX])
            logger.info("Agent barrier pass: ranks {ranks}")
            await server.broadcast((HPAgentResponse.BARRIER, self.seq))
        except Exception as e:
            logger.error(f"notify barrier fail. {e}")

    def handle_request(self, req) -> Any:
        if not isinstance(req, dict):
            return HPAgentResponse.INVALID

        event = req.get("event", None)
        if not event:
            return HPAgentResponse.UNKNOWN
        if event == HPAgentEvent.NOOP:
            return self.handle_noop(req)
        if event == HPAgentEvent.PING:
            return self.handle_ping(req)
        if event == HPAgentEvent.BARRIER:
            return self.handle_barrier(req)
        if event == HPAgentEvent.FAILURE:
            return self.handle_failure(req)

    def handle_noop(self, req):
        return HPAgentResponse.OK

    def handle_ping(self, req):
        return HPAgentResponse.PONG

    def handle_failure(self, req):
        rank = req.get("rank", None)
        if rank is None:
            return HPAgentResponse.INVALID
        seq = req.get("seq", -1)
        if seq < self.seq:
            logger.warning(
                f"seq({seq}) < current seq({self.seq}). Receive previous failure after restart."
            )
            return HPAgentResponse.OK
        try:
            self.store.append(HP_AGENT_FAILURE_PREFIX, f"{rank},")
        except Exception as e:
            logger.error(f"store.set({HP_AGENT_BARRIER_PREFIX}) fail. {e}")
            return HPAgentResponse.ERROR
        return HPAgentResponse.OK

    def handle_barrier(self, req):
        rank = req.get("rank", None)
        if rank is None:
            return HPAgentResponse.INVALID
        try:
            self.store.append(HP_AGENT_BARRIER_PREFIX, f"{rank},")
        except Exception as e:
            logger.error(f"store.set({HP_AGENT_BARRIER_PREFIX}) fail. {e}")
            return HPAgentResponse.ERROR
        return HPAgentResponse.OK
