import os
import queue

from hyperpod_elastic_agent.ipc import (
    HyperPodIPCException,
    InProcessRestartSocketClient,
    RestartMode,
)

from ..logger import get_logger
from ..utils import debug_msg
from .hp_agent_event import HPAgentEvent, HPAgentResponse

logger = get_logger(__name__)


class HPAgentK8sAPI:
    def __init__(self, client: InProcessRestartSocketClient):
        self.client = client
        self.seq = 0

    def validate_env(self, env, seq):
        self.validate_world_size(env, seq)
        self.validate_rank(env, seq)

    def validate_rank(self, env, seq):
        assert "RANK" in env, f"RANK not found in {env}"
        rank = int(env["RANK"])
        current_rank = int(os.environ["RANK"])
        logger.info(
            debug_msg(
                f"Receive RANK={rank} and Environment RANK: {current_rank}",
                rank=current_rank,
                seq=seq,
            )
        )

    def validate_world_size(self, env, seq):
        assert "WORLD_SIZE" in env, f"WORLD_SIZE not found in {env}"
        world_size = int(env["WORLD_SIZE"])
        current_world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        logger.info(
            debug_msg(
                f"Receive RANK={world_size} and Environment RANK: {current_world_size}",
                rank=rank,
                seq=seq,
            )
        )

    def set_env(self, env, seq):
        self.validate_env(env, seq)
        for k, v in env.items():
            os.environ[k] = v

    def hyperpod_barrier(self, rank, timeout=None, **kw):
        resp = self.client.hyperpod_barrier()
        action, body = resp
        assert action == "job_start", "unrecognize action: {action}"
        assert "worker_envs" in body
        env = body.get("worker_envs")
        seq = env.get("JOB_RESTART_COUNT", None)
        assert seq is not None, f"cannot find sequence number from {env=}"
        self.seq = int(seq)
        rank = int(os.environ["RANK"])
        logger.info(
            debug_msg(f"hyperpod_barrier response: {resp}", rank=rank, seq=self.seq)
        )
        self.set_env(env, self.seq)
        # send RunAck to the HyperPod infra
        self.client.hyperpod_past_rcb_barrier()
        return self.seq

    def hyperpod_wait(self, timeout=None):
        try:
            resp = self.client.hyperpod_wait_fault(timeout=timeout)
        except HyperPodIPCException:
            # From ipc implementation: https://code.amazon.com/packages/HyperPodElasticAgent/blobs/ee6370f224d0ca23c270c3da0561f65165e95a70/--/src/amzn_hyper_pod_elastic_agent/ipc/client.py?__session=eyJwa2ciOiJIeXBlclBvZEVsYXN0aWNBZ2VudCIsInRhYnMiOlt7ImZpbGUiOiJzcmMvYW16bl9oeXBlcl9wb2RfZWxhc3RpY19hZ2VudC9oeXBlcnBvZF9lbGFzdGljX2FnZW50LnB5In0seyJmaWxlIjoic3JjL2Ftem5faHlwZXJfcG9kX2VsYXN0aWNfYWdlbnQvaXBjL2NsaWVudC5weSNMMjIzIn0seyJmaWxlIjoic3JjL2Ftem5faHlwZXJfcG9kX2VsYXN0aWNfYWdlbnQvaXBjL19faW5pdF9fLnB5In1dfQ%3D%3D#L223
            # once the ipc client hit timeout, it will raise a HyperPodIPCException
            # exception.
            return HPAgentResponse.OK

        assert isinstance(resp, tuple)
        action, body = resp
        rank = int(os.environ["RANK"])
        logger.info(
            debug_msg(f"hyperpod_wait got resp {resp}", rank=rank, seq=self.seq)
        )
        if action == "job_fault":
            return HPAgentResponse.FAILURE, self.seq

        logger.warning(f"unrecognize response {resp=}")
        return HPAgentResponse.UNKNOWN

    def hyperpod_send(
        self, event: HPAgentEvent, rank: int, plr_restart: bool = False, *a, **kw
    ):
        restart_mode = (
            RestartMode.PROCESS_LEVEL_RESTART
            if plr_restart
            else RestartMode.IN_PROCESS_RESTART
        )
        logger.info(
            debug_msg(
                f"sending fault to agent for restart mode: {restart_mode}",
                rank=rank,
                seq=self.seq,
            )
        )
        self.client.hyperpod_send_fault(rank, restart_mode)

    def hyperpod_notify_labels(self, labels: dict[str, str]):
        self.client.hyperpod_notify_labels(labels)

    def hyperpod_wait_rank_info(self):
        return self.client.hyperpod_wait_rank_info()

    def __enter__(self):
        return self

    def __exit__(self, *e):
        pass
