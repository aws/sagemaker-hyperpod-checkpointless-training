import argparse
import importlib
import logging
import os
import pathlib
import random
import time
from typing import Optional

import torch
from hyperpod_elastic_agent.ipc import InProcessRestartSocketClient
from hp_inprocess.health_check import CudaHealthCheck
from hp_inprocess.wrap import HPCallWrapper, HPWrapper
from megatron.core import parallel_state
from megatron.core.distributed.distributed_data_parallel import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
)
from megatron.core.transformer import TransformerConfig

from hyperpod_checkpointless_training.inprocess.elastic.hp_agent_api import HPAgentK8sAPI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inprocess Restart Optimal Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--size",
        default=64,
        type=int,
        help="model hidden size",
    )
    parser.add_argument(
        "--layers",
        default=4,
        type=int,
        help="number of layers",
    )
    parser.add_argument(
        "--log-interval",
        default=10,
        type=int,
        help="logging interval",
    )
    parser.add_argument(
        "--chkpt-interval",
        default=5,
        type=int,
        help="checkpointing interval",
    )
    parser.add_argument(
        "--total-iterations",
        default=50,
        type=int,
        help="total training iterations",
    )
    parser.add_argument(
        "--seed",
        default=5678,
        type=int,
        help="random seed, time-based if None",
    )
    parser.add_argument(
        "--path",
        default="/tmp/",
        type=str,
        help="directory for the checkpoint file",
    )
    parser.add_argument(
        "--trace_file_path",
        default=None,
        type=str,
        help="directory for the tracefile file",
    )
    parser.add_argument(
        "--fault-prob",
        default=0.02,
        type=float,
        help="fault injection probability",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="device",
    )
    parser.add_argument(
        "--log-level",
        type=lambda s: logging._nameToLevel[s.upper()],
        default=logging.INFO,
        help="logging level",
    )

    return parser.parse_args()


args = parse_args()


class HPAgentK8sAPIFactory:
    def __call__(self):
        client = InProcessRestartSocketClient()
        return HPAgentK8sAPI(client)


class Trainer:
    def __init__(self, args):
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.args = args
        self.device = self.setup_device(args)
        self.model = self.setup_model(args, self.device)
        self.optim = self.setup_optim(self.model)
        self.checkpoint = None
        self.base_store = None
        random.seed(self.args.seed * self.world_size + self.rank)

    def setup_device(self, args):
        if args.device == "cuda":
            torch.cuda.set_device(self.local_rank)
            return torch.device("cuda")
        return torch.device("cpu")

    def setup_base_store(self):
        return torch.distributed.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=int(os.environ["MASTER_PORT"]) + 1,
            world_size=int(os.environ["WORLD_SIZE"]),
            is_master=(int(os.environ["RANK"]) == 0),
            multi_tenant=True,
            wait_for_workers=True,
            use_libuv=True,
        )

    def setup_model(self, args, device):
        size = args.size
        layers = args.layers
        layers = [torch.nn.Linear(size, size) for _ in range(layers)]
        return torch.nn.Sequential(*layers).to(device)

    def setup_optim(self, model):
        return torch.optim.Adam(model.parameters(), lr=1e-5)

    @property
    def backend(self):
        return "nccl" if self.args.device == "cuda" else "gloo"

    @HPWrapper(
        health_check=CudaHealthCheck(),
        hp_api_factory=HPAgentK8sAPIFactory(),
        abort_timeout=60.0,
        trace_file_path=args.trace_file_path,
    )
    def fit(self, call_wrapper: Optional[HPCallWrapper] = None):
        iteration = 0
        call_wrapper.step_upon_restart = 0
        torch.cuda.set_device(self.local_rank)
        s = time.perf_counter()
        if not self.base_store:
            self.base_store = self.setup_base_store()

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        store = torch.distributed.PrefixStore(
            str(call_wrapper.seq.get()), self.base_store
        )
        torch.distributed.init_process_group(
            self.backend,
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        importlib.reload(parallel_state)
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

        # FIXME: This is a workaround solution to solve potential NCCL hanging
        # issue after restart. If some ranks fail before the first NCCL calls,
        # other healthy ranks won't be able to be aborted by fault_handling_thread.
        # Therefore, we call a barrier right after the initialization to prevent
        # from the above case.
        torch.distributed.barrier()
        torch.cuda.synchronize()
        print(
            f"[RANK:{self.rank}] init_process_group duration={time.perf_counter() - s}"
        )
        total_iterations = self.args.total_iterations
        ddp_config = DistributedDataParallelConfig()
        self.ddp = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=self.args.layers),
            ddp_config,
            self.model,
        )

        checkpoint = self.load()

        if checkpoint:
            self.model.load_state_dict(checkpoint["model"])
            self.optim.load_state_dict(checkpoint["optim"])
            iteration = checkpoint["iteration"]
            print(f"Load checkpoint {iteration=}, rank={self.rank=}")

        chkpt_interval = self.args.chkpt_interval
        for iteration in range(iteration, total_iterations):
            self.run(iteration, call_wrapper)
            call_wrapper.step_upon_restart += 1
            # save checkpoint
            if iteration % chkpt_interval == chkpt_interval - 1:
                if self.rank == 0:
                    checkpoint = {
                        "model": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                        "iteration": iteration,
                    }
                    print(f"Save checkpoint {iteration=}, rank={self.rank=}")
                    with call_wrapper.atomic_lock:
                        self.save(checkpoint)

    def run(self, iteration, call_wrapper):
        size = self.args.size
        rank = self.rank
        inp = torch.rand(size, size).to(self.device)

        # forward/backward/optimize
        self.model.zero_grad()
        out = self.ddp(inp)
        loss = out.square().mean()
        loss.backward()

        # simulate fail
        if iteration >= 3 and random.random() < self.args.fault_prob:
            raise RuntimeError(f"example fault at {iteration=} from {rank=}")

        torch.distributed.barrier()
        with call_wrapper.atomic_lock:
            self.optim.step()

        loss.item()
        print(iteration, f"{rank=} {iteration=} {loss.item()=}")

    def interval_print0(self, iteration, *a, **kw):
        rank = self.rank
        log_interval = self.args.log_interval
        if rank == 0 and iteration % log_interval == log_interval - 1:
            print(*a, **kw)

    def load(self):
        checkpoint_path = pathlib.Path(self.args.path) / "checkpoint.pt"
        if checkpoint_path.exists():
            return torch.load(checkpoint_path)

    def save(self, checkpoint):
        checkpoint_path = pathlib.Path(self.args.path) / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)


def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=args.log_level,
    )
    logging.info(f"{args}")
    trainer = Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
