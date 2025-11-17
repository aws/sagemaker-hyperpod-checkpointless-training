import os
import threading
import time

import torch
import torch.distributed as dist

from hyperpod_checkpointless_training.inprocess.abort import AbortTorchDistributed, HPAbortTorchDistributed


def fake_shutdown_process_group_backend(group, device):
    time.sleep(100)


AbortTorchDistributed.shutdown_process_group_backend = (
    fake_shutdown_process_group_backend
)


def worker():
    abort = HPAbortTorchDistributed()
    abort(None, timeout=1)


if __name__ == "__main__":
    # NOTE:
    # This test is trying to simulate NCCL abort hanging. Once NCCL abort
    # hanging, the process should raise a timeout error and exitcode != 0

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    abort = HPAbortTorchDistributed()
    dist.init_process_group(backend="nccl")
    dist.barrier()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # do while loop here rather than thread.join. We want to simulate thread is
    # able to exit the process.
    while True:
        time.sleep(1)
