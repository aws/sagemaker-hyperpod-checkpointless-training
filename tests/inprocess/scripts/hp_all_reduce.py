import os

import torch
import torch.distributed as dist


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def run(rank, world_size):
    setup()
    tensor = torch.ones(10).to(rank)
    print(f"Rank {rank} starting with tensor: {tensor}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} has tensor: {tensor}")
    cleanup()


def main():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    run(rank, world_size)


if __name__ == "__main__":
    main()
