# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""Memory tracker."""

import gc
import os
from typing import Any, Tuple

try:
    import psutil
    process = psutil.Process(os.getpid())
    base_mem_usage = process.memory_info().data
    last_mem_usage = base_mem_usage
except: # noqa
    process = None
    base_mem_usage = None
    last_mem_usage = None

import torch
from torch.distributed._shard.sharded_tensor import ShardedTensor

# pylint: disable=global-statement
dtype_to_bit = {
    torch.float32: 32,
    torch.float64: 64,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
    torch.bool: 1,
}


_GB = 1024**3
_FORMAT = "7.4f"


def memory_status(  # pylint: disable=too-many-locals
    tag: str = "",
    reset_max: bool = True,
    sync: bool = True,
    writers: Tuple[Any] = (),
) -> Tuple[float]:
    """Memory status gpu."""
    rank = int(os.getenv("RANK", -1))
    local_rank = torch.cuda.current_device()
    seq = int(os.getenv("JOB_RESTART_COUNT", 0))

    if sync:
        torch.cuda.synchronize()

    free_memory, total_memory = torch.cuda.mem_get_info()
    total_used_str = f"Raw: {free_memory / (1024**3):.2f} free / {total_memory / (1024**3):.2f} total GB."

    # Convert to GB for printing.
    alloced = torch.cuda.memory_allocated(device=local_rank) / _GB
    max_alloced = torch.cuda.max_memory_allocated(device=local_rank) / _GB
    cached = torch.cuda.memory_reserved(device=local_rank) / _GB
    max_cached = torch.cuda.max_memory_reserved(device=local_rank) / _GB

    msg = (
        f"[GPU MEMORY] (torch, rank, device) = ({torch.__version__}, {rank}, {local_rank}),"
        f" (alloc, max_alloc, cache, max_cache) = ({alloced:{_FORMAT}}, {max_alloced:{_FORMAT}},"
        f" {cached:{_FORMAT}}, {max_cached:{_FORMAT}}) GB. {total_used_str} [{tag:10s}]"
    )

    if reset_max:
        torch.cuda.reset_peak_memory_stats()

    usage = {
        "allocated": alloced,
        "max_allocated": max_alloced,
        "max_reserved": max_cached,
        "reserved": cached,
    }
    for writer in writers:
        writer.add_scalars(f"GPUMemoryGB/{tag}", usage, seq)

    return msg, usage


def recursive_iter_with_visited(obj, visited):
    """
    Recusively tracking all the references of a given object
    This is useful to understand what object is holding a tensor from release
    """
    refs = gc.get_referrers(obj)
    print(f"refs {id(refs)} all ids {[id(ref) for ref in refs]}")
    visited.add(id(refs))
    for ref in refs:
        if id(ref) not in visited:
            visited.add(id(ref))
            if isinstance(ref, torch.Tensor):
                print(
                    f"Find CYCLE tensor reference {id(ref)}, dtype {ref.dtype} device {ref.device}, size {ref.size()}, requires_grad {ref.requires_grad}, dict {ref.__dict__}"
                )
                continue
            if isinstance(ref, list):
                for item in ref:
                    try:
                        print(
                            f"list item type {type(item)}, item {item}, id {id(item)}"
                        )
                    except:  # noqa
                        print(f"list item type {type(item)}")
            else:
                try:
                    print(f"item type {type(ref)}, item {ref}, id {id(ref)}")
                except:  # noqa
                    print(f"item type {type(ref)}")
            recursive_iter_with_visited(ref, visited)


def gc_collect():
    while gc.collect():
        pass


def dump_gpu_tensor_reference(tensor_ids, tag):
    gc_collect()
    objects = gc.get_objects()
    gpu_tensors = [
        obj for obj in objects if is_meaningful_tensor(obj) and id(obj) in tensor_ids
    ]
    visited = set()
    visited.add(id(gpu_tensors))
    visited.add(id(objects))
    for tensor in gpu_tensors:
        print(
            f"[{tag}][TENSOR] dtype {tensor.dtype} device {tensor.device}, size {tensor.size()}, requires_grad {tensor.requires_grad},  dict {tensor.__dict__}"
        )
        recursive_iter_with_visited(tensor, visited)


def is_meaningful_tensor(obj):
    try:
        if isinstance(obj, ShardedTensor):
            return False
        return isinstance(obj, torch.Tensor) and obj.is_cuda and obj.requires_grad
    except:
        pass


def get_tensor_ids():
    gc_collect()
    objects = gc.get_objects()
    gpu_tensors_ids = [id(obj) for obj in objects if is_meaningful_tensor(obj)]
    return set(gpu_tensors_ids)


def memory_status_cpu(  # pylint: disable=too-many-locals
    tag: str = "", writers: Tuple[Any] = ()
) -> Tuple[float]:
    """Memory status cpu."""
    rank = int(os.getenv("RANK", -1))
    local_rank = torch.cuda.current_device()
    seq = int(os.getenv("JOB_RESTART_COUNT", 0))

    global last_mem_usage
    global base_mem_usage  # pylint: disable=global-variable-not-assigned

    gc_collect()

    objects = gc.get_objects()
    tensors = [
        obj for obj in objects if isinstance(obj, torch.Tensor) and not obj.is_cuda
    ]
    torch_usage = 0
    for t in tensors:  # pylint: disable=invalid-name
        torch_usage += t.numel() * dtype_to_bit[t.dtype]
    # total_usage = psutil.virtual_memory()[3] # This will get the total usage for all processes
    current_usage = process.memory_info().data
    total_usage = current_usage - base_mem_usage
    usage_change = current_usage - last_mem_usage
    last_mem_usage = current_usage

    torch_usage /= _GB
    total_usage /= _GB
    usage_change /= _GB
    base_usage = base_mem_usage / _GB

    msg = f"[CPU MEMORY]@{seq:04d} \
        (torch, rank, device) = ({torch.__version__}, {rank}, {local_rank}), \
        (torch tensor, mem, change since last measurement, base) = ({torch_usage:{_FORMAT}}, \
        {total_usage:{_FORMAT}}, {usage_change:{_FORMAT}}, {base_usage:{_FORMAT}}): \
        {tag}"

    usage = {
        "base": base_usage,
        "delta": usage_change,
        "torch": torch_usage,
        "total": total_usage,
    }
    for writer in writers:
        writer.add_scalars(f"CPUMemoryGB/{tag}", usage, usage)

    return msg, usage
