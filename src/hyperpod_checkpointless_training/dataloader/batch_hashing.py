# type: ignore

# Standard Library
import hashlib
import io
from typing import Callable

# Third Party
import numpy as np
import torch
from torch import Tensor


def _create_checksum_str(payload: bytes, hash_func: Callable = hashlib.md5) -> str:
    """
    Creates a checksum string from the specified bytes using the given hash function.

    Args:
        payload: The bytes data to create a checksum for
        hash_func: The hash function to use (default: hashlib.md5)

    Returns:
        Hex string representation of the checksum
    """
    return hash_func(payload, usedforsecurity=False).hexdigest()


def tensor_to_hash(tensor, hash_func: Callable = hashlib.md5):
    """
    Converts a tensor to a deterministic checksum string.

    Args:
        tensor: The PyTorch tensor to hash
        hash_func: The hash function to use (default: hashlib.md5)

    Returns:
        Hex string representation of the tensor hash
    """
    # if bfloat16, cast the tensor to float before converting to numpy
    # need to do this because numpy doesn't support bf16
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()

    np_array = tensor.numpy()
    buffer = io.BytesIO()
    np.save(buffer, np_array, allow_pickle=False)
    serialized_data = buffer.getvalue()
    md5_hash = _create_checksum_str(serialized_data, hash_func)
    return md5_hash


def batch_to_hash(data, prefix=""):
    """
    Recursively converts a batch (dict, list, or tensor) to a deterministic checksum string.

    Args:
        data: The data to hash (can be dict, list, tensor, or scalar)
        prefix: String prefix for nested key naming (default: "")

    Returns:
        Hex string representation of the batch hash
    """
    hashes = []
    if isinstance(data, Tensor):
        hashes.append(f"{prefix}=" + tensor_to_hash(data))
    elif isinstance(data, list) and data:
        for i, item in enumerate(data):
            key = f"{prefix}.{i}" if prefix else str(i)
            hashes.append(batch_to_hash(item, key))
    elif isinstance(data, dict) and data:
        for key in sorted(data.keys()):
            _key = f"{prefix}.{key}" if prefix else key
            hashes.append(batch_to_hash(data[key], _key))
    else:
        hashes.append(f"{prefix}=" + str(data))
    return _create_checksum_str(str(hashes).encode("utf-8"))


def compute_batch_with_hash(batch, step) -> str:
    """
    Computes the hash of a batch including the global step for ordering validation.

    Args:
        batch: The training batch data to hash
        step: The global training step number

    Returns:
        Hex string representation of the batch hash with step information
    """
    batch_with_step = {"global_step": step, "batch_data": batch}
    return batch_to_hash(batch_with_step)
