import gc
import os
import random
import time
from collections.abc import Collection, Mapping
from typing import Any, Callable

import cloudpickle
import numpy as np
import torch
from lightning.pytorch.trainer.states import TrainerFn
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedBase
from megatron.core.tensor_parallel.random import (
    _set_cuda_rng_state,
    get_cuda_rng_tracker,
)
from megatron.core.optimizer import ChainedOptimizer
from torch.distributed.checkpoint._nested_dict import (
    FLATTEN_MAPPING,
    unflatten_state_dict,
)
from torch.distributed.checkpoint._traverse import (
    OBJ_PATH,
    STATE_DICT_ITEM,
    _keep_visiting_tensors,
)
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from nemo.lightning.pytorch.callbacks import PEFT

from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.parameter_update_lock import ParameterUpdateLock
from hyperpod_checkpointless_training.inprocess.tools.memory_tracker import memory_status
from hyperpod_checkpointless_training.inprocess.utils import debug_msg
from hyperpod_checkpointless_training.nemo_plugins.load_balancer import check_available_replica, get_rank_maps
from hyperpod_checkpointless_training.nemo_plugins.memory_checksum import MemoryChecksumManager
from hyperpod_checkpointless_training.nemo_plugins.utils import init_process_group

hp_logger = get_logger()

"""Original Copyright Meta Platforms, Inc. and affiliates under the BSD License"""
"""Modifications Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved"""
def traverse_state_dict(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None],
    keep_traversing: Callable[[STATE_DICT_ITEM], bool] = _keep_visiting_tensors,
) -> None:
    """Invoke ``visitor`` for each value recursively in ``state_dict``.

    This function is migrated from https://github.com/pytorch/pytorch/blob/v2.6.0/torch/distributed/checkpoint/_traverse.py#L36

    The original function will discard keys when dict is empty. For example,

     >>> 'lr_schedulers': [
     ...   {'base_lrs': [0.00010041137340851942, 0.00010041137340851942],
     ...    'last_epoch': 6,
     ...    'verbose': False,
     ...    '_step_count': 7,
     ...    '_get_lr_called_within_step': False,
     ...    '_last_lr': [3.012341202255583e-07, 3.01231107884356e-07],
     ...    'lr_lambdas': [{}, {}]
     ...    }
     ...  ]

     The original function will discard 'lr_lambdas' because it has empty dictionary.
    """

    def _is_terminal(value: STATE_DICT_ITEM) -> bool:
        values: Collection[STATE_DICT_ITEM]
        if isinstance(value, Mapping):
            return False
        elif isinstance(value, list):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if isinstance(value, Mapping):
            # our fix is here. if value is empty, we call visitor.
            if value:
                for k, v in value.items():
                    _traverse_obj(path + (str(k),), v)
            else:
                visitor(path, value)
        elif _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)


def flatten_state_dict(
    state_dict: STATE_DICT_TYPE,
) -> tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    """Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.

    This function is migrated from https://github.com/pytorch/pytorch/blob/v2.6.0/torch/distributed/checkpoint/_nested_dict.py#L32C5-L33C1

    Note that the original implementation may discard elements once the path
    has empty Mapping. In this case, we may restore function peers fail. See
    the example in `traverse_state_dict`.
    """
    flattened = {}
    mappings = {}

    def flat_copy(path: OBJ_PATH, value: Any) -> None:
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    traverse_state_dict(
        state_dict, flat_copy, lambda x: isinstance(x, (torch.Tensor, ShardedBase))
    )
    return flattened, mappings


@torch.no_grad()
def offload_state_dict_to_cpu(state_dict):
    state_dict, mappings = flatten_state_dict(state_dict)
    # Extract the keys whose tensor is a cuda tensors which might be used to do p2p send.
    tensor_meta, _, _ = extract_tensors_from_flatten_state_dict(state_dict)
    tensor_keys = [t[0] for t in tensor_meta]

    for key, value in state_dict.items():
        if isinstance(value, ShardedBase):
            value.data = value.data.cpu()
        if isinstance(value, torch.Tensor):
            state_dict[key] = value.cpu()


    state_dict = unflatten_state_dict(state_dict, mappings)
    return state_dict, tensor_keys


def extract_tensors_from_flatten_state_dict(flatted_checkpoint, keys_to_extract=None):
    """
    Separate tensors from non-tensor data in flattened checkpoint for P2P transfer.

    Splits checkpoint into tensor metadata, actual tensors, and non-tensor data
    to enable efficient distributed transfer during checkpointless recovery.

    Returns:
        tuple: (tensor_meta, tensors, checkpoint_no_tensor) for separate handling
    """
    tensor_meta = []
    tensors = []
    checkpoint_no_tensor = {}
    device = torch.cuda.current_device()

    for key, value in flatted_checkpoint.items():
        if isinstance(value, ShardedBase):
            value = value.data

        # Include CPU tensors if they're critical parameters or in keys_to_extract
        # checkpoint might have device misplacement
        # example: state_dict.module.decoder.layers.0.mlp.router.weight
        should_extract = (keys_to_extract is not None and key in keys_to_extract) or (
            isinstance(value, torch.Tensor) and
            (
                (value.is_cuda and value.get_device() == device) or (not value.is_cuda and ("state_dict" in key or "param_state" in key))
            )
        )

        if should_extract:
            # Store tensor metadata and actual tensor separately
            tensor_meta.append((key, value.size()))
            tensors.append(value)
            checkpoint_no_tensor[key] = None

        elif isinstance(value, torch.Tensor) and value.is_cuda:
            raise ValueError(
                f"Found CUDA tensor that does not belong to current device. Current device {device}, tensor device {value.get_device()}"
            )
        else:
            # Non-tensor data (scalars, configs, etc.)
            checkpoint_no_tensor[key] = value

    return tensor_meta, tensors, checkpoint_no_tensor


def convert_to_saved_to_local_dtype(saved_flatted_state_dict, local_flatted_state_dict):
    """Convert save checkpoint dtype to the local checkpoint dtype if mismatch."""

    for k, local_tensor in local_flatted_state_dict.items():
        if k not in saved_flatted_state_dict or saved_flatted_state_dict[k] is None:
            continue
        if not isinstance(local_tensor, torch.Tensor):
            continue

        if local_tensor.dtype != saved_flatted_state_dict[k].dtype:
            hp_logger.debug(
                debug_msg(
                    f"Converting {k} tensor from type {saved_flatted_state_dict[k].dtype} to type {local_tensor.dtype}"
                )
            )
            new_tensor = saved_flatted_state_dict[k].to(local_tensor.dtype)
            saved_flatted_state_dict[k] = new_tensor


def fill_tensor_back_to_flatten_state_dict_in_place(
    tensor_meta, tensors, checkpoint_no_tensor
):
    if len(tensor_meta) != len(tensors):
        raise ValueError(
            f"tensor_meta and tensors should be 1-1 mapping, getting {len(tensor_meta)} tensor_meta but {len(tensors)} tensors"
        )
    for tensor_meta_data, tensor in zip(tensor_meta, tensors):
        tensor_name = tensor_meta_data[0]
        if tensor.size() != tensor_meta_data[1]:
            raise ValueError(
                f"Mismatch tensor shape for {tensor_name}: meta has {tensor_meta_data[1]} but actual size is {tensor.size()}"
            )
        checkpoint_no_tensor[tensor_name] = tensor


def load_saved_to_local(saved_tensors, local_tensors, mismatching_indexes=None):
    if mismatching_indexes is None:
        mismatching_indexes = []
    local_index = 0
    for saved_index, saved_tensor in enumerate(saved_tensors):
        # Need to make sure local_tensors is up to date with the saved_tensors
        # We will need to use local_tensors to reconstruct the dict to load
        if saved_index in mismatching_indexes:
            local_tensors.insert(saved_index, saved_tensor)
        else:
            local_tensors[local_index].copy_(saved_tensor)
        local_index += 1
    if len(saved_tensors) != len(local_tensors):
        raise RuntimeError(
            f"Mismatching tensors during loading, saved_tensors {len(saved_tensors)}, local_tensors {len(local_tensors)}"
        )


def validate_tensor_meta_match(saved_meta, new_meta, strict=False):
    """
    Validate compatibility between saved and current tensor metadata for P2P recovery.

    Ensures tensor shapes and keys match between healthy and failed ranks to enable
    safe checkpoint transfer during checkpointless recovery.

    Args:
        saved_meta: Tensor metadata from healthy rank's saved checkpoint
        new_meta: Tensor metadata from current rank's model state
        strict: If True, requires exact match; if False, allows subset matching

    Returns:
        list or None: Indexes of mismatching tensors, or None if all match
    """
    if strict:
        # Strict mode: exact match required
        if saved_meta != new_meta:
            saved_meta_set = set(saved_meta)
            new_meta_set = set(new_meta)
            raise ValueError(
                f"Different saved_meta and new_meta, save vs new: {saved_meta_set.difference(new_meta_set)} \n new vs save: {new_meta_set.difference(saved_meta_set)} "
            )
        return None
    else:
        # If there are exact keys in saved tensor meta, it is allowed
        for item in new_meta:
            if item not in saved_meta:
                raise ValueError(f"tensor {item} from new_meta does not exist in {saved_meta}")

        # Create a sublist of saved_meta containing only items that are in new_meta
        saved_meta_subset = [item for item in saved_meta if item in new_meta]

        # Compare the sequence
        if saved_meta_subset != new_meta:
            raise ValueError(
                "The sequence of tensors does not match between the save/load state_dict"
            )

        # Find indexes of items in saved_meta that don't exist in new_meta
        mismatching_indexes = [
            index for index, item in enumerate(saved_meta) if item not in new_meta
        ]

        # Validate that missing tensors are not critical model/optimizer parameters
        for index in mismatching_indexes:
            if (
                "state_dict" in saved_meta[index][0]
                or "param_state" in saved_meta[index][0]
            ):
                raise ValueError(
                    f"Missing model/opt tensor in local state_dict {saved_meta[index][0]}"
                )

        return mismatching_indexes if len(mismatching_indexes) > 0 else None


def remove_model_checkpoint_callbacks(checkpoint):
    """
    Remove ModelCheckpoint callback data from checkpoint to avoid tensor metadata mismatches.

    ModelCheckpoint callbacks are not essential for recovery and can cause tensor metadata
    inconsistencies between saved and local checkpoints during peer-to-peer recovery.

    Args:
        checkpoint: Checkpoint dictionary to clean
    """
    if "callbacks" in checkpoint:
        checkpoint["callbacks"] = {
            k: v for k, v in checkpoint["callbacks"].items()
            if not k.startswith("ModelCheckpoint")
        }
        hp_logger.debug(debug_msg("Removed ModelCheckpoint callbacks from checkpoint"))


class CheckpointManager:
    """
    Manages in-memory checkpoints and peer-to-peer recovery for checkpointless fault tolerance.

    CORE RESPONSIBILITIES:

    1. **In-Memory Checkpoint Management**: Saves and manages NeMo model checkpoints in memory
       for fast recovery without disk I/O during checkpointless recovery scenarios.

    2. **Recovery Feasibility Validation**: Determines if checkpointless recovery is possible
       by validating global step consistency, rank health, and model state integrity.

    3. **Peer-to-Peer Recovery Orchestration**: Coordinates checkpoint transfer between healthy
       and failed ranks using distributed communication for fast recovery.

    4. **RNG State Management**: Preserves and restores random number generator states across
       Python, NumPy, PyTorch, and Megatron for deterministic recovery.
    """

    def __init__(self, enable_checksum: bool = False, enable_offload: bool = False):
        """
        Initialize CheckpointManager with checksum validation and state tracking.

        Args:
            enable_checksum: Enable model state checksum validation for integrity checks
        """
        self.rng_states = None
        self.checksum_manager = MemoryChecksumManager(enable_checksum=enable_checksum)
        self.parameter_update_lock = ParameterUpdateLock()

        # In-memory checkpoint storage
        self._checkpoint = None
        self._checkpoint_keys_to_extract = None
        self.global_step = None

        # Recovery feasibility tracking
        self._checkpointless_recovery_feasible = False
        self.checkpointless_recovery_verified = (
            False  # should be set to false everytime after failure
        )
        self.failed_rank_info = None
        self.global_step_info = None
        self._log_memory_status = False
        self.enable_offload = enable_offload

    def checkpointless_recovery_feasible(
        self, trainer, include_checksum_verification=True
    ):
        # Skip checkpointless recovery. Should only be enabled for testing.
        if os.getenv("IS_CKPT_ONLY", "0") == "1":
            return False

        if not self.checkpointless_recovery_verified:
            self._checkpointless_recovery_feasible = (
                self.validate_checkpointless_restore(
                    trainer, include_checksum_verification=include_checksum_verification
                )
            )
        return self._checkpointless_recovery_feasible

    def reset_checkpointless_recovery_validation(self):
        self.checkpointless_recovery_verified = False
        self.failed_rank_info = None
        self.global_step_info = None

    def save_checkpoint(self, trainer):
        """
        Save NeMo model checkpoint in memory for potential checkpointless recovery.

        Called by CheckpointlessCallback at batch end or during exception handling
        to create recovery points without disk I/O overhead.
        """
        start = time.perf_counter()
        if self._log_memory_status:
            msg, _ = memory_status("before saving")
        else:
            msg = ""
        hp_logger.debug(debug_msg(f"Checkpoint manager saving checkpoint, {msg}"))

        # Create in-memory checkpoint compatible with NeMo format
        self._checkpoint = self.get_nemo_in_memory_checkpoint(trainer)
        self.global_step = trainer.global_step

        if self._log_memory_status:
            msg, _ = memory_status("after saving")
        else:
            msg = ""
        hp_logger.debug(
            debug_msg(
                f"Checkpoint manager finished saving checkpoint, execute time {time.perf_counter() - start}, {msg}"
            )
        )

    def delete_checkpoint(self):
        if self._log_memory_status:
            msg, _ = memory_status("before deleting")
        else:
            msg = ""
        hp_logger.debug(debug_msg(f"Checkpoint manager deleting checkpoint, {msg}"))
        self._checkpoint = None
        self.global_step = None
        self.rng_states = None  # Clear after loading

        # Cleanup cache after delete checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if self._log_memory_status:
            msg, _ = memory_status("after deleting")
        else:
            msg = ""
        hp_logger.info(debug_msg(f"Checkpoint manager finished deleting checkpoint, {msg}"))

    def store_rng_states(self):
        # Python random states
        random_state = random.getstate()
        # Numpy random states
        np_random_state = np.random.get_state()
        # CPU rng state
        cpu_rng_state = torch.get_rng_state().cuda()
        # GPU rng state
        cuda_rng_state = torch.cuda.get_rng_state().cuda()
        # Megatron cuda rng state tracker
        cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
        cuda_rng_state_tracker_states = {
            k: v.cuda() for k, v in cuda_rng_state_tracker.items()
        }

        self.rng_states = [
            random_state,
            np_random_state,
            cpu_rng_state,
            cuda_rng_state,
            cuda_rng_state_tracker_states,
        ]

    def load_rng_states(self):
        """
        Restore all RNG states for deterministic recovery continuation.

        Restores Python, NumPy, PyTorch, and Megatron RNG states to ensure
        training continues with identical random sequences after recovery.
        """
        if self.rng_states is None:
            hp_logger.warning(debug_msg("No RNG states available, skip loading...."))
            return

        # Unpack stored RNG states
        (
            random_state,
            np_random_state,
            cpu_rng_state,
            cuda_rng_state,
            cuda_rng_state_tracker_states,
        ) = self.rng_states

        # Restore all RNG states for deterministic continuation
        random.setstate(random_state)  # Python random
        np.random.set_state(np_random_state)  # NumPy random
        torch.set_rng_state(cpu_rng_state.cpu())  # PyTorch CPU RNG
        _set_cuda_rng_state(cuda_rng_state.cpu())  # PyTorch GPU RNG

        # Restore Megatron tensor parallel RNG tracker
        cuda_rng_state_tracker_states = {
            k: v.cpu() for k, v in cuda_rng_state_tracker_states.items()
        }
        get_cuda_rng_tracker().set_states(cuda_rng_state_tracker_states)

        hp_logger.debug(debug_msg("RNG states loaded...."))

    def maybe_offload_checkpoint(self):
        """Offload checkpoint from GPU to CPU memory if offload is enabled."""
        if self._checkpoint is not None and self.enable_offload:
            hp_logger.debug(debug_msg("Offload checkpoint"))
            self._checkpoint, self._checkpoint_keys_to_extract = offload_state_dict_to_cpu(self._checkpoint)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def distribute_rng(self, src, dst):
        # Distribute py/numpy random states
        py_random_states = self._distribute_non_tensor_checkpoint(
            self.rng_states[:2], src, dst
        )
        self.rng_states[:2] = py_random_states
        # Distribute rng state tensors
        self._transfer_tensors_between_ranks(self.rng_states[2:-1], src, dst)
        # Distribute Megatron rng tracker state tensors
        self._transfer_tensors_between_ranks([self.rng_states[-1]], src, dst)

    @torch.no_grad()
    def get_nemo_in_memory_checkpoint(
        self,
        trainer,
        is_loading=False,
        only_model_weights=False,
    ):
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
        hp_logger.debug(
            debug_msg("start to simulating NeMo checkpoint behavior.......")
        )

        strategy = trainer.strategy
        torch.cuda.synchronize()

        if (
            "optimizer_states" in checkpoint
            and strategy.trainer.state.fn == TrainerFn.FITTING
        ):
            # Clear the optimizer states. This handles the case where ckpt_save_optimizer=False
            # Ideally, the optimizer state dicts should not be generated in this case
            checkpoint["optimizer_states"] = {}

            # replace unsharded optimizer_states with sharded dict.
            # note that if trainer.save_checkpoint(path, save_weights_only=True) is called,
            # the checkpoint will contain only model weights. Optimizer states will be omitted.
            if strategy.ckpt_save_optimizer and not only_model_weights:
                checkpoint["optimizer"] = [
                    strategy.optimizer_sharded_state_dict(is_loading=is_loading)
                ]

        hp_logger.debug(debug_msg("simulating NeMo checkpoint behavior finished...."))

        return checkpoint

    def restore_from_peer(
        self, trainer, rank_maps, local_checkpoint, saved_checkpoint, only_model_weights=False, keys_to_extract=None
    ) -> None:
        """
        Restore checkpoint from healthy peer ranks via P2P transfer.

        Args:
            trainer: PyTorch Lightning trainer instance
            rank_maps: List of (src_rank, dst_rank) tuples for P2P transfer
            local_checkpoint: Current rank's checkpoint structure
            saved_checkpoint: Healthy rank's saved checkpoint (None on failed ranks)
            only_model_weights: If True, restore only model weights without optimizer
            keys_to_extract: Specific checkpoint keys to extract for P2P transfer

        Returns:
            dict or None: Restored checkpoint if successful, None if checksum fails
        """
        pickle_key = "hyper_parameters.config"

        local_flatted_state_dict, _ = flatten_state_dict(local_checkpoint)
        local_tensor_meta, local_tensors, _ = extract_tensors_from_flatten_state_dict(
            local_flatted_state_dict
        )
        cpu_tensor_map = {}

        # Collect saved checkpoint structure on the healthy ranks only
        # On failure ranks, it will be updated in-place through P2P
        # Saved checkpoint will be loaded through the PTL workflow
        if self.parameter_update_lock.is_healthy():
            if saved_checkpoint is None:
                raise ValueError(
                    debug_msg(
                        "Healthy rank does not have checkpoint stored in the manager"
                    )
                )
            hp_logger.debug(debug_msg("Healthy rank self loading checkpoint..."))
            saved_flatted_state_dict, saved_flatten_mapping = flatten_state_dict(saved_checkpoint)

            saved_tensor_meta, saved_tensors, saved_checkpoint_no_tensor = extract_tensors_from_flatten_state_dict(
                saved_flatted_state_dict, keys_to_extract
            )

            mismatching_indexes = validate_tensor_meta_match(saved_tensor_meta, local_tensor_meta)
            # Healthy rank will do in-place load here, while failure rank will load through P2P
            load_saved_to_local(
                saved_tensors, local_tensors, mismatching_indexes=mismatching_indexes
            )

        else:
            saved_tensor_meta = None
            saved_checkpoint_no_tensor = None
            saved_flatten_mapping = None

        for rank_map in rank_maps:
            src_rank, dst_rank = rank_map
            cur_rank = torch.distributed.get_rank()
            if cur_rank in [src_rank, dst_rank]:
                hp_logger.debug(
                    debug_msg(f"Recover checkpoint from [{src_rank}] to [{dst_rank}]")
                )
                # Move CPU tensors to GPU for P2P transfer in src_rank and dst_rank
                # Other ranks does not need p2p comm
                for i in range(len(local_tensors)):
                    if not local_tensors[i].is_cuda:
                        cpu_tensor_map[i] = local_tensors[i]
                        local_tensors[i] = local_tensors[i].cuda()
                hp_logger.debug(debug_msg(f"total {len(cpu_tensor_map)} tensors moved from CPU to GPU"))

                if cur_rank == src_rank:
                    self.distribute_rng(src_rank, dst_rank)
                    # Sender rank does not store any info, simply sending the data to peer
                    if pickle_key in saved_checkpoint_no_tensor:
                        saved_checkpoint_no_tensor[pickle_key] = cloudpickle.dumps(
                            saved_checkpoint_no_tensor[pickle_key]
                        )
                    objects = (
                        saved_tensor_meta,
                        saved_checkpoint_no_tensor,
                        saved_flatten_mapping,
                    )
                    self._distribute_non_tensor_checkpoint(objects, src_rank, dst_rank)

                    self._transfer_tensors_between_ranks(
                        local_tensors,
                        src_rank,
                        dst_rank,
                        async_send=False,
                        skip_scalar=False,
                    )
                else:
                    if self.rng_states is None:
                        # Initialize rng states for spare nodes
                        self.store_rng_states()
                    self.distribute_rng(src_rank, dst_rank)
                    (
                        saved_tensor_meta,
                        saved_checkpoint_no_tensor,
                        saved_flatten_mapping,
                    ) = self._distribute_non_tensor_checkpoint(None, src_rank, dst_rank)
                    mismatching_indexes = validate_tensor_meta_match(
                        saved_tensor_meta, local_tensor_meta
                    )
                    # Fill missing tensors with empties
                    if mismatching_indexes is not None:
                        for index in mismatching_indexes:
                            local_tensors.insert(
                                index,
                                torch.empty(
                                    saved_tensor_meta[index][1],
                                    device=torch.cuda.current_device(),
                                ),
                            )

                    if pickle_key in saved_checkpoint_no_tensor:
                        saved_checkpoint_no_tensor[pickle_key] = cloudpickle.loads(
                            saved_checkpoint_no_tensor[pickle_key]
                        )

                    self._transfer_tensors_between_ranks(
                        local_tensors,
                        src_rank,
                        dst_rank,
                        async_send=False,
                        skip_scalar=False,
                    )
                hp_logger.debug(
                    debug_msg(f"Checkpoint sent from [{src_rank}] to [{dst_rank}]")
                )
        torch.cuda.synchronize()
        hp_logger.debug(debug_msg("Checkpoint send/recv finished"))

        # Only need for faulty ranks, healthy rank are loaded locally
        if not self.parameter_update_lock.is_healthy():
            # Restore original CPU tensor reference
            for i, cpu_tensor in cpu_tensor_map.items():
                cpu_tensor.copy_(local_tensors[i], non_blocking=False)

        checksum_integrity_passed = self.checksum_manager.verify_global_checksum(
            trainer,
            self.parameter_update_lock.param_update_completed,
            self.parameter_update_lock.first_step,
        )
        if not checksum_integrity_passed:
            return None

        self.load_rng_states()

        fill_tensor_back_to_flatten_state_dict_in_place(
            saved_tensor_meta, local_tensors, saved_checkpoint_no_tensor
        )

        checkpoint = unflatten_state_dict(
            saved_checkpoint_no_tensor, saved_flatten_mapping
        )

        # Remove the low p tensors
        del checkpoint["state_dict"]

        # Remove opt state from checkpoint
        if not only_model_weights:
            self._cleanup_optimizer_states(checkpoint, trainer)
        else:
            # For base model recovery, ensure optimizer key exists but is empty
            if "optimizer" not in checkpoint:
                checkpoint["optimizer"] = []
            if "optimizer_states" not in checkpoint:
                checkpoint["optimizer_states"] = {}

        return checkpoint

    def validate_checkpointless_restore(
        self, trainer, include_checksum_verification=True
    ):
        """
        Validate if it is feasible for checkpointless restore, conditions are
        - Global step match
        - There is enough replica to recovery
        - Checkpoint checksum matches

        Returns:
            bool: True if checkpointless restore is valid, False otherwise
        """
        hp_logger.debug(
            debug_msg(
                f"startinng validate_checkpointless_restore with first_step {self.parameter_update_lock.first_step}"
            )
        )

        if not torch.distributed.is_initialized():
            hp_logger.debug(
                debug_msg("init_process_group in validate_checkpointless_restore")
            )
            init_process_group(trainer)

        self.checkpointless_recovery_verified = True
        self.failed_rank_info, self.global_step_info = self.sync_rank_and_step_info(
            trainer
        )
        if not self.failed_rank_info:
            hp_logger.warning(
                debug_msg("No failed nodes detected but there was an in proc restart.")
            )
            # return False

        _, global_step_match = self.validate_global_step(
            self.global_step_info, self.failed_rank_info
        )
        hp_logger.debug(debug_msg(f"validate_global_step match {global_step_match}"))
        if not global_step_match:
            return False

        if not check_available_replica(self.failed_rank_info):
            hp_logger.info(
                debug_msg(
                    "Unable to find replicas to restore, falling back to loading from checkpoint."
                )
            )
            return False

        if include_checksum_verification:
            checksum_integrity_passed = self.checksum_manager.verify_global_checksum(
                trainer,
                self.parameter_update_lock.param_update_completed,
                self.parameter_update_lock.first_step,
            )
        else:
            checksum_integrity_passed = True

        if not checksum_integrity_passed:
            hp_logger.info(
                debug_msg(
                    "checksum missmatch, falling back to loading from checkpoint."
                )
            )
            return False

        hp_logger.info(
            debug_msg("Successfully validated checkpoint-less recovery feasibility")
        )
        return True

    @torch.no_grad()
    def try_checkpointless_load(self, trainer):
        """
        Attempt checkpointless recovery by loading state from peer ranks.

        Main entry point for checkpointless recovery that:
        1. Validates recovery feasibility
        2. Orchestrates peer-to-peer checkpoint transfer
        3. Cleans up in-memory checkpoints

        Returns:
            dict or None: Restored checkpoint if successful, None if fallback needed
        """
        hp_logger.debug(debug_msg("Running try_checkpointless_load"))
        checkpoint = None

        # Attempt checkpointless recovery if feasible
        if self.checkpointless_recovery_feasible(
            trainer, include_checksum_verification=False
        ):
            hp_logger.info(debug_msg("Running checkpointless load."))
            rank_maps = get_rank_maps(self.failed_rank_info.keys())
            # Collect the local checkpoint structure, on all ranks
            local_checkpoint = self.get_nemo_in_memory_checkpoint(
                trainer,
                is_loading=True,
            )
            # Remove ModelCheckpoint callback data to avoid tensor metadata mismatches
            remove_model_checkpoint_callbacks(local_checkpoint)
            checkpoint = self.restore_from_peer(
                trainer, rank_maps, local_checkpoint, self._checkpoint, keys_to_extract=self._checkpoint_keys_to_extract
            )
            hp_logger.info(debug_msg("Checkpointless load finished"))

        # Always clean up in-memory checkpoints after recovery attempt
        self.delete_checkpoint()

        return checkpoint

    def validate_global_step(self, global_step_info, failed_rank_info):
        """
        Validate global steps across the world to ensure all healthy ranks are aligned
        at the same step and return the max of non-zero step. Returns the global step and
        a flag denoting whether or not the global step matches across all ranks.

        Note that we do it early rather than rely on the checkpointless since we want
        data module setup overlap with framework as the setup is async in original dataloader.
        """
        failed_ranks = list(failed_rank_info.keys())
        step_list = [
            step for i, step in enumerate(global_step_info) if i not in failed_ranks
        ]

        # Edge case where all ranks are in failed_ranks
        if len(step_list) == 0 and all(step == 0 for step in global_step_info):
            return 0, True
        elif len(step_list) == 0:
            hp_logger.info(
                debug_msg(
                    "All ranks are failure_ranks, need to restore from checkpoint"
                )
            )
            return max(global_step_info), False

        if all(step_list[0] == step for step in step_list):
            return step_list[0], True
        else:
            hp_logger.info(
                debug_msg("global step mismatch, need to restore from checkpoint")
            )
            return max(step_list), False

    def sync_rank_and_step_info(self, trainer):
        """
        # All gather faulty ranks info from the world
        # the output will look like below, where 2 is the failing rank:
        # {
        #   2: [0, 2, 4, 6],  # Rank 0 is in a data parallel group with ranks 0, 2, 4, 6
        # }
        """

        rank_dict = {}

        if self.global_step is None:
            # Only spare node should enter this as it is starting from scratch
            self.global_step = trainer.global_step

        # collect global step from all ranks
        rank_dict[torch.distributed.get_rank()] = [self.global_step]

        # collect the peers only for the newly introduced spare,
        # self.param_update_completed means we are reusing the same node for restarting
        if not self.parameter_update_lock.is_healthy():
            inter_dist_opt_group = (
                parallel_state.get_inter_distributed_optimizer_instance_group()
            )
            candidate_ranks = torch.distributed.get_process_group_ranks(
                inter_dist_opt_group
            )
            rank_dict[torch.distributed.get_rank()].append(candidate_ranks)

        # Gather rank info from all ranks
        gathered_rank_info = [None] * torch.distributed.get_world_size()
        hp_logger.debug(debug_msg("Gathering Rank info"))
        torch.distributed.all_gather_object(gathered_rank_info, rank_dict)

        rank_info = {}
        global_step_info = []
        for rank_dict in gathered_rank_info:
            for rank, info in rank_dict.items():
                global_step_info.append(info[0])
                # Collect failed rank info if available
                if len(info) > 1:
                    rank_info.update({rank: info[1]})
        hp_logger.debug(
            debug_msg(f"Updated rank info {rank_info}, global step {global_step_info}")
        )

        return rank_info, global_step_info

    def _transfer_tensors_between_ranks(
        self, tensor_checkpoints, src, dst, async_send=True, skip_scalar=True
    ):
        """
        Transfer tensor checkpoints from source rank to destination rank.

        Args:
            tensor_checkpoints: List of tensor checkpoints to transfer
            src: Source rank to transfer from
            dst: Destination rank to transfer to
        """
        cur_rank = torch.distributed.get_rank()
        send_call = torch.distributed.isend if async_send else torch.distributed.send

        for ckpt in tensor_checkpoints:
            if isinstance(ckpt, dict):
                for tensor in ckpt.values():
                    if len(tensor.size()) == 0 and skip_scalar:
                        hp_logger.debug(
                            debug_msg(
                                f"src={src}, dst={dst} getting scalar tensor {tensor}, skipping..."
                            )
                        )
                        continue
                    if cur_rank == src:
                        # Asynchronously send the tensor
                        send_call(tensor, dst)
                    if cur_rank == dst:
                        torch.distributed.recv(tensor, src)
            elif isinstance(ckpt, list):
                for tensor in ckpt:
                    if len(tensor.size()) == 0 and skip_scalar:
                        hp_logger.debug(
                            debug_msg(
                                f"src={src}, dst={dst} getting scalar tensor {tensor}, skipping..."
                            )
                        )
                        continue
                    if cur_rank == src:
                        # Asynchronously send the tensor
                        send_call(tensor, dst)
                    if cur_rank == dst:
                        torch.distributed.recv(tensor, src)
            else:
                if cur_rank == src:
                    # Asynchronously send the tensor
                    send_call(ckpt, dst)
                if cur_rank == dst:
                    torch.distributed.recv(ckpt, src)

    def _distribute_non_tensor_checkpoint(self, non_tensor_checkpoint, src, dst):
        """
        Distribute non-tensor checkpoint data from source rank to destination rank.

        Args:
            non_tensor_checkpoint: Non-tensor checkpoint data to distribute
            src: Source rank to distribute from
            dst: Destination rank to distribute to

        Returns:
            The distributed non-tensor checkpoint data
        """
        cur_rank = torch.distributed.get_rank()
        if cur_rank == dst:
            objects = [None]
        else:
            objects = [non_tensor_checkpoint]
        if cur_rank == src:
            torch.distributed.send_object_list(objects, dst)
        if cur_rank == dst:
            torch.distributed.recv_object_list(objects, src)
        return objects[0]

    def _cleanup_optimizer_states(self, checkpoint, trainer):
        """
        Remove optimizer parameter states from checkpoint to reduce memory usage.

        Handles both ChainedOptimizer state and regular structures.

        ChainedOptimizer will have state_dict like
        {"0": {"optimizer": ...
                "param_state": ...
                "param_state_sharding_type": ...
                }
            "1": ...
        }
        """
        PARAM_STATE_KEYS = ["param_state", "param_state_sharding_type"]

        if not trainer.optimizers:
            raise RuntimeError("No optimizers found in trainer")

        for opt_state in checkpoint["optimizer"]:
            if not any(key in opt_state for key in PARAM_STATE_KEYS):
                for chained_opt_state in opt_state.values():
                    for key in PARAM_STATE_KEYS:
                        chained_opt_state.pop(key, None)
            else:
                for key in PARAM_STATE_KEYS:
                    opt_state.pop(key, None)


class PEFTCheckpointManager(CheckpointManager):
    """
    Manages checkpoints for PEFT (Parameter-Efficient Fine-Tuning) with separate base and adapter handling.

    Extends CheckpointManager to optimize PEFT workflows by:
    - Saving base model weights once and reusing across training
    - Storing only adapter weights in regular checkpoints
    - Supporting separate recovery paths for base model and adapters
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params_to_save = set()
        self.base_model_weights = None
        self.base_model_keys_to_extract = None
        self._base_model_offloaded = False

    def maybe_save_base_model(self, trainer):
        """Save base model weights once, filtering out adapter parameters."""
        # Base model should only be saved once
        if self.base_model_weights is not None:
            return

        msg, _ = memory_status("Before base model saving")
        hp_logger.debug(debug_msg(f"{msg}"))
        self.base_model_weights = self.get_nemo_in_memory_checkpoint(trainer, is_loading=False, only_model_weights=True)
        # self.base_model_weights.pop("callbacks")
        # Filter base model weights
        self.base_model_weights["state_dict"] = {
            k: v for k, v in self.base_model_weights["state_dict"].items() if not self.is_adapter_key(k)
        }
        msg, _ = memory_status("After base modelsaving")
        hp_logger.debug(debug_msg(f"Checkpoint manager finished saving base model checkpoint, {msg}"))

    def maybe_offload_checkpoint(self):
        """Offload base model weights from GPU to CPU; adapter weights are negligible."""
        def _should_offload():
            return self.base_model_weights is not None \
                and self.enable_offload \
                and not self._base_model_offloaded

        if _should_offload():
            self.base_model_weights, self.base_model_keys_to_extract = offload_state_dict_to_cpu(
                self.base_model_weights
            )
            self._base_model_offloaded = True

    @torch.no_grad()
    def try_base_model_checkpointless_load(self, trainer):
        """
        Attempt PEFT base model weights checkpointless recovery by loading state from peer ranks.

        Main entry point for checkpointless recovery that:
        1. Validates recovery feasibility
        2. Orchestrates peer-to-peer checkpoint transfer
        3. Note that we never clean up the base model weights.

        Returns:
            dict or None: Restored checkpoint if successful, None if fallback needed
        """
        start = time.perf_counter()
        hp_logger.info(debug_msg("Running PEFT Base Model checkpointless load"))

        checkpoint = None

        # Attempt checkpointless recovery if feasible
        # Note this is called during resume. The adapter model + optimizers weights are not
        # created at this moment. Thus no filtering needed.
        if self.checkpointless_recovery_feasible(trainer, include_checksum_verification=False):
            rank_maps = get_rank_maps(self.failed_rank_info.keys())
            local_checkpoint = self.get_nemo_in_memory_checkpoint(
                trainer,
                is_loading=True,
                only_model_weights=True,
            )

            checkpoint = self.restore_from_peer(
                trainer,
                rank_maps,
                local_checkpoint,
                self.base_model_weights,
                only_model_weights=True,
                keys_to_extract=self.base_model_keys_to_extract,
            )

            hp_logger.debug(debug_msg(f"Base model checkpointless restore, execute time {time.perf_counter() - start}"))

        # Only clean up if we did not offload the base model
        if not self._base_model_offloaded:
            self.base_model_weights = None

        # Cleanup memory more aggresivelly after delete object
        gc.collect()
        torch.cuda.empty_cache()

        return checkpoint

    def is_adapter_key(self, key):
        """
        Check if state dict key belongs to adapter parameters.

        Args:
            key: State dict key (string or tuple)

        Returns:
            bool: True if key is adapter parameter, False if base model
        """
        if isinstance(key, tuple):
            return key[1].requires_grad
        return key in self.params_to_save or ".adapter." in key or key.endswith(".adapters")

    def save_checkpoint(self, trainer):
        """
        Save NeMo PEFT adapater model checkpoint in memory for potential checkpointless recovery.

        Called by CheckpointlessCallback at batch end or during exception handling
        to create recovery points without disk I/O overhead.
        """
        start = time.perf_counter()
        if self._log_memory_status:
            msg, _ = memory_status("before saving")
        else:
            msg = ""
        hp_logger.debug(debug_msg(f"Checkpoint manager saving checkpoint, {msg}"))

        self.maybe_save_base_model(trainer)

        # Create in-memory checkpoint compatible with NeMo format
        self._checkpoint = self.get_nemo_in_memory_checkpoint(trainer)

        filter_start = time.perf_counter()
        # Update trainer's status
        if self.base_model_weights is not None:
            for k in self._checkpoint.keys():
                if k in ["optimizer", "optimizer_states", "state_dict"]:
                    continue
                self.base_model_weights[k] = self._checkpoint[k]
        # Filter adapter model weights
        self._checkpoint["state_dict"] = {
            k: v for k, v in self._checkpoint["state_dict"].items() if self.is_adapter_key(k)
        }
        hp_logger.debug(
            debug_msg(f"Checkpoint manager Filtering, execute time {time.perf_counter() - filter_start}, {msg}")
        )

        self.global_step = trainer.global_step

        if self._log_memory_status:
            msg, _ = memory_status("after saving")
        else:
            msg = ""
        hp_logger.debug(
            debug_msg(
                f"Checkpoint manager finished saving checkpoint, execute time {time.perf_counter() - start}, {msg}"
            )
        )

    @torch.no_grad()
    def try_checkpointless_load(self, trainer):
        """
        Attempt PEFT adapter weights checkpointless recovery by loading state from peer ranks.

        Main entry point for checkpointless recovery that:
        1. Validates recovery feasibility
        2. Orchestrates peer-to-peer checkpoint transfer
        3. Cleans up in-memory checkpoints

        Returns:
            dict or None: Restored checkpoint if successful, None if fallback needed
        """
        start = time.perf_counter()
        hp_logger.debug(debug_msg("Running PEFT adapter checkpointless load"))
        checkpoint = None

        # Attempt checkpointless recovery if feasible
        if self.checkpointless_recovery_feasible(trainer, include_checksum_verification=False):
            rank_maps = get_rank_maps(self.failed_rank_info.keys())
            # Collect the local checkpoint structure, on all ranks
            local_checkpoint = self.get_nemo_in_memory_checkpoint(
                trainer,
                is_loading=True,
            )

            # Remove ModelCheckpoint callback data to avoid tensor metadata mismatches
            remove_model_checkpoint_callbacks(local_checkpoint)
            # Filter adapter model weights
            local_checkpoint["state_dict"] = {
                k: v for k, v in local_checkpoint["state_dict"].items() if self.is_adapter_key(k)
            }

            checkpoint = self.restore_from_peer(
                trainer, rank_maps, local_checkpoint, self._checkpoint, keys_to_extract=self._checkpoint_keys_to_extract
            )

            if trainer.state.fn == TrainerFn.FITTING:
                # Load optimizer
                orig_ckpt_load_optimizer = trainer.strategy.ckpt_load_optimizer
                trainer.strategy.ckpt_load_optimizer = True
                trainer.strategy.load_optimizer_state_dict(checkpoint, selective_restore=False)
                trainer.strategy.ckpt_load_optimizer = orig_ckpt_load_optimizer
                lr_schedulers = checkpoint["lr_schedulers"]
                for config, lrs_state in zip(trainer.lr_scheduler_configs, lr_schedulers):
                    hp_logger.debug(debug_msg("Loading lr_schedulers for PEFT"))
                    config.scheduler.load_state_dict(lrs_state)

        # Always clean up in-memory checkpoints after recovery attempt
        self.delete_checkpoint()

        hp_logger.debug(
            debug_msg(f"PEFT Adapter model checkpointless restore, execute time {time.perf_counter() - start}")
        )
        return checkpoint
