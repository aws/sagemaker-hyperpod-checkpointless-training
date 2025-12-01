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

import hashlib

import torch
import torch.distributed
from megatron.core.optimizer import ChainedOptimizer

from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg
from hyperpod_checkpointless_training.nemo_plugins.utils import get_sharded_tensor_states

hp_logger = get_logger()


class MemoryChecksumManager:
    """
    Manager for computing and verifying checksums of model parameters and optimizer states.

    Attributes:
        enable_checksum (bool): Whether checksum verification is enabled
        memory_checksum (str): The stored checksum of the memory state
    """

    def __init__(self, enable_checksum=False):
        self.enable_checksum = enable_checksum
        self.memory_checksum = None

    @torch.no_grad()
    def _compute_checksum(self, trainer) -> str:
        """
        Compute a checksum of the model parameters and optimizer state.

        This method computes a SHA-256 hash of the model parameters and optimizer states,
        aligning with the state handling in distribute_checkpoint() from restore.py.
        (There is some duplication of code here, and intentionally not refactoring it till
        we gather some more data and to keep this change manageable)

        The hash includes:
        1. Model parameters from the sharded state dict
        2. Optimizer tensor states

        The checksum is computed with the following logic, we will measure and optimize further
        Another potential idea is to take samples at predefined strides to keep it smaller, but
        we will measure with this approach.
        1. Converting each tensor to CPU numpy array
        2. Converting the numpy array to bytes
        3. Updating a running SHA-256 hash with these bytes

        Args:
            trainer: The PyTorch Lightning trainer instance

        Returns:
            str: A hexadecimal string representing the checksum
        """
        hasher = hashlib.sha256()

        # Add param/optimizer states to checksum
        for optimizer in trainer.optimizers:
            # Handle ChainedOptimizer case
            if isinstance(optimizer.mcore_optimizer, ChainedOptimizer):
                optimizers = optimizer.mcore_optimizer.chained_optimizers
            else:
                optimizers = [optimizer.mcore_optimizer]

            # Process optimizer tensor states using the same approach as distribute_checkpoint
            for opt in optimizers:
                # Get sharded tensor states
                opt_param_tensors, opt_state_tensors = get_sharded_tensor_states(opt)
                for _, param in opt_param_tensors.items():
                    if torch.is_tensor(param):
                        param_data = param.cpu().numpy().tobytes()
                        hasher.update(param_data)
                opt_state_tensors = [
                    tensor
                    for subdict in opt_state_tensors.values()
                    for tensor in subdict.values()
                ]
                for param in opt_state_tensors:
                    if torch.is_tensor(param):
                        param_data = param.cpu().numpy().tobytes()
                        hasher.update(param_data)
        return hasher.hexdigest()

    def store_checksum(self, trainer):
        """
        Store the current memory state checksum if checksum is enabled.

        This method computes a checksum of the current model parameters and optimizer states
        and stores it for later verification. The checksum is only computed and stored if
        checksum verification is enabled.

        This should be called at the end of a training loop, after parameter updates have been
        completed, to capture the state that should be preserved across restarts.

        Args:
            trainer: The PyTorch Lightning trainer instance
        """
        if self.enable_checksum:
            self.memory_checksum = self._compute_checksum(trainer)

    def _verify_local_checksum(
        self, trainer, param_update_completed, first_step
    ) -> bool:
        """
        This method checks if the current state of model parameters and optimizer states
        matches the checksum that was stored at the end of the previous training loop.

        The verification is only performed if:
        1. Checksum verification is enabled
        2. Parameter update was completed (i.e., we're in a state where the checksum should be valid)

        If the verification fails, it indicates that the GPU memory state may be corrupted,
        and the system should fall back to loading from checkpoint.

        Args:
            trainer: The PyTorch Lightning trainer instance
            param_update_completed: Whether parameter update was completed in the previous iteration

        Returns:
            bool: True if checksum verification passed, False otherwise
        """

        # If checksum is not enabled skip verification
        if not self.enable_checksum:
            return True  # skip verification

        # If parameter update was not completed, then it means failure happened before optimizer
        # step stage. Hence, the healthy node would have completed. Nothing to verify
        if not param_update_completed:
            return True  # skip verification

        # For node replacement, the newly come in nodes should skip the check.
        if first_step:
            return True

        cur_rank = torch.distributed.get_rank()

        # If memory_checksum is None, verification fails
        if self.memory_checksum is None:
            hp_logger.warning(
                debug_msg(
                    f"Memory checksum is None on rank {cur_rank} but enable_checksum is True. "
                    f"This may indicate a problem with the checksum computation or storage."
                )
            )
            hp_logger.info(
                debug_msg(
                    "Missing memory checksum, falling back to loading from checkpoint."
                )
            )
            return False

        # Compute current checksum and compare with stored checksum
        current_checksum = self._compute_checksum(trainer)
        if current_checksum != self.memory_checksum:
            hp_logger.warning(
                debug_msg(
                    f"Memory checksum mismatch on rank {cur_rank}. "
                    f"Expected: {self.memory_checksum}, Got: {current_checksum}. "
                    f"GPU memory state may be corrupted."
                )
            )
            hp_logger.info(
                debug_msg(
                    "Checksum mismatch detected, falling back to loading from checkpoint."
                )
            )
            return False

        return True

    def verify_global_checksum(
        self, trainer, param_update_completed, first_step
    ) -> bool:
        """
        Verify that the memory state matches the previously recorded checksum across all ranks.

        If any rank fails the verification, all ranks will fall back to loading from checkpoint
        to maintain consistency across the distributed training process. See collective below.

        Args:
            trainer: The PyTorch Lightning trainer instance
            param_update_completed: Whether parameter update was completed in the previous iteration

        Returns:
            bool: True if checksum verification passed on all ranks, False otherwise
        """

        if not self.enable_checksum:
            return True  # skip verification

        # First verify the local memory checksum
        local_checksum_passed = self._verify_local_checksum(
            trainer, param_update_completed, first_step
        )
        hp_logger.debug(debug_msg(f"Local checksum passed {local_checksum_passed}"))
        ## IMPORTANT: DO NOT EARLY RETURN HERE. Every rank must go through allreduce, to tell
        ## everyone else that it encounered an error.

        # Perform a collective operation to ensure all ranks passed the checksum verification
        # Convert boolean to tensor for all_reduce operation
        device = torch.cuda.current_device()
        local_tensor = torch.tensor([1 if local_checksum_passed else 0], device=device)

        # Use all_reduce with MIN operation (logical AND across all ranks)
        # The default PG should have all ranks
        torch.distributed.all_reduce(local_tensor, op=torch.distributed.ReduceOp.MIN)

        # Convert back to boolean - if any rank failed (0), the result will be 0
        global_checksum_passed = bool(local_tensor.item())

        # Log if this rank passed but others failed
        cur_rank = torch.distributed.get_rank()
        if local_checksum_passed and not global_checksum_passed:
            hp_logger.info(
                debug_msg(
                    f"Rank {cur_rank} passed checksum verification, but some other rank failed. "
                    f"Falling back to loading from checkpoint for consistency across all ranks."
                )
            )
        hp_logger.debug(debug_msg(f"global checksum passed {global_checksum_passed}"))

        return global_checksum_passed
