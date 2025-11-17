# type: ignore
"""Memory-map Config."""

# Standard Library
import os
from dataclasses import dataclass
from typing import Callable

# Third Party
import torch
from torch import distributed as dist
from megatron.core.num_microbatches_calculator import get_num_microbatches


# First Party
from hyperpod_checkpointless_training.dataloader.mmap.cache_read_dataloader import (
    CacheResumeReadDataLoader,
)
from hyperpod_checkpointless_training.dataloader.mmap.prefetched_dataloader import (
    CacheResumePrefetchedDataLoader,
)
from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg

logger = get_logger()


@dataclass
class MMAPConfig:
    """Base Config."""

    pass


@dataclass
class CacheResumeMMAPConfig(MMAPConfig):
    """Cache-resume MMAP Config."""

    cache_dir: str = "/dev/shm/pdl_cache"
    prefetch_length: int = 10
    val_prefetch_length: int = 10
    lookback_length: int = 2
    checkpoint_frequency: int = None
    model_parallel_group: object = None
    enable_batch_encryption: bool = False

    def create(
        self,
        dataloader_init_callable: Callable,
        parallel_state_util,
        step: int,
        is_data_loading_rank: Callable,
        create_model_parallel_group_callable: Callable,
        name: str = "Train",
        is_val: bool = False,
        cached_len: int = 0,
    ):
        """
        Create and return the configured dataloader.

        Args:
            dataloader_init_callable: The dataloader initialization function.
            parallel_state_util: The parallel state utility instance.
            step: The data step to resume from.
            create_model_parallel_group_callable: A callable to the create_model_parallel_group
            name: The name of the dataloader.
            is_val: A boolean used to toogle between train and val. Returns true for val
        Returns:
            An MMAP wrapped dataloader.
        """
        # Validate required parameters
        if step is None:
            raise ValueError("step must be provided.")
        if self.checkpoint_frequency is None:
            logger.warning("Cache performance degraded since checkpoint frequency is not set")

        if self.model_parallel_group is None:
            self.model_parallel_group = create_model_parallel_group_callable()

        # Set up cache directory
        cache_dir_suffix = "val" if is_val else "train"
        cache_dir = os.path.join(self.cache_dir, cache_dir_suffix)
        step = 0 if is_val else step
        force_cold_start = True if is_val else False
        prefetch_length = self.val_prefetch_length if is_val else self.prefetch_length
        synchronized_length = 0

        if cached_len:
            cached_len, step = self._adjust_for_epoch_resume(cached_len, step)

        # Create the appropriate dataloader based on parallel state
        if is_data_loading_rank():
            data_loader = CacheResumePrefetchedDataLoader(
                step=step,
                dataloader_init_callable=dataloader_init_callable,
                model_parallel_group=self.model_parallel_group,
                parallel_state_util=parallel_state_util,
                force_cold_start=force_cold_start,
                model_checkpoint_frequency=self.checkpoint_frequency,
                lookback_length=self.lookback_length,
                prefetch_length=prefetch_length,
                cache_dir=cache_dir,
                enable_batch_encryption=self.enable_batch_encryption,
            )
            if not cached_len:
                data_loader.initialize_data_loader()
                synchronized_length = len(data_loader.data_loader)

        else:
            data_loader = CacheResumeReadDataLoader(
                dataloader_init_callable=dataloader_init_callable,
                model_parallel_group=self.model_parallel_group,
                step=step,
                prefetch_length=prefetch_length,
                parallel_state_util=parallel_state_util,
                force_cold_start=force_cold_start,
                cache_dir=cache_dir,
                enable_batch_encryption=self.enable_batch_encryption,
            )

        if not cached_len:
            length_tensor = torch.tensor(synchronized_length, dtype=torch.long, device='cuda')
            tp_group = parallel_state_util.parallel_state.get_tensor_model_parallel_group()
            dist.all_reduce(length_tensor, op=dist.ReduceOp.MAX, group=tp_group)
            synchronized_length = length_tensor.item()
    
        # Set data_loader length
        data_loader.set_length(max(cached_len, synchronized_length))

        logger.info(
            debug_msg(f"Created {name} {data_loader.__class__.__name__} with length: {len(data_loader)}")
        )
        return data_loader
    
    def _adjust_for_epoch_resume(self, cached_len: int, step: int) -> tuple[int, int]:
        """Adjust cached_len and step when resuming within an epoch or after an epoch."""
        num_microbatches = get_num_microbatches()
        epoch_length = cached_len
        # we need to adjust step in epoch so that DL skips to correct batch after new epoch
        batch_in_epoch = (step * num_microbatches) % epoch_length
        # remaining_batches is normally handled by megatron data sampler but we override the sampler in our example
        # https://github.com/NVIDIA-NeMo/NeMo/blob/55635b9a353daa75f92c329cf71c9017f59ca22b/nemo/collections/common/data/data_samplers.py#L90-L101   
        remaining_batches = epoch_length - batch_in_epoch
        logger.info(debug_msg(f"Original Cached Len: {cached_len} Overriding cached_len to {remaining_batches}, and starting from batch {batch_in_epoch}"))
        return remaining_batches, batch_in_epoch
