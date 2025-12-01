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

# type: ignore
"""LLM DataModule - First PyTorch Lightning DataModule with SkipDataLoader."""
# Standard Library
import os
from typing import Callable, List, Optional

# Third Party
import lightning.pytorch as pl
import torch.utils.data
from nemo.collections.llm.gpt.data.utils import _reconfigure_limit_batches
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from torch.utils.data import DataLoader

try:
    from megatron.core import parallel_state
    from megatron.core.num_microbatches_calculator import get_num_microbatches
except (ImportError, ModuleNotFoundError):
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

# First Party
from hyperpod_checkpointless_training.dataloader.skip_dataloader import SkipDataLoader
from hyperpod_checkpointless_training.inprocess.logger import get_logger

from .hf_dataset import HuggingFaceDataset

logger = get_logger(__name__)

DEFAULT_SEED = 42


# All Megatron datamodules require a data sampler attachment. This serves as a workaround
# for that requirement. This helps us eliminate the need for having a mock dataset
#
# For our transform_dataloader passthrough, we assume the user dataloader is already
# properly configured, since transform_dataloader would normally handle sampling.
#
# Note: To integrate with the Megatron sampler, special handling would be required since
# add_megatron_sampler creates a new dataloader from the initialized dataset of the
# passed-in dataloader. We would need to:
# 1. Bypass this call in the normal flow
# 2. In our dataloader_init_callable, call this ourselves with appropriate args to
#    properly hide this dataloader initialization.


# Reference implementation : - NeMo/nemo/collections/vlm/data/data_module.py
class OurDataSampler(MegatronDataSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform_dataloader(self, dataloader: DataLoader) -> DataLoader:
        return dataloader


class LLMDataModule(pl.LightningDataModule):
    """
    First DataModule that creates a dataset from the provided path and wraps
    the train_dataloader with SkipDataLoader.

    This is an example user data module

    When lazy init is enabled,
        we initialize the respective datasets during the train/val_dataloader calls
        rather than setup.
        This is useful when dataset setup is slow and we want to hide that latency with MMAP.
    """

    def __init__(
        self,
        dataset_path: str = "/fsx/datasets/c4/en/hf-tokenized/llama3/train",
        val_dataset_path: str = None,
        micro_batch_size: int = None,
        global_batch_size: int = None,
        seq_length: int = 4096,
        num_workers: int = 4,
        partition: str = "train",
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Callable = None,
        rampup_batch_size: Optional[List[int]] = None,
        lazy_init: bool | None = None,
        keep_in_memory: bool = True,
    ):
        """
        Initialize the LLM DataModule.

        Args:
            dataset_path: Path to the HuggingFace tokenized dataset
            batch_size: Local batch size for the DataLoader (batch size per GPU)
            micro_batch_size: Micro batch size for gradient accumulation
            global_batch_size: Global batch size across all GPUs
            seq_length: Sequence length for Megatron data sampler
            num_workers: Number of worker processes for data loading
            partition: Dataset partition to load (e.g., "train", "validation")
            pin_memory: Whether to pin memory in DataLoader
            shuffle: Whether to shuffle the dataset
            drop_last: Whether to drop the last incomplete batch
            skip_batches: Number of batches to skip at the beginning
            collate_fn: Collate function for batches
            rampup_batch_size: Optional rampup batch size for Megatron
            lazy_init: Enable lazy dataset initialization
                If True, the dataset is lazily initialized.
                If None, defaults to the `DM_LAZY_INIT` environment variable if set,
                otherwise True if unset.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.val_dataset_path = val_dataset_path
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.seq_length = seq_length
        self.num_workers = num_workers
        self.partition = partition
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.rampup_batch_size = rampup_batch_size
        self.keep_in_memory = keep_in_memory
        if lazy_init is not None:
            self.lazy_init = lazy_init
        else:
            self.lazy_init = int(os.environ.get("DM_LAZY_INIT", 1)) > 0

        # use val dataset by default if no path provided
        if self.val_dataset_path is None:
            self.val_dataset_path = self.dataset_path

        # Will be initialized in setup() if not lazy_init, otherwise in *_dataloader()
        self.train_dataset = None
        self.val_dataset = None

        self.data_sampler = OurDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

        logger.debug(f"Initialized MegatronDataSampler: {self.data_sampler}")

    def _get_dp_rank_and_size(self):
        """
        Calculate dp_rank and dp_size using parallel_state directly.

        Returns:
            Tuple of (dp_rank, dp_size)
        """

        dp_size = parallel_state.get_data_parallel_world_size()
        dp_rank = parallel_state.get_data_parallel_rank()

        return dp_rank, dp_size

    def _setup_train_dataset(self):
        """
        Initialize the training dataset.

        Returns:
            HuggingFaceDataset
        """
        return HuggingFaceDataset(
            input_path=self.dataset_path,
            partition=self.partition,
            seq_length=self.seq_length,
            keep_in_memory=self.keep_in_memory,
        )

    def _setup_val_dataset(self):
        """
        Initialize the validation dataset.

        Returns:
            HuggingFaceDataset
        """
        return HuggingFaceDataset(
            input_path=self.val_dataset_path,
            partition=self.partition,
            seq_length=self.seq_length,
            keep_in_memory=self.keep_in_memory,
        )

    def setup(self, stage: Optional[str] = None):
        """
        Setup the datasets for training and validation.

        If lazy init is enabled,
            setup is skipped and datasets will be initialized in train/val_dataloader.

        Args:
            stage: Stage of training ('fit', 'validate', 'test', or 'predict')
        """
        if self.lazy_init:
            logger.info("Lazy initialization enabled. Skipping setup.")
            return
        if stage == "fit" or stage is None:
            logger.info(f"Setting up datasets from path: {self.dataset_path}")

            self.train_dataset = self._setup_train_dataset()

            if self.val_dataset_path == self.dataset_path:
                self.val_dataset = self.train_dataset
            else:
                self.val_dataset = self._setup_val_dataset()

            logger.info(
                f"Setup complete. Train dataset size: {len(self.train_dataset)}"
            )

        if stage == 'validate':

            logger.info(f"Setting up datasets from path: {self.val_dataset_path}")

            self.val_dataset = self._setup_val_dataset()

            logger.info(
                f"Setup complete. Val dataset size: {len(self.val_dataset)}"
            )


    def train_dataloader(self, target_load_step: int = 0) -> SkipDataLoader:
        """
        Create the training DataLoader using SkipDataLoader with distributed sampling.

        If lazy init is enabled, will initialize the dataset.

        Args:
            target_load_step: The step to which the dataloader should load to.
                MMAP will pass this value to this callable. User must implement this functionality.

        Returns:
            SkipDataLoader that inherits from DataLoader
        """
        if self.lazy_init and self.train_dataset is None:
            self.train_dataset = self._setup_train_dataset()

        if self.train_dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        dp_rank, dp_size = self._get_dp_rank_and_size()

        # Reason: add_megatron_sampler will create a new dataloader and a new batch sampler
        # that is aware of the dp_rank/size. Since we override transform and so
        # add_megatron_sampler is not called, we need to handle this on our end,
        # and create one, passing it into our dataloader.
        # Reference: - NeMo/nemo/lightning/data.py

        # Create DistributedSampler
        sampler = torch.utils.data.DistributedSampler(
            self.train_dataset,
            shuffle=self.shuffle,
            seed=DEFAULT_SEED,
            rank=dp_rank,
            num_replicas=dp_size,
        )

        logger.debug(
            f"Created DistributedSampler: dp_rank={dp_rank}, dp_size={dp_size}, shuffle={self.shuffle}"
        )

        # Create SkipDataLoader directly (it inherits from DataLoader)
        # Here we pass in the micro_batch_size so that we can mimic the dataloader length as defined by megatron
        # Length is defined as num_samples / (MBS * DP_size)
        # https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/lightning/data.py#L306
        return SkipDataLoader(
            dataset=self.train_dataset,
            sampler=sampler,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            _skip_batches=target_load_step,
        )

    def val_dataloader(self, target_load_step: int = 0) -> SkipDataLoader:
        """
        Create the validation DataLoader.

        If lazy init is enabled, will initialize the dataset.

        Args:
            target_load_step: The step to which the dataloader should load to.
                MMAP will pass this value to this callable, generally 0.
                User must implement this functionality.

        Returns:
            Standard DataLoader for validation
        """
        if self.lazy_init and self.val_dataset is None:
            self.val_dataset = self._setup_val_dataset()

        if self.val_dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        return SkipDataLoader(
            dataset=self.val_dataset,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False,  # Don't shuffle validation data
            drop_last=False,  # Don't drop last batch for validation
            _skip_batches=target_load_step,
        )

    # Recongiure limit batches and Megatron Data sampler are functions which help us to calculate the
    # num_microbatches and reconfigure the trainer.limit_train_batches and trainer.limit_val_batches
    # in terms of num_microbatches
    def reconfigure_limit_batches(self):
        """
        Reconfigure trainer.limit_train_batches and trainer.limit_val_batches in terms of num of microbatches.
        """
        if self.trainer is None:
            logger.warning("Trainer not set, skipping limit batches reconfiguration")
            return

        # Override limit_train_batches in terms of num of microbatches
        self.trainer.limit_train_batches = _reconfigure_limit_batches(
            self.trainer.limit_train_batches, self.train_dataset
        )
        # Override limit_val_batches to be a multiple of num microbatches to prevent val_step from exiting
        #   in between a step
        self.trainer.limit_val_batches = _reconfigure_limit_batches(
            self.trainer.limit_val_batches, self.val_dataset
        )

        # Override num sanity steps to be a multiple of num of microbatches
        self.trainer.num_sanity_val_steps *= get_num_microbatches()
