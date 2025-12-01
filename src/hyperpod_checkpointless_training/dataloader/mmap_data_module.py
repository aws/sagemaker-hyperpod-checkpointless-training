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
"""Checkpointless DataModule - Wrap Train DataLoader with MMAP."""

# Standard Library
from typing import Optional, Callable
from functools import partial

# Third Party
import lightning.pytorch as pl
from torch.utils.data import DataLoader

# First Party
from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.dataloader.config import MMAPConfig
from hyperpod_checkpointless_training.dataloader.utils import FakeDataset
from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

logger = get_logger(__name__)


class MMAPDataModule(pl.LightningDataModule):
    """
    Applies MMAP data loading capabilities to user DataModule.

    Currently tested parallelism order is "tp-pp-cp-ep-dp" and "tp-cp-ep-dp-pp"
    """

    def __init__(
        self,
        data_module: pl.LightningDataModule,
        mmap_config: MMAPConfig,
        parallel_state_util=MegatronParallelStateUtil(),
        is_data_loading_rank: Callable = None,
    ):
        """
        Initialize the MMAP DataModule.

        Args:
            data_module: The underlying DataModule to wrap (e.g., LLMDataModule)
            mmap_config: The MMAP configuration to use for creating dataloaders
            parallel_state_util: Parallel state utility
            is_data_loading_rank: Function to determine if this is a data loading rank
        """
        super().__init__()
        self.data_module = data_module
        self.mmap_config = mmap_config
        self._parallel_state_util = parallel_state_util
        self._is_data_loading_rank = is_data_loading_rank or self._parallel_state_util.is_tp_0
        self.global_step = 0
        self.cached_train_dl_len = 0
        self.cached_val_dl_len = 0

    def load_checkpoint(self, checkpoint):
        logger.debug(f"setting global step {checkpoint['global_step']}")
        self.global_step = checkpoint["global_step"]

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying data module i.e the LLM Data Module.

        Args:
            name: Attribute name to access

        Returns:
            Attribute value from the underlying data module
        """
        try:
            return getattr(self.data_module, name)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def setup(self, stage: Optional[str] = None):
        """
        Setup the underlying data module.

        Args:
            stage: Stage of training ('fit', 'validate', 'test', or 'predict')
        """

        if self._is_data_loading_rank():
            self.data_module.setup(stage)

    def train_dataloader(self) -> DataLoader:
        """
        Create the training DataLoader with MMAP wrapping.

        This method wraps the underlying data module's train_dataloader
        with the configured MMAP data loader.

        Returns:
            MMAP-wrapped DataLoader or the original DataLoader based on config
        """
        if self._is_data_loading_rank():
            dataloader_init_callable = partial(self.data_module.train_dataloader)
        else:
            dataloader_init_callable = partial(self.fake_dataloader)
        dataloader = self.mmap_config.create(
            dataloader_init_callable,
            self._parallel_state_util,
            self.global_step,
            self._is_data_loading_rank,
            self._parallel_state_util.create_model_parallel_group,
            name="Train",
            is_val=False,
            cached_len=self.cached_train_dl_len
        )
        # after resume, we may reduce the dataloader len, but we don't want to overwrite the original length
        self.cached_train_dl_len = max(len(dataloader), self.cached_train_dl_len)
        return dataloader

    # Implementing the val_data_loader with mmap config

    def val_dataloader(self):
        """
        Create the validation DataLoader with MMAP wrapping.

        This method wraps the underlying data module's val_dataloader
        with the configured MMAP data loader.

        Returns:
            MMAP-wrapped DataLoader or the original DataLoader based on config
        """
        if self._is_data_loading_rank():
            dataloader_init_callable = partial(self.data_module.val_dataloader)
        else:
            dataloader_init_callable = partial(self.fake_dataloader)
        dataloader = self.mmap_config.create(
            dataloader_init_callable,
            self._parallel_state_util,
            self.global_step,
            self._is_data_loading_rank,
            self._parallel_state_util.create_model_parallel_group,
            name="Val",
            is_val=True,
            cached_len=self.cached_val_dl_len
        )
        self.cached_val_dl_len = len(dataloader)
        return dataloader

    def test_dataloader(self):
        """
        Create the test DataLoader if the underlying data module supports it.

        Returns:
            Test DataLoader from the underlying data module
        """
        if hasattr(self.data_module, "test_dataloader"):
            return self.data_module.test_dataloader()
        return None

    def predict_dataloader(self):
        """
        Create the predict DataLoader if the underlying data module supports it.

        Returns:
            Predict DataLoader from the underlying data module
        """
        if hasattr(self.data_module, "predict_dataloader"):
            return self.data_module.predict_dataloader()
        return None

    def reconfigure_limit_batches(self):
        reconfigure_method = getattr(
            self.data_module, "reconfigure_limit_batches", None
        )
        if reconfigure_method is not None:
            logger.info(
                "Delegating reconfigure_limit_batches to underlying data module"
            )
            if getattr(self, "trainer", None) is not None:
                self.data_module.trainer = self.trainer
            reconfigure_method()

    @property
    def data_sampler(self):
        """
        Expose the underlying data module's data sampler to NeMo.

        This allows NeMo's MegatronStrategy to detect the data sampler
        and initialize the microbatch calculator properly.
        """
        return getattr(self.data_module, "data_sampler", None)

    def get_underlying_data_module(self) -> pl.LightningDataModule:
        """
        Get the underlying data module.

        Returns:
            The wrapped data module
        """
        return self.data_module

    def fake_dataloader(self, target_load_step: int = 0) -> DataLoader:
        # we can ignore target_load_step here
        return DataLoader(
            dataset=FakeDataset(),
            batch_size=1,
            num_workers=0,
        )

    def state_dict(self):
        """
        Get the state dictionary of the MMAP DataModule.

        Returns:
            A dictionary containing cached dataloader lengths.
        """
        return {"cached_train_dl_len": self.cached_train_dl_len, "cached_val_dl_len": self.cached_val_dl_len}

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary into MMAP DataModule.

        Args:
            state_dict: The state dictionary to load
        """
        self.cached_train_dl_len = state_dict["cached_train_dl_len"]
        self.cached_val_dl_len = state_dict["cached_val_dl_len"]
