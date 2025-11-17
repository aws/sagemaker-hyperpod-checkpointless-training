# type: ignore
"""Dataloader utils."""

import threading
import torch

from torch.utils.data import DataLoader, IterableDataset, Dataset
from lightning.pytorch.core.datamodule import LightningDataModule
from hyperpod_checkpointless_training.inprocess.utils import debug_msg
from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.abort import HPDataLoaderManager

hp_logger = get_logger(__name__)


class FakeDataset(Dataset):
    def __init__(self, seq_length: int = 8192, dataset_length: int = 20365052):
        self.seq_length = seq_length
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        return {
            "tokens": torch.zeros(self.seq_length, dtype=torch.long),
            "labels": torch.zeros(self.seq_length, dtype=torch.long),
            "loss_mask": torch.ones(self.seq_length, dtype=torch.float),
            "position_ids": torch.arange(self.seq_length, dtype=torch.long),
        }


class CheckpointlessDataModule(LightningDataModule):
    """Checkpointless Datamodule to wrap original dataloader with Checkpointless support."""

    def __init__(
        self,
        cfg: object = None,
        data_module: LightningDataModule = None,
        global_batch_size: int = None,
        micro_batch_size: int = None,
        seq_length: int = None,
        enable_inprocess: bool = False,
        always_use_original: bool = False,
    ):
        super().__init__()
        if data_module is None:
            raise ValueError("data_module must be provided")
        self.cfg = cfg
        self.data_module = data_module
        self.global_batch_size = global_batch_size or data_module.global_batch_size
        self.micro_batch_size = micro_batch_size or data_module.micro_batch_size
        self.seq_length = seq_length or data_module.seq_length
        self._use_original = False
        self.enable_inprocess = enable_inprocess
        self.always_use_original = always_use_original
        self.checkpointless_train_dataloader = None
        self.checkpointless_val_dataloader = None
        self._dataloader_manager = HPDataLoaderManager()

    def setup(self, stage=None):
        if self.use_original:
            self.data_module.setup(stage)

    @property
    def use_original(self):
        return (
            self.always_use_original or self._use_original or not self.enable_inprocess
        )

    @use_original.setter
    def use_original(self, use):
        self._use_original = use

    def _wrap_with_checkpointless(self, dataloader):
        dataloader_class = getattr(
            self, "checkpointless_dataloader_class", CheckpointlessTrainingDataloader
        )
        checkpointless_dataloader = dataloader_class(dataloader)
        # Register dataloader upon creation
        self._dataloader_manager.register(checkpointless_dataloader)
        return checkpointless_dataloader

    def train_dataloader(self, stage=None):
        if self.use_original:
            self.checkpointless_train_dataloader = self._wrap_with_checkpointless(
                self.data_module.train_dataloader()
            )
            return self.checkpointless_train_dataloader

        global_batch_size = self.global_batch_size
        micro_batch_size = self.micro_batch_size

        from megatron.core import parallel_state

        self.dp_rank = parallel_state.get_data_parallel_rank()
        self.dp_world_size = parallel_state.get_data_parallel_world_size()

        if global_batch_size % (self.dp_world_size * micro_batch_size) != 0:
            raise RuntimeError(
                f"`global_batch_size` ({global_batch_size}) is not divisible by "
                f"`micro_batch_size` ({micro_batch_size}) x `data_parallel_size` ({self.dp_world_size})"
            )
        batch_size = global_batch_size // self.dp_world_size

        dataset = FakeDataset(seq_length=self.seq_length)

        return DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
        )

    def load_checkpoint(self, checkpoint):
        if hasattr(self.data_module, "load_checkpoint"):
            self.data_module.load_checkpoint(checkpoint)

    def val_dataloader(self, stage=None):
        if self.use_original:
            val_dataloaders = self.data_module.val_dataloader()
            if isinstance(val_dataloaders, list):
                self.checkpointless_val_dataloader = []
                for val_dl in val_dataloaders:
                    self.checkpointless_val_dataloader.append(self._wrap_with_checkpointless(val_dl))
            else:
                self.checkpointless_val_dataloader = self._wrap_with_checkpointless(val_dataloaders)
            return self.checkpointless_val_dataloader

        return self.train_dataloader(stage)  # type: ignore

    def __getattr__(self, name):
        # Pipe all unknown attributes to the wrapped data_module.
        return getattr(self.data_module, name)

    def state_dict(self):
        return self.data_module.state_dict()
    
    def load_state_dict(self, state_dict):
        self.data_module.load_state_dict(state_dict)

class CheckpointlessTrainingDataloader:

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._stop_event = threading.Event()

    def stop(self):
        try:
            hp_logger.debug(
                debug_msg(f"Stopping dataloader of type {type(self.dataloader)}")
            )

            if hasattr(self.dataloader, "stop"):
                hp_logger.debug(
                    debug_msg(
                        f"Calling underlying dataloader stop {id(self.dataloader)}"
                    )
                )
                self.dataloader.stop()

            self._cleanup_custom_components()

            hp_logger.debug(
                debug_msg(f"Completed cleanup for dataloader {id(self.dataloader)}")
            )
        except Exception as e:
            hp_logger.error(
                debug_msg(
                    f"Error during dataloader {id(self.dataloader)} cleanup: {str(e)}"
                )
            )
        finally:
            self._stop_event.set()

    def _cleanup_custom_components(self):
        """
        Hook method for subclass-specific cleanup.
        """
        pass

    def __iter__(self):
        try:
            data_iter = iter(self.dataloader)
            while True:
                batch = next(data_iter)

                if not batch:
                    hp_logger.debug("Encountered empty/None batch, waiting for stop...")
                    self._stop_event.wait(timeout=30)
                    return
                yield batch
        except Exception as e:
            hp_logger.error(
                debug_msg(f"Error during dataloader generating batches {e}")
            )

    def __getattr__(self, name):
        # Pipe all unknown attributes to the wrapped dataloader.
        return getattr(self.dataloader, name)
    
    # note this is necessary for proper multi_epoch training given we override the megatron datasampler in our example
    def __len__(self):
        return len(self.dataloader)


class DummyDataset(IterableDataset):
    """Dummy Dataset."""

    def __init__(self, seqlen=2048):
        super().__init__()
        self.seqlen = seqlen

    def __iter__(self):
        while True:
            yield torch.ones(self.seqlen, dtype=torch.int64)
