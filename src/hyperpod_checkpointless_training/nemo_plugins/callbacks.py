import time
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override

from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.parameter_update_lock import ParameterUpdateLock
from hyperpod_checkpointless_training.inprocess.utils import debug_msg
from nemo.lightning.pytorch.callbacks import PEFT

hp_logger = get_logger()


class CheckpointlessCallback(Callback):
    """
    Lightning callback that integrates NeMo training with Checkpointless Training's fault tolerance system.

    Manages step tracking, checkpoint saving, and parameter update coordination
    for in-process restart capabilities.

    CORE RESPONSIBILITIES:

    1. **Training Step Lifecycle Management**: Tracks training progress and coordinates
       with ParameterUpdateLock to enable/disable checkpointless recovery based on
       training state (first step vs subsequent steps).

    2. **Checkpoint State Coordination**: Manages in-memory checkpoint saving at the
       end of each successful training step, including model checksums, RNG states,
       and global step tracking for deterministic recovery.
    """

    def __init__(
        self,
        enable_inprocess: bool = False,
        enable_checkpointless: bool = False,
        enable_checksum: bool = False,
        clean_tensor_hook: bool = False,
        clean_lightning_module: bool = False,
    ):
        """
        Initialize the callback with fault tolerance features.

        Args:
            enable_inprocess: Enable in-process restart capabilities
            enable_checkpointless: Enable checkpointless recovery (requires inprocess)
            enable_checksum: Enable model state checksum validation (requires checkpointless)
            clean_tensor_hook: Clear tensor hooks from all GPU tensors during cleanup (expensive)
            clean_lightning_module: Enable this to free GPU memory after each GPU restarts
        """
        self.enable_inprocess = enable_inprocess
        self.enable_checkpointless = enable_checkpointless
        # enable_checksum will be consumed in wrapper.checkpoint_manager
        self.enable_checksum = enable_checksum
        self.clean_tensor_hook = clean_tensor_hook
        self.clean_lightning_module = clean_lightning_module
        self.tried_adapter_checkpointless = False

        if self.enable_checkpointless:
            assert (
                enable_inprocess
            ), "checkpointless can not be enabled without inprocess"
        if enable_checksum:
            assert (
                self.enable_checkpointless
            ), "checksum can not be enabled without checkpointless"

    def get_wrapper_from_trainer(self, trainer):
        """Get the HPCallWrapper instance from the trainer for fault tolerance coordination."""
        return trainer.wrapper

    @override
    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, *args, **kwargs
    ):
        """
        Called at the start of each training batch.

        Updates step tracking and marks recovery as complete:
        - For PEFT: Applies transform and attempts adapter checkpointless restore
        - Sets first_step=False (recovery finished, enables checkpointless recovery)
        - Increments step counter for restart decision logic
        """
        if hasattr(trainer.strategy, "is_peft") and trainer.strategy.is_peft():
            peft = self.get_peft_callback(trainer)
            if peft is not None:
                # Sometimes Nemo/PTL might not be calling the transform during the resume. It will be
                # a no-op if it is already called.
                peft._maybe_apply_transform(trainer)
                params_to_save = peft.params_to_save if hasattr(peft, "params_to_save") else set()
                self.try_adapter_checkpointless_restore(trainer, params_to_save)
            else:
                raise RuntimeError(debug_msg("peft is enabled but can not find peft callback"))

        param_update_lock = ParameterUpdateLock()
        # Recovery is finished at this point, mark first_step False
        param_update_lock.first_step = False
        # Increment step_upon_restart at the beginning of each step
        self.get_wrapper_from_trainer(trainer).step_upon_restart += 1


    def get_peft_callback(self, trainer):
        for callback in trainer.callbacks:
            if isinstance(callback, PEFT):
                return callback
        return None

    def try_adapter_checkpointless_restore(self, trainer, params_to_save):
        if self.tried_adapter_checkpointless:
            return
        checkpoint_manager = trainer.strategy.get_wrapper().checkpoint_manager
        checkpoint_manager.try_checkpointless_load(trainer)
        checkpoint_manager.params_to_save = params_to_save
        self.tried_adapter_checkpointless = True

    @override
    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Release parameter update lock at end of each training batch.

        Lock release timing ensures checkpointless recovery can proceed after
        parameter updates complete.
        """
        if self.enable_inprocess and self.enable_checkpointless:
            ParameterUpdateLock().__exit__(None, None, None)
            hp_logger.debug(
                debug_msg(f"Lock released")
            )
