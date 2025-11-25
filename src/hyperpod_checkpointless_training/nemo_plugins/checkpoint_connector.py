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

import time

from hyperpod_checkpointless_training.nemo_plugins.resume import CheckpointlessAutoResume
from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg

from lightning.pytorch.utilities.migration.utils import _pl_migrate_checkpoint
from lightning.pytorch.trainer.connectors.checkpoint_connector import _CheckpointConnector
from pathlib import Path

hp_logger = get_logger()


def set_adapter_model_ckpt_path(trainer, model_ckpt_path):
    """
    Set model_ckpt_path on WrappedAdapterIO for PEFT checkpointless recovery.

    Prevents adapter_metadata corruption by restoring model_ckpt_path lost during
    checkpointless recovery (not part of checkpoint state).

    Args:
        trainer: PyTorch Lightning trainer instance
        model_ckpt_path: Path to base model checkpoint
    """
    if not model_ckpt_path:
        return

    # Currently only checks trainer.strategy._checkpoint_io
    locations_to_check = [
        ('trainer.strategy._checkpoint_io', getattr(trainer.strategy, '_checkpoint_io', None)),
    ]

    # Also check nested _checkpoint_io (for async case)
    for name, obj in list(locations_to_check):
        if obj and hasattr(obj, '_checkpoint_io'):
            locations_to_check.append((f'{name}._checkpoint_io', obj._checkpoint_io))

    hp_logger.debug(debug_msg(f"Setting model_ckpt_path={model_ckpt_path} on WrappedAdapterIO objects where model_ckpt_path is None"))

    for location_name, obj in locations_to_check:
        if obj:
            obj_type = type(obj).__name__
            hp_logger.debug(debug_msg(f"Checking {location_name}: type={obj_type}, has_model_ckpt_path={hasattr(obj, 'model_ckpt_path')}"))

            # Only update if it's WrappedAdapterIO type and has model_ckpt_path attribute
            if obj_type == 'WrappedAdapterIO' and hasattr(obj, 'model_ckpt_path'):
                try:
                    current_value = getattr(obj, 'model_ckpt_path', 'NOT_SET')
                    # Only update if current value is None
                    if current_value is None:
                        obj.model_ckpt_path = model_ckpt_path
                        hp_logger.debug(debug_msg(f"Set model_ckpt_path at {location_name} (WrappedAdapterIO): None -> {model_ckpt_path}"))
                    else:
                        hp_logger.debug(debug_msg(f"Skipping {location_name} - model_ckpt_path is not None: {current_value}"))
                except Exception as e:
                    hp_logger.warning(debug_msg(f"Failed to set model_ckpt_path at {location_name} (WrappedAdapterIO): {e}"))
            else:
                hp_logger.debug(debug_msg(f"Skipping {location_name} - not WrappedAdapterIO type or no model_ckpt_path attribute (type={obj_type})"))


def try_checkpointless_resume(trainer, checkpoint_path):
    """
    Attempt checkpointless recovery before falling back to disk checkpoint.

    Args:
        trainer: PyTorch Lightning trainer instance
        checkpoint_path: Path to disk checkpoint for fallback

    Returns:
        dict or None: Loaded checkpoint if checkpointless recovery succeeds, None otherwise
    """
    checkpoint_manager = trainer.strategy.get_wrapper().checkpoint_manager
    if trainer.strategy.is_peft():
        loaded_checkpoint = checkpoint_manager.try_base_model_checkpointless_load(trainer)
    else:
        loaded_checkpoint = checkpoint_manager.try_checkpointless_load(trainer)

    if loaded_checkpoint is not None:
        loaded_checkpoint = _pl_migrate_checkpoint(loaded_checkpoint, checkpoint_path)

        if trainer.strategy.is_peft():
            # Fix for ckptless recovery: Ensure checkpoint_io has model_ckpt_path to prevent adapter_metadata corruption
            # This value gets lost during ckptless recovery since it is saved as a singluar value (not part of a ckpt)
            # Ref: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/lightning/pytorch/callbacks/peft.py#L523
            if (hasattr(trainer, 'fresume') and hasattr(trainer.fresume, 'restore_config') and
                trainer.fresume.restore_config and hasattr(trainer.fresume.restore_config, 'path') and
                trainer.fresume.restore_config.path):
                set_adapter_model_ckpt_path(trainer, trainer.fresume.restore_config.path)
            else:
                hp_logger.warning(debug_msg(f"Could not find restore_config path to set model_ckpt_path."))

        return loaded_checkpoint

    strategy = trainer.strategy
    if hasattr(strategy.trainer, "fresume"):
        # Only CheckpointlessAutoResume support delayed call
        if not isinstance(strategy.trainer.fresume, CheckpointlessAutoResume):
            raise ValueError(
                f"CheckpointlessMegatronStrategy requires trainer.fresume to be a CheckpointlessAutoResume, but getting {type(strategy.trainer.fresume)}"
            )
        hp_logger.debug(debug_msg("Delayed calling resume to find checkpoint path"))
        strategy.trainer.fresume.setup(strategy.trainer, force_setup=True)
        if not trainer.ckpt_path:
            hp_logger.debug(debug_msg("Calling delayed selective restore call."))
            trainer.strategy.selective_restore()
        checkpoint_path = trainer.ckpt_path

    if checkpoint_path is None:
        hp_logger.warning(debug_msg("Checkpoint path is None"))

    return None

class CheckpointlessCompatibleConnector(_CheckpointConnector):
    """
    Attempts to pre-load the checkpoint file to memory, with the source path determined in this priority:

    1. try checkpoint-less recovery
    2. if checkpointless return None, fallback to parent.resume_start()
    """

    def resume_start(self, checkpoint_path=None) -> None:
        self.start_time = time.perf_counter()
        loaded_checkpoint = try_checkpointless_resume(self.trainer, checkpoint_path)
        if loaded_checkpoint is not None:
            self._loaded_checkpoint = loaded_checkpoint
            return

        # Fallback to parent implementation
        super().resume_start(self.trainer.ckpt_path)

    def resume_end(self):
        super().resume_end()
        hp_logger.info(f"CheckpointlessConnector Load cost : {time.perf_counter() - self.start_time}")

        if hasattr(self.trainer.datamodule, "load_checkpoint"):
            self.trainer.datamodule.load_checkpoint({"global_step": self.trainer.global_step})
