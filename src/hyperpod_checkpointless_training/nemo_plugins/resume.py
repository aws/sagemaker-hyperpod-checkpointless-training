from typing import Union

import lightning.fabric as fl
import lightning.pytorch as pl
from nemo.lightning.resume import AutoResume

from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg

hp_logger = get_logger()


class CheckpointlessAutoResume(AutoResume):
    def setup(
        self, trainer: Union[pl.Trainer, fl.Fabric], model=None, force_setup=False
    ):
        """
        Conditionally delay AutoResume setup to enable checkpointless recovery validation.

        This method overrides NeMo's AutoResume.setup() to implement a two-phase initialization:
        1. Initial call (force_setup=False): Skips checkpoint path resolution to allow
        CheckpointManager to validate checkpointless recovery feasibility first
        2. Forced call (force_setup=True): Executes normal AutoResume logic if checkpointless
        recovery is not feasible

        The delay is necessary because checkpointless recovery validation requires Megatron
        parallel state initialization, which happens after this setup would normally run.

        Args:
            force_setup: If True, bypass delay and execute AutoResume setup immediately.
                        Set to True by CheckpointConnector when checkpointless recovery
                        is not feasible and checkpoint path must be resolved.

        Returns:
            None. Sets trainer.ckpt_path to None if delaying, or resolves checkpoint path
            via parent setup() if force_setup=True.

        Note:
            When force_setup=False, trainer.ckpt_path is explicitly set to None to prevent
            premature checkpoint loading before checkpointless recovery validation completes.
        """
        if not force_setup:
            hp_logger.info(
                debug_msg(
                    "Skipping AutoResume setup for possible checkpoint-less recovery...."
                )
            )
            trainer.ckpt_path = None
            return
        hp_logger.info(debug_msg("Running AutoResume setup...."))
        super().setup(trainer, model=model)
