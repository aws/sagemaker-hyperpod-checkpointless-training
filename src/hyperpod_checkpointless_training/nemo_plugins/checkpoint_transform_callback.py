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

from lightning.pytorch.callbacks import Callback
from typing_extensions import override

from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg
from nemo.lightning.pytorch.callbacks import PEFT

from hyperpod_checkpointless_training.inprocess.parameter_update_lock import ParameterUpdateLock
import torch

hp_logger = get_logger()

class CheckpointTransform(Callback):

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
        peft = self.get_peft_callback(trainer)
        if peft is not None:
            peft._maybe_apply_transform(trainer)
        else:
            raise RuntimeError(debug_msg("peft is enabled but can not find peft callback"))
    


    def get_peft_callback(self, trainer):
        for callback in trainer.callbacks:
            if isinstance(callback, PEFT):
                return callback
        return None
