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
import lightning.pytorch as pl
from hyperpod_checkpointless_training.inprocess.logger import get_logger

logger = get_logger()


class DataModuleEpochCallback(Callback):

    def __init__(self):
        pass

    # we need this change to support trainer.reload_dataloaders_every_n_epochs
    # when an epoch change happens after a fault, the mmap datamodule global step will continue to be the same as when 
    # loaded from checkpoint. We need to update it on epoch end since this is required in setup_data and called before
    # epoch_start hook. 
    @override
    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        try:
            # case with MMAP and Checkpointless Wrap
            if hasattr(trainer.datamodule, 'data_module'):
                trainer.datamodule.data_module.global_step = trainer.global_step
            # case MMAP only
            else:
                trainer.datamodule.global_step = trainer.global_step
        except Exception:
            logger.debug("Does not use MMAP, skipping setting global step")
