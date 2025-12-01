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

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import lightning.pytorch as pl
from hyperpod_checkpointless_training.nemo_plugins.datamodule_epoch_callback import DataModuleEpochCallback


def test_on_train_epoch_end_direct_global_step():
    # Setup
    callback = DataModuleEpochCallback()
    trainer = MagicMock(spec=pl.Trainer)
    trainer.global_step = 42

    # Create real datamodule object with direct global_step
    class DataModule:
        def __init__(self):
            self.global_step = 0
    
    trainer.datamodule = DataModule()

    # Call the method
    callback.on_train_epoch_end(trainer, None)

    # Assert
    assert trainer.datamodule.global_step == 42


def test_on_train_epoch_end_nested_global_step():
    # Setup
    callback = DataModuleEpochCallback()
    trainer = MagicMock(spec=pl.Trainer)
    trainer.global_step = 42

    class InnerDataModule:
        def __init__(self):
            self.global_step = 0

    inner_datamodule = InnerDataModule()

    class DataModule:
        def __init__(self, inner_datamodule):
            self.data_module = inner_datamodule
    trainer.datamodule = DataModule(inner_datamodule)

    # Call the method
    callback.on_train_epoch_end(trainer, None)

    # Assert
    assert trainer.datamodule.data_module.global_step == 42




def test_on_train_epoch_end_no_global_step():
    # Setup
    callback = DataModuleEpochCallback()
    trainer = MagicMock(spec=pl.Trainer)
    trainer.global_step = 42

    # Create mock datamodule without global_step
    datamodule = MagicMock()
    trainer.datamodule = datamodule

    # Call the method - should not raise any errors
    callback.on_train_epoch_end(trainer, None)
