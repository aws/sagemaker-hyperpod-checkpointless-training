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
from unittest.mock import Mock, patch, MagicMock
import time
from hyperpod_checkpointless_training.nemo_plugins.checkpoint_connector import CheckpointlessCompatibleConnector
from hyperpod_checkpointless_training.nemo_plugins.checkpoint_connector import try_checkpointless_resume
from hyperpod_checkpointless_training.nemo_plugins.checkpoint_connector import set_adapter_model_ckpt_path
from hyperpod_checkpointless_training.nemo_plugins.resume import CheckpointlessAutoResume


class TestCheckpointlessCompatibleConnector:
    @pytest.fixture
    def mock_trainer(self):
        trainer = Mock()
        trainer.strategy = Mock()
        return trainer

    @pytest.fixture
    def connector(self, mock_trainer):
        return CheckpointlessCompatibleConnector(trainer=mock_trainer)

    @pytest.fixture
    def mock_checkpoint(self):
        return {
            "state": "test",
            "pytorch-lightning_version": "2.0.0",
            "global_step": 123,
            "epoch": 1,
            "state_dict": {},
        }

    @patch("time.perf_counter")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_connector.try_checkpointless_resume")
    def test_resume_start_checkpointless_success_with_none_path(self, mock_try_checkpointless, mock_perf_counter, connector):
        """Test resume_start when checkpointless recovery succeeds and path is None."""
        mock_checkpoint = {"state_dict": {}, "optimizer": []}
        mock_try_checkpointless.return_value = mock_checkpoint
        mock_perf_counter.return_value = 100.0

        connector.resume_start(checkpoint_path=None)

        mock_try_checkpointless.assert_called_once_with(connector.trainer, None)
        assert connector._loaded_checkpoint == mock_checkpoint
        assert connector.start_time == 100.0  

    @patch("time.perf_counter")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_connector.try_checkpointless_resume")
    def test_resume_start_checkpointless_fails_with_valid_path(self, mock_try_checkpointless, mock_perf_counter, connector):
        """Test resume_start when checkpointless recovery fails and valid path provided."""
        checkpoint_path = "/path/to/checkpoint"
        mock_try_checkpointless.return_value = None
        connector.trainer.ckpt_path = checkpoint_path
        mock_perf_counter.return_value = 100.0

        with patch.object(connector.__class__.__bases__[0], 'resume_start') as mock_parent_resume:
            connector.resume_start(checkpoint_path=checkpoint_path)

            mock_try_checkpointless.assert_called_once_with(connector.trainer, checkpoint_path)
            mock_parent_resume.assert_called_once_with(checkpoint_path)
            assert connector.start_time == 100.0

    @patch("time.perf_counter")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_connector.try_checkpointless_resume")
    def test_resume_start_no_checkpoint_path(self, mock_try_checkpointless, mock_perf_counter, connector):
        """Test resume_start when no checkpoint path is provided."""
        mock_try_checkpointless.return_value = None
        connector.trainer.ckpt_path = None
        mock_perf_counter.return_value = 100.0

        with patch.object(connector.__class__.__bases__[0], 'resume_start') as mock_parent_resume:
            connector.resume_start(checkpoint_path=None)

            mock_try_checkpointless.assert_called_once_with(connector.trainer, None)
            mock_parent_resume.assert_called_once_with(None)
            assert connector.start_time == 100.0

    @patch("time.perf_counter")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_connector.try_checkpointless_resume")
    def test_resume_start_with_checkpoint_path_load_returns_none(self, mock_try_checkpointless, mock_perf_counter, connector):
        """Test resume_start when checkpointless fails and parent load returns None."""
        checkpoint_path = "path/to/checkpoint"
        mock_try_checkpointless.return_value = None
        connector.trainer.ckpt_path = checkpoint_path
        mock_perf_counter.return_value = 100.0

        with patch.object(connector.__class__.__bases__[0], 'resume_start') as mock_parent_resume:
            connector.resume_start(checkpoint_path=checkpoint_path)

            mock_try_checkpointless.assert_called_once_with(connector.trainer, checkpoint_path)
            mock_parent_resume.assert_called_once_with(checkpoint_path)

    @patch("hyperpod_checkpointless_training.inprocess.logger.get_logger")
    @patch("time.perf_counter")
    def test_resume_end_with_datamodule(self, mock_perf_counter, mock_logger, connector):
        """Test resume_end with datamodule that has load_checkpoint method."""
        connector.start_time = 100.0
        mock_perf_counter.return_value = 105.0
        connector.trainer.global_step = 500
        connector.trainer.datamodule = Mock()
        connector.trainer.datamodule.load_checkpoint = Mock()

        with patch.object(connector.__class__.__bases__[0], 'resume_end') as mock_parent_resume_end:
            connector.resume_end()

            mock_parent_resume_end.assert_called_once()
            connector.trainer.datamodule.load_checkpoint.assert_called_once_with({"global_step": 500})

    @patch("hyperpod_checkpointless_training.inprocess.logger.get_logger")
    @patch("time.perf_counter")
    def test_resume_end_without_datamodule(self, mock_perf_counter, mock_logger, connector):
        """Test resume_end without datamodule."""
        connector.start_time = 100.0
        mock_perf_counter.return_value = 105.0
        connector.trainer.datamodule = None

        with patch.object(connector.__class__.__bases__[0], 'resume_end') as mock_parent_resume_end:
            connector.resume_end()

            mock_parent_resume_end.assert_called_once()

class TestSetAdapterModelCkptPath:
    @pytest.fixture
    def mock_trainer(self):
        trainer = Mock()
        trainer.strategy = Mock()
        return trainer

    def test_set_adapter_model_ckpt_path_no_path(self, mock_trainer):
        """Test set_adapter_model_ckpt_path with empty model_ckpt_path"""
        set_adapter_model_ckpt_path(mock_trainer, None)
        set_adapter_model_ckpt_path(mock_trainer, "")
        # Should return early without doing anything

    def test_set_adapter_model_ckpt_path_wrapped_adapter_io_success(self, mock_trainer):
        """Test set_adapter_model_ckpt_path with WrappedAdapterIO and None model_ckpt_path"""
        mock_checkpoint_io = Mock()
        mock_checkpoint_io.__class__.__name__ = "WrappedAdapterIO"
        mock_checkpoint_io.model_ckpt_path = None
        mock_trainer.strategy._checkpoint_io = mock_checkpoint_io

        set_adapter_model_ckpt_path(mock_trainer, "/path/to/model.ckpt")

        assert mock_checkpoint_io.model_ckpt_path == "/path/to/model.ckpt"

    def test_set_adapter_model_ckpt_path_existing_value_not_overwritten(self, mock_trainer):
        """Test set_adapter_model_ckpt_path doesn't overwrite existing model_ckpt_path"""
        mock_checkpoint_io = Mock()
        mock_checkpoint_io.__class__.__name__ = "WrappedAdapterIO"
        mock_checkpoint_io.model_ckpt_path = "/existing/path.ckpt"
        mock_trainer.strategy._checkpoint_io = mock_checkpoint_io

        set_adapter_model_ckpt_path(mock_trainer, "/path/to/model.ckpt")

        # Should not overwrite existing value
        assert mock_checkpoint_io.model_ckpt_path == "/existing/path.ckpt"

    def test_set_adapter_model_ckpt_path_non_wrapped_adapter_io(self, mock_trainer):
        """Test set_adapter_model_ckpt_path with non-WrappedAdapterIO object"""
        mock_checkpoint_io = Mock()
        mock_checkpoint_io.__class__.__name__ = "RegularCheckpointIO"
        mock_trainer.strategy._checkpoint_io = mock_checkpoint_io

        set_adapter_model_ckpt_path(mock_trainer, "/path/to/model.ckpt")
        # Should skip non-WrappedAdapterIO objects without error


class TestTryCheckpointlessResume:
    @pytest.fixture
    def mock_trainer(self):
        trainer = Mock()
        strategy = Mock()
        wrapper = Mock()
        checkpoint_manager = Mock()

        wrapper.checkpoint_manager = checkpoint_manager
        strategy.get_wrapper.return_value = wrapper
        strategy.is_peft.return_value = False
        trainer.strategy = strategy
        trainer.ckpt_path = None

        return trainer

    @patch("hyperpod_checkpointless_training.inprocess.logger.get_logger")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_connector._pl_migrate_checkpoint")
    def test_checkpointless_load_returns_checkpoint(self, mock_migrate, mock_logger, mock_trainer):
        """Test when checkpointless load returns a checkpoint"""
        checkpoint = {"state": "test"}
        migrated_checkpoint = {"state": "migrated"}
        mock_trainer.strategy.is_peft.return_value = False

        mock_trainer.strategy.get_wrapper().checkpoint_manager.try_checkpointless_load.return_value = checkpoint
        mock_migrate.return_value = migrated_checkpoint

        result = try_checkpointless_resume(mock_trainer, "test_path")

        mock_migrate.assert_called_once_with(checkpoint, "test_path")
        assert result == migrated_checkpoint

    @patch("hyperpod_checkpointless_training.inprocess.logger.get_logger")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_connector._pl_migrate_checkpoint")
    def test_checkpointless_load_peft_model(self, mock_migrate, mock_logger, mock_trainer):
        """Test when checkpointless load returns a checkpoint for PEFT model"""
        checkpoint = {"state": "test"}
        migrated_checkpoint = {"state": "migrated"}
        mock_trainer.strategy.is_peft.return_value = True

        mock_trainer.strategy.get_wrapper().checkpoint_manager.try_base_model_checkpointless_load.return_value = checkpoint
        mock_migrate.return_value = migrated_checkpoint

        result = try_checkpointless_resume(mock_trainer, "test_path")

        mock_migrate.assert_called_once_with(checkpoint, "test_path")
        assert result == migrated_checkpoint

    @patch("hyperpod_checkpointless_training.inprocess.logger.get_logger")
    def test_checkpointless_load_returns_none(self, mock_logger, mock_trainer):
        """Test when checkpointless load returns None and no fresume attribute"""
        mock_trainer.strategy.is_peft.return_value = False
        mock_trainer.strategy.get_wrapper().checkpoint_manager.try_checkpointless_load.return_value = None
        if hasattr(mock_trainer.strategy.trainer, 'fresume'):
            del mock_trainer.strategy.trainer.fresume

        result = try_checkpointless_resume(mock_trainer, "test_path")

        assert result is None

    @patch("hyperpod_checkpointless_training.inprocess.logger.get_logger")
    def test_checkpointless_load_returns_none_invalid_fresume_type(self, mock_logger, mock_trainer):
        """Test when checkpointless load returns None and fresume is not CheckpointlessAutoResume"""
        mock_trainer.strategy.is_peft.return_value = False
        mock_trainer.strategy.get_wrapper().checkpoint_manager.try_checkpointless_load.return_value = None
        mock_trainer.strategy.trainer.fresume = Mock()

        with pytest.raises(ValueError, match="CheckpointlessMegatronStrategy requires trainer.fresume to be a CheckpointlessAutoResume"):
            try_checkpointless_resume(mock_trainer, "test_path")

    @patch("hyperpod_checkpointless_training.inprocess.logger.get_logger")
    def test_checkpointless_load_returns_none_valid_fresume_type(self, mock_logger, mock_trainer):
        """Test when checkpointless load returns None and fresume is CheckpointlessAutoResume"""
        mock_trainer.strategy.is_peft.return_value = False
        mock_trainer.strategy.get_wrapper().checkpoint_manager.try_checkpointless_load.return_value = None

        mock_fresume = Mock(spec=CheckpointlessAutoResume)
        mock_trainer.strategy.trainer.fresume = mock_fresume
        mock_trainer.ckpt_path = "/new/checkpoint/path"

        result = try_checkpointless_resume(mock_trainer, "test_path")

        mock_fresume.setup.assert_called_once_with(mock_trainer.strategy.trainer, force_setup=True)
        assert result is None

    @patch("hyperpod_checkpointless_training.inprocess.logger.get_logger")
    def test_checkpointless_load_peft_with_fresume_no_ckpt(self, mock_logger, mock_trainer):
        """Test PEFT model with fresume and selective restore with no ckpt available"""
        mock_trainer.strategy.is_peft.return_value = True
        mock_trainer.strategy.get_wrapper().checkpoint_manager.try_base_model_checkpointless_load.return_value = None

        mock_fresume = Mock(spec=CheckpointlessAutoResume)
        mock_trainer.strategy.trainer.fresume = mock_fresume
        mock_trainer.ckpt_path = None

        result = try_checkpointless_resume(mock_trainer, "test_path")

        mock_fresume.setup.assert_called_once_with(mock_trainer.strategy.trainer, force_setup=True)
        mock_trainer.strategy.selective_restore.assert_called_once()
        assert result is None

    @patch("hyperpod_checkpointless_training.inprocess.logger.get_logger")
    def test_checkpointless_load_peft_with_fresume(self, mock_logger, mock_trainer):
        """Test PEFT model with fresume and selective restore with ckpt available"""
        mock_trainer.strategy.is_peft.return_value = True
        mock_trainer.strategy.get_wrapper().checkpoint_manager.try_base_model_checkpointless_load.return_value = None

        mock_fresume = Mock(spec=CheckpointlessAutoResume)
        mock_trainer.strategy.trainer.fresume = mock_fresume
        mock_trainer.ckpt_path = "/new/checkpoint/path"

        result = try_checkpointless_resume(mock_trainer, "test_path")

        mock_fresume.setup.assert_called_once_with(mock_trainer.strategy.trainer, force_setup=True)
        mock_trainer.strategy.selective_restore.assert_not_called()
        assert result is None
