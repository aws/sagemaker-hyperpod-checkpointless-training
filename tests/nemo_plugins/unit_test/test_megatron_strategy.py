import pytest
import torch
from unittest.mock import Mock, patch
from hyperpod_checkpointless_training.nemo_plugins.megatron_strategy import CheckpointlessMegatronStrategy
from hyperpod_checkpointless_training.nemo_plugins.fault_injection import HPFaultInjectionCallback
from nemo.lightning.pytorch.callbacks import PEFT

class TestCheckpointlessMegatronStrategy:
    @pytest.fixture
    def mock_trainer(self):
        trainer = Mock()
        trainer.wrapper = Mock()
        trainer.wrapper.checkpoint_manager = Mock()
        trainer.wrapper.finalize = Mock()
        trainer.wrapper.abort = Mock()
        trainer.wrapper.abort.instances = []
        trainer.callbacks = []
        trainer.lightning_module = Mock()
        return trainer

    @pytest.fixture
    def strategy(self):
        return CheckpointlessMegatronStrategy()

    def test_init(self, strategy):
        """Test strategy initialization."""
        assert strategy.base_store is None

    @patch.object(CheckpointlessMegatronStrategy.__bases__[0], 'setup')
    def test_setup_with_fault_injection_callback(self, mock_parent_setup, strategy, mock_trainer):
        """Test setup when HPFaultInjectionCallback is present."""
        fault_callback = Mock(spec=HPFaultInjectionCallback)
        other_callback = Mock()
        mock_trainer.callbacks = [other_callback, fault_callback]
        strategy.trainer = mock_trainer
        
        strategy.setup(mock_trainer)
        
        mock_parent_setup.assert_called_once_with(mock_trainer)
        fault_callback.on_rcb_start.assert_called_once_with(mock_trainer, mock_trainer.lightning_module)
        mock_trainer.wrapper.finalize.register_attributes.assert_called_once_with(mock_trainer)

    @patch.object(CheckpointlessMegatronStrategy.__bases__[0], 'setup')
    def test_setup_without_fault_injection_callback(self, mock_parent_setup, strategy, mock_trainer):
        """Test setup when no HPFaultInjectionCallback is present."""
        other_callback = Mock()
        mock_trainer.callbacks = [other_callback]
        strategy.trainer = mock_trainer
        
        strategy.setup(mock_trainer)
        
        mock_parent_setup.assert_called_once_with(mock_trainer)
        mock_trainer.wrapper.finalize.register_attributes.assert_called_once_with(mock_trainer)

    @patch.object(CheckpointlessMegatronStrategy.__bases__[0], 'setup')
    def test_setup_with_abort_instances(self, mock_parent_setup, strategy, mock_trainer):
        """Test setup with abort instances that have register_trainer method."""
        mock_abort1 = Mock()
        mock_abort1.register_trainer = Mock()
        mock_abort2 = Mock()
        # mock_abort2 doesn't have register_trainer method
        
        mock_trainer.wrapper.abort.instances = [mock_abort1, mock_abort2]
        strategy.trainer = mock_trainer
        
        strategy.setup(mock_trainer)
        
        mock_parent_setup.assert_called_once_with(mock_trainer)
        mock_abort1.register_trainer.assert_called_once_with(mock_trainer)
        # mock_abort2.register_trainer should not be called since it doesn't exist

    def test_get_wrapper(self, strategy, mock_trainer):
        """Test get_wrapper method."""
        strategy.trainer = mock_trainer
        
        result = strategy.get_wrapper()
        
        assert result == mock_trainer.wrapper

    @patch.object(CheckpointlessMegatronStrategy.__bases__[0], 'load_model_state_dict')
    def test_load_model_state_dict_with_state_dict(self, mock_parent_load, strategy):
        """Test load_model_state_dict when state_dict is present."""
        checkpoint = {"state_dict": {"model.weight": torch.randn(2, 2)}, "other": "data"}
        
        strategy.load_model_state_dict(checkpoint, strict=True)
        
        mock_parent_load.assert_called_once_with(checkpoint, strict=True)

    def test_load_model_state_dict_without_state_dict(self, strategy):
        """Test load_model_state_dict when state_dict is not present."""
        checkpoint = {"other": "data"}
        
        # Should return None without calling parent method
        result = strategy.load_model_state_dict(checkpoint, strict=True)
        
        assert result is None

    @patch.object(CheckpointlessMegatronStrategy.__bases__[0], 'load_model_state_dict')
    def test_load_model_state_dict_default_strict(self, mock_parent_load, strategy):
        """Test load_model_state_dict with default strict parameter."""
        checkpoint = {"state_dict": {"model.weight": torch.randn(2, 2)}}
        
        strategy.load_model_state_dict(checkpoint)
        
        mock_parent_load.assert_called_once_with(checkpoint, strict=True)

    def test_is_peft_with_peft_callback(self, strategy, mock_trainer):
        """Test is_peft method when PEFT callback is present."""
        peft_callback = Mock(spec=PEFT)
        other_callback = Mock()
        mock_trainer.callbacks = [other_callback, peft_callback]
        strategy.trainer = mock_trainer
        
        result = strategy.is_peft()
        
        assert result is True

    def test_is_peft_without_peft_callback(self, strategy, mock_trainer):
        """Test is_peft method when no PEFT callback is present."""
        other_callback = Mock()
        mock_trainer.callbacks = [other_callback]
        strategy.trainer = mock_trainer
        
        result = strategy.is_peft()
        
        assert result is False

    def test_is_peft_empty_callbacks(self, strategy, mock_trainer):
        """Test is_peft method when callbacks list is empty."""
        mock_trainer.callbacks = []
        strategy.trainer = mock_trainer
        
        result = strategy.is_peft()
        
        assert result is False
