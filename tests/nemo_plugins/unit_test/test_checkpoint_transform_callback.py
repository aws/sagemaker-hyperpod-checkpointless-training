import pytest
from unittest.mock import Mock, MagicMock
import lightning.pytorch as pl
from nemo.lightning.pytorch.callbacks import PEFT

from hyperpod_checkpointless_training.nemo_plugins.checkpoint_transform_callback import CheckpointTransform


def test_on_train_batch_start_with_peft_callback():
    """Test on_train_batch_start when PEFT callback is present."""
    # Setup
    callback = CheckpointTransform()
    trainer = MagicMock(spec=pl.Trainer)
    pl_module = MagicMock()
    batch = MagicMock()
    batch_idx = 0
    
    # Create a mock PEFT callback
    peft_callback = MagicMock(spec=PEFT)
    peft_callback._maybe_apply_transform = MagicMock()
    
    # Add PEFT callback to trainer's callbacks
    trainer.callbacks = [peft_callback]
    
    # Call the method
    callback.on_train_batch_start(trainer, pl_module, batch, batch_idx)
    
    # Assert that _maybe_apply_transform was called with trainer
    peft_callback._maybe_apply_transform.assert_called_once_with(trainer)


def test_on_train_batch_start_without_peft_callback():
    """Test on_train_batch_start raises RuntimeError when PEFT callback is missing."""
    # Setup
    callback = CheckpointTransform()
    trainer = MagicMock(spec=pl.Trainer)
    pl_module = MagicMock()
    batch = MagicMock()
    batch_idx = 0
    
    # No PEFT callback in trainer's callbacks
    trainer.callbacks = []
    
    # Call the method and expect RuntimeError
    with pytest.raises(RuntimeError, match="peft is enabled but can not find peft callback"):
        callback.on_train_batch_start(trainer, pl_module, batch, batch_idx)


def test_on_train_batch_start_with_non_peft_callbacks():
    """Test on_train_batch_start raises RuntimeError when only non-PEFT callbacks are present."""
    # Setup
    callback = CheckpointTransform()
    trainer = MagicMock(spec=pl.Trainer)
    pl_module = MagicMock()
    batch = MagicMock()
    batch_idx = 0
    
    # Add some non-PEFT callbacks
    other_callback1 = MagicMock()
    other_callback2 = MagicMock()
    trainer.callbacks = [other_callback1, other_callback2]
    
    # Call the method and expect RuntimeError
    with pytest.raises(RuntimeError, match="peft is enabled but can not find peft callback"):
        callback.on_train_batch_start(trainer, pl_module, batch, batch_idx)


def test_get_peft_callback_returns_peft_when_present():
    """Test get_peft_callback returns PEFT callback when it exists."""
    # Setup
    callback = CheckpointTransform()
    trainer = MagicMock(spec=pl.Trainer)
    
    # Create a mock PEFT callback
    peft_callback = MagicMock(spec=PEFT)
    other_callback = MagicMock()
    
    # Add callbacks to trainer
    trainer.callbacks = [other_callback, peft_callback]
    
    # Call the method
    result = callback.get_peft_callback(trainer)
    
    # Assert
    assert result is peft_callback


def test_get_peft_callback_returns_first_peft_when_multiple():
    """Test get_peft_callback returns first PEFT callback when multiple exist."""
    # Setup
    callback = CheckpointTransform()
    trainer = MagicMock(spec=pl.Trainer)
    
    # Create multiple PEFT callbacks
    peft_callback1 = MagicMock(spec=PEFT)
    peft_callback2 = MagicMock(spec=PEFT)
    other_callback = MagicMock()
    
    # Add callbacks to trainer
    trainer.callbacks = [other_callback, peft_callback1, peft_callback2]
    
    # Call the method
    result = callback.get_peft_callback(trainer)
    
    # Assert - should return the first PEFT callback
    assert result is peft_callback1


def test_get_peft_callback_returns_none_when_not_present():
    """Test get_peft_callback returns None when PEFT callback doesn't exist."""
    # Setup
    callback = CheckpointTransform()
    trainer = MagicMock(spec=pl.Trainer)
    
    # Add only non-PEFT callbacks
    other_callback1 = MagicMock()
    other_callback2 = MagicMock()
    trainer.callbacks = [other_callback1, other_callback2]
    
    # Call the method
    result = callback.get_peft_callback(trainer)
    
    # Assert
    assert result is None


def test_get_peft_callback_returns_none_with_empty_callbacks():
    """Test get_peft_callback returns None when callbacks list is empty."""
    # Setup
    callback = CheckpointTransform()
    trainer = MagicMock(spec=pl.Trainer)
    
    # Empty callbacks list
    trainer.callbacks = []
    
    # Call the method
    result = callback.get_peft_callback(trainer)
    
    # Assert
    assert result is None
