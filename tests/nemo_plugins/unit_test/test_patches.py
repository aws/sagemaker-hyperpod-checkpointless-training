import sys
from unittest.mock import Mock, patch

from hyperpod_checkpointless_training.nemo_plugins.patches import (
    patch_megatron_optimizer,
    suppress_no_sync_warning,
)


def test_suppress_no_sync_warning_installs_hook():
    """Test that suppress_no_sync_warning adds custom hook"""
    original_hook = sys.unraisablehook
    suppress_no_sync_warning()
    
    assert sys.unraisablehook != original_hook
    
    # Restore original hook
    sys.unraisablehook = original_hook


def test_suppress_no_sync_warning_suppresses_ddp_exceptions():
    """Test that DDP no_sync exceptions are suppressed"""
    original_hook = sys.unraisablehook
    mock_default_hook = Mock()
    
    with patch('sys.__unraisablehook__', mock_default_hook):
        suppress_no_sync_warning()
        
        # Create mock unraisable with DDP no_sync
        mock_unraisable = Mock()
        mock_object = Mock()
        mock_object.__repr__ = Mock(return_value="DistributedDataParallel.no_sync")
        mock_unraisable.object = mock_object
        
        sys.unraisablehook(mock_unraisable)
        
        # Should be suppressed
        mock_default_hook.assert_not_called()
    
    # Restore original hook
    sys.unraisablehook = original_hook


def test_suppress_no_sync_warning_passes_through_other_exceptions():
    """Test that non-DDP exceptions are passed to default hook"""
    original_hook = sys.unraisablehook
    mock_default_hook = Mock()
    
    with patch('sys.__unraisablehook__', mock_default_hook):
        suppress_no_sync_warning()
        
        # Create mock unraisable without DDP no_sync
        mock_unraisable = Mock()
        mock_object = Mock()
        mock_object.__repr__ = Mock(return_value="some other exception")
        mock_unraisable.object = mock_object
        
        sys.unraisablehook(mock_unraisable)
        
        mock_default_hook.assert_called_once_with(mock_unraisable)
    
    # Restore original hook
    sys.unraisablehook = original_hook


@patch('hyperpod_checkpointless_training.nemo_plugins.patches.MixedPrecisionOptimizer')
def test_patch_megatron_optimizer_patches_step_method(mock_optimizer_class):
    """Test that patch_megatron_optimizer patches the step method"""
    original_step = Mock()
    mock_optimizer_class.step_with_ready_grads = original_step
    
    patch_megatron_optimizer()
    
    # Verify method was replaced
    assert mock_optimizer_class.step_with_ready_grads != original_step
