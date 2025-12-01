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

from dataclasses import field, dataclass
import os
import signal
import unittest
from unittest.mock import ANY, MagicMock, patch, PropertyMock, call

import pytest
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.connectors.data_connector import CombinedLoader
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
import torch.nn as nn

from hyperpod_checkpointless_training.nemo_plugins.fault_injection import HPFaultInjectionCallback
from hyperpod_checkpointless_training.nemo_plugins.error_injection_utils import (
    TestFaultConfig,
    ErrorInjectionForwardHook,
    ErrorInjectionBackwardHook,
    RandomErrorInjector,
    find_middle_transformer_layer,
    FaultInjector,
)


class TestHPFaultInjectionCallback(unittest.TestCase):
    def setUp(self):
        # Setup common mocks
        self.trainer_mock = MagicMock()
        self.pl_module_mock = MagicMock()

        # Mock strategy.root_device to return a valid torch device
        self.trainer_mock.strategy.root_device = torch.device('cpu')
        self.trainer_mock.strategy.cluster_environment.global_rank.return_value = 0

        self.trainer_mock.fit_loop = MagicMock()
        self.trainer_mock.fit_loop.hp_wrapper = MagicMock()
        self.trainer_mock.fit_loop.hp_wrapper.step_upon_restart = 0

        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'MASTER_ADDR': 'localhost',
            'MASTER_PORT': '12345',
            'WORLD_SIZE': '4',
            'RANK': '0',
            'LOCAL_RANK': '0',
            'JOB_RESTART_COUNT': '1'
        })
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_init_default(self):
        """Test initialization with default parameters."""
        callback = HPFaultInjectionCallback()
        self.assertEqual(callback.test_fault_config.fault_prob_after_bwd, 0)
        self.assertEqual(callback.test_fault_config.fault_prob_between_lock, 0)
        self.assertEqual(callback.test_fault_config.fault_prob_during_fwd, 0)
        self.assertEqual(callback.test_fault_config.fault_prob_during_bwd, 0)
        self.assertEqual(callback.test_fault_config.fault_ranks, set())
        self.assertEqual(callback.test_fault_config.steps_before_fault, 10000)
        self.assertIsNone(callback.hook_handles)
        self.assertIsNone(callback.random_error_injector)
        self.assertIsNone(callback.fault_injector)
        self.assertFalse(callback.error_signal_handler_registered)

    def test_init_custom_config(self):
        """Test initialization with custom fault configuration."""
        config = TestFaultConfig(
            fault_type="plr",
            fault_prob_after_bwd=0.1,
            fault_prob_between_lock=0.2,
            fault_prob_during_fwd=0.3,
            fault_prob_during_bwd=0.4,
            fault_prob_random=0.5,
            fault_ranks={1, 2},
            steps_before_fault=100
        )
        callback = HPFaultInjectionCallback(test_fault_config=config)
        self.assertEqual(callback.test_fault_config.fault_type, "plr")
        self.assertEqual(callback.test_fault_config.fault_prob_after_bwd, 0.1)
        self.assertEqual(callback.test_fault_config.fault_prob_between_lock, 0.2)
        self.assertEqual(callback.test_fault_config.fault_prob_during_fwd, 0.3)
        self.assertEqual(callback.test_fault_config.fault_prob_during_bwd, 0.4)
        self.assertEqual(callback.test_fault_config.fault_prob_random, 0.5)
        self.assertEqual(callback.test_fault_config.fault_ranks, {1, 2})
        self.assertEqual(callback.test_fault_config.steps_before_fault, 100)

    @patch("hyperpod_checkpointless_training.nemo_plugins.fault_injection.find_middle_transformer_layer")
    def test_register_error_injection_hooks_with_module_attribute(self, mock_find_middle_transformer_layer):
        """Test _register_error_injection_hooks when pl_module has a module attribute."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.test_fault_config.fault_prob_during_fwd = 0.1
        callback.test_fault_config.fault_prob_during_bwd = 0.1

        # Create a mock module
        mock_module = MagicMock(spec=nn.Module)
        self.pl_module_mock.module = mock_module

        # Set up the mock for find_middle_transformer_layer
        mock_transformer_layer = MagicMock(spec=nn.Module)
        mock_find_middle_transformer_layer.return_value = mock_transformer_layer

        # Call the method
        callback._register_error_injection_hooks(self.trainer_mock, self.pl_module_mock)

        # Assertions
        mock_find_middle_transformer_layer.assert_called_once_with(mock_module)
        self.assertEqual(len(callback.hook_handles), 2)
        mock_transformer_layer.register_forward_hook.assert_called_once()
        mock_transformer_layer.register_full_backward_hook.assert_called_once()

    @patch("hyperpod_checkpointless_training.nemo_plugins.fault_injection.find_middle_transformer_layer")
    def test_register_error_injection_hooks_without_module_attribute(self, mock_find_middle_transformer_layer):
        """Test _register_error_injection_hooks when pl_module doesn't have a module attribute."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.test_fault_config.fault_prob_during_fwd = 0.1
        callback.test_fault_config.fault_prob_during_bwd = 0.1

        # Remove module attribute from pl_module_mock
        if hasattr(self.pl_module_mock, 'module'):
            delattr(self.pl_module_mock, 'module')

        # Set up the mock for find_middle_transformer_layer
        mock_transformer_layer = MagicMock(spec=nn.Module)
        mock_find_middle_transformer_layer.return_value = mock_transformer_layer

        # Call the method
        callback._register_error_injection_hooks(self.trainer_mock, self.pl_module_mock)

        # Assertions
        mock_find_middle_transformer_layer.assert_called_once_with(self.pl_module_mock)
        self.assertEqual(len(callback.hook_handles), 2)
        mock_transformer_layer.register_forward_hook.assert_called_once()
        mock_transformer_layer.register_full_backward_hook.assert_called_once()

    @patch("hyperpod_checkpointless_training.nemo_plugins.fault_injection.find_middle_transformer_layer")
    def test_register_error_injection_hooks_no_transformer_layer(self, mock_find_middle_transformer_layer):
        """Test _register_error_injection_hooks when no transformer layer is found."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.test_fault_config.fault_prob_during_fwd = 0.1
        callback.test_fault_config.fault_prob_during_bwd = 0.1

        # Remove module attribute from pl_module_mock if it exists
        if hasattr(self.pl_module_mock, 'module'):
            delattr(self.pl_module_mock, 'module')

        # Create mock methods for register_forward_hook and register_full_backward_hook
        self.pl_module_mock.register_forward_hook = MagicMock(return_value=MagicMock())
        self.pl_module_mock.register_full_backward_hook = MagicMock(return_value=MagicMock())

        # Set up the mock for find_middle_transformer_layer to return None
        mock_find_middle_transformer_layer.return_value = None

        # Call the method
        callback._register_error_injection_hooks(self.trainer_mock, self.pl_module_mock)

        # Assertions
        mock_find_middle_transformer_layer.assert_called_once()
        self.assertEqual(len(callback.hook_handles), 2)
        # When find_middle_transformer_layer returns None, target_layer is set to module_to_search (pl_module_mock)
        self.pl_module_mock.register_forward_hook.assert_called_once()
        self.pl_module_mock.register_full_backward_hook.assert_called_once()

    @patch("hyperpod_checkpointless_training.nemo_plugins.fault_injection.find_middle_transformer_layer")
    def test_register_error_injection_hooks_zero_probabilities(self, mock_find_middle_transformer_layer):
        """Test _register_error_injection_hooks when fault probabilities are zero."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.test_fault_config.fault_prob_during_fwd = 0
        callback.test_fault_config.fault_prob_during_bwd = 0

        # Call the method
        callback._register_error_injection_hooks(self.trainer_mock, self.pl_module_mock)

        # Assertions
        mock_find_middle_transformer_layer.assert_called_once()
        self.assertEqual(len(callback.hook_handles), 0)

    @patch("signal.signal")
    @patch("hyperpod_checkpointless_training.nemo_plugins.fault_injection.RandomErrorInjector")
    def test_start_random_error_injector(self, mock_random_error_injector_class, mock_signal):
        """Test _start_random_error_injector method."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.test_fault_config.fault_prob_random = 0.1

        # Mock the RandomErrorInjector instance
        mock_injector = MagicMock()
        mock_random_error_injector_class.return_value = mock_injector

        # Call the method
        callback._start_random_error_injector(self.trainer_mock)

        # Assertions
        mock_signal.assert_called_once_with(signal.SIGUSR1, ANY)
        mock_random_error_injector_class.assert_called_once_with(callback.test_fault_config, self.trainer_mock)
        mock_injector.start.assert_called_once()
        self.assertTrue(callback.error_signal_handler_registered)
        self.assertEqual(callback.random_error_injector, mock_injector)

    @patch("signal.signal")
    @patch("hyperpod_checkpointless_training.nemo_plugins.fault_injection.RandomErrorInjector")
    def test_start_random_error_injector_zero_prob(self, mock_random_error_injector_class, mock_signal):
        """Test _start_random_error_injector method when fault_prob_random is zero."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.test_fault_config.fault_prob_random = 0

        # Call the method
        callback._start_random_error_injector(self.trainer_mock)

        # Assertions
        mock_signal.assert_not_called()
        mock_random_error_injector_class.assert_not_called()
        self.assertFalse(callback.error_signal_handler_registered)
        self.assertIsNone(callback.random_error_injector)

    def test_on_after_grad_clipping(self):
        """Test on_after_grad_clipping method."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.fault_injector = MagicMock()
        callback.test_fault_config.fault_prob_after_bwd = 0.1

        # Mock trainer's attributes
        self.trainer_mock.strategy.cluster_environment.global_rank.return_value = 0
        self.trainer_mock.fit_loop.hp_wrapper.seq = MagicMock()
        self.trainer_mock.fit_loop.hp_wrapper.seq.get.return_value = 1
        self.trainer_mock.global_step = 10

        # Call the method
        callback.on_after_grad_clipping(self.trainer_mock, self.pl_module_mock)

        # Assertions
        callback.fault_injector.maybe_inject.assert_called_once_with(
            fault_prob=callback.test_fault_config.fault_prob_after_bwd,
            msg="Simulating failure after backward",
        )

    def test_on_train_batch_end(self):
        """Test on_train_batch_end method."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.fault_injector = MagicMock()
        callback.test_fault_config.fault_prob_between_lock = 0.1

        # Mock trainer's attributes
        self.trainer_mock.strategy.cluster_environment.global_rank.return_value = 0
        self.trainer_mock.fit_loop.hp_wrapper.seq = MagicMock()
        self.trainer_mock.fit_loop.hp_wrapper.seq.get.return_value = 1
        self.trainer_mock.global_step = 10

        # Call the method
        callback.on_train_batch_end(self.trainer_mock, self.pl_module_mock, {}, {}, 0)

        # Assertions
        callback.fault_injector.maybe_inject.assert_called_once_with(
            fault_prob=callback.test_fault_config.fault_prob_between_lock,
            msg="Simulating failure between lock",
        )

    def test_on_exception(self):
        """Test on_exception method."""
        # Setup
        callback = HPFaultInjectionCallback()
        exception = RuntimeError("Test exception")

        # Call the method
        with patch("hyperpod_checkpointless_training.nemo_plugins.fault_injection.hp_logger") as mock_logger:
            callback.on_exception(self.trainer_mock, self.pl_module_mock, exception)

        # Assertions
        mock_logger.debug.assert_called_once()
        self.assertIn("Exception observed in fault_inj callback", mock_logger.debug.call_args[0][0])

    def test_on_fit_end(self):
        """Test on_fit_end method."""
        # Setup
        callback = HPFaultInjectionCallback()

        # Create mock hook handles
        mock_handle1 = MagicMock()
        mock_handle2 = MagicMock()
        callback.hook_handles = [mock_handle1, mock_handle2]

        # Create mock random error injector
        mock_injector = MagicMock()
        callback.random_error_injector = mock_injector

        # Call the method
        with patch("hyperpod_checkpointless_training.nemo_plugins.fault_injection.hp_logger") as mock_logger:
            callback.on_fit_end(self.trainer_mock, self.pl_module_mock)

        # Assertions
        mock_handle1.remove.assert_called_once()
        mock_handle2.remove.assert_called_once()
        mock_injector.stop.assert_called_once()
        mock_injector.join.assert_called_once()
        self.assertIsNone(callback.hook_handles)
        self.assertIsNone(callback.random_error_injector)
        mock_logger.info.assert_called_once()
        self.assertIn("Fault injection hooks and threads cleaned up", mock_logger.info.call_args[0][0])

    @patch(
        "hyperpod_checkpointless_training.nemo_plugins.fault_injection.HPFaultInjectionCallback._register_error_injection_hooks"
    )
    @patch(
        "hyperpod_checkpointless_training.nemo_plugins.fault_injection.HPFaultInjectionCallback._start_random_error_injector"
    )
    def test_on_rcb_start_with_fault_ranks(self, mock_start_random_error_injector, mock_register_error_injection_hooks):
        """Test on_rcb_start method when rank is in fault_ranks."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.test_fault_config.fault_ranks = {0, 1}
        self.trainer_mock.strategy.cluster_environment.global_rank.return_value = 0

        # Call the method
        with patch("hyperpod_checkpointless_training.nemo_plugins.fault_injection.hp_logger") as mock_logger:
            callback.on_rcb_start(self.trainer_mock, self.pl_module_mock)

        # Assertions
        mock_register_error_injection_hooks.assert_called_once_with(self.trainer_mock, self.pl_module_mock)
        mock_logger.debug.assert_called_once()
        mock_start_random_error_injector.assert_called_once_with(self.trainer_mock)
        self.assertIsInstance(callback.fault_injector, FaultInjector)
        self.assertIn("Fault injection hooks registered for rank", mock_logger.debug.call_args[0][0])

    @patch(
        "hyperpod_checkpointless_training.nemo_plugins.fault_injection.HPFaultInjectionCallback._register_error_injection_hooks"
    )
    @patch(
        "hyperpod_checkpointless_training.nemo_plugins.fault_injection.HPFaultInjectionCallback._start_random_error_injector"
    )
    def test_on_rcb_start_not_in_fault_ranks(self, mock_start_random_error_injector, mock_register_error_injection_hooks):
        """Test on_rcb_start method when rank is not in fault_ranks."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.test_fault_config.fault_ranks = {1, 2}
        self.trainer_mock.strategy.cluster_environment.global_rank.return_value = 0

        # Call the method
        with patch("hyperpod_checkpointless_training.nemo_plugins.fault_injection.hp_logger") as mock_logger:
            callback.on_rcb_start(self.trainer_mock, self.pl_module_mock)

        # Assertions
        mock_start_random_error_injector.assert_not_called()
        mock_register_error_injection_hooks.assert_not_called()
        self.assertIsNone(callback.fault_injector)
        mock_logger.debug.assert_not_called()

    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.FaultInjector.maybe_inject")
    def test_on_after_grad_clipping_no_fault_injector(self, mock_maybe_inject):
        """Test on_after_grad_clipping when hp_wrapper is not available."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.test_fault_config.fault_prob_after_bwd = 0.1

        # Mock trainer's attributes without hp_wrapper
        self.trainer_mock.strategy.cluster_environment.global_rank.return_value = 0
        self.trainer_mock.fit_loop = MagicMock()
        # No hp_wrapper attribute
        self.trainer_mock.global_step = 10

        # Call the method
        callback.on_after_grad_clipping(self.trainer_mock, self.pl_module_mock)

        # Assertions
        mock_maybe_inject.assert_not_called()

    @patch("hyperpod_checkpointless_training.nemo_plugins.error_injection_utils.FaultInjector.maybe_inject")
    def test_on_train_batch_end_no_fault_injector(self, mock_maybe_inject):
        """Test on_train_batch_end when hp_wrapper is not available."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.test_fault_config.fault_prob_between_lock = 0.1

        # Mock trainer's attributes without hp_wrapper
        self.trainer_mock.strategy.cluster_environment.global_rank.return_value = 0
        self.trainer_mock.fit_loop = MagicMock()
        # No hp_wrapper attribute
        self.trainer_mock.global_step = 10

        # Call the method
        callback.on_train_batch_end(self.trainer_mock, self.pl_module_mock, {}, {}, 0)

        # Assertions
        mock_maybe_inject.assert_not_called()

    @patch('signal.signal')
    def test_error_signal_handler(self, mock_signal):
        """Test the error signal handler function."""
        # Setup
        callback = HPFaultInjectionCallback()
        callback.test_fault_config.fault_prob_random = 0.1

        # Mock trainer's attributes
        self.trainer_mock.strategy.cluster_environment.global_rank.return_value = 0
        self.trainer_mock.fit_loop.hp_wrapper.seq = MagicMock()
        self.trainer_mock.fit_loop.hp_wrapper.seq.get.return_value = 1
        self.trainer_mock.global_step = 10

        # Call _start_random_error_injector to register the signal handler
        callback._start_random_error_injector(self.trainer_mock)

        # Get the signal handler function
        signal_handler = mock_signal.call_args[0][1]

        # Test the signal handler
        with self.assertRaises(RuntimeError) as context:
            signal_handler(signal.SIGUSR1, None)
        self.assertIn("Simulating random failure", str(context.exception))


if __name__ == '__main__':
    unittest.main()
