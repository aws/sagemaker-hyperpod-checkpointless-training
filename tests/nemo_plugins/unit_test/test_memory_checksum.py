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

import unittest
from unittest.mock import MagicMock, patch, Mock
import torch
import numpy as np

from hyperpod_checkpointless_training.nemo_plugins.memory_checksum import MemoryChecksumManager


# Mock the imported modules
class MockChainedOptimizer:
    pass

# Patch the imports at module level
@patch('megatron.core.optimizer.ChainedOptimizer', MockChainedOptimizer)
class TestMemoryChecksumManager(unittest.TestCase):
    def setUp(self):
        # Create mock model
        self.mock_model = MagicMock()

        # Create mock model/optimizer tensor states
        self.mock_param1 = torch.tensor([0.1, 0.2, 0.3])
        self.mock_param2 = torch.tensor([0.4, 0.5, 0.6])
        self.mock_sharded_tensor = {
            0: self.mock_param1,
            1: self.mock_param2
        }, {0: {1: self.mock_param1}}

        # Create mock mcore optimizer
        self.mock_mcore_optimizer = MagicMock()

        # Create mock optimizer
        self.mock_optimizer = MagicMock()
        self.mock_optimizer.mcore_optimizer = self.mock_mcore_optimizer

        # Create mock trainer
        self.mock_trainer = MagicMock()
        self.mock_trainer.model = [self.mock_model]  # Model is now accessed as trainer.model[0]
        self.mock_trainer.optimizers = [self.mock_optimizer]

    def test_init(self):
        # Test initialization with default value
        manager = MemoryChecksumManager()
        self.assertFalse(manager.enable_checksum)
        self.assertIsNone(manager.memory_checksum)

        # Test initialization with enable_checksum=True
        manager = MemoryChecksumManager(enable_checksum=True)
        self.assertTrue(manager.enable_checksum)
        self.assertIsNone(manager.memory_checksum)

    def test_store_checksum_disabled(self):
        # Test store_checksum when checksum is disabled
        manager = MemoryChecksumManager(enable_checksum=False)
        manager.store_checksum(self.mock_trainer)
        self.assertIsNone(manager.memory_checksum)

    @patch("hyperpod_checkpointless_training.nemo_plugins.memory_checksum.MemoryChecksumManager._compute_checksum")
    def test_store_checksum_enabled(self, mock_compute_checksum):
        # Test store_checksum when checksum is enabled
        mock_compute_checksum.return_value = 'test_checksum'

        manager = MemoryChecksumManager(enable_checksum=True)
        manager.store_checksum(self.mock_trainer)

        mock_compute_checksum.assert_called_once_with(self.mock_trainer)
        self.assertEqual(manager.memory_checksum, 'test_checksum')

    def test_verify_global_checksum_disabled(self):
        # Test verify_global_checksum when checksum is disabled
        manager = MemoryChecksumManager(enable_checksum=False)
        result = manager.verify_global_checksum(self.mock_trainer, True, False)
        self.assertTrue(result)

    def test_verify_global_checksum_param_update_not_completed(self):
        # Test verify_global_checksum when param_update_completed is False
        with patch('torch.distributed.get_rank', return_value=0), \
             patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.all_reduce', side_effect=lambda tensor, **kwargs: None), \
             patch('torch.cuda.current_device', return_value=0 if torch.cuda.is_available() else None):
            manager = MemoryChecksumManager(enable_checksum=True)
            result = manager.verify_global_checksum(self.mock_trainer, False, False)
            self.assertTrue(result)

    @patch("hyperpod_checkpointless_training.nemo_plugins.memory_checksum.hp_logger")
    def test_verify_global_checksum_memory_checksum_none(self, mock_logging):
        # Test verify_global_checksum when memory_checksum is None
        with patch('torch.distributed.get_rank', return_value=0), \
             patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.all_reduce', side_effect=lambda tensor, **kwargs: None), \
             patch('torch.cuda.current_device', return_value=0 if torch.cuda.is_available() else None):
            manager = MemoryChecksumManager(enable_checksum=True)
            result = manager.verify_global_checksum(self.mock_trainer, True, False)
            self.assertFalse(result)
            mock_logging.warning.assert_called()

    @patch("hyperpod_checkpointless_training.nemo_plugins.memory_checksum.MemoryChecksumManager._compute_checksum")
    def test_verify_global_checksum_match(self, mock_compute_checksum):
        # Test verify_global_checksum when checksum matches
        mock_compute_checksum.return_value = 'test_checksum'

        with patch('torch.distributed.get_rank', return_value=0), \
             patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.all_reduce', side_effect=lambda tensor, **kwargs: None), \
             patch('torch.cuda.current_device', return_value=0 if torch.cuda.is_available() else None):

            manager = MemoryChecksumManager(enable_checksum=True)
            manager.memory_checksum = 'test_checksum'
            result = manager.verify_global_checksum(self.mock_trainer, True, False)

            self.assertTrue(result)
            mock_compute_checksum.assert_called_once_with(self.mock_trainer)

    @patch("hyperpod_checkpointless_training.nemo_plugins.memory_checksum.MemoryChecksumManager._compute_checksum")
    @patch("hyperpod_checkpointless_training.nemo_plugins.memory_checksum.hp_logger")
    def test_verify_global_checksum_mismatch(self, mock_logging, mock_compute_checksum):
        # Test verify_global_checksum when checksum doesn't match
        mock_compute_checksum.return_value = 'different_checksum'

        with patch('torch.distributed.get_rank', return_value=0), \
             patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.all_reduce', side_effect=lambda tensor, **kwargs: None), \
             patch('torch.cuda.current_device', return_value=0 if torch.cuda.is_available() else None):

            manager = MemoryChecksumManager(enable_checksum=True)
            manager.memory_checksum = 'test_checksum'
            result = manager.verify_global_checksum(self.mock_trainer, True, False)

            self.assertFalse(result)
            mock_compute_checksum.assert_called_once_with(self.mock_trainer)
            mock_logging.warning.assert_called()

    @patch("hyperpod_checkpointless_training.nemo_plugins.memory_checksum.MemoryChecksumManager._verify_local_checksum")
    @patch("hyperpod_checkpointless_training.nemo_plugins.memory_checksum.hp_logger")
    def test_verify_global_checksum_local_pass_global_fail(self, mock_logging, mock_verify_local):
        # Test verify_global_checksum when local checksum passes but global checksum fails
        mock_verify_local.return_value = True

        with patch('torch.distributed.get_rank', return_value=0), \
             patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.cuda.current_device', return_value=0 if torch.cuda.is_available() else None):

            # Mock all_reduce to set the tensor to 0 (indicating a failure on another rank)
            def mock_all_reduce(tensor, **kwargs):
                tensor.fill_(0)

            with patch('torch.distributed.all_reduce', side_effect=mock_all_reduce):
                manager = MemoryChecksumManager(enable_checksum=True)
                result = manager.verify_global_checksum(self.mock_trainer, True, False)

                self.assertFalse(result)
                mock_verify_local.assert_called_once_with(self.mock_trainer, True, False)
                mock_logging.info.assert_called()

    @patch('megatron.core.optimizer.ChainedOptimizer', MockChainedOptimizer)
    def test_compute_checksum(self):
        # Create tensor states that match the expected format
        param_tensors = {
            0: torch.tensor([0.1, 0.2, 0.3]),
            1: torch.tensor([0.4, 0.5, 0.6])
        }
        state_tensors = {
            0: {1: torch.tensor([0.7, 0.8, 0.9])}
        }

        # Create mock optimizer structure
        mock_mcore_optimizer = Mock()
        mock_optimizer = Mock()
        mock_optimizer.mcore_optimizer = mock_mcore_optimizer

        # Set up the trainer with the optimizer
        self.mock_trainer.optimizers = [mock_optimizer]

        # Mock get_sharded_tensor_states at the module level
        with patch(
            "hyperpod_checkpointless_training.nemo_plugins.memory_checksum.get_sharded_tensor_states",
            return_value=(param_tensors, state_tensors),
        ):

            # Test the checksum computation
            manager = MemoryChecksumManager()

            # Compute first checksum
            checksum1 = manager._compute_checksum(self.mock_trainer)

            # Modify a tensor
            original_value = param_tensors[0].clone()
            param_tensors[0].fill_(2.0)

            # Compute second checksum
            checksum2 = manager._compute_checksum(self.mock_trainer)

            # Reset tensor and compute third checksum
            param_tensors[0].copy_(original_value)
            checksum3 = manager._compute_checksum(self.mock_trainer)

            # Verify checksums
            self.assertNotEqual(checksum1, checksum2)
            self.assertEqual(checksum1, checksum3)


if __name__ == '__main__':
    unittest.main()
