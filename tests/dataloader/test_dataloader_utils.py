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

# type: ignore
import unittest
from unittest.mock import patch, MagicMock

from hyperpod_checkpointless_training.dataloader.utils import (
    CheckpointlessTrainingDataloader,
    CheckpointlessDataModule,
    DummyDataset,
    FakeDataset,
)
import torch
from torch.utils.data import DataLoader


class TestCheckpointlessDataModule(unittest.TestCase):

    def setUp(self):
        self.parallel_state_patcher = patch("megatron.core.parallel_state")
        self.mock_parallel_state = self.parallel_state_patcher.start()
        self.mock_parallel_state.get_data_parallel_rank.return_value = 0
        self.mock_parallel_state.get_data_parallel_world_size.return_value = 1

        self.global_batch_size = 32
        self.micro_batch_size = 4
        self.max_length = 2048

        self.mock_data_module = MagicMock()
        self.mock_data_module.train_dataloader.return_value = DataLoader(DummyDataset())

        self.checkpointless_datamodule = CheckpointlessDataModule(
            cfg=MagicMock(),
            data_module=self.mock_data_module,
            global_batch_size=self.global_batch_size,
            micro_batch_size=self.micro_batch_size,
            seq_length=self.max_length,
            enable_inprocess=True,
        )

    def test_init(self):
        self.assertEqual(self.checkpointless_datamodule.data_module, self.mock_data_module)
        self.assertFalse(self.checkpointless_datamodule.use_original)

    def test_init_validation(self):
        with self.assertRaises(ValueError) as context:
            self.checkpointless_datamodule = CheckpointlessDataModule(
                cfg=MagicMock(),
                global_batch_size=self.global_batch_size,
                micro_batch_size=self.micro_batch_size,
                seq_length=self.max_length,
                enable_inprocess=True,
            )

    def test_setup(self):
        self.checkpointless_datamodule.setup()
        self.assertFalse(self.checkpointless_datamodule.data_module.setup.called)

        self.checkpointless_datamodule.use_original = True
        self.checkpointless_datamodule.setup()
        self.checkpointless_datamodule.data_module.setup.assert_called_once()

    def test_use_original_property(self):
        self.assertFalse(self.checkpointless_datamodule.use_original)
        self.checkpointless_datamodule.use_original = True
        self.assertTrue(self.checkpointless_datamodule.use_original)

    def test_train_dataloader_original_mode(self):
        self.checkpointless_datamodule.use_original = True

        # Reset the mock to use the actual class
        self.checkpointless_datamodule.checkpointless_dataloader_class = CheckpointlessTrainingDataloader

        dataloader = self.checkpointless_datamodule.train_dataloader()
        self.assertIsInstance(dataloader, CheckpointlessTrainingDataloader)
        self.assertEqual(
            dataloader.dataloader, self.mock_data_module.train_dataloader()
        )
        self.assertEqual(self.checkpointless_datamodule.checkpointless_train_dataloader, dataloader)

    def test_train_dataloader_checkpointless_mode(self):
        self.checkpointless_datamodule.use_original = False
        dataloader = self.checkpointless_datamodule.train_dataloader()
        self.assertIsInstance(dataloader, DataLoader)
        expected_batch_size = self.global_batch_size
        self.assertEqual(dataloader.batch_size, expected_batch_size)

    def test_train_dataloader_invalid_batch_size(self):
        # Set up batch sizes that aren't compatible
        self.global_batch_size = (
            33  # Not divisible by (dp_world_size * micro_batch_size)
        )
        self.micro_batch_size = 4

        invalid_checkpointless_datamodule = CheckpointlessDataModule(
            cfg=MagicMock(),
            data_module=self.mock_data_module,
            global_batch_size=self.global_batch_size,
            micro_batch_size=self.micro_batch_size,
            seq_length=self.max_length,
            enable_inprocess=True,
        )

        with self.assertRaises(RuntimeError) as context:
            invalid_checkpointless_datamodule.train_dataloader()

    def test_val_dataloader(self):
        # Test non-original mode
        val_dataloader = self.checkpointless_datamodule.val_dataloader()
        self.assertIsInstance(val_dataloader, DataLoader)

        # Test original mode
        self.checkpointless_datamodule.use_original = True
        self.checkpointless_datamodule.val_dataloader()
        self.checkpointless_datamodule.data_module.val_dataloader.assert_called_once()

    def test_load_checkpoint(self):
        checkpoint = MagicMock()
        self.checkpointless_datamodule.load_checkpoint(checkpoint)
        self.checkpointless_datamodule.data_module.load_checkpoint.assert_called_once_with(checkpoint)

    def test_dataloader_manager_initialization(self):
        """Test that dataloader manager is properly initialized"""
        from hyperpod_checkpointless_training.dataloader.utils import HPDataLoaderManager

        self.assertIsInstance(self.checkpointless_datamodule._dataloader_manager, HPDataLoaderManager)

    @patch("hyperpod_checkpointless_training.dataloader.utils.HPDataLoaderManager")
    def test_dataloader_registration(self, mock_manager_class):
        """Test that dataloaders are registered with the manager"""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager

        # Create a new instance to test registration
        test_data_module = CheckpointlessDataModule(
            cfg=MagicMock(),
            data_module=self.mock_data_module,
            global_batch_size=self.global_batch_size,
            micro_batch_size=self.micro_batch_size,
            seq_length=self.max_length,
            enable_inprocess=True,
        )

        # Set to original mode and create dataloader
        test_data_module.use_original = True
        dataloader = test_data_module.train_dataloader()

        # Verify that register was called on the manager
        mock_manager.register.assert_called_once_with(dataloader)

    def test_train_dataloader_with_custom_dataloader_class(self):
        """Test that train_dataloader calls custom checkpointless_dataloader_class with data_module.train_dataloader()"""
        # Create a mock dataloader class
        mock_dataloader_class = MagicMock()
        mock_dataloader_instance = MagicMock()
        mock_dataloader_class.return_value = mock_dataloader_instance

        # Create a child module with the custom dataloader class attribute
        class ChildDataModule(CheckpointlessDataModule):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.checkpointless_dataloader_class = mock_dataloader_class

        # Create instance of child module
        child_data_module = ChildDataModule(
            cfg=MagicMock(),
            data_module=self.mock_data_module,
            global_batch_size=self.global_batch_size,
            micro_batch_size=self.micro_batch_size,
            seq_length=self.max_length,
            enable_inprocess=True,
        )

        # Set to original mode and create dataloader
        child_data_module.use_original = True
        dataloader = child_data_module.train_dataloader()

        # Verify that the custom dataloader class was called with data_module.train_dataloader()
        expected_train_dataloader = self.mock_data_module.train_dataloader()
        mock_dataloader_class.assert_called_once_with(expected_train_dataloader)

        # Verify that the returned dataloader is the mock instance
        self.assertEqual(dataloader, mock_dataloader_instance)

    def tearDown(self):
        self.parallel_state_patcher.stop()


class TestCheckpointlessTrainingDataloader(unittest.TestCase):
    def setUp(self):
        self.mock_dataloader = MagicMock()
        self.batch_dataloader = CheckpointlessTrainingDataloader(self.mock_dataloader)

    def test_init(self):
        self.assertEqual(self.batch_dataloader.dataloader, self.mock_dataloader)
        self.assertFalse(self.batch_dataloader._stop_event.is_set())

    def test_iterator_with_valid_batch(self):
        """Test iterator with valid batches"""
        mock_batch = MagicMock()
        self.mock_dataloader.__iter__.return_value = iter([mock_batch])

        iterator = iter(self.batch_dataloader)
        first_batch = next(iterator)
        self.assertEqual(first_batch, mock_batch)

    def test_iterator_with_none_batch(self):
        """Test iterator behavior with None batch"""
        self.mock_dataloader.__iter__.return_value = iter([None])
        iterator = iter(self.batch_dataloader)

        # Set the stop event before iteration to avoid waiting
        self.batch_dataloader._stop_event.set()

        # Convert iterator to list - should complete immediately
        batches = list(iterator)
        self.assertEqual(len(batches), 0)

    def test_stop(self):
        """Test the stop method"""
        # Test with dataloader having stop method
        self.mock_dataloader.stop = MagicMock()

        self.batch_dataloader.stop()

        # Verify stop was called on underlying dataloader
        self.mock_dataloader.stop.assert_called_once()
        # Verify stop event was set
        self.assertTrue(self.batch_dataloader._stop_event.is_set())

    def test_stop_handles_exceptions(self):
        """Test stop method handles exceptions gracefully"""
        self.mock_dataloader.stop = MagicMock(side_effect=Exception("Test error"))

        # Should not raise exception
        self.batch_dataloader.stop()

        # Verify stop event was still set
        self.assertTrue(self.batch_dataloader._stop_event.is_set())

    def test_iterator_empty_batch(self):
        """Test iteration with empty batch"""
        mock_batches = [MagicMock(), None, MagicMock()]
        self.mock_dataloader.__iter__.return_value = iter(mock_batches)

        # Create iterator
        iterator = iter(self.batch_dataloader)

        # Get first batch
        first_batch = next(iterator)
        self.assertEqual(first_batch, mock_batches[0])

        # Set stop event before getting None batch
        self.batch_dataloader._stop_event.set()

        # Try to get next batch (which would be None)
        remaining_batches = list(
            iterator
        )  # Should return empty list since stop event is set
        self.assertEqual(len(remaining_batches), 0)

    def test_cleanup_custom_components(self):
        """Test the _cleanup_custom_components hook method"""
        # This is a hook method that should be overridable by subclasses
        # Test that it can be called without error
        self.batch_dataloader._cleanup_custom_components()
        # Should not raise any exceptions

    def test_len(self):
        """Test that length is properly delegated to wrapped dataloader"""
        self.mock_dataloader.__len__.return_value = 42
        self.assertEqual(len(self.batch_dataloader), 42)
        self.mock_dataloader.__len__.assert_called_once()


class TestFakeDataset(unittest.TestCase):
    def setUp(self):
        self.fake_dataset = FakeDataset(seq_length=2048)

    def test_len(self):
        """Test that FakeDataset has length 20365052"""
        self.assertEqual(len(self.fake_dataset), 20365052)

    def test_getitem_returns_dict(self):
        """Test that accessing items returns expected dict structure"""
        item = self.fake_dataset[0]
        self.assertIsInstance(item, dict)
        self.assertIn("tokens", item)
        self.assertIn("labels", item)
        self.assertIn("loss_mask", item)
        self.assertIn("position_ids", item)


class TestDummyDataset(unittest.TestCase):
    def setUp(self):
        self.dummy_dataset = DummyDataset(seqlen=2048)

    def test_init(self):
        self.assertEqual(self.dummy_dataset.seqlen, 2048)

    def test_iterator(self):
        iterator = iter(self.dummy_dataset)
        batch = next(iterator)
        self.assertIsInstance(batch, torch.Tensor)
        self.assertEqual(batch.shape, (2048,))


if __name__ == "__main__":
    unittest.main()
