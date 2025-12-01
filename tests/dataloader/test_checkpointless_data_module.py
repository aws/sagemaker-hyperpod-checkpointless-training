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
from unittest.mock import Mock, patch, MagicMock


class MockLightningDataModule:
    def __init__(self, *args, **kwargs):
        pass


class MockDataLoader:
    def __init__(self, dataset, batch_size=1, num_workers=1, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers


class MockDataset:
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("FakeDataset has no items")


from hyperpod_checkpointless_training.dataloader.mmap_data_module import (
    MMAPDataModule,
    FakeDataset,
)


class TestMMAPDataModule(unittest.TestCase):

    def setUp(self):
        self.mock_data_module = Mock()
        self.mock_mmap_config = Mock()
        self.mock_parallel_state_util = Mock()
        self.mock_is_data_loading_rank = Mock()

        self.mmap_module = MMAPDataModule(
            data_module=self.mock_data_module,
            mmap_config=self.mock_mmap_config,
            parallel_state_util=self.mock_parallel_state_util,
            is_data_loading_rank=self.mock_is_data_loading_rank,
        )

    def tearDown(self):
        del self.mock_data_module
        del self.mock_mmap_config
        del self.mock_parallel_state_util
        del self.mock_is_data_loading_rank
        del self.mmap_module

    def test_init(self):
        self.assertEqual(self.mmap_module.data_module, self.mock_data_module)
        self.assertEqual(self.mmap_module.mmap_config, self.mock_mmap_config)
        self.assertEqual(self.mmap_module._parallel_state_util, self.mock_parallel_state_util)
        self.assertEqual(self.mmap_module._is_data_loading_rank, self.mock_is_data_loading_rank)

    def test_setup_data_loading_rank(self):
        self.mock_is_data_loading_rank.return_value = True

        self.mmap_module.setup("fit")

        self.mock_is_data_loading_rank.assert_called_once()
        self.mock_data_module.setup.assert_called_once_with("fit")

    def test_setup_non_data_loading_rank(self):
        self.mock_is_data_loading_rank.return_value = False

        self.mmap_module.setup("fit")

        self.mock_is_data_loading_rank.assert_called_once()
        self.mock_data_module.setup.assert_not_called()

    def test_train_dataloader_data_loading_rank(self):
        self.mock_is_data_loading_rank.return_value = True
        mock_dataloader = Mock()
        mock_dataloader.__len__ = Mock(return_value=100)
        self.mock_mmap_config.create.return_value = mock_dataloader

        result = self.mmap_module.train_dataloader()

        self.mock_is_data_loading_rank.assert_called_once()
        self.mock_mmap_config.create.assert_called_once()
        self.assertEqual(result, mock_dataloader)

        call_args = self.mock_mmap_config.create.call_args
        dataloader_callable = call_args[0][0]
        self.assertEqual(call_args[0][1], self.mock_parallel_state_util)
        self.assertEqual(call_args[1]["name"], "Train")

        dataloader_callable()
        self.mock_data_module.train_dataloader.assert_called_once()

    def test_train_dataloader_non_data_loading_rank(self):
        self.mock_is_data_loading_rank.return_value = False
        mock_dataloader = Mock()
        mock_dataloader.__len__ = Mock(return_value=100)
        self.mock_mmap_config.create.return_value = mock_dataloader

        result = self.mmap_module.train_dataloader()

        self.mock_is_data_loading_rank.assert_called_once()
        self.mock_mmap_config.create.assert_called_once()
        self.assertEqual(result, mock_dataloader)

        call_args = self.mock_mmap_config.create.call_args
        dataloader_callable = call_args[0][0]
        self.assertEqual(call_args[0][1], self.mock_parallel_state_util)
        self.assertEqual(call_args[1]["name"], "Train")

        fake_dataloader = dataloader_callable()
        self.assertEqual(fake_dataloader.batch_size, 1)
        self.assertEqual(fake_dataloader.num_workers, 0)
        self.assertIsInstance(fake_dataloader.dataset, FakeDataset)
        self.mock_data_module.train_dataloader.assert_not_called()

    def test_val_dataloader(self):
        self.mock_is_data_loading_rank.return_value = True
        mock_val_dataloader = Mock()
        mock_val_dataloader.__len__ = Mock(return_value=100)
        self.mock_mmap_config.create.return_value = mock_val_dataloader

        result = self.mmap_module.val_dataloader()

        self.mock_mmap_config.create.assert_called_once()
        self.assertEqual(result, mock_val_dataloader)

        call_args = self.mock_mmap_config.create.call_args
        dataloader_callable = call_args[0][0]
        self.assertEqual(call_args[1]["name"], "Val")

        dataloader_callable()
        self.mock_data_module.val_dataloader.assert_called_once()

    def test_get_underlying_data_module(self):
        result = self.mmap_module.get_underlying_data_module()

        self.assertEqual(result, self.mock_data_module)

    def test_load_checkpoint(self):
        """Test load_checkpoint method sets global_step."""
        checkpoint = {"global_step": 150}
        
        self.mmap_module.load_checkpoint(checkpoint)
        
        self.assertEqual(self.mmap_module.global_step, 150)

    def test_train_dataloader_caches_length(self):
        """Test that train_dataloader caches the dataloader length."""
        self.mock_is_data_loading_rank.return_value = True
        mock_dataloader = Mock()
        mock_dataloader.__len__ = Mock(return_value=100)
        self.mock_mmap_config.create.return_value = mock_dataloader
        
        result = self.mmap_module.train_dataloader()
        
        self.assertEqual(result, mock_dataloader)
        self.assertEqual(self.mmap_module.cached_train_dl_len, 100)

    def test_val_dataloader_caches_length(self):
        """Test that val_dataloader caches the dataloader length."""
        self.mock_is_data_loading_rank.return_value = True
        mock_dataloader = Mock()
        mock_dataloader.__len__ = Mock(return_value=50)
        self.mock_mmap_config.create.return_value = mock_dataloader
        
        result = self.mmap_module.val_dataloader()
        
        self.assertEqual(result, mock_dataloader)
        self.assertEqual(self.mmap_module.cached_val_dl_len, 50)

    def test_cached_length_passed_to_mmap_config(self):
        """Test that cached lengths are passed to mmap_config.create."""
        # Set initial cached lengths
        self.mmap_module.cached_train_dl_len = 75
        self.mmap_module.cached_val_dl_len = 25
        
        mock_train_dl = Mock()
        mock_train_dl.__len__ = Mock(return_value=80)
        mock_val_dl = Mock()
        mock_val_dl.__len__ = Mock(return_value=30)
        
        self.mock_mmap_config.create.side_effect = [mock_train_dl, mock_val_dl]
        
        # Call train_dataloader
        self.mmap_module.train_dataloader()
        
        # Verify cached_len was passed for train
        train_call_args = self.mock_mmap_config.create.call_args_list[0]
        self.assertEqual(train_call_args[1]["cached_len"], 75)
        
        # Call val_dataloader
        self.mmap_module.val_dataloader()
        
        # Verify cached_len was passed for val
        val_call_args = self.mock_mmap_config.create.call_args_list[1]
        self.assertEqual(val_call_args[1]["cached_len"], 25)

    def test_state_dict(self):
        """Test state_dict method returns cached lengths."""
        self.mmap_module.cached_train_dl_len = 150
        self.mmap_module.cached_val_dl_len = 75
        
        result = self.mmap_module.state_dict()
        
        expected = {
            "cached_train_dl_len": 150,
            "cached_val_dl_len": 75
        }
        self.assertEqual(result, expected)

    def test_load_state_dict(self):
        """Test load_state_dict method loads cached lengths."""
        state_dict = {
            "cached_train_dl_len": 200,
            "cached_val_dl_len": 100
        }
        
        self.mmap_module.load_state_dict(state_dict)
        
        self.assertEqual(self.mmap_module.cached_train_dl_len, 200)
        self.assertEqual(self.mmap_module.cached_val_dl_len, 100)

    def test_cached_length_persistence_across_calls(self):
        """Test that cached lengths persist and update across multiple dataloader calls."""
        mock_train_dl1 = Mock()
        mock_train_dl1.__len__ = Mock(return_value=100)
        mock_train_dl2 = Mock()
        mock_train_dl2.__len__ = Mock(return_value=120)
        
        mock_val_dl1 = Mock()
        mock_val_dl1.__len__ = Mock(return_value=50)
        mock_val_dl2 = Mock()
        mock_val_dl2.__len__ = Mock(return_value=60)
        
        self.mock_mmap_config.create.side_effect = [
            mock_train_dl1, mock_val_dl1, mock_train_dl2, mock_val_dl2
        ]
        
        # First round of calls
        self.mmap_module.train_dataloader()
        self.mmap_module.val_dataloader()
        
        self.assertEqual(self.mmap_module.cached_train_dl_len, 100)
        self.assertEqual(self.mmap_module.cached_val_dl_len, 50)
        
        # Second round of calls - should pass cached lengths and update them
        self.mmap_module.train_dataloader()
        self.mmap_module.val_dataloader()
        
        self.assertEqual(self.mmap_module.cached_train_dl_len, 120)
        self.assertEqual(self.mmap_module.cached_val_dl_len, 60)
        
        # Verify all create calls were made with correct cached_len parameters
        create_calls = self.mock_mmap_config.create.call_args_list
        self.assertEqual(len(create_calls), 4)
        
        # First train call - no cached length
        self.assertEqual(create_calls[0][1]["cached_len"], 0)
        # First val call - no cached length
        self.assertEqual(create_calls[1][1]["cached_len"], 0)
        # Second train call - uses cached length from first call
        self.assertEqual(create_calls[2][1]["cached_len"], 100)
        # Second val call - uses cached length from first call
        self.assertEqual(create_calls[3][1]["cached_len"], 50)


class TestFakeDataset(unittest.TestCase):

    def setUp(self):
        self.fake_dataset = FakeDataset()

    def tearDown(self):
        del self.fake_dataset

    def test_len(self):
        result = len(self.fake_dataset)

        self.assertEqual(result, 20365052)

    def test_getitem_returns_data(self):
        result = self.fake_dataset[0]

        self.assertIn("tokens", result)
        self.assertIn("labels", result)
        self.assertIn("loss_mask", result)
        self.assertIn("position_ids", result)
