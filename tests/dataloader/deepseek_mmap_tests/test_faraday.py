# type: ignore
import unittest
from unittest.mock import Mock
from functools import partial

from hyperpod_checkpointless_training.dataloader.mmap_data_module import MMAPDataModule
from hyperpod_checkpointless_training.dataloader.config import CacheResumeMMAPConfig
from hyperpod_checkpointless_training.dataloader.mmap.utils import MockDataLoader


class TestMMAPDataModule(unittest.TestCase):

    def setUp(self):
        """Setup test environment with mock objects."""
        # Mock LLM data module
        self.mock_data_module = Mock()
        self.mock_data_module.train_dataloader.return_value = Mock()
        self.mock_data_module.val_dataloader.return_value = Mock()

        # Mock MMAP config
        self.mock_mmap_config = Mock(spec=CacheResumeMMAPConfig)
        self.mock_mmap_config.create.return_value = MockDataLoader()

        # Mock parallel state util
        self.mock_parallel_state_util = Mock()
        self.mock_parallel_state_util.create_model_parallel_group.return_value = Mock()

        # Mock is_data_loading_rank function
        self.mock_is_data_loading_rank = Mock()

        # Create MMAPDataModule instance
        self.mmap_module = MMAPDataModule(
            data_module=self.mock_data_module,
            mmap_config=self.mock_mmap_config,
            parallel_state_util=self.mock_parallel_state_util,
            is_data_loading_rank=self.mock_is_data_loading_rank,
        )

    def test_train_dataloader_data_loading_rank(self):
        """Test train_dataloader when is_data_loading_rank=True."""
        self.mock_is_data_loading_rank.return_value = True

        result = self.mmap_module.train_dataloader()

        self.assertEqual(result, self.mock_mmap_config.create.return_value)

        self.mock_mmap_config.create.assert_called_once()
        call_args = self.mock_mmap_config.create.call_args

        dataloader_callable = call_args[0][0]
        self.assertIsInstance(dataloader_callable, partial)

        self.assertEqual(call_args[0][1], self.mock_parallel_state_util)
        self.assertEqual(call_args[0][2], 0)  # global_step
        self.assertEqual(call_args[0][3], self.mock_is_data_loading_rank)
        self.assertEqual(call_args[1]["name"], "Train")
        self.assertEqual(call_args[1]["is_val"], False)

    def test_train_dataloader_non_data_loading_rank(self):
        """Test train_dataloader when is_data_loading_rank=False."""
        self.mock_is_data_loading_rank.return_value = False

        result = self.mmap_module.train_dataloader()

        self.assertEqual(result, self.mock_mmap_config.create.return_value)

        self.mock_mmap_config.create.assert_called_once()
        call_args = self.mock_mmap_config.create.call_args

        dataloader_callable = call_args[0][0]
        self.assertIsInstance(dataloader_callable, partial)

        self.assertEqual(call_args[0][1], self.mock_parallel_state_util)
        self.assertEqual(call_args[0][2], 0)  # global_step
        self.assertEqual(call_args[0][3], self.mock_is_data_loading_rank)

    def test_val_dataloader_data_loading_rank(self):
        """Test val_dataloader when is_data_loading_rank=True."""
        self.mock_is_data_loading_rank.return_value = True

        result = self.mmap_module.val_dataloader()

        self.assertEqual(result, self.mock_mmap_config.create.return_value)

        self.mock_mmap_config.create.assert_called_once()
        call_args = self.mock_mmap_config.create.call_args

        dataloader_callable = call_args[0][0]
        self.assertIsInstance(dataloader_callable, partial)

        self.assertEqual(call_args[1]["name"], "Val")

        self.mmap_module.setup("fit")

        self.mock_data_module.setup.assert_called_once_with("fit")

    def test_setup_non_data_loading_rank(self):
        """Test setup method when is_data_loading_rank=False."""
        self.mock_is_data_loading_rank.return_value = False

        self.mmap_module.setup("fit")

        self.mock_data_module.setup.assert_not_called()
