# type: ignore
"""Unit tests for config.py module."""
import os
import unittest
from unittest.mock import Mock, patch

import torch

from hyperpod_checkpointless_training.dataloader.config import (
    MMAPConfig,
    CacheResumeMMAPConfig,
)


class TestMMAPConfig(unittest.TestCase):
    """Test cases for MMAPConfig base class."""

    def test_mmap_config_initialization(self):
        """Test that MMAPConfig can be initialized successfully."""
        config = MMAPConfig()
        self.assertIsInstance(config, MMAPConfig)


class TestCacheResumeMMAPConfig(unittest.TestCase):

    def setUp(self):
        """Setup test environment."""
        self.cache_dir = "/test/cache"
        self.config = CacheResumeMMAPConfig(
            cache_dir=self.cache_dir,
            prefetch_length=5,
            val_prefetch_length=5,
            lookback_length=5,
            checkpoint_frequency=10,
            enable_batch_encryption=False,
        )

        self.mock_dataloader_init = Mock()
        self.mock_parallel_state_util = Mock()
        self.mock_is_data_loading_rank = Mock()
        self.mock_create_model_parallel_group = Mock()

    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch("hyperpod_checkpointless_training.dataloader.config.CacheResumePrefetchedDataLoader")
    @patch("torch.distributed.get_rank")
    def test_create_prefetched_dataloader_training(
        self, mock_get_rank, mock_prefetched_loader, mock_tensor, mock_all_reduce
    ):
        """Test creates CacheResumePrefetchedDataLoader for training when is_data_loading_rank=True."""
        mock_get_rank.return_value = 0
        self.mock_is_data_loading_rank.return_value = True
        
        mock_pdl_instance = Mock()
        mock_pdl_instance.data_loader = Mock(__len__=Mock(return_value=100))
        mock_pdl_instance.__len__ = Mock(return_value=100)
        mock_prefetched_loader.return_value = mock_pdl_instance
        
        
        mock_tensor.return_value = Mock(item=Mock(return_value=100))
        self.mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = Mock()

        self.config.create(
            self.mock_dataloader_init,
            self.mock_parallel_state_util,
            100,
            self.mock_is_data_loading_rank,
            self.mock_create_model_parallel_group,
            is_val=False,
        )

        mock_prefetched_loader.assert_called_once()

    @patch("torch.distributed.get_rank")
    def test_create_method_missing_data_step(self, mock_get_rank):
        """Test create method raises ValueError when step is None."""
        mock_get_rank.return_value = 0
        self.mock_is_data_loading_rank.return_value = True
        config = CacheResumeMMAPConfig(checkpoint_frequency=50)

        mock_dataloader_callable = Mock()
        mock_parallel_state_util = Mock()
        mock_is_data_loading_rank = Mock(return_value=True)
        create_model_parallel_group_callable = Mock()

        with self.assertRaises(ValueError) as context:
            config.create(
                mock_dataloader_callable,
                mock_parallel_state_util,
                None,
                mock_is_data_loading_rank,
                create_model_parallel_group_callable,
            )

        self.assertIn("step must be provided", str(context.exception))

    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumeReadDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_read_dataloader_training(self, mock_get_rank, mock_read_loader, mock_tensor, mock_all_reduce):
        """Test creates CacheResumeReadDataLoader for training when is_data_loading_rank=False."""
        mock_get_rank.return_value = 0
        self.mock_is_data_loading_rank.return_value = False
        
        mock_cdl_instance = Mock()
        mock_cdl_instance.set_length = Mock()
        mock_cdl_instance.__len__ = Mock(return_value=100)
        mock_read_loader.return_value = mock_cdl_instance
        
        # Mock tensor operations for length synchronization
        mock_length_tensor = Mock()
        mock_length_tensor.item.return_value = 100
        mock_tensor.return_value = mock_length_tensor

        # Mock parallel state
        mock_tp_group = Mock()
        self.mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = mock_tp_group
        
        self.config.create(
            self.mock_dataloader_init,
            self.mock_parallel_state_util,
            100,
            self.mock_is_data_loading_rank,
            self.mock_create_model_parallel_group,
            is_val=False,
        )

        mock_read_loader.assert_called_once_with(
            dataloader_init_callable=self.mock_dataloader_init,
            model_parallel_group=self.mock_create_model_parallel_group.return_value,
            step=100,
            prefetch_length=5,
            parallel_state_util=self.mock_parallel_state_util,
            force_cold_start=False,
            cache_dir=os.path.join(self.cache_dir, "train"),
            enable_batch_encryption=False,
        )

    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumePrefetchedDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_prefetched_dataloader_validation(
        self, mock_get_rank, mock_prefetched_loader, mock_tensor, mock_all_reduce
    ):
        """Test creates CacheResumePrefetchedDataLoader for validation when is_data_loading_rank=True."""
        mock_get_rank.return_value = 0
        self.mock_is_data_loading_rank.return_value = True
        
        # Mock the dataloader and its length
        mock_underlying_dl = Mock()
        mock_underlying_dl.__len__ = Mock(return_value=100)
        
        mock_pdl_instance = Mock()
        mock_pdl_instance.data_loader = mock_underlying_dl
        mock_pdl_instance.__len__ = Mock(return_value=100)
        mock_pdl_instance.set_length = Mock()
        mock_prefetched_loader.return_value = mock_pdl_instance
        
        # Mock tensor operations for length synchronization
        mock_length_tensor = Mock()
        mock_length_tensor.item.return_value = 100
        mock_tensor.return_value = mock_length_tensor

        # Mock parallel state
        mock_tp_group = Mock()
        self.mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = mock_tp_group

        self.config.create(
            self.mock_dataloader_init,
            self.mock_parallel_state_util,
            100,
            self.mock_is_data_loading_rank,
            self.mock_create_model_parallel_group,
            is_val=True,
        )

        mock_prefetched_loader.assert_called_once_with(
            step=0,
            dataloader_init_callable=self.mock_dataloader_init,
            model_parallel_group=self.mock_create_model_parallel_group.return_value,
            parallel_state_util=self.mock_parallel_state_util,
            force_cold_start=True,
            model_checkpoint_frequency=10,
            lookback_length=5,
            prefetch_length=5,
            cache_dir=os.path.join(self.cache_dir, "val"),
            enable_batch_encryption=False,
        )

    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumeReadDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_read_dataloader_validation(self, mock_get_rank, mock_read_loader, mock_tensor, mock_all_reduce):
        """Test creates CacheResumeReadDataLoader for validation when is_data_loading_rank=False."""
        mock_get_rank.return_value = 0
        self.mock_is_data_loading_rank.return_value = False
        
        mock_cdl_instance = Mock()
        mock_cdl_instance.set_length = Mock()
        mock_cdl_instance.__len__ = Mock(return_value=100)
        mock_read_loader.return_value = mock_cdl_instance
        
        # Mock tensor operations for length synchronization
        mock_length_tensor = Mock()
        mock_length_tensor.item.return_value = 100
        mock_tensor.return_value = mock_length_tensor

        # Mock parallel state
        mock_tp_group = Mock()
        self.mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = mock_tp_group
        
        self.config.create(
            self.mock_dataloader_init,
            self.mock_parallel_state_util,
            100,
            self.mock_is_data_loading_rank,
            self.mock_create_model_parallel_group,
            is_val=True,
        )

        mock_read_loader.assert_called_once_with(
            dataloader_init_callable=self.mock_dataloader_init,
            model_parallel_group=self.mock_create_model_parallel_group.return_value,
            step=0,
            prefetch_length=5,
            parallel_state_util=self.mock_parallel_state_util,
            force_cold_start=True,
            cache_dir=os.path.join(self.cache_dir, "val"),
            enable_batch_encryption=False,
        )


class TestCacheResumeMMAPConfigWithLengthSetting(unittest.TestCase):
    """Test cases for CacheResumeMMAPConfig with length setting functionality."""

    def setUp(self):
        """Setup test environment."""
        self.cache_dir = "/test/cache"
        self.config = CacheResumeMMAPConfig(
            cache_dir=self.cache_dir,
            prefetch_length=5,
            val_prefetch_length=5,
            lookback_length=5,
            checkpoint_frequency=10,
            enable_batch_encryption=False,
        )

    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumePrefetchedDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_method_sets_length_for_pdl(
        self, mock_get_rank, mock_pdl_class, mock_tensor, mock_all_reduce
    ):
        """Test create method sets length on PrefetchedDataLoader."""
        mock_get_rank.return_value = 0
        
        # Mock the dataloader and its length
        mock_underlying_dl = Mock()
        mock_underlying_dl.__len__ = Mock(return_value=100)
        
        mock_pdl_instance = Mock()
        mock_pdl_instance.data_loader = mock_underlying_dl
        mock_pdl_instance.__len__ = Mock(return_value=100)
        mock_pdl_instance.set_length = Mock()
        mock_pdl_class.return_value = mock_pdl_instance

        # Mock tensor operations for length synchronization
        mock_length_tensor = Mock()
        mock_length_tensor.item.return_value = 100
        mock_tensor.return_value = mock_length_tensor

        # Mock parallel state
        mock_parallel_state_util = Mock()
        mock_tp_group = Mock()
        mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = mock_tp_group

        mock_dataloader_callable = Mock()
        mock_is_data_loading_rank = Mock(return_value=True)
        mock_create_model_parallel_group_callable = Mock()

        result = self.config.create(
            mock_dataloader_callable,
            mock_parallel_state_util,
            100,
            mock_is_data_loading_rank,
            mock_create_model_parallel_group_callable,
            cached_len=0,
        )

        # Verify the dataloader was created and length was set
        self.assertEqual(result, mock_pdl_instance)
        mock_pdl_instance.initialize_data_loader.assert_called_once()
        mock_pdl_instance.set_length.assert_called_once_with(100)
        
        # Verify tensor operations for length synchronization
        mock_tensor.assert_called_once_with(100, dtype=torch.long, device='cuda')
        mock_all_reduce.assert_called_once()

    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumeReadDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_method_sets_length_for_cdl(
        self, mock_get_rank, mock_cdl_class, mock_tensor, mock_all_reduce
    ):
        """Test create method sets length on CacheResumeReadDataLoader."""
        mock_get_rank.return_value = 1
        
        mock_cdl_instance = Mock()
        mock_cdl_instance.set_length = Mock()
        mock_cdl_instance.__len__ = Mock(return_value=100)
        mock_cdl_class.return_value = mock_cdl_instance

        # Mock tensor operations for length synchronization
        mock_length_tensor = Mock()
        mock_length_tensor.item.return_value = 75
        mock_tensor.return_value = mock_length_tensor

        # Mock parallel state
        mock_parallel_state_util = Mock()
        mock_tp_group = Mock()
        mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = mock_tp_group

        mock_dataloader_callable = Mock()
        mock_is_data_loading_rank = Mock(return_value=False)
        mock_create_model_parallel_group_callable = Mock()

        result = self.config.create(
            mock_dataloader_callable,
            mock_parallel_state_util,
            100,
            mock_is_data_loading_rank,
            mock_create_model_parallel_group_callable,
            cached_len=0,
        )

        # Verify the dataloader was created and length was set
        self.assertEqual(result, mock_cdl_instance)
        mock_cdl_instance.set_length.assert_called_once_with(75)
        
        # Verify tensor operations for length synchronization
        mock_tensor.assert_called_once_with(0, dtype=torch.long, device='cuda')  # CDL doesn't initialize, so 0
        mock_all_reduce.assert_called_once()

    @patch("hyperpod_checkpointless_training.dataloader.config.get_num_microbatches")
    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumePrefetchedDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_method_uses_cached_length(
        self, mock_get_rank, mock_pdl_class, mock_tensor, mock_all_reduce, mock_num_mbs
    ):
        """Test create method uses cached_len when provided."""
        mock_get_rank.return_value = 0
        
        mock_pdl_instance = Mock()
        mock_pdl_instance.set_length = Mock()
        mock_pdl_instance.__len__ = Mock(return_value=50)
        mock_pdl_class.return_value = mock_pdl_instance

        mock_parallel_state_util = Mock()
        mock_dataloader_callable = Mock()
        mock_is_data_loading_rank = Mock(return_value=True)
        mock_create_model_parallel_group_callable = Mock()

        mock_num_mbs.return_value = 1

        # Call with cached_len provided
        result = self.config.create(
            mock_dataloader_callable,
            mock_parallel_state_util,
            100,
            mock_is_data_loading_rank,
            mock_create_model_parallel_group_callable,
            cached_len=150,
        )

        # step 100 for cached_len 150 should be 50 samples remaining (set length to samples remaining)
        mock_pdl_instance.set_length.assert_called_once_with(50)
        mock_tensor.assert_not_called()  # No tensor operations when cached_len is provided
        mock_all_reduce.assert_not_called()  # No all_reduce when cached_len is provided

    @patch("hyperpod_checkpointless_training.dataloader.config.get_num_microbatches")
    @patch("torch.distributed.all_reduce")
    @patch("torch.tensor")
    @patch(
        "hyperpod_checkpointless_training.dataloader.config.CacheResumePrefetchedDataLoader"
    )
    @patch("torch.distributed.get_rank")
    def test_create_method_max_cached_and_synchronized_length(
        self, mock_get_rank, mock_pdl_class, mock_tensor, mock_all_reduce, mock_num_mbs
    ):
        """Test create method uses cached_len when provided (no synchronization occurs)."""
        mock_get_rank.return_value = 0
        
        # Mock the dataloader and its length
        mock_underlying_dl = Mock()
        mock_underlying_dl.__len__ = Mock(return_value=200)  # This won't be used since cached_len is provided
        
        mock_pdl_instance = Mock()
        mock_pdl_instance.data_loader = mock_underlying_dl
        mock_pdl_instance.__len__ = Mock(return_value=50)
        mock_pdl_instance.set_length = Mock()
        mock_pdl_class.return_value = mock_pdl_instance

        # Mock parallel state
        mock_parallel_state_util = Mock()
        mock_tp_group = Mock()
        mock_parallel_state_util.parallel_state.get_tensor_model_parallel_group.return_value = mock_tp_group

        mock_dataloader_callable = Mock()
        mock_is_data_loading_rank = Mock(return_value=True)
        mock_create_model_parallel_group_callable = Mock()

        mock_num_mbs.return_value = 1

        # Call with cached_len=150 - no synchronization should occur
        result = self.config.create(
            mock_dataloader_callable,
            mock_parallel_state_util,
            100,
            mock_is_data_loading_rank,
            mock_create_model_parallel_group_callable,
            cached_len=150,
        )

        # step 100 for cached_len 150 should be 50 samples remaining (set length to samples remaining)
        mock_pdl_instance.set_length.assert_called_once_with(50)
        # Verify no tensor operations occurred since cached_len was provided
        mock_tensor.assert_not_called()
        mock_all_reduce.assert_not_called()

    def test_create_method_validation_step_none(self):
        """Test create method raises ValueError when step is None."""
        config = CacheResumeMMAPConfig()

        mock_dataloader_callable = Mock()
        mock_parallel_state_util = Mock()
        mock_is_data_loading_rank = Mock(return_value=True)
        mock_create_model_parallel_group_callable = Mock()

        with self.assertRaises(ValueError) as context:
            config.create(
                mock_dataloader_callable,
                mock_parallel_state_util,
                None,
                mock_is_data_loading_rank,
                mock_create_model_parallel_group_callable,
            )

        self.assertIn("step must be provided", str(context.exception))
