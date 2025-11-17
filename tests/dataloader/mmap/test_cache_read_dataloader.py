# type: ignore
"""Simple unit test for cache_read_dataloader module."""

import unittest
from unittest.mock import Mock, patch

from hyperpod_checkpointless_training.dataloader.mmap.cache_read_dataloader import (
    CacheOnlyReadDataLoader,
    CacheResumeReadDataLoader,
)
from hyperpod_checkpointless_training.dataloader.mmap.utils import (
    PrefetchedDataLoaderSignal,
)


class TestCacheOnlyReadDataLoader(unittest.TestCase):
    """Simple test for CacheOnlyReadDataLoader."""

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.dist.get_rank",
        return_value=0,
    )
    def test_initialization_happy_case(self, mock_get_rank):
        """Test that CacheOnlyReadDataLoader can be initialized successfully."""
        mock_dataloader_callable = Mock()

        loader = CacheOnlyReadDataLoader(
            step=10,
            dataloader_init_callable=mock_dataloader_callable,
            cache_dir="/tmp/test_cache",
            lookback_length=5,
            prefetch_length=3,
        )

        self.assertEqual(loader._original_step, 10)
        self.assertEqual(loader._lookback_length, 5)
        self.assertEqual(loader._prefetch_length, 3)
        self.assertEqual(loader._cache_dir, "/tmp/test_cache")
        self.assertTrue(loader._read_only)

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.dist.get_rank",
        return_value=0,
    )
    def test_init_method(self, mock_get_rank):
        """Test the init method calls parent and sets signals."""
        mock_dataloader_callable = Mock()

        loader = CacheOnlyReadDataLoader(
            step=10,
            dataloader_init_callable=mock_dataloader_callable,
            cache_dir="/tmp/test_cache",
        )

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)

        with patch.object(loader.__class__.__bases__[0], "init") as mock_parent_init:
            loader.init(mock_dl_signals)

            mock_parent_init.assert_called_once_with(mock_dl_signals)

            mock_dl_signals.set_start_to_fetch_signal.assert_called_once()
            mock_dl_signals.set_dl_step_signal.assert_called_once()

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.dist.get_rank",
        return_value=0,
    )
    @patch("time.sleep")
    def test_get_cached_batch_success(self, mock_sleep, mock_get_rank):
        """Test get_cached_batch returns batch when cache file exists."""

        mock_dataloader_callable = Mock()

        loader = CacheOnlyReadDataLoader(
            step=10,
            dataloader_init_callable=mock_dataloader_callable,
            cache_dir="/tmp/test_cache",
        )

        mock_cache = Mock()
        mock_batch = {"data": "test_batch"}
        mock_cache.get_content.return_value = mock_batch
        mock_cache.is_final_index.return_value = False
        loader._cache = mock_cache
        loader._step = 5

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_dl_signals.should_exit.return_value = False

        result = loader.get_cached_batch(None, mock_dl_signals)

        self.assertEqual(result, mock_batch)
        mock_cache.get_content.assert_called_once_with(5)
        mock_dl_signals.should_exit.assert_called_once()

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.dist.get_rank",
        return_value=0,
    )
    def test_get_cached_batch_exit_signal(self, mock_get_rank):
        """Test get_cached_batch raises RuntimeError when exit signal is set."""
        mock_dataloader_callable = Mock()

        loader = CacheOnlyReadDataLoader(
            step=10,
            dataloader_init_callable=mock_dataloader_callable,
            cache_dir="/tmp/test_cache",
        )

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_dl_signals.should_exit.return_value = True

        loader._logger = Mock()

        with self.assertRaises(RuntimeError) as context:
            loader.get_cached_batch(None, mock_dl_signals)

        self.assertIn("exit event set", str(context.exception))
        mock_dl_signals.should_exit.assert_called_once()
        mock_dl_signals.set_exit.assert_called_once()
        loader._logger.error.assert_called_once()

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.dist.get_rank",
        return_value=0,
    )
    def test_get_cached_batch_exception_filenotfound(self, mock_get_rank):
        """Test get_cached_batch handles FileNotFoundError properly."""
        mock_cache = Mock()
        mock_cache.get_content.side_effect = FileNotFoundError
        mock_cache.is_final_index.return_value = False
        mock_dataloader_callable = Mock()

        step = 10
        cdl = CacheOnlyReadDataLoader(
            step=step,
            dataloader_init_callable=mock_dataloader_callable,
            cache_dir="/tmp/test_cache",
        )
        cdl._cache = mock_cache
        cdl._wait_cache_batch_log_interval = 0

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_dl_signals.should_exit.side_effect = [False, True]

        cdl._logger = Mock()

        with self.assertRaises(RuntimeError) as context:
            cdl.get_cached_batch(None, mock_dl_signals)

        self.assertIn("exit event set", str(context.exception))

        self.assertTrue(mock_dl_signals.should_exit.call_count == 2)
        cdl._logger.error.assert_called_once()
        cdl._cache.get_content.assert_called_once_with(step)
        self.assertTrue(mock_dl_signals.should_exit.call_count == 2)


class TestCacheResumeReadDataLoader(unittest.TestCase):
    """Simple test for CacheOnlyReadDataLoader."""

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.cache_read_dataloader.CacheResumeReadDataLoader.init"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.dist.get_rank",
        return_value=0,
    )
    def test_initialization_happy_case(self, mock_get_rank, mock_init, mock_cache_init):
        """Test that CacheOnlyReadDataLoader can be initialized successfully."""
        mock_dataloader_callable = Mock()
        mock_model_parallel_group = Mock()

        loader = CacheResumeReadDataLoader(
            step=10,
            dataloader_init_callable=mock_dataloader_callable,
            model_parallel_group=mock_model_parallel_group,
            cache_dir="/tmp/test_cache",
            lookback_length=5,
            prefetch_length=3,
        )

        self.assertEqual(loader._original_step, 10)
        self.assertEqual(loader._lookback_length, 5)
        self.assertEqual(loader._prefetch_length, 3)
        self.assertEqual(loader._cache_dir, "/tmp/test_cache")
        self.assertTrue(loader._read_only)
        mock_cache_init.assert_called_once()
