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
import os
import queue
import unittest
from unittest.mock import Mock, call, patch

import torch

from hyperpod_checkpointless_training.dataloader.mmap.cache import TTLMMAPCache
from hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader import (
    PassthroughCacheDataLoader,
)
from hyperpod_checkpointless_training.dataloader.mmap.utils import (
    FINAL_DATA_SIGNAL_ELEMENT,
    MockDataLoader,
    MockParallelState,
    MockParallelStateUtil,
    PrefetchedDataLoaderSignal,
)


class TestPassthroughCacheDataLoader(unittest.TestCase):

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def setUp(self, mock_world_size, mock_rank):
        self.cache_dir = "/tmp/test_cache"
        self.passthrough_dl = self.create_mock_pass_dl(cache_dir=self.cache_dir)

    def create_mock_pass_dl(
        self, cache_dir: str = "/tmp/test_cache", tp_size: int | None = None
    ):
        mock_dataloader_init = lambda target_load_step=None: MockDataLoader()
        mock_parallel_state_util = MockParallelStateUtil()
        if tp_size:
            mock_parallel_state_util.parallel_state.tp_size = tp_size

        passthrough_dl = PassthroughCacheDataLoader(
            step=0,
            dataloader_init_callable=mock_dataloader_init,
            cache_dir=cache_dir,
            parallel_state_util=mock_parallel_state_util,
        )

        # Set up the parallel state properly
        passthrough_dl._parallel_state = mock_parallel_state_util.parallel_state
        passthrough_dl._pdl_ranks_in_dp = mock_parallel_state_util.pdl_ranks_in_mp_group
        return passthrough_dl

    def test_init(self):
        self.assertEqual(self.passthrough_dl._step, 0)
        self.assertEqual(self.passthrough_dl._cache_dir, self.cache_dir)
        self.assertEqual(self.passthrough_dl._read_only, True)
        self.assertEqual(self.passthrough_dl._lookback_length, 0)
        self.assertEqual(self.passthrough_dl._prefetch_length, 0)

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.get_world_size", return_value=8)
    def test_pdl_per_node_group(
        self, mock_world_size, mock_rank, mock_is_init, mock_new_group
    ):
        os.environ["LOCAL_WORLD_SIZE"] = "8"

        passthrough_dl = self.create_mock_pass_dl(cache_dir=self.cache_dir, tp_size=4)

        mock_new_group.return_value = Mock()
        group = passthrough_dl.pdl_per_node_group
        self.assertIsNotNone(group)

    @patch("torch.distributed.get_rank", return_value=1)
    def test_skip_pdl_per_node_group_tp_size_large(self, mock_rank):
        """Test pdl_per_node_group returns None when tp_size >= local_world_size."""
        os.environ["LOCAL_WORLD_SIZE"] = "4"

        # Set up parallel state with tp_size >= local_world_size
        # tp_size >= 4
        passthrough_dl = self.create_mock_pass_dl(cache_dir=self.cache_dir, tp_size=8)

        group = passthrough_dl.pdl_per_node_group
        self.assertEqual(group, None)

    def test_is_data_rank(self):
        self.passthrough_dl._parallel_state = MockParallelState()
        is_data = self.passthrough_dl.is_data_rank()
        self.assertTrue(is_data)
        self.assertTrue(self.passthrough_dl.is_data_rank(0))

    @patch("threading.Thread")
    def test_start_fetch_thread(self, mock_thread):
        mock_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        self.passthrough_dl._start_fetch_thread(mock_signals)

        mock_thread.assert_called_once_with(
            target=self.passthrough_dl.fetch_data,
            args=(mock_signals,),
            daemon=False,
        )
        mock_thread_instance.start.assert_called_once()

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.TTLMMAPCache"
    )
    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_rank", return_vaue=0)
    def test_init_with_signals(
        self,
        mock_get_rank,
        mock_is_initialized,
        mock_world_size,
        mock_new_group,
        mock_cache_class,
    ):
        mock_cache = Mock(spec=TTLMMAPCache)
        mock_cache_class.return_value = mock_cache
        mock_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_new_group.return_value = Mock()

        # Set up environment for proper pdl_per_node_group calculation
        import os

        os.environ["LOCAL_WORLD_SIZE"] = "8"

        passdl = self.create_mock_pass_dl(cache_dir=self.cache_dir)

        with patch.object(passdl, "_start_fetch_thread") as mock_start_thread:
            passdl.init(mock_signals)

            mock_cache_class.assert_called_once()
            mock_cache.init.assert_called_once()
            mock_start_thread.assert_called_once_with(mock_signals)

    def test_fetch_data_exit(self):
        mock_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_signals.wait_dl_step.return_value = True
        self.passthrough_dl.fetch_data(mock_signals)
        mock_signals.wait_dl_step.assert_called_once()
        mock_signals.wait_start_to_fetch.assert_not_called()

    def test_fetch_data(self):
        mock_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_signals.wait_dl_step.return_value = False
        mock_signals.wait_start_to_fetch.return_value = False
        mock_signals.try_emit_to_output_queue.return_value = True

        self.passthrough_dl.data_loader = None
        self.passthrough_dl._dl_checkpoint_load_step = None

        with patch(
            "hyperpod_checkpointless_training.dataloader.mmap.utils.dl_init_lock"
        ):
            self.passthrough_dl.fetch_data(mock_signals)

            mock_signals.wait_dl_step.assert_called_once()
            mock_signals.wait_start_to_fetch.assert_called_once()
            mock_signals.try_emit_to_output_queue.assert_has_calls(
                [
                    call({}, logger=self.passthrough_dl._logger, name="PassDL"),
                    call(
                        FINAL_DATA_SIGNAL_ELEMENT,
                        logger=self.passthrough_dl._logger,
                        name="PassDL",
                    ),
                ]
            )

    def test_maybe_join_old_threads(self):
        mock_thread = Mock()
        self.passthrough_dl._fetch_thread = mock_thread

        self.passthrough_dl.maybe_join_old_threads()

        mock_thread.join.assert_called_once()

    def test_get_cached_batch(self):
        mock_elem = {"data": torch.tensor([1, 2, 3])}
        mock_signals = Mock(spec=PrefetchedDataLoaderSignal)

        result = self.passthrough_dl.get_cached_batch(mock_elem, mock_signals)

        self.assertEqual(result, mock_elem)

    def test_stop(self):
        mock_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_thread = Mock()
        self.passthrough_dl._fetch_thread = mock_thread
        self.passthrough_dl.data_loader = Mock()
        mock_cache = Mock(spec=TTLMMAPCache)
        self.passthrough_dl._cache = mock_cache

        self.passthrough_dl._stop(mock_signals)

        mock_signals.set_exit.assert_called_once()
        mock_thread.join.assert_called_once_with(timeout=30)
        mock_cache.set_final_index.assert_called_once_with(self.passthrough_dl._step)

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_stop_before_dataloader_init(self, mock_world, mock_rank):
        mock_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_dataloader_init = lambda target_load_step=None: MockDataLoader()
        mock_parallel_state_util = MockParallelStateUtil()

        # construct a PassDL, but simulate val dataloader where we don't initialize the dataloader yet
        passthrough_dl = PassthroughCacheDataLoader(
            step=0,
            dataloader_init_callable=mock_dataloader_init,
            cache_dir="dev/shm",
            parallel_state_util=mock_parallel_state_util,
        )
        passthrough_dl._stop(mock_signals)
        mock_signals.set_exit.assert_not_called()

    def test_iter_components_initialization(self):
        """Test that iteration components can be initialized properly."""
        mock_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_signals.should_exit.return_value = True  # Exit immediately

        # Test that we can create the signals object
        self.assertIsNotNone(mock_signals)

        # Test thread lock exists
        self.assertIsNotNone(self.passthrough_dl._thread_lock)

        # Test maybe_join_old_threads doesn't crash when no thread exists
        self.passthrough_dl._fetch_thread = None
        self.passthrough_dl.maybe_join_old_threads()  # Should not raise

    def test_iter_signal_handling(self):
        """Test signal handling components without full iteration."""
        # Test PrefetchedDataLoaderSignal creation
        signals = PrefetchedDataLoaderSignal(prefetch_length=2)
        self.assertIsNotNone(signals)
        self.assertIsNotNone(signals.output_queue)

        # Test that we can call should_exit
        self.assertFalse(signals.should_exit())

        # Test that we can set exit
        signals.set_exit()
        self.assertTrue(signals.should_exit())

    @patch("queue.Queue.get")
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader._stop"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch("torch.distributed.get_rank", return_value=0)
    def test_iter_outq_data_signal_elem(
        self, mock_get_rank, mock_init, mock_stop, mock_queue_get
    ):
        # output_queue.get() returns final data signal element from fetch thread
        mock_queue_get.return_value = FINAL_DATA_SIGNAL_ELEMENT

        passthrough_dl = self.create_mock_pass_dl(cache_dir=self.cache_dir)
        passthrough_dl._cache = Mock()
        passthrough_dl.dl_signals = PrefetchedDataLoaderSignal(prefetch_length=2)
        passdl_iter = iter(passthrough_dl)
        with self.assertRaises(StopIteration):
            next(passdl_iter)

        mock_queue_get.assert_called_once()
        mock_stop.assert_called_once()

    @patch("queue.Queue.get")
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader._stop"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch("torch.distributed.get_rank", return_value=0)
    def test_iter_outq_return_step_dur_train(
        self, mock_get_rank, mock_init, mock_stop, mock_queue_get
    ):
        # tests a few iterations during normal training
        elem_list = [0, 1, 2]
        iterations = len(elem_list)
        mock_queue_get.side_effect = elem_list

        mock_cache = Mock(spec=TTLMMAPCache)

        mock_cache.is_final_index.side_effect = [False for _ in range(iterations)]

        passthrough_dl = self.create_mock_pass_dl(cache_dir=self.cache_dir)
        passthrough_dl._dl_checkpoint_load_step = 0
        passthrough_dl._cache = mock_cache
        passthrough_dl.dl_signals = PrefetchedDataLoaderSignal(prefetch_length=2)

        passdl_iter = iter(passthrough_dl)

        # verify that returned element is as expected
        for expected_elem in elem_list:
            elem = next(passdl_iter)
            self.assertTrue(elem == expected_elem)

        self.assertTrue(mock_queue_get.call_count == iterations)
        self.assertTrue(mock_cache.is_final_index.call_count == iterations)
        # verify stop is not called as loop should not have completed
        mock_stop.assert_not_called()

    @patch("queue.Queue.get")
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader._stop"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch("torch.distributed.get_rank", return_value=0)
    def test_cleanup_expired_files_start_on_dl_load_step(
        self, mock_get_rank, mock_init, mock_stop, mock_queue_get
    ):
        # tests a few iterations during normal training
        elem_list = [0, 1, 2, 3, 4, 5]
        iterations = len(elem_list)
        mock_queue_get.side_effect = elem_list

        mock_cache = Mock(spec=TTLMMAPCache)

        mock_cache.is_final_index.side_effect = [False for _ in range(iterations)]

        passthrough_dl = self.create_mock_pass_dl(cache_dir=self.cache_dir)
        passthrough_dl._dl_checkpoint_load_step = 3
        passthrough_dl._cache = mock_cache
        passthrough_dl.dl_signals = PrefetchedDataLoaderSignal(prefetch_length=2)

        passdl_iter = iter(passthrough_dl)

        # verify that returned element is as expected
        for expected_elem in elem_list:
            elem = next(passdl_iter)
            self.assertTrue(elem == expected_elem)

            # step is incremented after next() so we need to decrement by 1 for right time to check
            if passthrough_dl._step - 1 < passthrough_dl._dl_checkpoint_load_step:
                # verify that start cleanup thread is not yet called
                mock_cache.start_cleanup_expired_files_thread.assert_not_called()
            if passthrough_dl._step - 1 == passthrough_dl._dl_checkpoint_load_step:
                # verify that start cleanup thread is called at _dl_checkpoint_load_step
                mock_cache.start_cleanup_expired_files_thread.assert_called_once()

        # verify that start cleanup thread is only called once
        mock_cache.start_cleanup_expired_files_thread.assert_called_once()
        self.assertTrue(mock_queue_get.call_count == iterations)
        self.assertTrue(mock_cache.is_final_index.call_count == iterations)
        # verify stop is not called as loop should not have completed
        mock_stop.assert_not_called()

    @patch("queue.Queue.get")
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader._stop"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch("torch.distributed.get_rank", return_value=0)
    def test_iter_outq_exception_qempty(
        self, mock_get_rank, mock_init, mock_stop, mock_queue_get
    ):
        # test standard case where we wait for output_queue to get populated before a value is yielded
        # output_queue.get() raises a queue.Empty() exception twice, and then yields an actual value
        expected_elem = 3
        mock_queue_get.side_effect = [queue.Empty(), queue.Empty(), expected_elem]

        mock_cache = Mock(spec=TTLMMAPCache)

        mock_cache.is_final_index.side_effect = [False, False, True]

        passthrough_dl = self.create_mock_pass_dl(cache_dir=self.cache_dir)
        passthrough_dl._cache = mock_cache
        passthrough_dl.dl_signals = PrefetchedDataLoaderSignal(prefetch_length=2)
        passthrough_dl._wait_elem_log_interval = 0

        passdl_iter = iter(passthrough_dl)

        elem = next(passdl_iter)
        self.assertTrue(elem == expected_elem)

        self.assertTrue(mock_queue_get.call_count == 3)
        self.assertTrue(mock_cache.is_final_index.call_count == 3)
        mock_stop.assert_not_called()

    @patch("queue.Queue.get")
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader._stop"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch("torch.distributed.get_rank", return_value=0)
    def test_iter_final_outq_return_exception(
        self, mock_get_rank, mock_init, mock_stop, mock_queue_get
    ):
        # output_queue.get() returns an exception element from fetch thread
        mock_queue_get.return_value = Exception()

        mock_cache = Mock(spec=TTLMMAPCache)

        mock_cache.is_final_index.return_value = False

        passthrough_dl = self.create_mock_pass_dl(cache_dir=self.cache_dir)
        passthrough_dl._cache = mock_cache
        passthrough_dl.dl_signals = PrefetchedDataLoaderSignal(prefetch_length=2)

        passdl_iter = iter(passthrough_dl)

        # verify exception element is re-raised properly
        with self.assertRaises(Exception):
            next(passdl_iter)

        mock_queue_get.assert_called_once()
        mock_cache.is_final_index.assert_called_once()
        mock_stop.assert_called_once()

    def test_get_num_fills(self):
        pdl_cache_sizes = {0: 5, 1: 3, 2: 0, 3: 7}

        result = PassthroughCacheDataLoader._get_num_fills(pdl_cache_sizes)

        self.assertEqual(result, 3)

    def test_get_num_fills_empty(self):
        pdl_cache_sizes = {0: 0, 1: 0}

        result = PassthroughCacheDataLoader._get_num_fills(pdl_cache_sizes)

        self.assertEqual(result, 0)

    def test_get_num_fills_all_empty(self):
        pdl_cache_sizes = {}

        result = PassthroughCacheDataLoader._get_num_fills(pdl_cache_sizes)

        self.assertEqual(result, 0)

    def test_del_with_cleanup(self):
        mock_dataloader = Mock()
        mock_dataloader.clean_up = Mock()
        self.passthrough_dl.data_loader = mock_dataloader

        self.passthrough_dl.__del__()

        mock_dataloader.clean_up.assert_called_once()

    def test_del_without_cleanup(self):
        mock_dataloader = Mock()
        del mock_dataloader.clean_up
        self.passthrough_dl.data_loader = mock_dataloader

        try:
            self.passthrough_dl.__del__()
        except AttributeError:
            self.fail("__del__ should handle missing clean_up method gracefully")

    def test_del_with_exception(self):
        mock_dataloader = Mock()
        mock_dataloader.clean_up.side_effect = Exception("Test exception")
        self.passthrough_dl.data_loader = mock_dataloader

        # Capture the warning to prevent it from appearing in test output
        with patch.object(self.passthrough_dl._logger, "warning") as mock_warning:
            # Suppress the actual warning output during testing
            with patch("logging.getLogger"):
                self.passthrough_dl.__del__()
            mock_warning.assert_called_once()

        # Clean up to prevent the warning from appearing during teardown
        self.passthrough_dl.data_loader = None

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_getattr_no_data_loader(self, mock_world_size, mock_rank):
        passthrough_dl = self.create_mock_pass_dl()

        with self.assertRaises(AttributeError):
            _ = passthrough_dl.some_attribute

    def test_getattr_data_loader_none(self):
        self.passthrough_dl.data_loader = None

        with self.assertRaises(AttributeError):
            _ = self.passthrough_dl.some_attribute

    def test_getattr_success(self):
        mock_dataloader = Mock()
        mock_dataloader.some_attribute = "test_value"
        self.passthrough_dl.data_loader = mock_dataloader

        result = self.passthrough_dl.some_attribute

        self.assertEqual(result, "test_value")

    def test_is_data_rank_with_different_ranks(self):
        self.passthrough_dl._parallel_state = MockParallelState()
        self.passthrough_dl._parallel_state.tp_size = 4

        self.assertTrue(self.passthrough_dl.is_data_rank(0))
        self.assertFalse(self.passthrough_dl.is_data_rank(1))
        self.assertFalse(self.passthrough_dl.is_data_rank(2))
        self.assertFalse(self.passthrough_dl.is_data_rank(3))
        self.assertTrue(self.passthrough_dl.is_data_rank(4))

    def test_get_pdl_ranks_in_mp(self):
        self.passthrough_dl._parallel_state = MockParallelState()
        self.passthrough_dl._pdl_ranks_in_dp = {0: [0, 1, 2]}

        result = self.passthrough_dl.get_pdl_ranks_in_mp

        self.assertEqual(result, [0, 1, 2])

    def test_len_with_dataloader_initialized(self):
        """Test __len__ when data_loader is available and _length is None."""
        mock_dataloader = MockDataLoader()
        mock_dataloader.data = [{"data": f"item_{i}"} for i in range(50)]
        self.passthrough_dl.data_loader = mock_dataloader
        self.passthrough_dl._length = None

        result = len(self.passthrough_dl)

        self.assertEqual(result, 50)

    def test_set_length_functionality(self):
        """Test the set_length method functionality."""
        initial_length = self.passthrough_dl._length
        self.assertIsNone(initial_length)

        self.passthrough_dl.set_length(200)

        self.assertEqual(self.passthrough_dl._length, 200)
        self.assertEqual(len(self.passthrough_dl), 200)

    def test_len_caching_behavior(self):
        """Test that cached length takes precedence over dataloader length."""
        mock_dataloader = MockDataLoader()
        mock_dataloader.data = [{"data": f"item_{i}"} for i in range(30)]
        self.passthrough_dl.data_loader = mock_dataloader

        # Set cached length different from dataloader length
        self.passthrough_dl.set_length(75)

        result = len(self.passthrough_dl)

        # Should return cached length, not dataloader length
        self.assertEqual(result, 75)
        self.assertNotEqual(result, 30)

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader._stop"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    def test_iter_force_cold_start(
        self, mock_world_size, mock_rank, mock_cache_init, mock
    ):
        """Test that force_cold_start=True triggers the expected initialization sequence."""
        # Create dataloader with force_cold_start=True
        passthrough_dl = self.create_mock_pass_dl()
        passthrough_dl._force_cold_start = True

        # Mock the required methods and objects
        mock_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_signals.should_exit.return_value = True  # Exit immediately
        mock_signals.output_queue = Mock()
        mock_signals.output_queue.get.return_value = FINAL_DATA_SIGNAL_ELEMENT
        mock_cache = Mock(spec=TTLMMAPCache)
        passthrough_dl._cache = mock_cache
        passthrough_dl.dl_signals = mock_signals  # Set dl_signals on instance

        with (
            patch.object(passthrough_dl, "maybe_join_old_threads") as mock_join,
            patch.object(passthrough_dl, "init") as mock_init,
            patch(
                "hyperpod_checkpointless_training.dataloader.mmap.utils.PrefetchedDataLoaderSignal",
                return_value=mock_signals,
            ) as mock_signal_class,
        ):

            # Start iteration
            iterator = iter(passthrough_dl)

            # Try to get next item to trigger cache checks
            with self.assertRaises(StopIteration):
                next(iterator)

            # Verify all expected methods were called in order
            mock_join.assert_called_once()
            mock_signals.set_start_to_fetch_signal.assert_called_once()
            mock_signals.set_dl_step_signal.assert_called_once()
