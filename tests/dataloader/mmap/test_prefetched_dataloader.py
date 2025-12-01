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
from unittest.mock import Mock, patch, MagicMock, call
import queue
import threading
import time

from hyperpod_checkpointless_training.dataloader.mmap.prefetched_dataloader import (
    CacheOnlyPrefetchedDataLoader,
    CacheResumePrefetchedDataLoader,
)
from hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader import (
    PassthroughCacheDataLoader,
)

from hyperpod_checkpointless_training.dataloader.mmap.utils import (
    PrefetchedDataLoaderSignal,
    FINAL_DATA_SIGNAL_ELEMENT,
    RestartMode,
    MockParallelStateUtil,
    MockDataLoader,
)


class TestCacheOnlyPrefetchedDataLoader(unittest.TestCase):

    def create_mock_pdl(
        self,
        cache_dir: str = "/tmp/test_cache",
        tp_size: int | None = None,
        *args,
        **kwargs,
    ):
        mock_dataloader_init = lambda target_load_step=None: MockDataLoader()
        mock_parallel_state_util = MockParallelStateUtil()
        if tp_size:
            mock_parallel_state_util.parallel_state.tp_size = tp_size

        pdl = CacheOnlyPrefetchedDataLoader(
            step=0,
            dataloader_init_callable=mock_dataloader_init,
            cache_dir=cache_dir,
            parallel_state_util=mock_parallel_state_util,
            *args,
            **kwargs,
        )

        # Set up the parallel state properly
        pdl._parallel_state = mock_parallel_state_util.parallel_state
        pdl._pdl_ranks_in_dp = mock_parallel_state_util.pdl_ranks_in_mp_group
        pdl._pdl_per_node_group = mock_parallel_state_util.pdl_ranks_in_mp_group

        # Mock the logger properly
        mock_logger = Mock()
        mock_logger.info = Mock()
        pdl._logger = mock_logger

        return pdl

    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_maybe_join_old_threads(self, mock_get_context, mock_get_rank):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()

        loader = CacheOnlyPrefetchedDataLoader(
            step=10, dataloader_init_callable=mock_dataloader_callable
        )

        mock_thread = Mock()
        loader._save_finished_thread = mock_thread

        with patch.object(
            loader.__class__.__bases__[0], "maybe_join_old_threads"
        ) as mock_parent:
            loader.maybe_join_old_threads()

            mock_parent.assert_called_once()
            mock_thread.join.assert_called_once()

    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_spawn_workers(self, mock_get_context, mock_get_rank):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()

        with patch("threading.Thread") as mock_thread_class:
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread

            loader = CacheOnlyPrefetchedDataLoader(
                step=10, dataloader_init_callable=mock_dataloader_callable
            )

            # Call spawn_workers explicitly since threads are created in __iter__, not __init__
            loader.spawn_workers()

            mock_thread_class.assert_called_with(target=loader.mp_init, daemon=True)
            mock_thread.start.assert_called_once()

    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_mp_init(self, mock_get_context, mock_get_rank):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_event = Mock()
        mock_queue = Mock()
        mock_process = Mock()

        mock_context.Event.return_value = mock_event
        mock_context.Queue.return_value = mock_queue
        mock_context.Process.return_value = mock_process
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()

        with patch("threading.Thread"):
            loader = CacheOnlyPrefetchedDataLoader(
                step=10,
                dataloader_init_callable=mock_dataloader_callable,
                saving_length=2,
            )

        loader.worker_events = []
        loader.mp_task_queues = []
        loader.mp_task_result_queues = []
        loader.workers = []

        loader.mp_init()

        self.assertEqual(len(loader.worker_events), 2)
        self.assertEqual(len(loader.mp_task_queues), 2)
        self.assertEqual(len(loader.mp_task_result_queues), 2)
        self.assertEqual(len(loader.workers), 2)

        self.assertEqual(mock_process.start.call_count, 2)

    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_init_common(self, mock_get_context, mock_get_rank):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()

        loader = CacheOnlyPrefetchedDataLoader(
            step=10, dataloader_init_callable=mock_dataloader_callable, saving_length=2
        )

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)

        with patch("threading.Thread") as mock_thread_class:
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread

            loader.init_common(mock_dl_signals)

            self.assertIsInstance(mock_dl_signals.saving_queue, queue.Queue)
            mock_thread_class.assert_called_with(
                target=loader.save_finished, args=(mock_dl_signals,), daemon=True
            )
            mock_thread.start.assert_called_once()

    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_cache_only_init(self, mock_get_context, mock_get_rank):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()

        loader = CacheOnlyPrefetchedDataLoader(
            step=10, dataloader_init_callable=mock_dataloader_callable
        )

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        loader._logger = Mock()

        with patch.object(loader, "init_common") as mock_init_common:
            loader.cache_only_init(mock_dl_signals)

            mock_init_common.assert_called_once_with(mock_dl_signals)

    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_save_finished(self, mock_get_context, mock_get_rank):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()

        loader = CacheOnlyPrefetchedDataLoader(
            step=10, dataloader_init_callable=mock_dataloader_callable
        )

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_dl_signals.should_exit.side_effect = [False, False, True]

        mock_queue_obj = Mock()
        mock_queue_obj.get.side_effect = [
            (0, "/tmp/cache_path", 100, time.time()),
            queue.Empty(),
        ]
        mock_dl_signals.saving_queue = mock_queue_obj

        mock_result_queue = Mock()
        mock_result_queue.get.return_value = True
        loader.mp_task_result_queues = [mock_result_queue]

        mock_cache = Mock()
        loader._cache = mock_cache
        loader._logger = Mock()

        loader.save_finished(mock_dl_signals)

        mock_cache.promote_content.assert_called_once_with("/tmp/cache_path", 100)
        mock_dl_signals.try_emit_to_output_queue.assert_called_once_with(
            100, logger=loader._logger, name="PDL"
        )

    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_save_finished_final_signal(self, mock_get_context, mock_get_rank):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()

        loader = CacheOnlyPrefetchedDataLoader(
            step=10, dataloader_init_callable=mock_dataloader_callable
        )

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_dl_signals.should_exit.return_value = False

        mock_queue_obj = Mock()
        mock_queue_obj.get.return_value = (None, FINAL_DATA_SIGNAL_ELEMENT, None, None)
        mock_dl_signals.saving_queue = mock_queue_obj

        mock_result_queue = Mock()
        mock_result_queue.get.return_value = True
        loader.mp_task_result_queues = [mock_result_queue]

        loader._logger = Mock()

        loader.save_finished(mock_dl_signals)

        mock_dl_signals.try_emit_to_output_queue.assert_called_once_with(
            FINAL_DATA_SIGNAL_ELEMENT, logger=loader._logger, name="PDL"
        )

    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_save_finished_exception_qempty(self, mock_get_context, mock_get_rank):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()

        loader = CacheOnlyPrefetchedDataLoader(
            step=10, dataloader_init_callable=mock_dataloader_callable
        )

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_dl_signals.should_exit.side_effect = [False, True, True]

        mock_queue_obj = Mock()
        mock_queue_obj.get.return_value = (0, "fake_tmp_path", None, None)
        mock_dl_signals.saving_queue = mock_queue_obj

        mock_result_queue = Mock()
        mock_result_queue.get.side_effect = queue.Empty()
        loader.mp_task_result_queues = [mock_result_queue]

        loader._logger = Mock()

        loader.save_finished(mock_dl_signals)

        self.assertTrue(mock_dl_signals.should_exit.call_count == 3)

    @patch("queue.Queue.get")
    @patch("threading.Thread")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_iter(self, mock_world_size, mock_get_rank, mock_thread, mock_queue_get):
        mock_get_rank.return_value = 0
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        mock_queue_get.return_value = FINAL_DATA_SIGNAL_ELEMENT

        pdl = self.create_mock_pdl()
        pdl._cache = Mock()

        # Set up dl_signals with proper Queue objects and mocked methods
        dl_signals = PrefetchedDataLoaderSignal(prefetch_length=2)
        dl_signals.set_start_to_fetch_signal = Mock()
        dl_signals.set_dl_step_signal = Mock()
        PassthroughCacheDataLoader.init(
            pdl, dl_signals
        )  # Initialize through parent class

        pdl_iter = iter(pdl)
        with self.assertRaises(StopIteration):
            next(pdl_iter)

        # 1 from spawn_workers_thread.join (PDL.__iter__)
        # 1 from _fetch_thread.join (PassDL._stop)
        self.assertTrue(mock_thread_instance.join.call_count == 2)
        mock_queue_get.assert_called_once()
        dl_signals.set_start_to_fetch_signal.assert_called_once()
        dl_signals.set_dl_step_signal.assert_called_once()

    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_emit_final_data_signal(self, mock_get_context, mock_get_rank):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()

        loader = CacheOnlyPrefetchedDataLoader(
            step=10, dataloader_init_callable=mock_dataloader_callable
        )

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        loader._logger = Mock()

        loader.emit_final_data_signal(mock_dl_signals)

        mock_dl_signals.try_emit_to_saving_queue.assert_called_once_with(
            (None, FINAL_DATA_SIGNAL_ELEMENT, None, None),
            logger=loader._logger,
            name="PDL",
        )

    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    @patch("time.time")
    def test_get_cached_batch_success(self, mock_time, mock_get_context, mock_get_rank):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context
        mock_time.return_value = 1000.0

        mock_dataloader_callable = Mock()

        loader = CacheOnlyPrefetchedDataLoader(
            step=10, dataloader_init_callable=mock_dataloader_callable
        )

        mock_cache = Mock()
        mock_batch = {"data": "test_batch"}
        mock_cache.get_content.return_value = mock_batch
        loader._cache = mock_cache
        loader._step = 5

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_dl_signals.should_exit.return_value = False

        result = loader.get_cached_batch(None, mock_dl_signals)

        self.assertEqual(result, mock_batch)
        mock_cache.get_content.assert_called_once_with(5)

    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_get_cached_batch_exit_signal(self, mock_get_context, mock_get_rank):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()

        loader = CacheOnlyPrefetchedDataLoader(
            step=10, dataloader_init_callable=mock_dataloader_callable
        )

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_dl_signals.should_exit.return_value = True

        loader._logger = Mock()

        with self.assertRaises(RuntimeError) as context:
            loader.get_cached_batch(None, mock_dl_signals)

        self.assertIn("exit event set", str(context.exception))

    @patch("torch.distributed.get_rank")
    def test_get_cached_batch_exception_filenotfound(self, mock_get_rank):
        pdl = self.create_mock_pdl()
        mock_cache = Mock()
        mock_cache.get_content.side_effect = FileNotFoundError

        pdl._cache = mock_cache
        pdl._step = 5
        pdl._wait_cache_batch_log_interval = 0

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_dl_signals.should_exit.side_effect = [False, True]

        with self.assertRaises(RuntimeError):
            pdl.get_cached_batch(5, mock_dl_signals)

        pdl._cache.get_content.assert_called_once_with(5)
        self.assertTrue(mock_dl_signals.should_exit.call_count == 2)

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.prefetched_dataloader.CacheOnlyPrefetchedDataLoader._fetch"
    )
    @patch("torch.distributed.get_rank")
    def test_fetch_data(self, mock_get_rank, mock_fetch):
        pdl = self.create_mock_pdl()
        pdl._cache = Mock()
        pdl._dl_checkpoint_load_step = 0
        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_dl_signals.wait_dl_step.return_value = False

        pdl.fetch_data(mock_dl_signals)

        mock_fetch.assert_called_once_with(pdl.data_loader, mock_dl_signals)
        mock_dl_signals.try_emit_to_saving_queue.assert_called_once()

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.prefetched_dataloader.CacheOnlyPrefetchedDataLoader._fetch"
    )
    @patch("torch.distributed.get_rank")
    def test_fetch_data_exit(self, mock_get_rank, mock_fetch):
        pdl = self.create_mock_pdl()
        pdl._cache = Mock()
        pdl._dl_checkpoint_load_step = 0
        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        mock_dl_signals.wait_dl_step.return_value = True

        pdl.fetch_data(mock_dl_signals)

        mock_fetch.assert_not_called()

    @patch("torch.distributed.get_rank")
    def test_initialize_data_loader_when_not_initialized(self, mock_get_rank):
        """Test initialize_data_loader when data_loader is None."""
        mock_get_rank.return_value = 0

        mock_dataloader = MockDataLoader()
        mock_dataloader.name = "TestDataLoader"
        mock_dataloader_init = Mock(return_value=mock_dataloader)

        pdl = self.create_mock_pdl()
        pdl._dataloader_init_callable = mock_dataloader_init
        pdl._dl_checkpoint_load_step = 5
        pdl.data_loader = None

        with patch(
            "hyperpod_checkpointless_training.dataloader.mmap.utils.dl_init_lock"
        ):
            pdl.initialize_data_loader()

        # Verify dataloader was initialized
        mock_dataloader_init.assert_called_once_with(target_load_step=5)
        self.assertEqual(pdl.data_loader, mock_dataloader)

        # Verify logging
        pdl._logger.info.assert_called_with("TestDataLoader DataLoader init finished.")

    @patch("torch.distributed.get_rank")
    def test_initialize_data_loader_when_already_initialized(self, mock_get_rank):
        """Test initialize_data_loader when data_loader is already set."""
        mock_get_rank.return_value = 0

        existing_dataloader = MockDataLoader()
        mock_dataloader_init = Mock()

        pdl = self.create_mock_pdl()
        pdl._dataloader_init_callable = mock_dataloader_init
        pdl.data_loader = existing_dataloader

        pdl.initialize_data_loader()

        # Verify dataloader init was NOT called
        mock_dataloader_init.assert_not_called()

        # Verify the existing dataloader is unchanged
        self.assertEqual(pdl.data_loader, existing_dataloader)

        # Verify warning log
        pdl._logger.info.assert_called_with(
            "Called initialize_data_loader when dataloader already inited."
        )

    @patch("torch.distributed.get_rank")
    def test_initialize_data_loader_thread_safety(self, mock_get_rank):
        """Test initialize_data_loader thread safety with dl_init_lock."""
        mock_get_rank.return_value = 0

        mock_dataloader = MockDataLoader()
        mock_dataloader.name = "ThreadSafeDataLoader"
        mock_dataloader_init = Mock(return_value=mock_dataloader)

        pdl = self.create_mock_pdl()
        pdl._dataloader_init_callable = mock_dataloader_init
        pdl._dl_checkpoint_load_step = 3
        pdl.data_loader = None

        # Mock the lock to verify it's being used
        with patch(
            "hyperpod_checkpointless_training.dataloader.mmap.prefetched_dataloader.dl_init_lock"
        ) as mock_lock:
            # Set up the mock as a proper context manager
            mock_lock_instance = Mock()
            mock_lock.__enter__ = Mock(return_value=mock_lock_instance)
            mock_lock.__exit__ = Mock(return_value=None)

            pdl.initialize_data_loader()

            # Verify lock was acquired
            mock_lock.__enter__.assert_called_once()
            mock_lock.__exit__.assert_called_once()

        # Verify dataloader was initialized within the lock
        mock_dataloader_init.assert_called_once_with(target_load_step=3)
        self.assertEqual(pdl.data_loader, mock_dataloader)

    @patch("torch.distributed.get_rank")
    def test_initialize_data_loader_with_different_target_steps(self, mock_get_rank):
        """Test initialize_data_loader with different target load steps."""
        mock_get_rank.return_value = 0

        test_cases = [0, 1, 100, 999]

        for target_step in test_cases:
            with self.subTest(target_step=target_step):
                mock_dataloader = MockDataLoader()
                mock_dataloader.name = f"DataLoader_Step_{target_step}"
                mock_dataloader_init = Mock(return_value=mock_dataloader)

                pdl = self.create_mock_pdl()
                pdl._dataloader_init_callable = mock_dataloader_init
                pdl._dl_checkpoint_load_step = target_step
                pdl.data_loader = None

                with patch(
                    "hyperpod_checkpointless_training.dataloader.mmap.utils.dl_init_lock"
                ):
                    pdl.initialize_data_loader()

                mock_dataloader_init.assert_called_once_with(
                    target_load_step=target_step
                )
                self.assertEqual(pdl.data_loader, mock_dataloader)
                pdl._logger.info.assert_called_with(
                    f"DataLoader_Step_{target_step} DataLoader init finished."
                )

    @patch("torch.distributed.get_rank")
    def test_initialize_data_loader_idempotent(self, mock_get_rank):
        """Test that initialize_data_loader is idempotent - calling multiple times has same effect."""
        mock_get_rank.return_value = 0

        mock_dataloader = MockDataLoader()
        mock_dataloader.name = "IdempotentDataLoader"
        mock_dataloader_init = Mock(return_value=mock_dataloader)

        pdl = self.create_mock_pdl()
        pdl._dataloader_init_callable = mock_dataloader_init
        pdl._dl_checkpoint_load_step = 7
        pdl.data_loader = None

        # First call - should initialize
        with patch(
            "hyperpod_checkpointless_training.dataloader.mmap.utils.dl_init_lock"
        ):
            pdl.initialize_data_loader()

        mock_dataloader_init.assert_called_once_with(target_load_step=7)
        self.assertEqual(pdl.data_loader, mock_dataloader)

        # Reset mock to verify second call behavior
        mock_dataloader_init.reset_mock()
        pdl._logger.reset_mock()

        # Second call - should not reinitialize
        pdl.initialize_data_loader()

        mock_dataloader_init.assert_not_called()
        self.assertEqual(pdl.data_loader, mock_dataloader)  # Should still be the same
        pdl._logger.info.assert_called_with(
            "Called initialize_data_loader when dataloader already inited."
        )


class TestCacheResumePrefetchedDataLoader(unittest.TestCase):

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.prefetched_dataloader.CacheResumePrefetchedDataLoader.cache_resume_init"
    )
    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_initialization(
        self, mock_get_context, mock_get_rank, mock_cache_resume, mock_init
    ):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()
        mock_model_parallel_group = Mock()

        loader = CacheResumePrefetchedDataLoader(
            step=10,
            dataloader_init_callable=mock_dataloader_callable,
            model_parallel_group=mock_model_parallel_group,
            cache_dir="/tmp/test_cache",
            lookback_length=5,
            prefetch_length=3,
        )

        self.assertEqual(loader.model_parallel_group, mock_model_parallel_group)
        mock_cache_resume.assert_called_once()
        mock_init.assert_called_once()

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch("torch.distributed.get_group_rank")
    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_cache_resume_init(
        self, mock_get_context, mock_get_rank, mock_get_group_rank, mock_init
    ):
        mock_get_rank.return_value = 0
        mock_get_group_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()
        mock_model_parallel_group = Mock()

        # First patch the __init__ to prevent auto-initialization
        with patch.object(
            CacheResumePrefetchedDataLoader, "__init__", return_value=None
        ) as mock_initialize:
            loader = CacheResumePrefetchedDataLoader.__new__(
                CacheResumePrefetchedDataLoader
            )

            # Manually set up required attributes that would normally be set in __init__
            loader._thread_lock = threading.Lock()
            loader._rank_id = 0
            loader._step = 10
            loader._prefetch_length = 3
            loader._logger = Mock()
            loader.model_parallel_group = mock_model_parallel_group
            loader._dataloader_init_callable = mock_dataloader_callable

        # Set up mock cache with all required methods
        mock_cache = MagicMock()
        mock_cache.__len__.return_value = 5
        mock_cache.get_content_indices.return_value = [10, 11, 12]
        mock_cache.all_gather_cache_size.return_value = ([5], RestartMode.WARM_START)
        mock_cache.prune_cache_init.return_value = None
        loader._cache = mock_cache

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)
        loader._step = 10
        loader._prefetch_length = 3
        loader._rank_id = 0

        loader.get_pdl_ranks_in_mp = Mock(return_value=[0, 1])
        loader._dl_checkpoint_load_step = 0
        loader._idx_prefetched = 0

        with (
            patch.object(loader, "init_common") as mock_init_common,
            patch.object(loader, "get_local_cache_size") as mock_get_cache_size,
            patch.object(loader, "_get_num_fills") as mock_get_num_fills,
            patch.object(loader, "_prune_cache") as mock_prune_cache,
            patch.object(loader, "_fill_output_queue_from_cache") as mock_fill_queue,
        ):

            mock_get_cache_size.return_value = (5, RestartMode.WARM_START)
            mock_get_num_fills.return_value = 3

            loader.cache_resume_init(mock_dl_signals)

            mock_init_common.assert_called_once_with(mock_dl_signals)
            mock_cache.prune_cache_init.assert_called_once_with(10, 3)
            mock_cache.all_gather_cache_size.assert_called_once_with(
                5, loader.get_pdl_ranks_in_mp, mock_model_parallel_group
            )
            mock_get_num_fills.assert_called_once_with([5])
            mock_prune_cache.assert_called_once_with(3)
            mock_fill_queue.assert_called_once_with(mock_dl_signals, 10)

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.prefetched_dataloader.CacheResumePrefetchedDataLoader.cache_resume_init"
    )
    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_prune_cache(
        self, mock_get_context, mock_get_rank, mock_cache_resume, mock_init
    ):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()
        mock_model_parallel_group = Mock()

        loader = CacheResumePrefetchedDataLoader(
            step=10,
            dataloader_init_callable=mock_dataloader_callable,
            model_parallel_group=mock_model_parallel_group,
        )

        mock_cache = MagicMock()
        mock_cache.__len__.return_value = 2
        loader._cache = mock_cache
        loader._step = 10
        loader._prefetch_length = 5
        loader._rank_id = 0
        loader._logger = Mock()

        with patch("torch.distributed.get_global_rank") as mock_get_global_rank:
            mock_get_global_rank.return_value = 0

            loader._prune_cache(3)

            mock_cache.prune_cache_init.assert_called_once_with(10, 3)

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.prefetched_dataloader.CacheResumePrefetchedDataLoader.cache_resume_init"
    )
    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_fill_output_queue_from_cache(
        self, mock_get_context, mock_get_rank, mock_cache_resume, mock_init
    ):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()
        mock_model_parallel_group = Mock()

        loader = CacheResumePrefetchedDataLoader(
            step=10,
            dataloader_init_callable=mock_dataloader_callable,
            model_parallel_group=mock_model_parallel_group,
        )

        mock_cache = Mock()
        mock_cache.get_content_indices.return_value = [8, 9, 10, 11]
        loader._cache = mock_cache

        mock_dl_signals = Mock(spec=PrefetchedDataLoaderSignal)

        loader._fill_output_queue_from_cache(mock_dl_signals, 10)

        expected_calls = [
            call(8, name="PDL"),
            call(9, name="PDL"),
            call(10, name="PDL"),
            call(11, name="PDL"),
        ]
        mock_dl_signals.try_emit_to_output_queue.assert_has_calls(expected_calls)

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.prefetched_dataloader.CacheResumePrefetchedDataLoader.cache_resume_init"
    )
    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_get_local_cache_size_cold_start(
        self, mock_get_context, mock_get_rank, mock_cache_resume, mock_init
    ):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()
        mock_model_parallel_group = Mock()

        loader = CacheResumePrefetchedDataLoader(
            step=10,
            dataloader_init_callable=mock_dataloader_callable,
            model_parallel_group=mock_model_parallel_group,
        )

        result = loader.get_local_cache_size(0)

        self.assertEqual(result, (0, RestartMode.COLD_START))

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader.PassthroughCacheDataLoader.init"
    )
    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.prefetched_dataloader.CacheResumePrefetchedDataLoader.cache_resume_init"
    )
    @patch("torch.distributed.get_rank")
    @patch("torch.multiprocessing.get_context")
    def test_get_local_cache_size_warm_start(
        self, mock_get_context, mock_get_rank, mock_cache_resume, mock_init
    ):
        mock_get_rank.return_value = 0
        mock_context = Mock()
        mock_get_context.return_value = mock_context

        mock_dataloader_callable = Mock()
        mock_model_parallel_group = Mock()

        loader = CacheResumePrefetchedDataLoader(
            step=10,
            dataloader_init_callable=mock_dataloader_callable,
            model_parallel_group=mock_model_parallel_group,
        )

        result = loader.get_local_cache_size(5)

        self.assertEqual(result, (5, RestartMode.WARM_START))
