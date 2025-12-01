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
import queue
import threading
import unittest
from unittest.mock import Mock, patch

import torch

from hyperpod_checkpointless_training.dataloader.mmap.utils import (
    WORKER_SAVE_FINISH,
    MockDataLoader,
    PrefetchedDataLoaderSignal,
    RestartMode,
    check_shared_memory_batch,
    save_worker,
)
from hyperpod_checkpointless_training.inprocess.logger import get_logger


class TestRestartMode(unittest.TestCase):

    def test_cold_start_value(self):
        self.assertEqual(RestartMode.COLD_START.value, "cold_start")

    def test_warm_start_value(self):
        self.assertEqual(RestartMode.WARM_START.value, "warm_start")


class TestPrefetchedDataLoaderSignal(unittest.TestCase):

    def test_init(self):
        signal = PrefetchedDataLoaderSignal(prefetch_length=5)
        self.assertIsInstance(signal.exit_event, threading.Event)
        self.assertIsInstance(signal.dl_step_signal, threading.Event)
        self.assertIsInstance(signal.signal_start_to_fetch, threading.Event)
        self.assertIsInstance(signal.output_queue, queue.Queue)
        self.assertIsNone(signal.saving_queue)
        self.assertEqual(signal.output_queue.maxsize, 5)

    def test_set_exit(self):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        self.assertFalse(signal.should_exit())
        signal.set_exit()
        self.assertTrue(signal.should_exit())

    def test_should_exit(self):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        self.assertFalse(signal.should_exit())
        signal.exit_event.set()
        self.assertTrue(signal.should_exit())

    def test_set_dl_step_signal(self):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        self.assertFalse(signal.dl_step_signal.is_set())
        signal.set_dl_step_signal()
        self.assertTrue(signal.dl_step_signal.is_set())

    def test_is_dl_step_set(self):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        self.assertFalse(signal.is_dl_step_set())
        signal.dl_step_signal.set()
        self.assertTrue(signal.is_dl_step_set())

    def test_set_start_to_fetch_signal(self):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        self.assertFalse(signal.signal_start_to_fetch.is_set())
        signal.set_start_to_fetch_signal()
        self.assertTrue(signal.signal_start_to_fetch.is_set())

    @patch("time.sleep")
    def test_wait_or_exit_with_signal_set(self, mock_sleep):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        wait_event = threading.Event()
        wait_event.set()

        result = signal.wait_or_exit(wait_event, "TEST")
        self.assertFalse(result)
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_wait_or_exit_with_exit_set(self, mock_sleep):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        wait_event = threading.Event()
        signal.set_exit()

        result = signal.wait_or_exit(wait_event, "TEST")
        self.assertTrue(result)

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_or_exit_with_warning(self, mock_time, mock_sleep):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        wait_event = threading.Event()

        mock_time.side_effect = [0, 101, 101, 101]
        mock_sleep.side_effect = [None, None, signal.set_exit()]

        result = signal.wait_or_exit(wait_event, "TEST")
        self.assertTrue(result)

    def test_wait_or_exit_no_signal(self):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        result = signal.wait_or_exit(None, "TEST")
        self.assertFalse(result)

    @patch("time.sleep")
    def test_wait_dl_step(self, mock_sleep):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        signal.dl_step_signal.set()

        result = signal.wait_dl_step()
        self.assertFalse(result)

    @patch("time.sleep")
    def test_wait_start_to_fetch(self, mock_sleep):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        signal.signal_start_to_fetch.set()

        result = signal.wait_start_to_fetch()
        self.assertFalse(result)

    @patch("hyperpod_checkpointless_training.dataloader.mmap.utils.dist.get_rank")
    def test_try_emit_to_queue_success(self, mock_get_rank):
        mock_get_rank.return_value = 0
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        test_queue = queue.Queue(maxsize=2)
        mock_logger = Mock()

        result = signal.try_emit_to_queue("test_data", test_queue, mock_logger, "TEST")

        self.assertFalse(result)
        self.assertEqual(test_queue.get(), "test_data")
        mock_logger.debug.assert_called_with("Successfully put batch in queue.")

    @patch("hyperpod_checkpointless_training.dataloader.mmap.utils.dist.get_rank")
    def test_try_emit_to_queue_exit_signal(self, mock_get_rank):
        mock_get_rank.return_value = 0
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        test_queue = queue.Queue(maxsize=2)
        mock_logger = Mock()
        signal.set_exit()

        result = signal.try_emit_to_queue("test_data", test_queue, mock_logger, "TEST")

        self.assertTrue(result)
        mock_logger.debug.assert_called_with("Received exit signal")

    @patch("hyperpod_checkpointless_training.dataloader.mmap.utils.dist.get_rank")
    @patch("time.sleep")
    def test_try_emit_to_queue_full_with_warning(self, mock_sleep, mock_get_rank):
        mock_get_rank.return_value = 0
        signal = PrefetchedDataLoaderSignal(prefetch_length=1)
        test_queue = queue.Queue(maxsize=0)
        mock_logger = Mock()

        def side_effect(*args, **kwargs):
            if mock_logger.warning.call_count >= 1:
                signal.set_exit()
            raise queue.Full()

        test_queue.put = Mock(side_effect=side_effect)

        result = signal.try_emit_to_queue("test_data", test_queue, mock_logger, "TEST")

        self.assertTrue(result)
        mock_logger.warning.assert_called()

    def test_try_emit_to_output_queue(self):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        mock_logger = Mock()

        with patch.object(signal, "try_emit_to_queue") as mock_try_emit:
            mock_try_emit.return_value = False
            result = signal.try_emit_to_output_queue("test_data", mock_logger, "TEST")

            self.assertFalse(result)
            mock_try_emit.assert_called_once_with(
                "test_data", signal.output_queue, mock_logger, "TEST"
            )

    def test_try_emit_to_saving_queue(self):
        signal = PrefetchedDataLoaderSignal(prefetch_length=3)
        test_queue = queue.Queue()
        mock_logger = Mock()

        with patch.object(signal, "try_emit_to_queue") as mock_try_emit:
            with patch.object(signal, "saving_queue", test_queue):
                mock_try_emit.return_value = False
                result = signal.try_emit_to_saving_queue(
                    "test_data", mock_logger, "TEST"
                )

                self.assertFalse(result)
                mock_try_emit.assert_called_once_with(
                    "test_data", test_queue, mock_logger, "TEST"
                )

    def test_fill_queue(self):
        test_queue = queue.Queue()
        PrefetchedDataLoaderSignal.fill_queue(3, test_queue)

        self.assertEqual(test_queue.qsize(), 3)
        for _ in range(3):
            self.assertEqual(test_queue.get(), {})


class TestSaveWorker(unittest.TestCase):

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.utils.torch.set_num_threads"
    )
    @patch("hyperpod_checkpointless_training.dataloader.mmap.utils.torch.save")
    def test_save_worker_success(self, mock_torch_save, mock_set_threads):
        task_queue = Mock()
        worker_exit_event = Mock()
        mp_task_result_queue = Mock()

        mock_batch = {"data": "test"}
        mock_temp_path = "/tmp/test.pt"
        task_queue.get.side_effect = [(mock_batch, mock_temp_path), queue.Empty()]
        worker_exit_event.is_set.side_effect = [False, False, True, True]
        mp_task_result_queue.put.return_value = None

        save_worker(task_queue, worker_exit_event, mp_task_result_queue)

        mock_set_threads.assert_called_once_with(1)
        mock_torch_save.assert_called_once_with(mock_batch, mock_temp_path)
        mp_task_result_queue.put.assert_called_with(WORKER_SAVE_FINISH, timeout=1)

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.utils.torch.set_num_threads"
    )
    def test_save_worker_empty_queue(self, mock_set_threads):
        task_queue = Mock()
        worker_exit_event = Mock()
        mp_task_result_queue = Mock()

        task_queue.get.side_effect = queue.Empty()
        worker_exit_event.is_set.return_value = True

        save_worker(task_queue, worker_exit_event, mp_task_result_queue)

        mock_set_threads.assert_called_once_with(1)

    @patch(
        "hyperpod_checkpointless_training.dataloader.mmap.utils.torch.set_num_threads"
    )
    @patch("hyperpod_checkpointless_training.dataloader.mmap.utils.torch.save")
    def test_save_worker_no_temp_path(self, mock_torch_save, mock_set_threads):
        task_queue = Mock()
        worker_exit_event = Mock()
        mp_task_result_queue = Mock()

        mock_batch = {"data": "test"}
        task_queue.get.side_effect = [(mock_batch, None), queue.Empty()]
        worker_exit_event.is_set.side_effect = [False, False, True, True]
        mp_task_result_queue.put.return_value = None

        save_worker(task_queue, worker_exit_event, mp_task_result_queue)

        mock_set_threads.assert_called_once_with(1)
        mock_torch_save.assert_not_called()
        mp_task_result_queue.put.assert_called_with(WORKER_SAVE_FINISH, timeout=1)


class TestCheckSharedMemoryBatch(unittest.TestCase):

    def test_check_shared_memory_batch_basic(self):
        batch = {"data": "not_a_tensor", "number": 42}
        check_shared_memory_batch(batch, "TEST")

    def test_check_shared_memory_batch_not_in_shared_mem(self):
        batch = {"tensor": torch.arange(101)}
        logger = get_logger("shared_memory_check")
        with self.assertLogs(logger=logger, level="WARNING"):
            check_shared_memory_batch(batch, "TEST")


class TestMockConstructs(unittest.TestCase):

    def test_mock_dl(self):
        mock_dl = MockDataLoader(data=["a", "b"])
        dl_iter = iter(mock_dl)
        next(dl_iter)
        next(dl_iter)
        with self.assertRaises(StopIteration):
            next(dl_iter)
