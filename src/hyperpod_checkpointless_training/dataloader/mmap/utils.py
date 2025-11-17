# type: ignore
"""Memory-mapped dataloader utils."""

# Standard Library
import queue
import threading
import time
from enum import Enum
from typing import Any
from unittest.mock import Mock

# Third Party
import torch
import torch.distributed as dist

# First Party
from hyperpod_checkpointless_training.dataloader.encryption.encryption_manager import (
    KMSEncryptionManager,
)
from hyperpod_checkpointless_training.inprocess.logger import get_logger

# Used to signal the end of the output queue.
FINAL_DATA_SIGNAL_ELEMENT = "FINAL_DATA_SIGNAL_ELEMENT"
WORKER_SAVE_FINISH = "WORKER_SAVE_FINISH"


class RestartMode(Enum):
    """Recovery restart mode enum."""

    COLD_START = "cold_start"
    WARM_START = "warm_start"


dl_init_lock = threading.Lock()


class PrefetchedDataLoaderSignal:
    """
    A class to manage synchronization signals for prefetched data loading operations.
    Handles thread coordination and data queue management for parallel processing.
    """

    def __init__(self, prefetch_length: int):
        """
        Initialize signal management system with specified prefetch buffer size.

        Args:
            prefetch_length: The prefetch length.
        """
        self.exit_event = threading.Event()  # Signal to terminate all operations
        self.dl_step_signal = (
            threading.Event()
        )  # Signal for original dataloader step completion
        self.signal_start_to_fetch = threading.Event()  # Signal to begin fetching data
        self.output_queue = queue.Queue(prefetch_length)  # Queue for prefetched data
        self.saving_queue = None  # Queue for saving operations (initialized later)
        self._logger = get_logger(__name__)

    def set_exit(self):
        """Trigger the exit signal to stop all operations."""
        self.exit_event.set()

    def should_exit(self):
        """Check if exit signal has been triggered."""
        return self.exit_event.is_set()

    def set_dl_step_signal(self):
        """Signal completion of an DL step."""
        self.dl_step_signal.set()

    def is_dl_step_set(self):
        """Check if DL step signal is set."""
        return self.dl_step_signal.is_set()

    def wait_dl_step(self):
        """
        Wait for DL step completion or exit signal.

        Returns:
            True if exit signal is set.
        """
        return self.wait_or_exit(self.dl_step_signal, "DL_STEP_SIGNAL")

    def set_start_to_fetch_signal(self):
        """Signal that data fetching can begin."""
        self.signal_start_to_fetch.set()

    def wait_start_to_fetch(self):
        """
        Wait for fetch start signal or exit signal.

        Returns:
            True if exit signal is set.
        """
        return self.wait_or_exit(self.signal_start_to_fetch, "DL_START_TO_FETCH")

    def wait_or_exit(self, wait_signal: threading.Event | None = None, name: str = ""):
        """
        Wait for a specific signal while checking for exit condition.
        Logs a warning if waiting exceeds 100 seconds.

        Args:
            wait_signal: Signal to wait for

        Returns:
            True if exit signal is set, False otherwise
        """
        if not wait_signal:
            return self.exit_event.is_set()

        start_time = time.time()
        warning_threshold = 100  # seconds

        while not wait_signal.is_set():
            time.sleep(0.1)  # Short sleep to prevent CPU overuse

            # Check for long wait time
            elapsed_time = time.time() - start_time
            if elapsed_time > warning_threshold:
                self._logger.warning(f"Signal {name} wait time {elapsed_time:.1f}s.")
                start_time = time.time()

            if self.exit_event.is_set():
                break

        return self.exit_event.is_set()

    def try_emit_to_queue(
        self,
        mmaped_tensor: Any,
        emit_queue: queue.Queue | None = None,
        logger=None,
        name: str | None = None,
    ) -> bool:
        """Try pushing the tensor to the queue.

        If the queue is full, it will wait until it is available.

        Args:
            mmaped_tensor: The object to be pushed to the queue.
                Typically an mmaped tensor or a tuple of done_event,
                cache_path/FINAL_DATA_SIGNAL_ELEMENT, and additional data.
            emit_queue: The queue to push the object to.

        Returns:
            bool: True if the exit signal is set, False otherwise.
        """
        check_frequency = 1
        number_of_checks_between_warnings = 300
        should_exit = False
        number_of_warnings_emitted = 0
        cumulative_waiting_time = 0
        while True:
            if self.should_exit():
                if logger:
                    logger.debug("Received exit signal")
                should_exit = True
                break
            try:
                emit_queue.put(mmaped_tensor, timeout=check_frequency)
                if logger:
                    logger.debug("Successfully put batch in queue.")
            except queue.Full:
                cumulative_waiting_time += check_frequency

                if (
                    cumulative_waiting_time
                    > (number_of_warnings_emitted + 1)
                    * number_of_checks_between_warnings
                    * check_frequency
                ):
                    log_message = (
                        f"[{dist.get_rank()}]"
                        f"[{name}] Could not put batch in queue that is full within {cumulative_waiting_time}s "
                        f"(warnings emitted: {number_of_warnings_emitted + 1})."
                    )
                    if logger:
                        logger.warning(log_message)
                    number_of_warnings_emitted += 1
            else:
                # If the put succeeded, proceed to next element from iterator, otherwise try to put the same element
                # again.
                break
        return should_exit

    def try_emit_to_output_queue(
        self, mmaped_tensor: Any, logger=None, name: str | None = None
    ) -> bool:
        """Try pushing the tensor to the output queue.

        If the queue is full, it will wait until it is available.

        Args:
            mmaped_tensor: The object to be pushed to the queue.
                Typically an mmaped tensor or a tuple of done_event,
                cache_path/FINAL_DATA_SIGNAL_ELEMENT, and additional data.

        Returns:
            bool: True if the exit signal is set, False otherwise.
        """
        return self.try_emit_to_queue(mmaped_tensor, self.output_queue, logger, name)

    def try_emit_to_saving_queue(
        self, mmaped_tensor: Any, logger=None, name: str | None = None
    ) -> bool:
        """Try pushing the tensor to the saving queue.

        If the queue is full, it will wait until it is available.

        Args:
            mmaped_tensor: The object to be pushed to the queue.
                Typically an mmaped tensor or a tuple of done_event,
                cache_path/FINAL_DATA_SIGNAL_ELEMENT, and additional data.

        Returns:
            bool: True if the exit signal is set, False otherwise.
        """
        return self.try_emit_to_queue(mmaped_tensor, self.saving_queue, logger, name)

    @staticmethod
    def fill_queue(num_fills: int, target_queue: queue.Queue) -> None:
        """
        Fill the target queue with empty batch.

        Args:
            num_fills: Number of batches to put in target queue.
            target_queue: The target queue.
        """
        for _ in range(num_fills):
            target_queue.put({})


def save_worker(
    task_queue: torch.multiprocessing.Queue,
    worker_exit_event: torch.multiprocessing.Event,
    mp_task_result_queue: torch.multiprocessing.Queue,
    enable_batch_encryption: bool = False,
):
    """Torch save a tensor in a spawned process."""
    torch.set_num_threads(1)
    if enable_batch_encryption:
        encryption_manager = KMSEncryptionManager()
    while not worker_exit_event.is_set():
        try:
            task = task_queue.get(timeout=1)
        except queue.Empty:
            continue

        batch, temp_path = task
        if temp_path:
            if enable_batch_encryption:
                batch = encryption_manager.encrypt(batch)
            torch.save(batch, temp_path)
            del batch
        while not worker_exit_event.is_set():
            try:
                mp_task_result_queue.put(WORKER_SAVE_FINISH, timeout=1)
            except queue.Full:
                continue
            else:
                break


def check_shared_memory_batch(batch: dict, name: str = "") -> None:
    """
    Check for any tensors in the batch that are not in shared memory
    and log a warning message.

    Args:
        batch: Dictionary of batch to check.
        name: The name to describe what requires the batch in shared memory.
    """
    logger = get_logger("shared_memory_check")
    for k, t in batch.items():
        if not isinstance(t, torch.Tensor):
            continue
        if t.numel() > 100 and not t.is_shared():
            logger.warning(
                f"{name} requires batches in shared memory, "
                f"{k} is not shared with shape: {t.shape}, "
                "Performance might be impacted"
            )


class MockDataLoader:
    def __init__(self, data=None):
        self.name = "MockDataLoader"
        self.data = data or [
            {"data": torch.tensor([1, 2, 3])},
            {"data": torch.tensor([4, 5, 6])},
        ]
        self.index = 0
        self._dl_checkpoint_load_step = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        result = self.data[self.index]
        self.index += 1
        return result

    def __len__(self):
        return len(self.data)

    def clean_up(self):
        pass


class MockParallelState:
    def __init__(self):
        self.tp_rank = 0
        self.tp_size = 1
        self.cp_rank = 0
        self.dp_rank = 0
        self.pp_rank = 0
        self.mp_group = Mock()

    def get_tensor_model_parallel_rank(self):
        return self.tp_rank

    def get_tensor_model_parallel_world_size(self):
        return self.tp_size

    def get_context_parallel_rank(self):
        return self.cp_rank

    def get_data_parallel_rank(self):
        return self.dp_rank

    def get_pipeline_model_parallel_rank(self):
        return self.pp_rank

    def get_model_parallel_group(self):
        return self.mp_group

    def model_parallel_is_initialized(self):
        return True


class MockParallelStateUtil:
    def __init__(self):
        self.parallel_state = MockParallelState()
        self.pdl_ranks_in_mp_group = {0: [0, 1]}
