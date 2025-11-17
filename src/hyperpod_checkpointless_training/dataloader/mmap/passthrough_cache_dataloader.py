# type: ignore
"""Passthrough Cache Data Loader."""

# Standard Library
import datetime
import functools
import os
import queue
import sys
import threading
import time
from typing import Callable, List

# Third Party
import torch
import torch.distributed as dist

# First Party
from hyperpod_checkpointless_training.dataloader.encryption.encryption_manager import (
    KMSEncryptionManager,
)
from hyperpod_checkpointless_training.dataloader.mmap.cache import (
    MMAPCache,
    TTLMMAPCache,
)
from hyperpod_checkpointless_training.dataloader.mmap.utils import (
    FINAL_DATA_SIGNAL_ELEMENT,
    PrefetchedDataLoaderSignal,
    dl_init_lock,
)
from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg


class PassthroughCacheDataLoader:
    """The passthrough cache dataloader (PassDL). Used as a base for alternative cache dataloaders.

    Non terminal pp ranks should use this class. It prefetches data from the dataloader async
      and puts it into a queue. When data is needed, yields one from the queue.

    fetch_thread:
    dataloader  --> [d3] --> output_queue [d1, d2]
                                    |
                                    |
                                    v
                                  main ---> d0
    """

    def __init__(
        self,
        step: int,
        dataloader_init_callable: Callable,
        cache_dir: str = "/dev/shm/pdl_cache",
        lookback_length: int = 0,
        prefetch_length: int = 0,
        model_checkpoint_frequency: int | None = None,
        read_only: bool = True,
        force_cold_start: bool = False,
        debug_logging: bool = False,
        parallel_state_util=None,
        enable_batch_encryption: bool = False,
    ):
        """Initialize the Passthrough Cache Data Loader.

        Args:
            step: The current/resumed training step.
            dataloader_init_callable: A callable that will return a dataloader. The callable should return
                a dataloader. The init process will be done in the background.
            cache_dir: The cache directory that stores the cache batch, a model parallel id will be added
            lookback_length: The lookback length of the cache. cache_size = lookback_length + prefetch_length.
            prefetch_length: The prefetch length of the cache. cache_size = lookback_length + prefetch_length.
            model_checkpoint_frequency: The model checkpoint frequency.
            read_only: If the corresponding rank is just cache ready only rank. ie: no delete action.
            force_cold_start: Whether or not to force restart. All cache will be wiped.
            parallel_state_util: Util that provides the model parallelism info.
            enable_batch_encryption: Whether or not to enable batch encryption with KMS.
                If enabled, assumes AWS_KMS_KEY_ID is set in the environment.
        """
        self._debug_logging = debug_logging
        self._logger = get_logger("passthrough_cache_dataloader")
        self._rank_id = dist.get_rank()
        self._lookback_length = lookback_length

        self._read_only = read_only
        self._force_cold_start = force_cold_start
        self._original_step = step
        self._step = step
        self._prefetch_length = prefetch_length
        self._cache_dir = cache_dir
        self._model_checkpoint_frequency = model_checkpoint_frequency
        self._dataloader_init_callable = dataloader_init_callable
        self.data_loader = None
        self._thread_lock = threading.Lock()
        self._fetch_thread = None
        self._parallel_state_util = parallel_state_util
        self._enable_batch_encryption = enable_batch_encryption
        self._length = None
        self.encryption_manager = None
        self.dl_signals = None
        if self._enable_batch_encryption:
            self.encryption_manager = self._create_encryption_manager()
        # seconds between debug log for waits to not flood logs
        self._wait_elem_log_interval = 5.0
        self._wait_cache_batch_log_interval = 5.0

    def _create_encryption_manager(self):
        encryption_manager = KMSEncryptionManager()
        self._logger.info(
            f"[{self._rank_id}] Setup encryption manager with "
            f"AWS_KMS_KEY_ID={encryption_manager.kms_key_id}."
        )
        return encryption_manager

    def _start_fetch_thread(self, dl_signals: PrefetchedDataLoaderSignal):
        """
        Start the fetch thread.

        The fetch thread will fetch data from the dataloader
        and put it into the output queue.
        """
        self._fetch_thread = threading.Thread(
            target=self.fetch_data,
            args=(dl_signals,),
            daemon=False,
        )
        self._fetch_thread.start()

    def init(self, dl_signals: PrefetchedDataLoaderSignal):
        """
        Initialize the dataloader.

        Args:
            dl_signals: The dataloader signals for synchronization.
        """
        self.dl_signals = dl_signals
        self._step = self._original_step
        self._idx_prefetched = self._step - 1
        self._dl_checkpoint_load_step = self._step
        if self._parallel_state_util:
            self._parallel_state = self._parallel_state_util.parallel_state
            cp_rank = self._parallel_state.get_context_parallel_rank()
            dp_rank = self._parallel_state.get_data_parallel_rank()
            pp_rank = self._parallel_state.get_pipeline_model_parallel_rank()
            self._pdl_ranks_in_dp = self._parallel_state_util.pdl_ranks_in_mp_group
            cache_dir = os.path.join(
                self._cache_dir,
                f"dp{dp_rank}_pp{pp_rank}_cp{cp_rank}/",
            )
        else:
            cache_dir = self._cache_dir
        self._cache = TTLMMAPCache(
            cache_dir,
            read_only=self._read_only,
            force_cold_start=self._force_cold_start,
            pdl_per_node_group=self.pdl_per_node_group,
            lookback_length=self._lookback_length,
            prefetch_length=self._prefetch_length,
            model_checkpoint_frequency=self._model_checkpoint_frequency,
        )
        self._cache.init()
        self._start_fetch_thread(dl_signals)

    def is_data_rank(self, rank: int | None = None):
        """
        Returns if the specified rank is a data rank.
        A data rank is the one with tp0.
        """
        if not self._parallel_state.model_parallel_is_initialized():
            return False
        tp_rank = self._parallel_state.get_tensor_model_parallel_rank()
        if rank:
            tp_size = self._parallel_state.get_tensor_model_parallel_world_size()
            return rank % tp_size == 0
        return tp_rank == 0

    @functools.cached_property
    def get_pdl_ranks_in_mp(self) -> List[int]:
        dp_rank = self._parallel_state.get_data_parallel_rank()
        return self._pdl_ranks_in_dp[dp_rank]

    @functools.cached_property
    def pdl_per_node_group(self):
        """Get the PDL process group on each node.
        only needed for SLURM as KUBERNETES will clean the caches when the pods are deleted
        """
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 8))
        tp_size = self._parallel_state.get_tensor_model_parallel_world_size()
        if tp_size >= local_world_size:
            return None
        world_size = dist.get_world_size()
        num_nodes = world_size // local_world_size
        current_node_id = self._rank_id // local_world_size
        current_pdl_node_group = None
        all_groups: list[list[int]]
        all_groups = [[] for _ in range(num_nodes)]

        for rank in range(world_size):
            if self.is_data_rank(rank):
                all_groups[rank // local_world_size].append(rank)
        if len(all_groups[0]) <= 1:
            return None
        for i, group in enumerate(all_groups):
            # Some nodes may not have PDL.
            if not group:
                continue
            g = torch.distributed.new_group(
                group, backend="gloo", timeout=datetime.timedelta(3600)
            )
            if i == current_node_id:
                current_pdl_node_group = g

        return current_pdl_node_group

    def fetch_data(self, dl_signals: PrefetchedDataLoaderSignal):
        """Fetch data from the dataloader.

        Note:
        1. Before the dataloader init, it will need to wait for its step to be determined.
        2. Before the dataloader fetching, it will wait for the another signal.

        The fetch data will eventually be pushed to the self._output_queue.
        """
        # Wait for the dl init step to be set. During restart, this means it will wait
        # for the PDL to check how many caches exist.
        # When exit event is set, wait_dl_step returns True, so we shouldn't proceed with
        # dataloader init.
        if dl_signals.wait_dl_step():
            return

        # If the dataloader is already init, we resue the dataloader.
        if not self.data_loader:
            with dl_init_lock:
                self.data_loader = self._dataloader_init_callable(
                    target_load_step=self._dl_checkpoint_load_step
                )
                self._logger.debug(f"[{self._rank_id}] DataLoader init finished")
        dl_signals.wait_start_to_fetch()
        while True:
            if dl_signals.try_emit_to_output_queue(
                {}, logger=self._logger, name="PassDL"
            ):
                break

        # The dataset is exhausted. Exit now.
        dl_signals.try_emit_to_output_queue(
            FINAL_DATA_SIGNAL_ELEMENT, logger=self._logger, name="PassDL"
        )

    def maybe_join_old_threads(self, timeout=None):
        """Join old threads if they exist.

        Args:
            timeout: Maximum time to wait for thread to finish, in seconds.
                     If None (default), waits indefinitely.
        """
        if self._fetch_thread:
            self._fetch_thread.join(timeout=timeout)

    def __iter__(self):
        """Iterate over the output queue."""
        if self._force_cold_start:
            # val_dataloader __iter__
            with self._thread_lock:
                self.maybe_join_old_threads()
                dl_signals = PrefetchedDataLoaderSignal(self._prefetch_length)
                self.init(dl_signals)
                self.dl_signals.set_start_to_fetch_signal()
                self.dl_signals.set_dl_step_signal()
        else:
            # train_dataloader __iter__
            # We need to wait till worker queues are created to start fetching. cache init was completed in __init__
            with self._thread_lock:
                self.dl_signals.set_start_to_fetch_signal()
                self.dl_signals.set_dl_step_signal()
        try:
            last_log_time = time.time()
            while not self.dl_signals.should_exit():
                element_wait_start = time.time()
                last_log_time = element_wait_start
                while True:
                    try:
                        elem = self.dl_signals.output_queue.get(timeout=1)
                        is_final_elem = elem == FINAL_DATA_SIGNAL_ELEMENT
                        is_final_index = self._cache.is_final_index(self._step)

                        if is_final_elem or is_final_index:
                            # No more prefetched buffer and data loader is exhausted.
                            self.dl_signals.set_exit()
                            break
                        if isinstance(elem, int):
                            assert (
                                elem == self._step
                            ), f"rank {self._rank_id} Index from output queue is {elem}, but counter is {self._step}"
                            if self._step == self._dl_checkpoint_load_step:
                                # Start cleanup thread after all pre-existing caches are consumed
                                if hasattr(
                                    self._cache, "start_cleanup_expired_files_thread"
                                ):
                                    self._cache.start_cleanup_expired_files_thread()
                        replaced_elem = self.get_cached_batch(elem, self.dl_signals)
                        if replaced_elem:
                            elem = replaced_elem
                        if isinstance(elem, Exception):
                            self._logger.error(
                                "Re-raising exception received from prefetching thread."
                            )
                            raise elem
                    except queue.Empty:
                        # Check for final index even when queue is empty
                        if self._cache.is_final_index(self._step):
                            self.dl_signals.set_exit()
                            break
                        time.sleep(0.1)
                        cur_time = time.time()
                        if cur_time - last_log_time > self._wait_elem_log_interval:
                            self._logger.debug(
                                f"{self._rank_id}: Waiting for element (step={self._step}) for total {cur_time - element_wait_start}s."
                            )
                            last_log_time = cur_time
                        continue
                    else:
                        cur_time = time.time()
                        self._logger.debug(
                            f"{self._rank_id}: Element (step={self._step}) available after total {cur_time - element_wait_start}s."
                        )
                        self._step += 1
                        break
                if elem == FINAL_DATA_SIGNAL_ELEMENT:
                    # No more prefetched buffer and data loader is exhausted.
                    break
                element_wait_time = time.time() - element_wait_start

                self._logger.debug(
                    f"{self._rank_id} Received element from queue in {element_wait_time:.2f}s "
                    f"({self.dl_signals.output_queue.qsize()} batches prefetched)."
                )
                yield elem
                if self.is_data_rank():
                    self._cache.prune_cache(self._step)
        finally:
            # when generator is garbage collected, generator throws a GeneratorExit exception and finally block will execute
            # Generator is automatically unreferenced on a new epoch
            self._stop(self.dl_signals)

    def get_cached_batch(self, elem, dl_signals: PrefetchedDataLoaderSignal):
        """
        Return the element as is.
        """
        return elem

    @staticmethod
    def _get_num_fills(pdl_cache_sizes: dict) -> int:
        """Get the number of fills from the cache info."""
        num_fills = sys.maxsize

        non_empty_pdl_cache_size = MMAPCache.get_non_empty_values(pdl_cache_sizes)
        if non_empty_pdl_cache_size:
            num_fills = min(min(non_empty_pdl_cache_size), num_fills)

        return num_fills if num_fills != sys.maxsize else 0

    def stop(self):
        self._stop(self.dl_signals)

    def _stop(self, dl_signals: PrefetchedDataLoaderSignal):
        """Properly set stop signal to all threads/processes."""
        dataloader_name = (
            self.data_loader.name if hasattr(self.data_loader, "name") else ""
        )
        if not hasattr(self, "_cache"):
            self._logger.debug(
                debug_msg("Dataloader not yet initialized, skipping stop")
            )
            return
        self._logger.debug(f"{dataloader_name} exiting")

        if self.is_data_rank():
            self._cache.set_final_index(self._step)
        dl_signals.set_exit()
        self.maybe_join_old_threads(timeout=30)
        # wait till fetch joins to cleanup task results queue
        if hasattr(self, "mp_task_queues") and self.mp_task_queues:
            self.mp_task_queues = []
            self.mp_task_result_queues = []

    def __del__(self):
        try:
            if (
                hasattr(self, "data_loader")
                and self.data_loader
                and hasattr(self.data_loader, "clean_up")
            ):
                self.data_loader.clean_up()
        except Exception as e:
            if hasattr(self, "_logger"):
                self._logger.warning(
                    f"[{self._rank_id}] Failed to clean up dataloader: {e}"
                )

    def set_length(self, length):
        """Set the length of the dataloader."""
        self._length = length

    def __len__(self):
        if self._length is not None:
            return self._length
        return len(self.data_loader)

    def __getattr__(self, name):
        if "data_loader" not in vars(self):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'data_loader'"
            )
        # Pipe all unknown attributes to the wrapped dataloader.
        if self.data_loader is None:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
        return getattr(self.data_loader, name)
