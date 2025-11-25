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
"""Prefetched Data Loader."""

# Standard Library
import queue
import threading
import time
from typing import Callable

# Third Party
import torch
import torch.distributed as dist
import torch.multiprocessing

# First Party
from hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader import (
    PassthroughCacheDataLoader,
)
from hyperpod_checkpointless_training.dataloader.mmap.utils import (
    FINAL_DATA_SIGNAL_ELEMENT,
    PrefetchedDataLoaderSignal,
    RestartMode,
    check_shared_memory_batch,
    save_worker,
    dl_init_lock,
)
from hyperpod_checkpointless_training.inprocess.logger import get_logger


class CacheOnlyPrefetchedDataLoader(PassthroughCacheDataLoader):
    """The Prefetch class of a cache dataloader(PDL) for CACHE_ONLY.

    There should be at least two PDLs in one dp group. One for pp0tp0, one for ppNtp0.
    All tp0 will be using this class as they will be data caching rank and provide
    cache data to CDL.

    During training, PDL is responsible for:
    1. Fetching data ahead of time. broadcasting if needed
    2. Caching it into mmap files (lookback + prefetch)
    Note that the actual broadcasting is done throught broadcast dataloader(BDL) from dataloader_init_callable

    fetch_thread:
    dataloader  --> [d3] --> mp_task_queue [d1, d2]
                                |
                                |
                                v
    multiprocess save_worker:
                            torch.save -> [mmap files]
                                                |
                                                |
    save_finished_thread:                       v
                                            output_queue [step1, step2]
                                                |
                                                |
                                                |
    Main thread                                 v
            [batch_1.pt, batch_2.pt]  ---->    main   ---> d0
    """

    def __init__(
        self,
        step: int,
        dataloader_init_callable: Callable,
        cache_dir: str = "/dev/shm/pdl_cache",
        lookback_length: int = 10,
        prefetch_length: int = 10,
        model_checkpoint_frequency: int | None = None,
        saving_length: int = 2,
        force_cold_start: bool = False,
        parallel_state_util=None,
        enable_batch_encryption: bool = False,
    ):
        """Initialize the Prefetched Data Loader (PDL).

        Args:
            step: The current/resumed training step.
            dataloader_init_callable: A callable that will return a dataloader. The callable should return
                a dataloader. The init process will be done in the background.
            cache_dir: The cache directory that stores the cache batch, a model parallel id will be added.
            lookback_length: The lookback length of the cache. cache_size = lookback_length + prefetch_length.
            prefetch_length: The prefetch length of the cache. cache_size = lookback_length + prefetch_length.
            model_checkpoint_frequency: The model checkpoint frequency.
            saving_length: The saving length. The number of parallel process workers used to save the batches.
            force_cold_start: Whether or not to force restart. All cache will be wiped.
            parallel_state_util: Util that provides the model parallelism info.
            enable_batch_encryption: Whether or not to enable batch encryption with KMS.
                If enabled, assumes AWS_KMS_KEY_ID is set in the environment.
        """
        super().__init__(
            step=step,
            cache_dir=cache_dir,
            dataloader_init_callable=dataloader_init_callable,
            lookback_length=lookback_length,
            prefetch_length=prefetch_length,
            read_only=False,
            force_cold_start=force_cold_start,
            model_checkpoint_frequency=model_checkpoint_frequency,
            parallel_state_util=parallel_state_util,
            enable_batch_encryption=enable_batch_encryption,
        )
        self._saving_length = saving_length
        self._dataloader_init_callable = dataloader_init_callable
        self.workers = None
        self._save_finished_thread = None
        self._spawn_workers_thread = None
        self.mp_context = torch.multiprocessing.get_context(method="fork")
        self.workers = []
        self.mp_task_queues = []
        self.worker_events = []
        self.mp_task_result_queues = []
        self._logger = get_logger(__name__)

    def maybe_join_old_threads(self, timeout=None):
        """Join old threads if they exist.

        Args:
            timeout: Maximum time to wait for threads to finish, in seconds.
                     If None (default), waits indefinitely.
        """
        super().maybe_join_old_threads(timeout=timeout)
        if self._save_finished_thread:
            self._save_finished_thread.join(timeout=timeout)

    def init(self, dl_signals: PrefetchedDataLoaderSignal):
        """Initialize the dataloader."""
        super().init(dl_signals)
        self.cache_only_init(dl_signals)

    def spawn_workers(self):
        """Spawn the workers that save to cache."""
        self._spawn_workers_thread = threading.Thread(
            target=self.mp_init,
            daemon=True,
        )

        self._spawn_workers_thread.start()

    def mp_init(self):
        """Initialize the multiprocessing manager and workers for saving to the cache."""

        for i in range(self._saving_length):
            self.worker_events.append(self.mp_context.Event())
            self.mp_task_queues.append(self.mp_context.Queue(2))
            self.mp_task_result_queues.append(self.mp_context.Queue(2))
            self.workers.append(
                self.mp_context.Process(
                    target=save_worker,
                    args=(
                        self.mp_task_queues[i],
                        self.worker_events[i],
                        self.mp_task_result_queues[i],
                        self._enable_batch_encryption,
                    ),
                    daemon=True,
                )
            )
        for w in self.workers:
            w.start()

    def init_common(self, dl_signals: PrefetchedDataLoaderSignal):
        """Initialize common actions for CacheOnly and CacheResume."""

        dl_signals.saving_queue = queue.Queue(self._saving_length)
        self._save_finished_thread = threading.Thread(
            target=self.save_finished,
            args=(dl_signals,),
            daemon=True,
        )
        self._save_finished_thread.start()

    def cache_only_init(self, dl_signals: PrefetchedDataLoaderSignal):
        """Initialize the PDL for cache only mode."""

        self.init_common(dl_signals)

        self._logger.debug("CacheOnly PDL init finished")

    def save_finished(self, dl_signals: PrefetchedDataLoaderSignal):
        """Push the mmap tensor to output_queue after torch.save."""
        check_frequency = 1

        while not dl_signals.should_exit():
            try:
                elem = dl_signals.saving_queue.get(timeout=check_frequency)
                worker_index, cache_tmp_path, _idx_prefetched, _ = elem
                while worker_index is not None:
                    try:
                        result = self.mp_task_result_queues[worker_index].get(timeout=1)
                    except queue.Empty:
                        if dl_signals.should_exit():
                            self._logger.debug("Received exit signal")
                            result = None
                            break
                        continue
                    if result:
                        break
                if cache_tmp_path == FINAL_DATA_SIGNAL_ELEMENT:
                    dl_signals.try_emit_to_output_queue(
                        FINAL_DATA_SIGNAL_ELEMENT, logger=self._logger, name="PDL"
                    )
                    return
                if result:
                    self._cache.promote_content(cache_tmp_path, _idx_prefetched)
                    dl_signals.try_emit_to_output_queue(
                        _idx_prefetched, logger=self._logger, name="PDL"
                    )
            except queue.Empty:
                continue

    def __iter__(self):
        """
        Iterate over the PDL.

        Join the spawn workers thread if it exists to ensure all workers are ready before iterating.
        """
        if not self.workers:
            self.spawn_workers()
            if self._spawn_workers_thread:
                self._spawn_workers_thread.join()
                self._spawn_workers_thread = None
        yield from super().__iter__()

    def _stop(self, dl_signals: PrefetchedDataLoaderSignal):
        """Properly set stop signal to all threads/processes."""
        for worker_event in self.worker_events:
            worker_event.set()

        for worker in self.workers:
            worker.join()

        self.workers = []
        self.worker_events = []

        if hasattr(self, "_cache") and hasattr(
            self._cache, "stop_cleanup_expired_files_thread"
        ):
            self._cache.stop_cleanup_expired_files_thread()

        super()._stop(dl_signals)

    def initialize_data_loader(self):
        if self.data_loader is not None:
            self._logger.info(
                "Called initialize_data_loader when dataloader already inited."
            )
            return

        if self._force_cold_start:
            target_load_step = 0
        else:
            target_load_step = self._dl_checkpoint_load_step

        self._logger.info(
            f"DataLoader init starting, should init from {target_load_step}"
        )
        with dl_init_lock:
            self.data_loader = self._dataloader_init_callable(
                target_load_step=target_load_step
            )
        dataloader_name = (
            self.data_loader.name if hasattr(self.data_loader, "name") else ""
        )
        self._logger.info(f"{dataloader_name} DataLoader init finished.")

    def fetch_data(self, dl_signals: PrefetchedDataLoaderSignal):
        """
        Fetch thread does 3 things sequentially:
            1. Wait for resume step signal for original dataloader.
            2. Init original dataloader with a target step.
                Original dataloader should forward if needed.
            3. Once init, wait for the signal to start fetching.
            4. Fetches batches from original dataloader, For each batch:
                a. Saves it to mmap through multiprocessing worker.
                b. Puts it into output queue.
                c. Do necessary clean up if it is a data rank.
        """
        # Wait for dl init step is set, during restart this will mean it will wait for the PDL to
        # check how many caches exist.
        self._logger.info("PDL fetch_data started")
        # When exit event is set, wait_dl_step returns True, so we shouldn't proceed with
        # dataloader and signal to exit now.
        if dl_signals.wait_dl_step():
            self.emit_final_data_signal(dl_signals)
            return
        self._logger.info(
            f"Original dataloader init starting, should init from {self._dl_checkpoint_load_step}"
        )
        # Reuse dataloader if is is already inited.
        if not self.data_loader:
            self.initialize_data_loader()
        self._fetch(self.data_loader, dl_signals)
        self.emit_final_data_signal(dl_signals)

    def emit_final_data_signal(self, dl_signals: PrefetchedDataLoaderSignal):
        """
        Emit the final data element signal.

        Used to signal when the dataset is exhausted. Any signal should not be set here.
        Instead all signals should be set in the _stop() function call.
        """
        dl_signals.try_emit_to_saving_queue(
            (None, FINAL_DATA_SIGNAL_ELEMENT, None, None),
            logger=self._logger,
            name="PDL",
        )

    def _fetch(
        self,
        data_loader,
        dl_signals: PrefetchedDataLoaderSignal,
    ):
        """Fetches data from the dataloader."""
        next_start = time.time()
        should_exit = False

        # Make sure we don't exit immediately here even if the exit event is set. This is
        # to prevent the case where sender exits early while the receiver is still waiting.
        # Thus cause the hang in the first batch and unable to exit.
        dl_signals.wait_start_to_fetch()

        worker_index = 0
        for i, batch in enumerate(data_loader):
            if i == 0:
                self._logger.info("DataLoader init Finishes, first batch yields")
            self._idx_prefetched += 1
            cache_tmp_path = self._cache.create_staging_entry()
            check_shared_memory_batch(batch, name="PrefetchedDataLoader")
            should_exit = dl_signals.try_emit_to_queue(
                (batch, cache_tmp_path),
                emit_queue=self.mp_task_queues[worker_index],
                logger=self._logger,
                name="PDL",
            )
            del batch
            if should_exit:
                break

            next_end = time.time()
            self._logger.debug(
                f"Loaded item {self._idx_prefetched} in {next_end-next_start:.3f}s."
            )
            next_start = next_end
            should_exit = dl_signals.try_emit_to_saving_queue(
                (worker_index, cache_tmp_path, self._idx_prefetched, time.time()),
                logger=self._logger,
                name="PDL",
            )
            worker_index += 1
            worker_index %= self._saving_length
            if should_exit:
                break

    def get_cached_batch(self, elem, dl_signals: PrefetchedDataLoaderSignal):
        """
        PDL will wait for the cache file to be available.
        """
        element_wait_start = time.time()
        last_log_time = element_wait_start
        while True:
            if dl_signals.should_exit():
                self._logger.error(f"{self.__class__.__name__}: exit event set")
                raise RuntimeError(f"{self.__class__.__name__}: exit event set")
            try:
                batch = self._cache.get_content(self._step)
                if self._enable_batch_encryption:
                    batch = self.encryption_manager.decrypt(batch)
                    self._logger.debug(
                        f"[{self._rank_id}]: Decrypted batch for step {self._step}."
                    )
            except FileNotFoundError:
                time.sleep(0.1)
                cur_time = time.time()
                if cur_time - last_log_time > self._wait_cache_batch_log_interval:
                    self._logger.debug(
                        f"[{self._rank_id}]: Waiting for cache file (step={self._step}) for total {cur_time - element_wait_start}s."
                    )
                    last_log_time = cur_time
            else:
                cur_time = time.time()
                self._logger.debug(
                    f"[{self._rank_id}]: Cache file (step={self._step}) available after total {cur_time - element_wait_start}s."
                )
                break
        return batch


class CacheResumePrefetchedDataLoader(CacheOnlyPrefetchedDataLoader):
    def __init__(
        self,
        step: int,
        dataloader_init_callable: Callable,
        model_parallel_group,
        cache_dir: str = "/dev/shm/pdl_cache",
        lookback_length: int = 10,
        prefetch_length: int = 10,
        model_checkpoint_frequency: int | None = None,
        saving_length: int = 2,
        force_cold_start: bool = False,
        parallel_state_util=None,
        enable_batch_encryption: bool = False,
    ):
        """The Prefetch class of a cache dataloader(PDL) CACHE_RESUME.

        During restart, PDL is responsible for:
        1. Loading the valid cache from cache directory.

        There will be three restart mode:
        1. Cold start: There are no cache available within the dp group. dataloader will start from sratch
        2. Warm start: There are cache available in all pdls and we will use the cach before first before using
            the freshly fetched data.

        Args:
            step: The current/resumed training step.
            dataloader_init_callable: A callable that will return a dataloader. The callable should return
                a dataloader. The init process will be done in the background.
            model_parallel_group: The parallel group that contains "tp-cp-pp".
            cache_dir: The cache directory that stores the cache batch, a model parallel id will be added.
            lookback_length: The lookback length of the cache. cache_size = lookback_length + prefetch_length.
            prefetch_length: The prefetch length of the cache. cache_size = lookback_length + prefetch_length.
            model_checkpoint_frequency: The model checkpoint frequency.
            saving_length: The saving length. The number of parallel process workers used to save the batches.
                We default to 2, so we can have 2 batches saving in parallel.
            force_cold_start: Whether or not to force restart. All cache will be wiped.
            parallel_state_util: Util that provides the model parallelism info.
            enable_batch_encryption: Whether or not to enable batch encryption with KMS.
                If enabled, assumes AWS_KMS_KEY_ID is set in the environment.
        """
        super().__init__(
            step=step,
            cache_dir=cache_dir,
            dataloader_init_callable=dataloader_init_callable,
            lookback_length=lookback_length,
            prefetch_length=prefetch_length,
            saving_length=saving_length,
            force_cold_start=force_cold_start,
            model_checkpoint_frequency=model_checkpoint_frequency,
            parallel_state_util=parallel_state_util,
            enable_batch_encryption=enable_batch_encryption,
        )
        self.model_parallel_group = model_parallel_group
        if not self._force_cold_start:
            # wait to init during val_dataloader iter
            # for pp > 1, calling init in iter causes a hang since iter is called when activations are ready for a stage
            # however, we skip the collective during force cold start.
            with self._thread_lock:
                self.maybe_join_old_threads()
                dl_signals = PrefetchedDataLoaderSignal(self._prefetch_length)
                self.init(dl_signals)

    def init(self, dl_signals: PrefetchedDataLoaderSignal):
        """Initialize the dataloader."""
        PassthroughCacheDataLoader.init(self, dl_signals)
        self.cache_resume_init(dl_signals)

    def cache_resume_init(self, dl_signals: PrefetchedDataLoaderSignal):
        """Initialize the PDL for cache resume mode."""
        self.init_common(dl_signals)

        # 0. Check cache expiration with buffer.
        self._cache.cleanup_expired_files(use_cleanup_buffer=True)

        # 1. Cache validations.
        # 1.1. Model step validation. if cache does not contain model checkpoint step,
        #  delete cache. It also removes invalid data, ex: id < step or id > step + prefetch length.
        self._cache.prune_cache_init(self._step, self._prefetch_length)

        # 2. Exchange cache availability within a data parallel group.
        num_valid_cache = len(self._cache)
        # first PP PDL: the rank(tp 0) that sends/receives the full batch. ie: first pipeline + data replica
        # last PP PDL: the rank(tp 0) that receives the partial batch, ie: last pipeline.
        if not self.model_parallel_group:
            (
                pdl_cache_sizes,
                scenario,
            ) = self.get_local_cache_size(num_valid_cache)
        else:
            (
                pdl_cache_sizes,
                scenario,
            ) = self._cache.all_gather_cache_size(
                num_valid_cache,
                self.get_pdl_ranks_in_mp,
                self.model_parallel_group,
            )
            if dist.get_group_rank(self.model_parallel_group, self._rank_id) == 0:
                self._logger.info(
                    f"[{self._rank_id}] Cache rank status: {pdl_cache_sizes}"
                )
                self._logger.info(f"[{self._rank_id}] {scenario} starting soon ")

        # 3. Prune Caches
        if scenario == RestartMode.COLD_START:
            # Wipe all caches on all ranks.
            max_valid_num_cache = 0
        else:
            # Take the minimum cache size of non-empty pdl.
            max_valid_num_cache = self._get_num_fills(pdl_cache_sizes)

        self._logger.info(
            f"[{self._rank_id}] Number of valid caches found: {max_valid_num_cache}"
        )
        # Remove the exceeding portion.
        self._prune_cache(max_valid_num_cache)
        self._dl_checkpoint_load_step += max_valid_num_cache
        self._idx_prefetched += max_valid_num_cache

        self._fill_output_queue_from_cache(dl_signals, self._step)

        self._logger.debug(f"[{self._rank_id}] CacheResume PDL init finished")

    def _prune_cache(self, max_valid_num_cache):
        """Take the min of non-empty cache and prune the exceeding part."""
        if dist.get_global_rank(self.model_parallel_group, 0) == self._rank_id:
            self._logger.debug(
                f"[{self._rank_id}] max_valid_cache_sizes: {max_valid_num_cache}"
            )
        self._cache.prune_cache_init(self._step, max_valid_num_cache)
        cache_to_fill = len(self._cache)
        assert cache_to_fill <= self._prefetch_length

    def _fill_output_queue_from_cache(
        self, dl_signals: PrefetchedDataLoaderSignal, step: int
    ) -> None:
        """
        Initialize output queue from cache.

        Args:
            dl_signals: The PrefetchedDataLoaderSignal.
            step: The current step.
                We use this to determine whether to use our caches.
        """
        all_available_ids = self._cache.get_content_indices()
        self._logger.debug(f"Filling from cache with {len(all_available_ids)} batches.")
        if all_available_ids and sorted(all_available_ids)[0] <= step:
            for id in sorted(all_available_ids):
                dl_signals.try_emit_to_output_queue(id, name="PDL")

    def get_local_cache_size(self, cache_size):
        if cache_size == 0:
            return (0, RestartMode.COLD_START)
        else:
            return (cache_size, RestartMode.WARM_START)
