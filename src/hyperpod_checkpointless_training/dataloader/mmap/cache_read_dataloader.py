# type: ignore
"""Cache Read Data Loader."""

import time
from typing import Callable

from hyperpod_checkpointless_training.dataloader.mmap.passthrough_cache_dataloader import (
    PassthroughCacheDataLoader,
)
from hyperpod_checkpointless_training.dataloader.mmap.utils import (
    PrefetchedDataLoaderSignal,
    RestartMode,
)
from hyperpod_checkpointless_training.inprocess.logger import get_logger


class CacheOnlyReadDataLoader(PassthroughCacheDataLoader):
    """The cache reading class of a cache dataloader(CDL) for CACHE_ONLY.

    It will read the cache from the local mmap files. Any non tp0 rank will be a CDL.

    Similar to the base class, CDL will prefetch data from the dataloader async
    and put it into a queue. When a data is needed, pops one from the queue but the actual data will
    come from reading the mmap files at step i.


    fetch_thread:
    dataloader  --> [d3] --> output_queue [d1, d2]


    Main thread:
    [caches] ---->main ---> mmap tensor
    """

    def __init__(
        self,
        step: int,
        dataloader_init_callable: Callable,
        cache_dir: str = "/dev/shm/pdl_cache",
        lookback_length: int = 0,
        prefetch_length: int = 0,
        model_checkpoint_frequency: int | None = None,
        force_cold_start: bool = False,
        parallel_state_util=None,
        enable_batch_encryption: bool = False,
    ):
        """Initialize the Cache Only Read Data Loader.

        Args:
            step: The current/resumed training step.
            dataloader_init_callable: A callable that will return a dataloader. The callable should return
            cache_dir: The cache directory that stores the cache batch, a model parallel id will be added
            lookback_length: The lookback length of the cache. cache_size = lookback_length + prefetch_length.
            prefetch_length: The prefetch length of the cache. cache_size = lookback_length + prefetch_length.
            model_checkpoint_frequency: The model checkpoint frequency.
            mode: Either CACHE_ONLY or CACHE_RESUME mode.
                CACHE_ONLY will only cache the prefetch and use it for intranode fetching in terminal pp nodes.
                CACHE_RESUME will cache and make cache replicas by broadcasting to the other terminal nodes if needed.
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
            read_only=True,
            model_checkpoint_frequency=model_checkpoint_frequency,
            force_cold_start=force_cold_start,
            parallel_state_util=parallel_state_util,
            enable_batch_encryption=enable_batch_encryption,
        )
        self._logger = get_logger(__name__)

    def init(self, dl_signals: PrefetchedDataLoaderSignal):
        super().init(dl_signals)
        dl_signals.set_start_to_fetch_signal()
        dl_signals.set_dl_step_signal()

    def get_cached_batch(self, elem, dl_signals: PrefetchedDataLoaderSignal):
        """
        CDL will wait for the cache file to be available.
        """
        element_wait_start = time.time()
        last_log_time = element_wait_start
        while True:
            if dl_signals.should_exit() or self._cache.is_final_index(self._step):
                dl_signals.set_exit()
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
                        f"{self._rank_id}: Waiting for cache file (step={self._step}) for total {cur_time - element_wait_start}s."
                    )
                    last_log_time = cur_time
            else:
                cur_time = time.time()
                self._logger.debug(
                    f"{self._rank_id}: Cache file (step={self._step}) available after total {cur_time - element_wait_start}s."
                )
                break
        return batch


class CacheResumeReadDataLoader(CacheOnlyReadDataLoader):
    def __init__(
        self,
        step: int,
        dataloader_init_callable: Callable,
        model_parallel_group,
        cache_dir: str = "/dev/shm/pdl_cache",
        lookback_length: int = 0,
        prefetch_length: int = 0,
        model_checkpoint_frequency: int | None = None,
        force_cold_start: bool = False,
        parallel_state_util=None,
        enable_batch_encryption: bool = False,
    ):
        """The cache reading class of a cache dataloader(CDL) for CACHE_RESUME.

        It extends CacheOnlyReadDataLoader by providing cache recovery feature.
        During restart it will involve a all gather to check local cache status and fill the output queue.

        cache_size = lookback_length + prefetch_length + prefetch_size_after_pdl
        Args:
            step: The current/resumed training step.
            dataloader_init_callable: A callable that will return a dataloader. The callable should return
            cache_dir: The cache directory that stores the cache batch, a model parallel id will be added
            model_parallel_group: The parallel group that contains "tp-cp-pp".
            lookback_length: The lookback length of the cache.
            prefetch_length: The prefetch length of the cache.
            model_checkpoint_frequency: The model checkpoint frequency.
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
            model_checkpoint_frequency=model_checkpoint_frequency,
            force_cold_start=force_cold_start,
            parallel_state_util=parallel_state_util,
            enable_batch_encryption=enable_batch_encryption,
        )
        assert (
            model_parallel_group
        ), f"model_parallel_group cannot be {model_parallel_group} when using cache_resume mode"

        self.model_parallel_group = model_parallel_group

        # wait till CDL is fully constructed to initialize fetching
        if not self._force_cold_start:
            with self._thread_lock:
                self.maybe_join_old_threads()
                dl_signals = PrefetchedDataLoaderSignal(self._prefetch_length)
                self.init(dl_signals)

    def init(self, dl_signals: PrefetchedDataLoaderSignal):
        PassthroughCacheDataLoader.init(self, dl_signals)
        (
            pdl_cache_sizes,
            scenario,
        ) = self._cache.all_gather_cache_size(
            -1,
            self.get_pdl_ranks_in_mp,
            self.model_parallel_group,
        )

        if scenario == RestartMode.COLD_START:
            num_fills = 0
        else:
            num_fills = self._get_num_fills(pdl_cache_sizes)
        dl_signals.fill_queue(num_fills, dl_signals.output_queue)
        self._idx_prefetched += num_fills

        self._logger.debug(f"[{self._rank_id}] CDL init finished")

    def get_local_cache_size(self):
        cache_size = len(self._cache)
        if cache_size == 0:
            return (0, RestartMode.COLD_START)
        else:
            return (cache_size, RestartMode.WARM_START)
