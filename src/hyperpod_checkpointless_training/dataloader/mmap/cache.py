# type: ignore
"""Cache."""

import os
import shutil
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from filelock import FileLock

from hyperpod_checkpointless_training.dataloader.mmap.utils import RestartMode
from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg

logger = get_logger(__name__)


class CacheInterface(ABC):
    """
    Cache Interface.

    Interface to define a Cache implementation that handles sequential ordered content.

    Content can be stored in either of two stages: Staging or Ready.
        Staging:
            The Staging stage is an intermediate storage location for NotReady content.
            Content should be staged here first, before being moved to the Ready stage.
            This helps prevents content corruption.
        Ready:
            The Ready stage is the final storage location for Ready content.

    Assumptions:
        The content index should be obtainable from its respective path in Ready.
    """

    @abstractmethod
    def init(self) -> None:
        """Initialize the Cache."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of content entries in Ready."""

    @abstractmethod
    def create_staging_entry(self) -> str:
        """
        Create an entry in Staging.

        Returns:
            A string specifying the path to the placeholder Staging entry.
        """

    @abstractmethod
    def stage_content(self, content: Any) -> str:
        """
        Create an entry in Staging with the given content.

        Args:
            content: The content to be staged.

        Returns:
            A string specifying the path to the Staging entry.
        """

    @abstractmethod
    def promote_content(self, staging_entry_path: str, idx: int) -> None:
        """
        Promote a content entry from Staging to Ready with the path and specified index.

        Args:
            staging_entry_path: The path to the entry in Staging.
            idx: The index to promote content entry to.
        """

    @abstractmethod
    def get_content(self, idx: int) -> Any:
        """
        Get content by index from Ready.

        Args:
            idx: The index.

        Returns:
            The content with the specified index.
        """

    @abstractmethod
    def get_content_indices(self) -> List[int]:
        """
        Get list of Ready content indices.

        Returns:
            A list of Ready content indices.
        """

    @abstractmethod
    def prune_cache_init(self, min_idx: int, keep_ahead: int = 0) -> None:
        """
        Prunes the unneeded content in Ready from a previous existing cache
        given the minimum required content index and number of content to keep ahead.

        To be called after cache init and before usage.

        Args:
            min_idx: The minimum content index we want to keep in Ready.
            keep_ahead: The number of successive content indices we want to keep.
        """

    @abstractmethod
    def prune_cache(self, idx: int) -> None:
        """
        Prunes the unneeded content in Ready given the current index.

        Args:
            idx: The current content index.
        """

    @abstractmethod
    def set_final_index(self, idx: int) -> None:
        """
        Set the final index to be used in Ready.

        Args:
            idx: The index to set as final.
        """

    @abstractmethod
    def is_final_index(self, idx: int) -> bool:
        """
        Returns whether the index is the set final index.

        Args:
            idx: The index to check is final.
        """


class MMAPCache(CacheInterface):
    """
    Maintains MMAP cache at the given cache directory. This can be a directory in tmpfs
    or on file system. The class implementation is agnostic to the underlying
    storage used.
    """

    def __init__(
        self,
        cache_dir: str,
        batch_prefix: str = "batch",
        lookback_length: int = 10,
        prefetch_length: int = 10,
        model_checkpoint_frequency: int | None = None,
        read_only: bool = True,
        force_cold_start: bool = False,
        pdl_per_node_group: dist.ProcessGroup | None = None,
    ):
        """
        Initializes the Cache class.

        Args:
            cache_dir (str): The cache directory maintains cache at the given cache directory.
            batch_prefix (str): The prefix of the batch file.
            lookback_length (int): The lookback length for the cache.
            prefetch_length (int): The prefetch length for the cache.
            model_checkpoint_frequency (int | None): The model checkpoint frequency.
                Used to determine when to remove old cache files.
            read_only (bool): If True, the cache is read-only. If False, the cache is writable.
            force_cold_start (bool): If True, the cache is removed before starting the training.
            pdl_per_node_group: The process group of a given node for the PDL.
        """
        self.read_only = read_only
        self.cache_dir = cache_dir
        self.force_cold_start = force_cold_start
        # mainly used for saving batches
        self.cache_tmp_dir = os.path.join(cache_dir, "tmp")
        # used for synchronizing the completion of a step or stopping
        self.complete = os.path.join(cache_dir, "complete")
        self.pdl_per_node_group = pdl_per_node_group
        self.batch_prefix = batch_prefix
        self.lookback_length = lookback_length
        self.prefetch_length = prefetch_length
        self.max_cache_size = prefetch_length + lookback_length
        self.model_checkpoint_frequency = model_checkpoint_frequency

    def init(self):
        """
        Initializes the cache directory.

        If read_only is True, the cache directory is not created.
        """
        if not self.read_only:
            # 1. Validate if there is a cache_dir with unmatched rank id.
            # If it does, remove everything inside the cache_dir.
            self._validate_cache_dir()
            if self.pdl_per_node_group:
                dist.barrier(group=self.pdl_per_node_group)
            # 2. Maybe force cold start.
            self._try_removing_cache_dir()
            # clear cache tmp dir
            self._delete_dir(self.cache_tmp_dir)
            self._delete_dir(self.complete)
            # 3. Setup necessary directories.
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.cache_tmp_dir, exist_ok=True)
            os.makedirs(self.complete, exist_ok=True)
        logger.debug(debug_msg(f"Initialized data cache at {self.cache_dir}"))

    def __len__(self):
        """
        Returns the number of valid caches.
        """
        return len(self.get_content_indices())

    def _validate_cache_dir(self):
        """
        Validates the cache directory.

        Throw away all the full cache directory if corresponding
        cache directory id does not match. Here assume that if one
        rank mismatches, all other ranks within the group will also mismatch.
        """
        try:
            cache_dir = os.path.normpath(self.cache_dir)
            cache_dir_id = os.path.basename(cache_dir)
            pdl_cache_dir = os.path.dirname(cache_dir)
            # expect FileNotFound error here if cache dir does not exist
            all_cache_dir_ids = os.listdir(pdl_cache_dir)

            should_delete_dir = False
            if cache_dir_id not in all_cache_dir_ids:
                should_delete_dir = True
                if not self.pdl_per_node_group:
                    logger.info(
                        debug_msg(
                            f"None of existing cache directories match, Removing {pdl_cache_dir}!"
                        )
                    )
                    self._delete_dir(pdl_cache_dir)
                    return

            if not self.pdl_per_node_group:
                # exit early if we don't have multiple PDL in one node.
                # in CACHE_ONLY mode, we always force cold start after which removes the cache dir,
                # so we don't need to remove the cache dir here.
                return

            tensor = torch.tensor([should_delete_dir], dtype=torch.int)
            dist.all_reduce(
                tensor, torch.distributed.ReduceOp.SUM, group=self.pdl_per_node_group
            )
            # If there are multiple pdl in one node, check if none of the cache directories match.
            if (
                tensor.item() == dist.get_world_size(self.pdl_per_node_group)
                and dist.get_rank(self.pdl_per_node_group) == 0
            ):
                logger.info(
                    debug_msg(
                        f"None of existing cache directories match, Removing {pdl_cache_dir}!"
                    )
                )
                self._delete_dir(pdl_cache_dir)
        except FileNotFoundError:
            logger.error(
                f"Failed to validate cache directory '{self.cache_dir}': directory not found"
            )

    def _try_removing_cache_dir(self):
        """
        Removes the cache directory if it exists and force_cold_start is True.
        """
        if self.read_only:
            return
        if self.force_cold_start:
            delete_dir_status = self._delete_dir(self.cache_dir)
            if not delete_dir_status:
                logger.error(
                    debug_msg(
                        f"Failed to remove cache directory during cold start '{self.cache_dir}'"
                    )
                )

    def _get_batch_filename(self, idx: int) -> str:
        """
        Constructs the filename for a given batch step.

        Args:
            idx (int): The step number of the batch.

        Returns:
            str: The full path to the batch file.
        """
        filename = os.path.join(self.cache_dir, f"{self.batch_prefix}_{idx}.pt")
        return filename

    def get_content_indices(self) -> List[int]:
        """
        Get list of Ready batch indices saved in cache.

        Returns:
            A list of Ready batch indices saved in cache.
        """
        ids = [
            int(
                fname[len(self.batch_prefix) + 1 : -3]
            )  # Extract step number from 'batch_1.pt'
            for fname in os.listdir(self.cache_dir)
            if fname.startswith(self.batch_prefix) and fname.endswith(".pt")
        ]
        return ids

    def create_staging_entry(self) -> str:
        """
        Create an entry in Staging.

        Returns:
            A string specifying the path to the placeholder Staging entry.
        """
        return self._create_tmp_file()

    def stage_content(self, content: Any) -> str:
        """
        Create an entry in Staging with the specified content.
        """
        staging_path = self.create_staging_entry()
        torch.save(content, staging_path)
        return staging_path

    def _create_tmp_file(self) -> str:
        """Create a file in the cache_dir/tmp directory."""
        if self.read_only:
            raise ValueError(
                "Trying to create a tmp file when read_only=True is not supported."
            )
        with tempfile.NamedTemporaryFile(dir=self.cache_tmp_dir, delete=False) as tmp:
            return tmp.name

    def promote_content(self, staging_entry_path: str, idx: int) -> None:
        """
        Promote a content entry from Staging to Ready with the path and specified index.

        Args:
            staging_entry_path: The path to the entry in Staging.
            idx: The index to promote content entry to.
        """
        self._rename_tmp_file(staging_entry_path, idx)

    def _rename_tmp_file(self, tmp_path: str, idx: int) -> None:
        """
        Given a file in cache_dir/tmp directory, save it into the main cache directory.

        Args:
            tmp_path (str): The tmp file path
            idx (int): The step number of the batch.
        """
        if self.read_only:
            raise ValueError(
                "Trying to save a tmp file into main cache directory when read_only=True is not supported."
            )
        filename = self._get_batch_filename(idx)
        os.rename(tmp_path, filename)

    def get_content(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get MMAP'd batch for step idx.

        Args:
            idx (int): The step number of the batch.

        Returns:
            MMAP'd batch for step idx.
        """
        filename = self._get_batch_filename(idx)
        tensor = torch.load(filename, mmap=True)
        return tensor

    def _remove_oldest(self, train_step: int):
        """
        Remove the oldest batch if the number of batches exceeds the max cache size.

        Args:
            train_step (int): The current training step.
        """
        ids_present = self.get_content_indices()
        if len(ids_present) <= self.max_cache_size + 1:
            return
        id_to_remove = sorted(ids_present)[0]
        if id_to_remove >= train_step:
            return

        filename = self._get_batch_filename(id_to_remove)
        self._delete_file(filename)

    def _delete_dir(self, dirname: str) -> bool:
        """
        Delete the directory with the given dirname.

        Args:
            dirname (str): The dirname to delete.

        Returns:
            bool: Whether the deletion was successful.
        """
        lock = FileLock(dirname + ".lock")
        try:
            # simple lock to ensure that only one process can try to remove
            # the given directory at a time
            with lock.acquire(blocking=False):
                if os.path.exists(dirname):
                    shutil.rmtree(dirname)
        except FileNotFoundError:
            logger.warning(
                debug_msg(
                    f"Failed to delete directory '{dirname}': directory not found"
                )
            )
            return False
        except NotADirectoryError:
            logger.error(
                debug_msg(
                    f"Failed to delete directory '{dirname}': path is not a directory"
                )
            )
            return False
        except OSError as e:
            logger.error(debug_msg(f"Failed to delete directory '{dirname}': {e}"))
            return False
        else:
            logger.debug(debug_msg(f"Deleted directory '{dirname}'."))
            return True

    def _delete_file(self, filename: str) -> bool:
        """
        Delete the file with the given filename.

        Args:
            filename (str): The filename to delete.

        Returns:
            bool: Whether the deletion was successful.
        """
        lock = FileLock(filename + ".lock")
        try:
            # simple lock to ensure that only one process can try to remove
            # the given filename at a time
            with lock.acquire(blocking=False):
                if os.path.exists(filename):
                    os.remove(filename)
        except FileNotFoundError:
            logger.warning(
                debug_msg(f"Failed to delete file '{filename}': file not found")
            )
            return False
        except IsADirectoryError:
            logger.error(
                debug_msg(f"Failed to delete file '{filename}': path is not a file")
            )
            return False
        except OSError as e:
            logger.error(debug_msg(f"Failed to delete file '{filename}': {e}"))
            return False
        else:
            logger.debug(debug_msg(f"Deleted file '{filename}'."))
            return True

    def _remove_lookback(self, last_model_checkpoint_step: int):
        """
        Remove the last lookback interval of cache files w.r.t model checkpoint frequency.

        Args:
            last_model_checkpoint_step (int): The last model checkpoint step.
        """
        if self.model_checkpoint_frequency is None:
            logger.warning(
                "_remove_lookback() called when model checkpoint frequency is not set."
                "_remove_lookback() should only be called when model checkpoint frequency is set."
            )
            return

        try:
            threads = []  # parallelize os.remove using threads
            # remove at most one model checkpoint interval.
            for remove_file_idx in range(
                last_model_checkpoint_step,
                last_model_checkpoint_step
                + min(self.lookback_length, self.model_checkpoint_frequency),
            ):
                fname = self._get_batch_filename(remove_file_idx)
                if not os.path.exists(fname):
                    continue
                thread = threading.Thread(
                    target=self._delete_file,
                    args=(os.path.join(self.cache_dir, fname),),
                )
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
        except RuntimeError as e:
            logger.warning(f"RuntimeError encountered in _remove_lookback(): {e}")

    def prune_cache(self, idx: int) -> None:
        """
        Delete the oldest batch if model frequency is not provided.

        Otherwise, check the position of the training step.
        If within the lookback interval from last model checkpoint step, it will
        maybe remove the full last lookback interval.

        Args:
            idx (int): The current training step.
        """
        # CacheDL -> read_only should be True
        # BaseDL -> read_only should be True
        # PrefetchDL -> read_only should be False
        assert (
            not self.read_only
        ), "Error: prune_cache called when read_only is set to True!"

        train_step = idx

        # If model_checkpoint_frequency is not provided or lookback_length is larger than
        # the checkpoint interval, simply remove the oldest cache.
        if not self.model_checkpoint_frequency or (
            self.model_checkpoint_frequency < self.lookback_length
        ):
            self._remove_oldest(train_step)
        else:
            # given current PDL yield step, calculate the smallest potential
            # training step with small buffer
            lower_bound_training_step = train_step - 2

            if lower_bound_training_step < 0:
                return

            # Check the step with respect to the model checkpoint step.
            rest = lower_bound_training_step % self.model_checkpoint_frequency
            if rest < self.lookback_length:
                if rest == 0:
                    # remove last lookback
                    last_model_checkpoint_step = (
                        lower_bound_training_step - self.model_checkpoint_frequency
                    )
                    if last_model_checkpoint_step < 0:
                        return
                    self._remove_lookback(last_model_checkpoint_step)
            else:
                self._delete_file(self._get_batch_filename(lower_bound_training_step))

    def prune_cache_init(self, min_idx: int, keep_ahead: int = 0) -> None:
        """
        Prunes the Ready stage in cache given the min_idx and keep_ahead.
        Deletes the cache if min_idx is outside the cache and removes any invalid content.

        Args:
            min_idx: The minimum content index we want to keep in Ready.
            keep_ahead: The number of successive content indices we want to keep.
        """
        if self.read_only:
            raise ValueError(
                "Trying to validate an index when read_only=True is not supported."
            )
        all_ids = sorted(self.get_content_indices())

        if not all_ids:
            return

        if min_idx < all_ids[0] or min_idx > all_ids[-1]:
            logger.info(
                debug_msg(
                    f"{self.cache_dir} does not have cache for index {min_idx}. Invalidating ..."
                )
            )
            self._delete_dir(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.cache_tmp_dir, exist_ok=True)
            os.makedirs(self.complete, exist_ok=True)
        else:
            last_saved = min_idx - 1
            max_loop_ahead = min_idx + keep_ahead - 1
            threads = []  # parallelize os.remove using threads
            for file_id in all_ids:
                fname = self._get_batch_filename(file_id)
                if (
                    file_id > max_loop_ahead
                    or file_id < min_idx
                    # Not continuous
                    or file_id != last_saved + 1
                ):
                    thread = threading.Thread(
                        target=os.remove, args=(os.path.join(self.cache_dir, fname),)
                    )
                    thread.start()
                    threads.append(thread)
                else:
                    last_saved = file_id
            for thread in threads:
                thread.join()

    @staticmethod
    def get_non_empty_values(cache_size_dict: dict) -> List[int]:
        """
        Get the non empty values from the dict.

        Args:
            cache_size_dict: Dictionary containing cache sizes.

        Returns:
            A list containing positive cache sizes.
        """
        return [v for v in cache_size_dict.values() if v > 0]

    def all_gather_cache_size(
        self,
        cache_size: int,
        pdl_ranks_in_mp: List[int],
        model_parallel_group: dist.ProcessGroup,
    ) -> Tuple[dict, RestartMode]:
        """
        All gather to get the cache_size within a model replica group.

        Args:
            cache_size: The cache size.
            pdl_ranks_in_mp: The list of PDL ranks in model parallel group.
            model_parallel_group: The process group used for communication.

        Returns:
            A 2-tuple containing the cache sizes for pdl_ranks_in_mp and
            the RestartMode enum indicating the restart scenario.
        """
        if self.force_cold_start:
            return (0, RestartMode.COLD_START)

        model_parallel_global_ranks = dist.get_process_group_ranks(model_parallel_group)
        device = (
            torch.cuda.current_device()
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        cache_size_tensor = torch.tensor([cache_size], device=device)
        tensor_list = [
            torch.zeros(1, dtype=torch.int64, device=device)
            for _ in range(dist.get_world_size(model_parallel_group))
        ]

        dist.all_gather(tensor_list, cache_size_tensor, group=model_parallel_group)
        all_gather_buffer_size = [int(tensor.item()) for tensor in tensor_list]

        pdl_cache_sizes: dict[int, int] = {
            global_rank: cache_size
            for cache_size, global_rank in zip(
                all_gather_buffer_size,
                model_parallel_global_ranks,
            )
            if global_rank in pdl_ranks_in_mp
        }

        num_empty_pdl = sum(x == 0 for x in pdl_cache_sizes.values())

        if num_empty_pdl == 0:
            scenario = RestartMode.WARM_START
        else:
            scenario = RestartMode.COLD_START

        return (
            pdl_cache_sizes,
            scenario,
        )

    def set_final_index(self, idx: int) -> None:
        """
        Set the final index to be used in Ready.

        Args:
            idx: The index to set as final.
        """
        self._write_final(idx)

    def _write_final(self, step: int) -> None:
        """Write the `final` file for the given step.

        We write a file to the `complete` directory to synchronize when to stop
        for a given step within a node.
        """
        tmp_final = self.create_staging_entry()
        final_file = os.path.join(self.complete, f"{step}")
        os.rename(tmp_final, final_file)

    def is_final_index(self, idx: int) -> bool:
        """Return whether the index is the set final index."""
        step = idx
        if not os.path.exists(self.complete):
            return False

        all_files = os.listdir(self.complete)
        if not all_files:
            return False

        if len(all_files) > 1:
            raise ValueError(
                f"There should be only one complete step, but there are {all_files}"
            )

        filename = all_files[0]
        complete_step = int(filename.split("/")[-1])

        return step >= complete_step


class TTLMMAPCache(MMAPCache):
    """
    TTLMMapCache.

    An extension of MMAPCache that supports time-to-live (TTL) for cache files.
    """

    TTL_SECONDS = 3600.0  # 1 hour
    CHECK_INTERVAL = 900.0  # 15 minutes
    stop_event = threading.Event()

    def __init__(
        self,
        *args,
        ttl_seconds: float | None = None,
        check_interval: float | None = None,
        cleanup_buffer: float | None = None,
        **kwargs,
    ):
        """Initialize TTLMMAPCache."""
        super().__init__(*args, **kwargs)
        if ttl_seconds is not None:
            self.TTL_SECONDS = ttl_seconds
        if check_interval is not None:
            self.CHECK_INTERVAL = check_interval
        if cleanup_buffer is not None:
            self.cleanup_buffer = cleanup_buffer
        else:
            # default to 20% of TTL
            self.cleanup_buffer = 0.2 * self.TTL_SECONDS
        self.cleanup_thread = None
        logger.info(
            f"TTLMMAPCache has been initialized with a {self.TTL_SECONDS} second TTL and "
            f"a {self.CHECK_INTERVAL} second check interval. "
        )

    def check_ttl(self, filename: str, buffer: float = 0.0) -> bool:
        """
        Check if the file has expired based on the TTL.

        Args:
            filename (str): The filename to check.
            buffer (float): Optional time buffer in seconds.
                Shortens the effective TTL.

        Returns:
            bool: True if the file has expired, False otherwise.
        """
        if not os.path.exists(filename):
            return False
        file_stat = os.stat(filename)
        file_age = time.time() - file_stat.st_mtime
        return file_age > (self.TTL_SECONDS - buffer)

    def cleanup_expired_files(self, use_cleanup_buffer: bool = False):
        """
        Cleanup expired files in the cache directory.
        """
        # Check the earliest step file
        all_ids = sorted(self.get_content_indices())
        for idx in all_ids:
            fname = self._get_batch_filename(idx)
            logger.debug(f"Checking if file {fname} with {idx=} is expired...")
            buffer = self.cleanup_buffer if use_cleanup_buffer else 0.0
            if self.check_ttl(fname, buffer=buffer):
                logger.debug(f"Deleting expired file {fname} with {idx=}.")
                self._delete_file(fname)
            else:
                # We only need to check the earliest step file, as files are added in order.
                logger.debug(
                    f"File {fname} with {idx=} is not expired. Stopping check."
                )
                break

    def cleanup_expired_files_worker(self):
        """
        Worker function to cleanup expired files.
        """
        if self.read_only:
            return
        while not self.stop_event.is_set():
            self.cleanup_expired_files()
            # wait for check interval, but exit early if stop_event is set
            if self.stop_event.wait(self.CHECK_INTERVAL):
                break

    def start_cleanup_expired_files_thread(self):
        """
        Start a background thread to cleanup expired files.
        """
        if self.read_only:
            return
        if self.cleanup_thread is not None:
            return
        logger.debug("Starting cleanup thread")
        self.cleanup_thread = threading.Thread(
            target=self.cleanup_expired_files_worker, daemon=True
        )
        self.cleanup_thread.start()

    def stop_cleanup_expired_files_thread(self):
        """
        Stop the background thread to cleanup expired files.
        """
        if self.cleanup_thread is None:
            return
        self.stop_event.set()
        self.cleanup_thread.join()
        self.cleanup_thread = None
        self.stop_event.clear()
