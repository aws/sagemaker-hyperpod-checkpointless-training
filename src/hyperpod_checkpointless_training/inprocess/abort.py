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

# Portions taken from NVIDIA nvidia-resiliency-ext, Copyright Nvidia Corporation

import abc
import concurrent.futures
import contextlib
import gc
import inspect
import logging
import os
import re
import signal
import time
from functools import wraps
import threading
import psutil

import packaging.version
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from .logger import get_logger
from .utils import HPState as State
from .utils import debug_msg
from .compose import Compose
from .finalize import Finalize

from hyperpod_checkpointless_training.inprocess.parameter_update_lock import ParameterUpdateLock

from hyperpod_checkpointless_training.nemo_plugins.abort import (
    abort_megatron,
    abort_te,
    cleanup_rope,
    cleanup_ddp,
    reload_megatron_and_te
)

from hyperpod_checkpointless_training.nemo_plugins.callbacks import CheckpointlessCallback

logger = get_logger()


def log_exec(target):
    if callable(target):

        @wraps(target)
        def wrapper(*args, **kwargs):
            with _log_exec(target):
                ret = target(*args, **kwargs)

        return wrapper
    else:
        return _log_exec(target)


@contextlib.contextmanager
def _log_exec(target, offset=3):
    stack = inspect.stack()
    caller_frame = stack[offset]
    caller_modulename = inspect.getmodulename(caller_frame.filename)

    log = logging.getLogger(caller_modulename)
    rank = int(os.getenv("RANK", "0"))

    if callable(target):
        name = f"{target.__module__}.{target.__qualname__}"
    else:
        name = target

    log.debug(f"{rank=} starts execution: {name}")
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        log.debug(f"{rank=} ends execution: {name} [{elapsed=:.4e}]")


def torch_older_than(version):
    mat = re.match(r"(\d+\.\d+\.\d+).*", torch.__version__)
    if not mat:
        raise RuntimeError(f"Unknown PyTorch version {torch.__version__}")
    torch_version = packaging.version.Version(mat.group(1))
    return torch_version < packaging.version.Version(version)


class Abort(abc.ABC):
    r"""
    Abstract base class for ``abort`` argument for
    :py:class:`inprocess.Wrapper`.

    An instance of :py:class:`Abort` is triggered by a separate monitoring
    thread within :py:class:`inprocess.Wrapper` as part of the termination
    mechanism when a fault is detected. Its primary purpose is to unblock the
    main thread, which might be waiting for results from other distributed
    ranks that are either already terminated or unresponsive. For example, this
    could occur during a distributed collective operation attempting to
    communicate with a terminated rank.

    Multiple instances of :py:class:`Abort` could be composed with
    :py:class:`inprocess.Compose` to achieve the desired behavior.
    """

    @abc.abstractmethod
    def __call__(self, state: State) -> State:
        raise NotImplementedError


class AbortTorchDistributed(Abort):
    r"""
    Aborts PyTorch distributed collectives and destroys all PyTorch
    distributed process groups.

    Shuts down all process group backends (NCCL/Gloo) in parallel using
    separate threads, then destroys the process group.
    """

    @staticmethod
    def shutdown_all_process_group_backends():
        device = torch.device("cuda")
        process_groups = list(torch.distributed.distributed_c10d._world.pg_names)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(process_groups)
        ) as executor:
            futures = [
                executor.submit(
                    AbortTorchDistributed.shutdown_process_group_backend,
                    group,
                    device,
                )
                for group in process_groups
            ]
            done, not_done = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.ALL_COMPLETED,
            )

    @staticmethod
    def shutdown_process_group_backend(group, device):
        if isinstance(group, torch.distributed.ProcessGroup):
            backend = group._get_backend(device)

            if isinstance(
                backend,
                torch.distributed.distributed_c10d.ProcessGroupNCCL,
            ):
                if torch_older_than("2.6.0"):
                    backend._shutdown()
                else:
                    backend.abort()
            elif isinstance(
                backend,
                torch.distributed.distributed_c10d.ProcessGroupGloo,
            ) and hasattr(backend, "abort"):
                backend.abort()

    @log_exec
    def __call__(self, state: State) -> State:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            logger.info(debug_msg("Running torch distributed abort"))
            AbortTorchDistributed.shutdown_all_process_group_backends()
            torch.distributed.destroy_process_group()
        return state


def enable_monitor(func):
    @wraps(func)
    def wrapper(self, state: State, timeout=None, *args, **kwargs):
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            logger.warning(
                debug_msg("HPAbortTorchDistributed: Torch not initialized", rank=-1)
            )
            return state

        monitor_process = None
        event = None

        if os.environ.get("TORCH_ENABLE_ABORT_MONITOR_PROCESS") == "1":
            ctx_start = time.perf_counter()
            ctx = mp.get_context("spawn")
            event = ctx.Event()
            monitor_process = ctx.Process(
                target=HPAbortTorchDistributed._monitor,
                args=(timeout, event),
                daemon=True,
            )
            monitor_process.start()
            ctx_time = (time.perf_counter() - ctx_start) * 1000

            logger.debug(
                debug_msg(
                    f"HPAbortTorchDistributed: Process creation took {ctx_time:.2f}ms",
                    rank=(
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized()
                        else -1
                    ),
                )
            )

        # Call the original function
        result = func(self, state, *args, **kwargs)

        # Clean up monitoring process if it was created
        if monitor_process is not None:
            cleanup_start = time.perf_counter()
            event.set()
            monitor_process.join()
            cleanup_time = (time.perf_counter() - cleanup_start) * 1000

            logger.debug(
                debug_msg(
                    f"HPAbortTorchDistributed: Process cleanup took {cleanup_time:.2f}ms",
                    rank=(
                        torch.distributed.get_rank()
                        if torch.distributed.is_initialized()
                        else -1
                    ),
                )
            )

        return result

    return wrapper


class HPAbortTorchDistributed(AbortTorchDistributed):
    @staticmethod
    def _monitor(timeout, event):
        if event.wait(timeout=timeout):
            return
        # NOTE
        # We have to use SIGKILL to terminate the parent process because the
        # parent may be deadlock which is unable to handle other signal such
        # as SIGTERM to terminate itself. For example, the main process may
        # suffer from deadlock from cudaEventDestroy. In this case, we have to
        # use SIGKILL to terminate the parent process.
        logger.critical("shutdown_process_group hit timeout")
        os.kill(os.getppid(), signal.SIGKILL)

    @log_exec
    @enable_monitor
    def __call__(self, state: State, timeout=None) -> State:
        AbortTorchDistributed.shutdown_all_process_group_backends()
        return state

    @log_exec
    def post_comm_abort_cleanup(self):
        torch.distributed.destroy_process_group()


# This function is from NVRx 0.4.1
class AbortTransformerEngine(Abort):
    r'''
    Aborts TransformerEngine Userbuffer.

    '''

    def __call__(self, state: State, *a, **kw) -> State:
        logger.debug(debug_msg("Running transformer engine abort"))
        try:
            import transformer_engine.pytorch as te
        except Exception:
            logger.warning("Transformer Engine is not detected, skip TE abort.")
        else:
            te.module.base.destroy_ub()

        try:
            import transformer_engine.pytorch.fp8 as te_fp8
        except Exception:
            logger.warning("Transformer Engine is not detected, skip TE_FP8 abort.")
        else:
            # Clear a class-member containing a process group
            te_fp8.FP8GlobalStateManager.reset()

        return state


class HPCheckpointingAbort(Abort):
    """
    Manages checkpoint component cleanup during fault recovery.

    Saves checkpoint states, kills async checkpoint processes and
    checkpoint manager, nullifies checkpoint_io,
    cleans up ModelCheckpoint callbacks. Applies pre-communicator abort
    and post-communicator abort cleanup phases.
    """

    def __init__(self):
        self.trainer = None
        self._checkpoint_abort_lock = threading.Lock()

    def save_checkpoint(self):
        """
        Saves checkpoint state for potential checkpointless recovery:
        - Stores model checksums for integrity validation
        - Saves RNG states for deterministic recovery
        - Updates global step tracking
        - Releases parameter update lock to mark completion
        """
        with ParameterUpdateLock():
            checkpoint_manager = self.trainer.wrapper.checkpoint_manager
            checkpoint_manager.checksum_manager.store_checksum(self.trainer)
            checkpoint_manager.save_checkpoint(self.trainer)
            checkpoint_manager.store_rng_states()

    def cleanup_ckpt_manager(self):
        try:
            from megatron.core.dist_checkpointing.strategies import filesystem_async
            manager_proc_pid = filesystem_async._results_queue._manager._process.pid
            logger.info(f"Killing checkpoint manager with pid {manager_proc_pid}")
            os.kill(manager_proc_pid, signal.SIGKILL)
            filesystem_async._results_queue = None
        except Exception:
            logger.debug("No checkpoint manager to cleanup.")

    def cleanup_ckpt_processes(self):
        if hasattr(self.trainer.strategy.checkpoint_io, "async_calls_queue"):
            try:
                async_queue = self.trainer.strategy.checkpoint_io.async_calls_queue
                for active_call in async_queue.async_calls:
                    async_proc_pid = active_call.async_caller.process.pid
                    parent = psutil.Process(async_proc_pid)
                    for child in parent.children():
                        try:
                            logger.debug(f"Killing checkpoint worker process  {child.pid}")
                            child.kill()
                        except Exception as e:
                            logger.warning(f"Failed to clean async process child: {e}")
                    logger.info(f"Killing async checkpoint process {async_proc_pid}")
                    parent.kill()
            except Exception as e:
                logger.warning(f"Exception during checkpoint process cleanup: {e}")

    def reset_model_checkpoint_callback(self):
        for callback in self.trainer.callbacks:
            if "ModelCheckpoint" in callback.state_key:
                logger.info(f"Resetting ModelCheckpoint callback.")
                callback.deferred_ckpts_to_remove.clear()
                callback._last_global_step_saved = 0
                callback._last_checkpoint_saved = ""

    def post_comm_abort_cleanup(self):
        """Clean up checkpoint after communicator abort.

        Save checkpoint states, executes _extra_post_comm_abort_clean_up and recreates
        checkpoint IO to ensure clean state after communicator abort.

        Note:
            Must execute after NCCL abort since active collectives can block CUDA
            kernels and prevent garbage collection from acquiring pthread locks.
            Save checkpoint also calls cuda kernels so must execute after.
        """
        with self._checkpoint_abort_lock:
            try:
                if not ParameterUpdateLock().acquired:
                    self.save_checkpoint()
                self._extra_post_comm_abort_clean_up()
                self.trainer.strategy._checkpoint_io = None
            except Exception as e:
                logger.warning(debug_msg(f"Cannot execute post cc abort checkpoint cleanup {e}"))
            finally:
                self.trainer = None

    def _extra_post_comm_abort_clean_up(self):
        pass

    def register_trainer(self, trainer):
        with self._checkpoint_abort_lock:
            self.trainer = trainer

    def __call__(self, state=None, *args, **kwargs):
        with self._checkpoint_abort_lock:
            logger.debug(debug_msg("Running checkpoint abort"))
            try:
                if self.trainer is None:
                    raise ValueError("Unable to execute checkpoint abort as trainer not registered.")
                self.cleanup_ckpt_processes()
                self.cleanup_ckpt_manager()
                self.reset_model_checkpoint_callback()
                self._extra_cleanup()
            except Exception as e:
                logger.warning(debug_msg(f"Cannot execute HPCheckpointing abort {e}"))
        return state

    def _extra_cleanup(self):
        pass


class HPDataLoaderAbort(Abort):
    """
    Stop all the dataloaders Gloo communication.
    This callable will be called before the actual abort happens.
    """
    def __call__(self, state=None, *a, **kw) -> State:
        try:
            logger.debug(debug_msg("Running data loader abort"))
            HPDataLoaderManager().abort()
        except Exception as e:
            logger.warning(debug_msg(f"Failed to abort data loader: {e}"))
        return state


class HPDataLoaderManager:
    """Singleton manager class to handle data loader lifecycle and cleanup."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initialize the manager instance."""
        self._active_dataloaders = set()
        self._dataloaders_lock = threading.Lock()

    def register(self, dataloader: DataLoader):
        """Register a single dataloader instance if not already registered."""
        if dataloader is None:
            logger.warning(debug_msg("Skipping registration of None dataloader"))
            return

        with self._dataloaders_lock:
            if dataloader not in self._active_dataloaders:
                self._active_dataloaders.add(dataloader)

    def _cleanup_worker_processes(self):
        """Clean up data worker processes."""
        subprocesses = psutil.Process().children(recursive=True)
        logger.debug(debug_msg(f"Found {len(subprocesses)} subprocess(es)"))

        for process in subprocesses:
            try:
                if "pt_data_worker" in process.name():
                    logger.debug(debug_msg(f"Attempting to terminate worker process {process.pid} ({process.name()})"))
                    try:
                        process.kill()
                        logger.debug(debug_msg(f"Successfully killed process {process.pid}"))
                    except psutil.NoSuchProcess:
                        logger.debug(debug_msg(f"Process {process.pid} already terminated"))
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(debug_msg(f"Error accessing process {process.pid}: {str(e)}"))

    def abort(self):
        with self._dataloaders_lock:
            try:
                active_count = len(self._active_dataloaders)
                logger.debug(debug_msg(f"Start cleaning up {active_count} dataloaders"))

                for dl in list(self._active_dataloaders):
                    dl.stop()

                logger.debug(debug_msg(f"Completed cleanup of {active_count} dataloaders"))
            except Exception as e:
                logger.error(debug_msg(f"Error stopping dataloader: {str(e)}"))
            finally:
                self._cleanup_worker_processes()
                self._active_dataloaders.clear()

class CheckpointlessAbortManager:
    @staticmethod
    def get_default_checkpointless_abort() -> Compose:
        """
        Get the default abort compose instance as a static method.

        Returns:
            Compose: The default composed abort instance containing all abort components

        Example:
            default_abort = CheckpointlessAbortManager.get_default_checkpointless_abort()
        """
        return Compose(AbortTransformerEngine(), HPCheckpointingAbort(), HPAbortTorchDistributed(), HPDataLoaderAbort())

    @staticmethod
    def create_custom_abort(*abort_instances: Abort) -> Compose:
        """
        Create a custom abort compose with only the specified abort instances.

        Args:
            *abort_instances: Variable number of abort instances to include in the compose

        Returns:
            Compose: A new composed abort instance containing only the specified components

        Example:
            custom_abort = CheckpointlessAbortManager.create_custom_abort(
                HPDataLoaderAbort(),
                HPCheckpointingAbort()
            )
        """
        if not abort_instances:
            raise ValueError("At least one abort instance must be provided")

        return Compose(*abort_instances)

    @staticmethod
    def override_abort(abort_compose: Compose, abort_type: type, new_abort: Abort) -> Compose:
        """
        Replace a specific abort component in a Compose instance with a new component.

        Args:
            abort_compose: The original Compose instance
            abort_type: The type of abort to replace (e.g., HPCheckpointingAbort)
            new_abort: The new abort instance to use instead

        Returns:
            A new Compose instance with the component replaced

        Example:
            custom_abort = CheckpointlessAbortManager.override_abort(
                CheckpointlessAbortManager.get_default_checkpointless_abort(),
                HPCheckpointingAbort,
                CustomHPCheckpointingAbort()
            )
        """
        # Use getattr to safely get instances from the Compose object
        instances = getattr(abort_compose, 'instances', None)
        if instances is None:
            raise ValueError("abort_compose must have 'instances' attribute")

        new_instances = []

        for instance in instances:
            if isinstance(instance, abort_type):
                new_instances.append(new_abort)
            else:
                new_instances.append(instance)

        return Compose(*new_instances)


class CheckpointlessFinalizeCleanup(Finalize):
    """Performs comprehensive cleanup after fault detection to prepare for in-process restart.

    Executes framework-specific cleanup operations including Megatron/TransformerEngine abort,
    DDP cleanup, module reloading, and memory cleanup by destroying training component references.
    Optionally clears Lightning module to reduce GPU memory footprint during restart.
    """

    def __init__(self):
        self.trainer = None
        self._finalize_abort_lock = threading.Lock()

    def __call__(self, *a, **kw):
        _clean_tensor_hook = False
        _clean_lightning_module = False
        for callback in self.trainer.callbacks:
            if isinstance(callback, CheckpointlessCallback):
                _clean_tensor_hook = callback.clean_tensor_hook
                _clean_lightning_module = getattr(callback, 'clean_lightning_module', False)
        try:
            abort_megatron()
            abort_te()
            cleanup_rope(self.trainer.lightning_module)
            cleanup_ddp(self.trainer)
            reload_megatron_and_te()
            if _clean_lightning_module:
                self._maybe_clear_lightning_module(self.trainer)
            self._clear_target_class_attributes()
            logger.debug(debug_msg("Completed finalize cleanup"))
        except Exception as e:
            logger.error(debug_msg(f"Error during cleanup: {str(e)}"))
        finally:
            # Cleanup after calls
            with self._finalize_abort_lock:
                self.trainer = None

    # Call during HPRestartlessCallback#on_rcb_start
    def register_attributes(self, trainer):
        with self._finalize_abort_lock:
            self.trainer = trainer

    def _maybe_clear_lightning_module(self, trainer):
        if trainer is None:
            return
        if not hasattr(trainer, "strategy"):
            return
        try:
            logger.debug(debug_msg(f"try _maybe_clear_lightning_module"))
            if hasattr(trainer.strategy, "lightning_module") and trainer.strategy.lightning_module is not None:
                trainer.strategy.lightning_module.cpu()
                trainer.strategy._lightning_module = None
        except Exception as e:
            logger.warning(debug_msg(f"Error during clear_lightning_module (Can be ignored): {str(e)}"))

    def _clear_target_class_attributes(self):
        """Destroy instance references of core training components to clear memory"""

        objects_to_destroy = []
        TARGET_CLASSES = {
            'Trainer', 'GPTOSSModel', 'LlamaModel', 'CheckpointlessMegatronStrategy', 'DistributedDataParallel',
            '_ParamAndGradBuffer', '_ParamAndGradBucket', '_ParamAndGradBucketGroup', 'DDP', 'DistributedOptimizer', 'ChainedOptimizer',
            'Parameter', 'TransformerLayer', 'YarnRotaryEmbedding', 'TransformerBlock', 'MegatronParallel',
            '_LoggerConnector', '_SignalConnector',
        }
        for obj in gc.get_objects():
            obj_type = type(obj).__name__

            # Must match target_classes element exactly
            if obj_type in TARGET_CLASSES:
                objects_to_destroy.append(obj)

        logger.debug(debug_msg(f"Found {len(objects_to_destroy)} objects to destroy"))

        for obj in objects_to_destroy:
            obj_dict = getattr(obj, "__dict__", None)
            if not obj_dict:
                continue

            for attr_name in tuple(obj_dict.keys()):
                attr_value = obj_dict.get(attr_name, None)

                if attr_value is not None:
                    try:
                        clear_fn = getattr(attr_value, "clear", None)
                    except Exception as e:
                        clear_fn = None
                    if callable(clear_fn):
                        try:
                            clear_fn()
                        except Exception:
                            pass
                obj_dict[attr_name] = None
