import functools
import gc
import threading
import time
import traceback
from typing import Any, Callable, Optional

import torch
from viztracer import VizTracer

from .abort import Abort, HPAbortTorchDistributed
from .elastic.hp_agent_event import HPAgentEvent
from .exception import HealthCheckError, InternalError, RankShouldRestart
from .finalize import Finalize
from .health_check import HealthCheck
from .hp_fault_handling_thread import HPFaultHandlingThread
from .hp_monitor_thread import HPMonitorThread
from .logger import get_logger
from .param_utils import enforce_type, enforce_value, substitute_param_value
from .parameter_update_lock import ParameterUpdateLock
from .tools.memory_tracker import memory_status
from .utils import AtomicInt, HPState, debug_msg, log_exc, reraise_if_unraisable
from .compose import Compose
from hyperpod_checkpointless_training.inprocess.env_validation import EnvManager
from hyperpod_checkpointless_training.inprocess.tools import memory_tracker


class HPWrapper:
    """Python function wrapper that returns a :py:class:`HPCallWrapper`. This
    wrapper enables restart capabilities for a Restart Code Block (RCB). The
    implementation uses a context manager instead of a Python decorator because
    the call wrapper lacks information about the number of RCBs it should
    monitor. As a result, the HPWrapper cannot release global resources such as
    :py:class`HPMonitorThread` which are maintained the entire training process
    lifecycle from start to completion.

    Args:
        abort: Asynchronously aborts execution.
        finalize: Rank-local finalize.
        health_check: Rank-local health check.
        hp_api_factory: Creating a HyperPod API to interact with AWS HyperPod.
        abort_timeout: Timeout for abort call in fault controlling thread
        enabled: Enables the wrapper.
        trace_file_path: The path to the tracefile for viztracer profile
        async_raise_before_abort: enable raise before abort in fault controlling thread, default True
        early_abort_communicator: Abort communicator (NCCL/Gloo) before aborting dataloader, default False
        checkpoint_manager: Checkpoint manager instance for handling checkpoint operations
        check_memory_status: Enable memory status logging, default True
    """

    def __init__(
        self,
        *,
        abort: Optional[Abort] = Compose(HPAbortTorchDistributed()),
        finalize: Optional[Finalize] = None,
        health_check: Optional[HealthCheck] = None,
        hp_api_factory: Optional[Callable] = None,
        abort_timeout: Optional[float] = None,
        enabled: bool = True,
        trace_file_path: Optional[str] = None,
        async_raise_before_abort: bool = True,
        early_abort_communicator: bool = False,
        checkpoint_manager: Any = None,
        check_memory_status: bool = True,
    ):
        enforce_type("abort", (Abort, type(None)))
        enforce_type("finalize", (Finalize, type(None)))
        enforce_type("enabled", bool)
        enforce_type("abort_timeout", (float, type(None)))
        enforce_value(torch.distributed.is_available())
        self.enabled = enabled

        if not enabled:
            return

        # Environment Variable validation
        env_manager = EnvManager()
        env_manager.validate()

        self.abort = abort
        self.finalize = finalize
        self.health_check = health_check
        self.hp_api = hp_api_factory()
        self.seq = AtomicInt(-1)
        self.abort_timeout = abort_timeout
        self.trace_file_path = trace_file_path
        self.async_raise_before_abort = async_raise_before_abort
        self.early_abort_communicator = early_abort_communicator
        self.checkpoint_manager = checkpoint_manager
        self.check_memory_status = check_memory_status

    def __call__(self, fn):
        if not self.enabled:
            return fn

        @functools.wraps(fn)
        def wrapped(*a, **kw):
            with HPCallWrapper(self) as call_wrapper:
                return call_wrapper(fn, *a, **kw)

        return wrapped


class HPCallWrapper:
    """The :py:class:`HPCallWrapper` monitors and manages the state of a
    Restart Code Block (RCB). When an RCB throws an exception, the wrapper
    notifies all other ranks to restart and initiates abort of process groups.

    Args:
        wrapper: A wrapper which holds global inprocess restart settings.
    """

    def __init__(self, wrapper: HPWrapper):
        self.hp_api = wrapper.hp_api
        self.abort = wrapper.abort
        self.finalize = wrapper.finalize
        self.health_check = wrapper.health_check
        self.failure = threading.Event()
        self.stop_raising = threading.Event()
        self.atomic_lock = ParameterUpdateLock()
        # first_step will be set false after the very first in-process restart
        self.atomic_lock.first_step = True
        self.abort_timeout = wrapper.abort_timeout
        self.trace_file_path = wrapper.trace_file_path
        self.async_raise_before_abort = wrapper.async_raise_before_abort
        # checkpoint manager is hold by wrapper but managed by the underlying framework
        self.checkpoint_manager = wrapper.checkpoint_manager
        self.check_memory_status = wrapper.check_memory_status

        self.state = HPState()
        self.logger = get_logger()
        self.hp_fault_handling_thread = None
        self.hp_monitor_thread = None
        self.seq = wrapper.seq
        self.step_upon_restart = 0  # Will get reset to 0 upon restart and get incremented per each step, logic is in plugin
        if self.trace_file_path is not None:
            self.tracer = VizTracer(register_global=False, ignore_c_function=True)
            self.tracing_enabled = True
        else:
            self.tracer = None
            self.tracing_enabled = False
        self.is_tracing = False

    def enable_tracing(self):
        if not self.tracing_enabled or self.state.rank != 0 or self.is_tracing:
            return
        self.is_tracing = True
        self.tracer.start()

    def end_tracing(self):
        if not self.tracing_enabled or self.state.rank != 0 or not self.is_tracing:
            return
        self.is_tracing = False
        self.tracer.stop()
        path = f"{self.trace_file_path}/profile_{time.monotonic()}.json"
        self.logger.info(
            debug_msg(
                f"saving trace file to {path}",
                rank=self.state.rank,
                seq=self.seq,
                steps=self.step_upon_restart,
            )
        )
        self.tracer.save(path)

    def initialize_barrier(self):
        """This function is in charge of waiting HyperPod barrier once it
        encouter an exception from RCB.
        """
        try:
            self.start_hp_monitor_thread()
            self.state.get_distributed_vars()
            # Recovery from a failure, main thread will wait for everyone to be ready
            self.logger.debug(
                debug_msg(
                    "waiting outside hp_barrier",
                    rank=self.state.rank,
                    seq=self.seq,
                    steps=self.step_upon_restart,
                )
            )
            self.seq.set(self.hp_api.hyperpod_barrier(self.state.rank))
            # Healthy process distributed vars will remain unchanged Standby
            # node's distributed vars can only be set after barrier
            self.state.get_distributed_vars()
            self.logger.debug(
                debug_msg(
                    "passed hp_barrier",
                    rank=self.state.rank,
                    seq=self.seq,
                    steps=self.step_upon_restart,
                )
            )

            # NOTE that we have to re-create a new `HPFaultHandlingThread` in
            # each failure. The reason is that `HPFaultHandlingThread` utilizes
            # `async_abort_main_thread` to trigger main thread raising
            # `RankShouldRestart` error. In this case, main thread has to call
            # `HPFaultHandlingThread.join` to capture the exception; otherwise,
            # main thread won't receive the exception.
            assert self.hp_fault_handling_thread is None

            # setting a new hp_fault_handling_thread
            self.stop_raising.clear()
            self.failure.clear()
            self.start_hp_fault_handling_thread()
            self.step_upon_restart = 0
        except BaseException as hp_barrier_ex:
            self.logger.error(
                log_exc(
                    hp_barrier_ex,
                    "hp_barrier_ex",
                    rank=self.state.rank,
                    seq=self.seq,
                    steps=self.step_upon_restart,
                )
            )
            raise hp_barrier_ex

    def start_hp_fault_handling_thread(self):
        self.hp_fault_handling_thread = HPFaultHandlingThread(
            abort=self.abort,
            failure=self.failure,
            stop_raising=self.stop_raising,
            atomic_lock=self.atomic_lock,
            abort_timeout=self.abort_timeout,
            async_raise_before_abort=self.async_raise_before_abort,
            seq=self.seq,
            daemon=True,
        )
        self.hp_fault_handling_thread.start()

    def handle_fn_exception(self, call_ex):
        """Process exception from execution function or RCB.

        Args:
            call_ex: Exception from the monitoring function.

        Note that :py:class:`RankShouldRestart` won't be handled in this
        function because it is a BaseException.
        """
        try:
            self.logger.error(
                log_exc(
                    call_ex,
                    "call_ex",
                    rank=self.state.rank,
                    seq=self.seq,
                    steps=self.step_upon_restart,
                )
            )
            if self.step_upon_restart > 1:
                self.hp_api.hyperpod_send(
                    HPAgentEvent.FAILURE,
                    rank=self.state.rank,
                    seq=self.seq.get(),
                )
            else:
                # In-process fault handling would hang/fail before the NCCL is fully initialized, i.e. error happens on the first step
                # Restart with PLR if error happens in the first step
                self.logger.debug(
                    debug_msg(
                        f"triggering plr for step_upon_restart {self.step_upon_restart}",
                        rank=self.state.rank,
                        seq=self.seq,
                        steps=self.step_upon_restart,
                    )
                )
                self.hp_api.hyperpod_send(
                    HPAgentEvent.FAILURE,
                    rank=self.state.rank,
                    seq=self.seq.get(),
                    plr_restart=True,
                )
            self.failure.set()
            self.stop_raising.set()
            # NOTE:
            # We have to call `threading.Thread.join()` here for waiting
            # `RankShouldRestart`; otherwise, main thread won't receive the
            # exception. In this case, we are unable to execute restart handler.
            self.logger.debug(
                debug_msg(
                    "wait hp_fault_handling_thread join",
                    rank=self.state.rank,
                    seq=self.seq,
                    steps=self.step_upon_restart,
                )
            )
            self.hp_fault_handling_thread.join()
            self.hp_fault_handling_thread.shutdown()
        except Exception as other_ex:
            self.logger.critical(
                log_exc(
                    other_ex,
                    "other_ex",
                    rank=self.state.rank,
                    seq=self.seq,
                    steps=self.step_upon_restart,
                )
            )
            raise InternalError(f"{self.state}") from other_ex
        else:
            # Somehow the async exception is not captured by the main thread
            # We manual raise here
            self.logger.warning(
                debug_msg(
                    "handle_fn_exception finished without error, manually throw RankShouldRestart",
                    rank=self.state.rank,
                    seq=self.seq,
                    steps=self.step_upon_restart,
                )
            )
            raise RankShouldRestart()

    def handle_finalize(self, term_ex):
        """Execute finalize function in the restart handler.

        Args:
            term_ex: A termination exception. Should be `RankShouldRestart`
        """
        if not self.finalize:
            return
        try:
            self.finalize(self.state, term_ex.__cause__)
        except Exception as finalize_ex:
            self.logger.error(
                log_exc(
                    finalize_ex,
                    "finalize_ex",
                    rank=self.state.rank,
                    seq=self.seq,
                    steps=self.step_upon_restart,
                )
            )
            raise finalize_ex from term_ex

    def handle_health_check(self, term_ex):
        """Execute health check in the restart handler.

        Args:
            term_ex: A termination exception. Should be `RankShouldRestart`
        """
        if not self.health_check:
            return
        try:
            self.health_check(self.state, term_ex.__cause__)
        except Exception as health_ex:
            self.logger.error(
                log_exc(
                    health_ex,
                    "health_ex",
                    rank=self.state.rank,
                    seq=self.seq,
                    steps=self.step_upon_restart,
                )
            )
            try:
                raise health_ex from term_ex
            except Exception:
                raise HealthCheckError from health_ex

    def handle_gc(self):
        """Garbage collect until all intermediate variables are out of scope"""
        try:
            while gc.collect():
                pass
        except Exception as gc_ex:
            self.logger.error(
                log_exc(
                    gc_ex,
                    "gc_ex",
                    rank=self.state.rank,
                    seq=self.seq,
                    steps=self.step_upon_restart,
                )
            )

    def handle_unknown_error(self, term_ex):
        """Handle unknown exception from the executed RCB.

        Args:
            term_ex: A termination exception.
        """
        self.logger.critical(
            log_exc(
                term_ex,
                "term_ex",
                rank=self.state.rank,
                seq=self.seq,
                steps=self.step_upon_restart,
            )
        )
        raise InternalError(f"{self.state.rank}") from term_ex

    def restart(self, term_ex):
        """Restart handler to do some check before restart.

        Args:
            term_ex: A termination exception. Should be `RankShouldRestart`
        """
        self.enable_tracing()
        self.logger.error(
            log_exc(
                term_ex,
                "term_ex",
                rank=self.state.rank,
                seq=self.seq,
                steps=self.step_upon_restart,
            )
        )
        if self.step_upon_restart <= 1:
            # Trigger PLR when error happen within first step
            self.hp_api.hyperpod_send(
                HPAgentEvent.FAILURE,
                rank=self.state.rank,
                seq=self.seq.get(),
                plr_restart=True,
            )
        self.logger.debug(
            debug_msg(
                "RankShouldRestart captured",
                rank=self.state.rank,
                seq=self.seq,
                steps=self.step_upon_restart,
            )
        )
        self.shutdown_hp_fault_handling_thread()
        self.atomic_lock.first_step = False
        # The lock should be release regradless now
        self.atomic_lock.force_release()
        self.checkpoint_manager.reset_checkpointless_recovery_validation()
        if self.check_memory_status:
            msg, _ = memory_status(tag="Before Offload")
            self.logger.info(debug_msg(msg))

        self.checkpoint_manager.maybe_offload_checkpoint()
        self.logger.debug(
            debug_msg(
                "hp_fault_handling_thread shutdown",
                rank=self.state.rank,
                seq=self.seq,
                steps=self.step_upon_restart,
            )
        )

        self.handle_finalize(term_ex)
        self.handle_gc()
        self.handle_health_check(term_ex)
        if self.check_memory_status:
            msg, _ = memory_status(tag="After restart")
            self.logger.info(debug_msg(msg))

    def launch(self, fn, *a, **kw):
        """A function to execute the RCB.

        Args:
            fn: Function to be executed.
            a: Function arguments.
            kw: Function keyword arguments.
        """
        try:
            a, kw = substitute_param_value(fn, a, kw, {HPCallWrapper: self})
            return fn(*a, **kw)
        except Exception as train_ex:
            self.enable_tracing()
            formated_trackback = traceback.format_exception(train_ex)
            self.logger.debug(
                debug_msg(
                    f"formated_trackback {formated_trackback}",
                    rank=self.state.rank,
                    seq=self.seq,
                    steps=self.step_upon_restart,
                )
            )
            traceback.print_exc()
            self.handle_fn_exception(train_ex)

    def run(self, fn, *a, **kw):
        """A wrapper to execute the RCB

        Args:
            fn: Function to be executed.
            a: Function arguments.
            kw: Function keyword arguments.
        """
        while True:
            try:
                self.initialize_barrier()
                self.end_tracing()
                return self.launch(fn, *a, **kw)
            except RankShouldRestart as term_ex:
                self.restart(term_ex)
            except Exception as term_ex:
                self.handle_unknown_error(term_ex)

    @reraise_if_unraisable(RankShouldRestart)
    def start(self, fn, *a, **kw):
        """Wrapper function which is in charge of handling a RCB lifecycle.

        Args:
            fn: Function to be executed.
            a: Function arguments.
            kw: Function keyword arguments.
        """
        try:
            ret = self.run(fn, *a, **kw)
        except BaseException as exit_ex:
            self.logger.critical(
                log_exc(
                    exit_ex,
                    "exit_ex",
                    rank=self.state.rank,
                    seq=self.seq,
                    steps=self.step_upon_restart,
                )
            )
            raise exit_ex
        else:
            self.state.advance()
        return ret

    def start_hp_monitor_thread(self):
        if self.hp_monitor_thread and self.hp_monitor_thread.is_alive():
            return

        self.hp_monitor_thread = HPMonitorThread(
            hp_api=self.hp_api,
            failure=self.failure,
            seq=self.seq,
            daemon=True,
        )
        self.hp_monitor_thread.start()

    def __call__(self, fn, *a, **kw):
        """A context manager to return our wrapper function.

        Args:
            fn: Function to be executed.
        """
        return self.start(fn, *a, **kw)

    def __enter__(self):
        self.start_hp_monitor_thread()
        return self

    def __exit__(self, *e):
        self.shutdown()

    def shutdown(self):
        """Shutdown :py:class:`HPFaultHandlingThread` and :py:class:`HPMonitorThread`."""
        self.shutdown_hp_fault_handling_thread()
        self.shutdown_hp_monitor_thread()

    def shutdown_hp_fault_handling_thread(self):
        try:
            self.stop_raising.set()
            if self.hp_fault_handling_thread:
                self.hp_fault_handling_thread.shutdown()
                self.hp_fault_handling_thread.join()

            self.hp_fault_handling_thread = None
        except RankShouldRestart:
            self.shutdown_hp_fault_handling_thread()

    def shutdown_hp_monitor_thread(self):
        if self.hp_monitor_thread:
            self.hp_monitor_thread.shutdown()
            self.hp_monitor_thread.join()

        self.hp_monitor_thread = None
