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

import os
import signal
import sys
import threading
import time
from typing import Callable, Optional

from .exception import RankShouldRestart
from .logger import get_logger
from .utils import AtomicInt, HPState, async_raise, debug_msg, log_exc
from .compose import Compose
from .abort import HPDataLoaderAbort


def async_abort_main_thread(abort_signal=None, msg=None):
    if abort_signal is None:
        if msg is not None:
            DynamicRankShouldRestart = type(
                "RankShouldRestart",
                (RankShouldRestart,),
                {
                    "__init__": lambda self: super(
                        DynamicRankShouldRestart, self
                    ).__init__(msg)
                },
            )
            exc_type = DynamicRankShouldRestart
        else:
            exc_type = RankShouldRestart

        async_raise(threading.main_thread().ident, exc_type)
    else:
        os.kill(os.getpid(), abort_signal)


class _TimeoutRLock:
    def __init__(self, lock: threading.RLock, timeout: float):
        self.lock = lock
        self.acquired = False
        self.timeout = timeout

    def __enter__(self):
        self.acquired = self.lock.acquire(blocking=True, timeout=self.timeout)
        return self.acquired

    def __exit__(self, *e):
        if self.acquired:
            self.lock.release()
            self.acquired = False
        return False


class HPFaultHandlingThread(threading.Thread):
    """The :py:class:`HPFaultHandlingThread` is in charge of raising
    `RankShouldRestart` exception. The class relies on
    :py:class:`PyThreadState_SetAsyncExc` to inject `RankShouldRestart
    into main thread. Note that main thread will receive `RankShouldRestart`
    once main thread calls `HPFaultHandlingThread.join`.

    Args:
        failure: A threading.Event to notify :py:class`HPFaultHandlingThread`
            that RCB is failure.
        stop_raising: A threading.Event to notify :py:class`HPFaultHandlingThread`
             don't inject :py:class:`RankShouldRestart` multiple times.
        atomic_lock: An RLock to prevent interruption in RCB critical section.
    """

    def __init__(
        self,
        *a,
        abort=None,
        failure: threading.Event = None,
        stop_raising: threading.Event = None,
        atomic_lock: threading.RLock = None,
        abort_timeout: float = None,
        soft_timeout: float = 30,
        hard_timeout: float = 30,
        failure_check_interval: float = 1,
        async_raise_before_abort: bool = True,
        early_abort_communicator: bool = False,
        seq: AtomicInt = None,
        **kw,
    ):
        state = HPState()
        state.get_distributed_vars()
        assert state.rank > -1, "Rank information is not available"
        assert failure, "Failure threading.Event is empty"
        assert stop_raising, "StopRaising threading.Event is empty"
        assert atomic_lock, "Atomic lock is empty"

        self.state = state
        self.abort = abort
        self.failure = failure
        self.stop_raising = stop_raising
        self.atomic_lock = atomic_lock
        self.should_stop = threading.Event()
        self.logger = get_logger()
        self.soft_timeout = soft_timeout
        self.hard_timeout = hard_timeout
        self.failure_check_interval = failure_check_interval
        self.abort_timeout = abort_timeout
        self.async_raise_before_abort = async_raise_before_abort
        self.early_abort_communicator = early_abort_communicator
        self.seq = seq
        name = f"{type(self).__name__}-{self.state.rank}"
        super().__init__(*a, name=name, **kw)

    def run(self):
        """When the thread detects a failure event to be true, it will call
        async_abort_main_thread which is using `PyThreadState_SetAsyncExc` to
        inject :py:class:`RankShouldRestart` to main thread. Because the thread
        should `join` on the main thread, we should break the loop; otherwise,
        the main thread will hang.

        The :py:class:`HPFaultHandlingThread` undergoes a tear down and respawn
        cycle after each fault detection. This design forces proper exception
        handling, ensuring the main thread processes the :py:class:`RankShouldRestart`
        signal and executes its restart handler. Additionally, recreating the
        :py:class:`HPFaultHandlingThread` provides a clean slate for handling
        multiple failures that occur during the same training iteration.

        Restart Workflow

          1. RCB raises an :py:class:`Exception`

                                                               RCB
                          HPFaultHandlingThread         Raise an Exception
                                    |                           |
                                    |                           | | Notify other ranks
                                    |       failure.set()       | |
                                    |<--------------------------| V
                                    |                           | |
              failure.wait() pass | |                           | | Shutdown/join HPFaultHandlingThread
                                  | |                           | V
                                  V |                           |
                           abot() | |                           |
                                  | |  async_abort_main_thread  |
                                  V |-------------------------->| |
                                    |                           | | Handle RankShouldRestart
                                    |                           | V
            stop the monitor loop | |                           | |
                                  | |                           | | Restart done
                                  V |                           | V
                                    |                           | |
                                    |                           | | Wait hyperpod_barrier
                                    |                           | V
                                    |                           | |
                                    |                           | | New HPFaultHandlingThread
                                    V                           V V

         2. HPFaultHandlingThread receives a fault signal

         HPMonitorThread         HPFaultHandlingThread                 RCB
                |                           |                           |
                |                           |                           |
                |                           |                           |
                |       failure.set()       |                           |
                |-------------------------->|                           |
                |                           |                           |
                |     failure.wait() pass | |                           |
                |                         | |                           |
                |                         V |                           |
                |                 abort() | |                           |
                |                         | |  async_abort_main_thread  |
                |                         V |-------------------------->| |
                |                           |                           | | Handle RankShouldRestart
                |                           |                           | V
                |   stop the monitor loop | |                           | |
                |                         | |                           | | Restart done
                |                         V |                           | V
                |                           |                           | |
                |                           |                           | | Wait hyperpod_barrier
                |                           |                           | V
                |                           |                           | |
                |                           |                           | | New HPFaultHandlingThread
                V                           V                           V V
        """
        while not self.should_stop.is_set():
            # In our previous design,we want to rely on infrastrcuture to give us haning information.
            # However, this solution may be too slow. Therefore, we want to
            # add process_watchdog to handle hanging check.
            if self.failure.wait(timeout=self.failure_check_interval):
                self.logger.debug(
                    debug_msg("failure set", rank=self.state.rank, seq=self.seq)
                )
                self.handle_failure()
                break

    def handle_failure(self):
        """Handle failure event from AWS HyperPod signal.

        Note that this function will

        1. Terminate all process groups
        2. Inject `RankShouldRestart` to the main thread

        The implementation idea here is we try to acquire lock until timeout.
        We set a timeout here to prevent from deadlock once main thread holds
        lock without realeasing it. For example, if main thread is hanging for
        some reasons after acquiring lock, fault handling thread won't be able
        to abort the main thread because it is unable to get the lock.

        """
        with _TimeoutRLock(self.atomic_lock, self.soft_timeout):
            return self.try_abort()

    def try_abort(self):
        self.logger.info(debug_msg("Running abort", rank=self.state.rank, seq=self.seq))
        if self.early_abort_communicator:
            # place data loader abort after communicator abort
            self.abort = self.reorder_aborts(HPDataLoaderAbort, 0)

        if self.async_raise_before_abort:
            self.do_main_abort()
            self.do_abort()
            self.do_post_comm_abort_cleanup()
        else:
            self.do_abort()
            self.do_post_comm_abort_cleanup()
            self.do_main_abort()
        self.logger.debug(
            debug_msg(
                f"abort finished, stop_raising={self.stop_raising.is_set()}",
                rank=self.state.rank,
                seq=self.seq,
            )
        )
        if not self.stop_raising.wait(self.soft_timeout):
            self.do_spin_main_abort()

    def do_post_comm_abort_cleanup(self):
        if not self.abort or not hasattr(self.abort, 'instances'):
            return

        for instance in self.abort.instances:
            if hasattr(instance, 'post_comm_abort_cleanup'):
                instance.post_comm_abort_cleanup()

    def do_abort(self):
        if not self.abort:
            return

        try:
            self.logger.debug(
                debug_msg(
                    "running all aborts",
                    rank=self.state.rank,
                    seq=self.seq,
                )
            )
            self.abort(None, timeout=self.abort_timeout)
        except Exception as abort_ex:
            self.logger.critical(
                log_exc(abort_ex, "abort_ex", rank=self.state.rank, seq=self.seq)
            )

    def do_main_abort(self):
        self.async_abort_main_thread(
            msg=debug_msg("Restart", rank=self.state.rank, seq=self.seq)
        )

    def async_abort_main_thread(self, *args, **kwargs):
        async_abort_main_thread(*args, **kwargs)

    def do_spin_main_abort(self):
        """Spin run async_abort_main_thread until main thread receives :py:class:`RankShouldRestart`.

        Spin run async_abort_main_thread is for haning main thread. From the
        original Nvidia implementation, process_watchdog is for detecting
        whether current main thread is hanging. Once, process_watchdog timeout,
        the monitor thread will run :py:func:`async_abort_main_thread` until
        the main thread receives :py:class:`RankShouldRestart` exception.
        """
        self.logger.warning(
            debug_msg(
                "spin async_abort_main_thread",
                rank=self.state.rank,
                seq=self.seq,
            )
        )
        s = time.perf_counter()
        # Interval for Python interpreter thread switching
        switch_interval = 2 * sys.getswitchinterval()
        while not self.should_stop.is_set() and not self.stop_raising.is_set():
            e = time.perf_counter()
            d = e - s
            if d < self.hard_timeout:
                self.async_abort_main_thread()
            else:
                self.kill()
            # Adding a sleep so we do not throw exceptions too frequent
            time.sleep(switch_interval)

    def kill(self):
        self.logger.critical(
            debug_msg(
                "process does not have response",
                rank=self.state.rank,
                seq=self.seq,
            )
        )
        os.kill(os.getpid(), signal.SIGKILL)

    def shutdown(self):
        self.should_stop.set()

    def reorder_aborts(self, instance_type, target_index):
        if not self.abort or not hasattr(self.abort, 'instances'):
            return self.abort
        current_instances = list(self.abort.instances)

        # Validate target_index
        if not (0 <= target_index < len(current_instances)):
            raise ValueError(f"Target index {target_index} is out of range. Must be between 0 and {len(current_instances) - 1}")

        target_instance = None
        for i, instance in enumerate(current_instances):
            if isinstance(instance, instance_type):
                target_instance = current_instances.pop(i)
                break

        if target_instance:
            current_instances.insert(target_index, target_instance)

        return Compose(*current_instances)
