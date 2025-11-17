import threading
import time

from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg

hp_logger = get_logger()


class ParameterUpdateLock:
    """
    A multi-thread safe singleton context manager that coordinates parameter updates with the fault handling system.

    This class serves as a critical synchronization primitive in Checkpointless Training's fault tolerance architecture,
    ensuring that parameter updates can be safely interrupted and resumed during in-process restarts.
    It prevents race conditions between the main training thread and fault handling threads.

    CORE RESPONSIBILITIES:

    1. **Critical Section Protection**: Protects parameter update operations from being interrupted
       by fault handling threads during critical moments (e.g. parameter updates).

    2. **Recovery State Management**: Tracks the health and completion status of parameter updates
       to enable intelligent recovery decisions (checkpointless vs checkpoint-based recovery).

    3. **Training Step Lifecycle Tracking**: Distinguishes between first step (requires PLR) and
       subsequent steps (can use in-process restart) for optimal recovery strategy.

    STATE MANAGEMENT:

    - **param_update_lock** (threading.RLock): The actual lock that protects critical sections.
    - **param_update_completed** (bool): Tracks whether the current parameter update cycle completed successfully.
    - **first_step** (bool): Indicates if this is the first training step after initialization/restart.

    Usage:
        # In training loop (main thread only)
        with ParameterUpdateLock() as lock:
            # Protected parameter update operations
            optimizer.step()
            model.zero_grad()
            # Lock automatically released, param_update_completed set appropriately

        # In fault handling (any thread)
        lock = ParameterUpdateLock()
        if lock.is_healthy():
            # Safe to attempt checkpointless recovery
            pass
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """
        Initialize the ParameterUpdateLock singleton instance.

        Sets up the internal locks and state variables:
        - param_update_lock: RLock for protecting critical sections
        - _attr_lock: Lock for protecting internal state access
        - _acquired: Tracks if the main lock is currently held
        - _param_update_completed: Initially True (no update in progress)
        - _first_step: Initially True (haven't started first training step)
        """
        self.param_update_lock = threading.RLock()
        self._attr_lock = threading.Lock()
        self._acquired = False
        self.start_time = None
        self._param_update_completed = True
        self._first_step = True

    @property
    def param_update_completed(self):
        with self._attr_lock:
            return self._param_update_completed

    @param_update_completed.setter
    def param_update_completed(self, value: bool):
        with self._attr_lock:
            self._param_update_completed = value

    @property
    def first_step(self):
        with self._attr_lock:
            return self._first_step

    @first_step.setter
    def first_step(self, value: bool):
        with self._attr_lock:
            self._first_step = value

    def is_healthy(self):
        """
        Determine if the training process is in a healthy state for checkpointless recovery.

        A healthy state means:
        1. Not on the first step (first_step=False) - we have started at least one training step
        2. Last parameter update completed successfully (param_update_completed=True)

        This is used by CheckpointManager to decide whether checkpointless recovery is feasible.
        If unhealthy, the system will fall back to checkpoint-based recovery.

        Returns:
            bool: True if the process is healthy and can participate in checkpointless recovery
        """
        with self._attr_lock:
            return not self._first_step and self._param_update_completed

    @property
    def acquired(self):
        with self._attr_lock:
            return self._acquired

    def __enter__(self):
        """
        Enter the critical section for parameter updates.

        This method:
        1. Acquires the parameter update lock to prevent fault handling interruption
        2. Marks parameter update as in progress (param_update_completed=False)
        3. Records start time for performance monitoring

        CRITICAL: This method should only be called in two cases:
        1. Main thread executing optimizer.step.
        2. Fault handling thread saving checkpoint state in checkpoint abort.
        """
        self.start_time = time.time()

        hp_logger.debug(debug_msg("Acquiring parameter update lock"))

        # Acquire the lock
        self.acquire()
        self._acquired = True

        # Mark parameter update as in progress
        self.param_update_completed = False

        hp_logger.debug(debug_msg("Parameter update lock acquired"))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the critical section and update completion status.

        This method:
        1. Updates param_update_completed based on whether an exception occurred
        2. Always releases the parameter update lock (even if exceptions occurred)
        3. Logs the operation duration and outcome

        The completion status is crucial for recovery decisions:
        - If no exception: param_update_completed=True (safe for checkpointless recovery)
        - If exception: param_update_completed=False (may need checkpoint-based recovery)

        CRITICAL: This method must only be called from main thread during optimizer step
        or fault handling thread when saving checkpoint, matching __enter__.

        Args:
            exc_type: The exception type, if an exception was raised, otherwise None
            exc_val: The exception value, if an exception was raised, otherwise None
            exc_tb: The traceback, if an exception was raised, otherwise None

        Returns:
            False: Always returns False to ensure exceptions are propagated up the stack
        """
        try:
            if exc_type is None:
                # No exception occurred, mark parameter update as completed
                hp_logger.debug(debug_msg("Parameter update completed successfully"))
                self.param_update_completed = True
            else:
                # Exception occurred, parameter update did not complete
                hp_logger.warning(
                    debug_msg(
                        f"Parameter update failed with exception {exc_type.__name__}: {exc_val}"
                    )
                )
                # Explicitly set param_update_completed to False
                self.param_update_completed = False
        finally:
            # Always release the lock if we acquired it
            if self._acquired:
                duration = time.time() - self.start_time
                hp_logger.debug(
                    debug_msg(f"Releasing parameter update lock after {duration:.2f}s")
                )
                self.release()
                self._acquired = False

        # Don't suppress exceptions
        return False

    def force_release(self):
        """
        Forcefully release the parameter update lock during fault handling.

        This method is called by HPCallWrapper.restart() during the restart cleanup process
        to ensure the lock is released even if the main thread was interrupted during
        a parameter update operation.

        Unlike the normal release() method, this continues releasing until the lock
        is fully released, handling cases where the lock was acquired multiple times
        (due to RLock re-entrance) or in inconsistent states.

        This is essential for recovery because:
        1. Ensures no deadlocks during restart process
        2. Allows subsequent training iterations to acquire the lock normally
        3. Handles edge cases where main thread was interrupted mid-operation

        Note: This method can be called from any thread (typically fault handling thread)
        """
        while True:
            try:
                self.param_update_lock.release()
            except RuntimeError:
                self._acquired = False
                break

    # Below APIs are to match the threading.Rlock APIs
    def acquire(self, *args, **kwargs):
        """
        Acquire the underlying lock
        """
        return self.param_update_lock.acquire(*args, **kwargs)

    def release(self, *args, **kwargs):
        """
        Release the underlying parameter update lock directly.

        This provides direct access to the RLock for cases where the context manager
        interface is not used. Should be paired with acquire() calls.

        Args:
            *args, **kwargs: Passed directly to threading.RLock.release()

        Raises:
            RuntimeError: If the lock is not currently held by the calling thread
        """
        return self.param_update_lock.release(*args, **kwargs)
