import contextlib
import ctypes
import dataclasses
import logging
import os
import socket
import sys
import threading


def async_raise(tid, exc_type, event=None):
    """Raise exception asynchronously in target thread for fault handling."""
    if event is not None:
        event.wait()

    set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
    set_async_exc.argtypes = (ctypes.c_ulong, ctypes.py_object)
    set_async_exc.restype = ctypes.c_int

    if not sys.is_finalizing():
        res = set_async_exc(tid, exc_type)
    else:
        res = 1

    if res == 0:
        raise RuntimeError
    elif res > 1:
        set_async_exc(tid, None)
        raise RuntimeError


def delayed_async_raise(tid, exc_type):
    """Raise exception in target thread after event signal."""
    event = threading.Event()
    thread = threading.Thread(
        target=async_raise,
        args=(tid, exc_type, event),
        daemon=True,
    )
    thread.start()
    event.set()


@contextlib.contextmanager
def reraise_if_unraisable(exc_type):
    def wrap(fn):
        def wrapped(*args, **kwargs):
            fn(*args, **kwargs)
            reraising_callback(*args, **kwargs)

        return wrapped

    def reraising_callback(unraisable_hook_args):
        if (
            issubclass(unraisable_hook_args.exc_type, exc_type)
            and not sys.is_finalizing()
        ):
            log = logging.getLogger(__name__)
            log.debug(f"sys.unraisablehook raises {exc_type}")
            delayed_async_raise(threading.main_thread().ident, exc_type)

    original_unraisablehook = sys.unraisablehook
    sys.unraisablehook = wrap(sys.unraisablehook)
    yield
    sys.unraisablehook = original_unraisablehook


@dataclasses.dataclass
class HPState:
    rank: int
    world_size: int
    iteration: int

    def __init__(self):
        self.get_distributed_vars()
        self.iteration = 0

    def set_distributed_vars(self):
        raise RuntimeError("HPState state does not allow to set dist vars")

    def get_distributed_vars(self):
        self.rank = int(os.getenv("RANK", -1))
        self.world_size = int(os.getenv("WORLD_SIZE", -1))

    def advance(self):
        self.iteration += 1


def debug_msg(msg, rank=None, seq=None, steps=-1):
    """Format debug message with rank, sequence, step, and host information."""
    rank = int(os.getenv("RANK", -1)) if rank is None else rank
    seq = int(os.getenv("JOB_RESTART_COUNT", -1)) if seq is None else seq
    return f"[RANK:{rank}][SPARE:{os.getenv('SPARE', None)}][SEQ:{seq}][STEPS:{steps}][SPARE:{os.getenv('SPARE', None)}][PID:{os.getpid()}][TID:{threading.get_ident()}][HOST:{socket.gethostname()}] {msg}"


def format_exc(exc: BaseException):
    """Format exception chain for logging."""
    excs = [repr(exc)]
    while (exc := exc.__cause__) is not None:
        excs.append(repr(exc))
    return " <- ".join(excs)


def log_exc(exc, name, rank=-1, seq=-1, steps=-1):
    """Log exception with debug context information."""
    return debug_msg(f"{name}: {format_exc(exc)}", rank=rank, seq=seq, steps=steps)


class AtomicInt:
    """Thread-safe integer wrapper for atomic operations."""
    def __init__(self, i):
        self.i = i
        self.lock = threading.Lock()

    def set(self, x):
        with self.lock:
            self.i = x

    def get(self):
        with self.lock:
            return self.i

    def __str__(self):
        with self.lock:
            return f"{self.i}"

    def __repr__(self):
        return self.__str__()
