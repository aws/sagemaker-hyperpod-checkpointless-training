import inspect
import logging


class HyperPodCheckpointlessValidationError(Exception):
    r"""
    HyperPod Checkpointless Validation Error.
    """

    pass


class RankShouldRestart(BaseException):
    r"""
    :py:exc:`BaseException` asynchronously raised in the main thread to
    interrupt the execution of the function wrapped with
    :py:class:`inprocess.Wrapper`.
    """

    def __del__(self):
        log = logging.getLogger(__name__)
        if log.isEnabledFor(logging.DEBUG):
            from . import wrap

            stack = inspect.stack(context=0)

            if len(stack) > 1 and stack[1].filename != wrap.__file__:
                locations = [
                    f"{info.frame.f_code.co_filename}:{info.frame.f_lineno}"
                    for info in stack[1:]
                ]
                traceback = " <- ".join(locations)
                log.debug(f"{type(self).__name__} suppressed at {traceback}")
            del stack


class RestartError(Exception):
    r"""
    Base :py:exc:`Exception` for exceptions raised by
    :py:class:`inprocess.Wrapper`.
    """

    pass


class RestartAbort(BaseException):
    r"""
    A terminal Python :py:exc:`BaseException` indicating that the
    :py:class:`inprocess.Wrapper` should be aborted immediately, bypassing any
    further restart attempts.
    """

    pass


class HealthCheckError(RestartError):
    r"""
    :py:exc:`RestartError` exception to indicate that
    :py:class:`inprocess.health_check.HealthCheck` raised errors, and execution
    shouldn't be restarted on this distributed rank.
    """

    pass


class InternalError(RestartError):
    r"""
    :py:class:`inprocess.Wrapper` internal error.
    """

    pass


class TimeoutError(RestartError):
    r"""
    :py:class:`inprocess.Wrapper` timeout error.
    """

    pass
