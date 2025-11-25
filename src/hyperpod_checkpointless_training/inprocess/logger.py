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

# Standard Library
import logging
import os
import sys
from functools import lru_cache
from typing import Any, TypeVar

from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


@lru_cache
def get_log_level() -> int:
    """Get the log level as configured or the default"""
    log_level = os.environ.get("HPWRAPPER_LOG_LEVEL", default="info").lower()
    if log_level == "off":
        return logging.FATAL + 1
    if log_level == "fatal":
        # fatal added so that log level can take same values for cpp and py
        # fatal in cpp exceptions kills the process
        # so use fatal for that only
        return logging.FATAL
    if log_level == "error":
        return logging.ERROR
    if log_level == "warning":
        return logging.WARNING
    if log_level == "info":
        return logging.INFO
    if log_level in ["debug", "trace"]:
        return logging.DEBUG
    raise ValueError(
        f"Allowed HPWRAPPER_LOG_LEVELS are: info, trace, debug, warning, error, fatal, off. Got: {log_level}"
    )


class PackagePathFilter(logging.Filter):
    def filter(self, record: Any) -> bool:
        #python modules cleanup, no need to log anymore
        if sys is None or sys.path is None or os is None:
            return False 
        pathname = record.pathname
        record.relativepath = None       
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True


class StreamHandler(logging.StreamHandler):
    def handle(self, record):
        rv = self.filter(record)
        if rv:
            try:
                self.acquire()
                self.emit(record)
            finally:
                self.release()
        return rv


def get_logger(name: str = "hp_wrapper") -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "initialized", False):
        return logger  # already configured

    show_time = os.getenv("HPWRAPPER_LOG_SHOW_TIME", "false").lower()
    time = "" if show_time not in ["true", "1"] else "%(asctime)s.%(msecs)03d: "
    log_formatter = logging.Formatter(
        fmt=f"[{time}%(levelname).1s %(relativepath)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stdout_handler = StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    stdout_handler.addFilter(PackagePathFilter())
    logger.handlers = [stdout_handler]  # overwrite

    level = get_log_level()
    if level <= logging.FATAL:
        logger.setLevel(level)
    else:
        logger.disabled = True
    logger.propagate = False
    setattr(logger, "initialized", True)
    return logger
