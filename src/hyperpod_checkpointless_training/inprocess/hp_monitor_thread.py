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

import queue
import threading

from .elastic.hp_agent_event import HPAgentResponse
from .logger import get_logger
from .utils import AtomicInt, HPState, debug_msg


class HPMonitorThread(threading.Thread):
    """The :py:class:`HPMonitorThread` is for monitoring signal from
    AWS HyperPod. For example, when HyperPod cluster send a FAILURE signal,
    The :py:class:`HPMonitorThread` will notify :py:class:`HPFaultHandlingThread`
    to handle restart.

    Args:
        hp_api: Creating a HyperPod API to interact with AWS HyperPod.
        failure: A threading.Event to notify :py:class`HPFaultHandlingThread` that RCB is failure.
    """

    def __init__(
        self,
        *a,
        hp_api=None,
        failure: threading.Event = None,
        seq: AtomicInt = None,
        **kw,
    ):
        assert hp_api, "HyperPod api is empty"
        assert failure, "Failure threading.Event is empty"

        self.state = HPState()
        self.hp_api = hp_api
        self.failure = failure
        self.state.get_distributed_vars()
        self.should_stop = threading.Event()
        self.logger = get_logger()
        self.seq = seq
        name = f"{type(self).__name__}-{self.state.rank}"
        super().__init__(*a, name=name, **kw)

    def handle_tuple(self, signal):
        hdr, body = signal
        if hdr == HPAgentResponse.FAILURE:
            return self.handle_failure_signal(body)
        self.logger.warning(
            debug_msg(
                f"HPMonitorThread unrecognize signal: {signal}",
                rank=self.state.rank,
                seq=self.seq,
            )
        )

    def handle_response(self, signal):
        if signal == HPAgentResponse.OK:
            return
        if signal == HPAgentResponse.INVALID:
            self.logger.warning(
                debug_msg(
                    "HPMonitorThread receive invalid response",
                    rank=self.state.rank,
                    seq=self.seq,
                )
            )
            return
        if signal == HPAgentResponse.UNKNOWN:
            self.logger.warning(
                debug_msg(
                    "HPMonitorThread receive unknown response",
                    rank=self.state.rank,
                    seq=self.seq,
                )
            )
            return
        self.logger.warning(
            debug_msg(
                f"HPMonitorThread unrecognize signal: {signal}",
                rank=self.state.rank,
                seq=self.seq,
            )
        )

    def handle_signal(self, signal):
        """Handle signale from AWS HyperPod.

        Args:
            signal: A signal from AWS HyperPod.
        """
        if isinstance(signal, tuple):
            return self.handle_tuple(signal)
        elif isinstance(signal, HPAgentResponse):
            return self.handle_response(signal)
        self.logger.warning(
            debug_msg(
                f"HPMonitorThread unrecognize response: {signal}",
                rank=self.state.rank,
                seq=self.seq,
            )
        )

    def handle_failure_signal(self, seq):
        """Handle failure signal from AWS HyperPod.

        The implementation is only to set failure event to be true.
        """
        self.logger.debug(
            debug_msg(
                f"HPMonitorThread handle_failure_signal with seq {seq}",
                rank=self.state.rank,
                seq=self.seq,
            )
        )
        if seq >= self.seq.get():
            self.failure.set()

    def hyperpod_wait(self, timeout=1):
        """This is a listener to wait for AWS HyperPod signals."""
        while not self.should_stop.is_set():
            try:
                yield self.hp_api.hyperpod_wait(timeout=timeout)
            except queue.Empty:
                pass

    def run(self):
        """The entrypoint of this thread. The thread will continuously listen
        signals from AWS HyperPod.
        """
        self.logger.debug(
            debug_msg("HPMonitorThread is started", rank=self.state.rank, seq=self.seq)
        )
        for resp in self.hyperpod_wait():
            self.handle_signal(resp)

    def shutdown(self):
        """Shutdown the thread loop."""
        self.should_stop.set()
