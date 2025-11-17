import os
import queue
import threading
import unittest
from unittest.mock import Mock, patch

from hyperpod_checkpointless_training.inprocess.elastic.hp_agent_event import HPAgentResponse
from hyperpod_checkpointless_training.inprocess.hp_monitor_thread import HPMonitorThread
from hyperpod_checkpointless_training.inprocess.utils import AtomicInt


class TestHPMonitorThread(unittest.TestCase):
    def setUp(self):
        os.environ["HPWRAPPER_LOG_LEVEL"] = "debug"  # test with full logging
        # Mock dependencies
        self.mock_hp_api = Mock()
        self.failure_event = threading.Event()
        self.seq = AtomicInt(0)

        # Create HPMonitorThread instance
        self.monitor_thread = HPMonitorThread(
            hp_api=self.mock_hp_api, failure=self.failure_event, seq=self.seq
        )

    def tearDown(self):
        patch.stopall()
        if self.monitor_thread.is_alive():
            self.monitor_thread.shutdown()
            self.monitor_thread.join()

    def test_initialization(self):
        """Test proper initialization of HPMonitorThread"""
        self.assertEqual(self.monitor_thread.hp_api, self.mock_hp_api)
        self.assertEqual(self.monitor_thread.failure, self.failure_event)
        self.assertEqual(self.monitor_thread.seq, self.seq)
        self.assertIsInstance(self.monitor_thread.should_stop, threading.Event)
        self.assertFalse(self.monitor_thread.should_stop.is_set())

    def test_initialization_without_required_args(self):
        """Test initialization fails without required arguments"""
        with self.assertRaises(AssertionError):
            HPMonitorThread(failure=self.failure_event)

        with self.assertRaises(AssertionError):
            HPMonitorThread(hp_api=self.mock_hp_api)

    def test_handle_failure_signal(self):
        """Test handling of failure signal"""
        # Test with higher sequence number
        self.seq.set(2)
        self.monitor_thread.handle_failure_signal(3)
        self.assertTrue(self.failure_event.is_set())

        # Reset failure event
        self.failure_event.clear()

        # Test with lower sequence number
        self.monitor_thread.handle_failure_signal(1)
        self.assertFalse(self.failure_event.is_set())

    def test_handle_tuple_signal(self):
        """Test handling of tuple signals"""
        # Test failure signal
        self.seq.set(2)
        self.monitor_thread.handle_tuple((HPAgentResponse.FAILURE, 1))
        self.assertFalse(self.failure_event.is_set())  # seq is lower than current

        # Test unrecognized signal
        with patch.object(self.monitor_thread.logger, "warning") as mock_warning:
            self.monitor_thread.handle_tuple(("UNKNOWN", None))
            mock_warning.assert_called_once()

    def test_handle_response(self):
        """Test handling of HPAgentResponse signals"""
        # Test OK response
        self.monitor_thread.handle_response(HPAgentResponse.OK)
        self.assertFalse(self.failure_event.is_set())

        # Test INVALID response
        with patch.object(self.monitor_thread.logger, "warning") as mock_warning:
            self.monitor_thread.handle_response(HPAgentResponse.INVALID)
            mock_warning.assert_called_once()

        # Test UNKNOWN response
        with patch.object(self.monitor_thread.logger, "warning") as mock_warning:
            self.monitor_thread.handle_response(HPAgentResponse.UNKNOWN)
            mock_warning.assert_called_once()

    def test_handle_signal(self):
        """Test handle_signal method with different types of signals"""
        # Test tuple signal
        with patch.object(self.monitor_thread, "handle_tuple") as mock_handle_tuple:
            self.monitor_thread.handle_signal((HPAgentResponse.FAILURE, 1))
            mock_handle_tuple.assert_called_once_with((HPAgentResponse.FAILURE, 1))

        # Test HPAgentResponse signal
        with patch.object(
            self.monitor_thread, "handle_response"
        ) as mock_handle_response:
            self.monitor_thread.handle_signal(HPAgentResponse.OK)
            mock_handle_response.assert_called_once_with(HPAgentResponse.OK)

        # Test unrecognized signal
        with patch.object(self.monitor_thread.logger, "warning") as mock_warning:
            self.monitor_thread.handle_signal("unknown")
            mock_warning.assert_called_once()

    def test_hyperpod_wait(self):
        """Test hyperpod_wait generator"""
        # Mock hp_api.hyperpod_wait to return a value and then raise queue.Empty
        self.mock_hp_api.hyperpod_wait.side_effect = [
            HPAgentResponse.OK,
            queue.Empty(),
        ]

        # Get first two values from generator
        wait_gen = self.monitor_thread.hyperpod_wait(timeout=0.1)
        results = []
        for _ in range(2):
            try:
                results.append(next(wait_gen))
            except RuntimeError:
                break

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], HPAgentResponse.OK)

    def test_shutdown(self):
        """Test shutdown functionality"""
        self.monitor_thread.shutdown()
        self.assertTrue(self.monitor_thread.should_stop.is_set())

    def test_run(self):
        """Test run method"""
        # Mock hyperpod_wait to return some values
        mock_responses = [HPAgentResponse.OK, (HPAgentResponse.FAILURE, 10)]

        def mock_generator():
            for response in mock_responses:
                yield response
            self.monitor_thread.shutdown()

        with patch.object(
            self.monitor_thread, "hyperpod_wait", return_value=mock_generator()
        ):
            self.monitor_thread.start()
            self.monitor_thread.join()

        self.assertTrue(self.monitor_thread.should_stop.is_set())


if __name__ == "__main__":
    unittest.main()
