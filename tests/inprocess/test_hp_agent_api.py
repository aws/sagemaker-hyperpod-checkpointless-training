import os
import unittest
from unittest.mock import Mock

from hyperpod_elastic_agent.ipc import (
    HyperPodIPCException,
    InProcessRestartSocketClient,
    RestartMode,
)

from hyperpod_checkpointless_training.inprocess.elastic.hp_agent_api import (
    HPAgentEvent,
    HPAgentK8sAPI,
    HPAgentResponse,
)


class TestHPAgentK8sAPI(unittest.TestCase):
    def setUp(self):
        """Set up test environment with mock client and required env variables"""
        self.mock_client = Mock(spec=InProcessRestartSocketClient)
        self.api = HPAgentK8sAPI(self.mock_client)
        # Set required environment variables
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "4"

    def test_validate_env(self):
        """Test environment validation with valid inputs"""
        env = {"RANK": "0", "WORLD_SIZE": "4"}
        self.api.validate_env(env, 1)  # Should not raise any assertion

    def test_hyperpod_barrier(self):
        """Test barrier functionality with successful job start"""
        mock_response = (
            "job_start",
            {"worker_envs": {"JOB_RESTART_COUNT": "1", "RANK": "0", "WORLD_SIZE": "4"}},
        )
        self.mock_client.hyperpod_barrier.return_value = mock_response
        result = self.api.hyperpod_barrier(0)
        self.assertEqual(result, 1)

    def test_hyperpod_wait_ok(self):
        """Test wait functionality when IPC exception occurs"""
        self.mock_client.hyperpod_wait_fault.side_effect = HyperPodIPCException()
        result = self.api.hyperpod_wait()
        self.assertEqual(result, HPAgentResponse.OK)

    def test_hyperpod_wait_failure(self):
        """Test wait functionality with job fault response"""
        self.mock_client.hyperpod_wait_fault.return_value = ("job_fault", {})
        result = self.api.hyperpod_wait()
        self.assertEqual(result, (HPAgentResponse.FAILURE, 0))

    def test_hyperpod_send(self):
        """Test send functionality with default restart mode"""
        self.api.hyperpod_send(HPAgentEvent.BARRIER, 0)
        self.mock_client.hyperpod_send_fault.assert_called_once_with(
            0, RestartMode.IN_PROCESS_RESTART
        )

    def test_hyperpod_wait_unknown_action(self):
        """Test wait functionality with unknown response action"""
        self.mock_client.hyperpod_wait_fault.return_value = ("unknown_action", {})
        result = self.api.hyperpod_wait()
        self.assertEqual(result, HPAgentResponse.UNKNOWN)

    def test_hyperpod_wait_with_timeout(self):
        """Test wait functionality with timeout parameter"""
        timeout = 30
        self.mock_client.hyperpod_wait_fault.side_effect = HyperPodIPCException()
        result = self.api.hyperpod_wait(timeout=timeout)
        self.mock_client.hyperpod_wait_fault.assert_called_once_with(timeout=timeout)
        self.assertEqual(result, HPAgentResponse.OK)

    def test_hyperpod_send_with_plr_restart(self):
        """Test send functionality with process level restart"""
        self.api.hyperpod_send(HPAgentEvent.BARRIER, 0, plr_restart=True)
        self.mock_client.hyperpod_send_fault.assert_called_once_with(
            0, RestartMode.PROCESS_LEVEL_RESTART
        )

    def test_hyperpod_send_with_in_process_restart(self):
        """Test send functionality with in-process restart"""
        self.api.hyperpod_send(HPAgentEvent.BARRIER, 0, plr_restart=False)
        self.mock_client.hyperpod_send_fault.assert_called_once_with(
            0, RestartMode.IN_PROCESS_RESTART
        )

    def test_hyperpod_wait_none_response(self):
        """Test wait functionality when response is None"""
        self.mock_client.hyperpod_wait_fault.return_value = None
        with self.assertRaises(AssertionError):
            self.api.hyperpod_wait()

    def test_hyperpod_wait_invalid_response_format(self):
        """Test wait functionality with invalid response format"""
        self.mock_client.hyperpod_wait_fault.return_value = "invalid_response"
        with self.assertRaises(AssertionError):
            self.api.hyperpod_wait()

    def test_hyperpod_barrier_invalid_response(self):
        """Test barrier functionality with invalid response"""
        self.mock_client.hyperpod_barrier.return_value = ("invalid_action", {})
        with self.assertRaises(AssertionError):
            self.api.hyperpod_barrier(0)

    def test_hyperpod_barrier_missing_worker_envs(self):
        """Test barrier functionality when worker_envs is missing"""
        self.mock_client.hyperpod_barrier.return_value = ("job_start", {})
        with self.assertRaises(AssertionError):
            self.api.hyperpod_barrier(0)

    def test_hyperpod_barrier_missing_restart_count(self):
        """Test barrier functionality when JOB_RESTART_COUNT is missing"""
        self.mock_client.hyperpod_barrier.return_value = (
            "job_start",
            {"worker_envs": {}},
        )
        with self.assertRaises(AssertionError):
            self.api.hyperpod_barrier(0)

    def test_hyperpod_notify_labels(self):
        """Test notify labels functionality"""
        labels = {"key": "value"}
        self.api.hyperpod_notify_labels(labels)
        self.mock_client.hyperpod_notify_labels.assert_called_once_with(labels)

    def test_hyperpod_wait_rank_info(self):
        """Test wait rank info functionality"""
        expected_result = {"rank": 0, "world_size": 4}
        self.mock_client.hyperpod_wait_rank_info.return_value = expected_result
        result = self.api.hyperpod_wait_rank_info()
        self.assertEqual(result, expected_result)
        self.mock_client.hyperpod_wait_rank_info.assert_called_once()

    def test_context_manager(self):
        """Test the context manager functionality"""
        with self.api as api:
            self.assertIsInstance(api, HPAgentK8sAPI)


if __name__ == "__main__":
    unittest.main()
