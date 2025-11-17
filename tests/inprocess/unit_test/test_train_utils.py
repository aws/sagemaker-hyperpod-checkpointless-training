import unittest
from unittest.mock import MagicMock, patch

import os
import sys


# Create mock modules to prevent import errors
sys.modules['hyperpod_elastic_agent'] = MagicMock()
sys.modules['hyperpod_elastic_agent.ipc'] = MagicMock()
sys.modules['hyperpod_elastic_agent.ipc.client'] = MagicMock()
sys.modules['hyperpod_elastic_agent.ipc.socket'] = MagicMock()

# Create a mock for InProcessRestartSocketClient
mock_client_class = MagicMock()
mock_client_instance = MagicMock()
mock_client_class.return_value = mock_client_instance
sys.modules['hyperpod_elastic_agent.ipc.client'].InProcessRestartSocketClient = mock_client_class

# Now import the modules that use InProcessRestartSocketClient
from hyperpod_checkpointless_training.inprocess.train_utils import HPAgentK8sAPIFactory, k8s_apis, wait_rank


class TestWaitRank(unittest.TestCase):
    def setUp(self):
        """Setup test environment."""
        self.mock_k8s_apis = MagicMock()
        self.mock_k8s_apis.hyperpod_wait_rank_info = MagicMock()
        self.patcher = patch("hyperpod_checkpointless_training.inprocess.train_utils.k8s_apis", self.mock_k8s_apis)
        self.patcher.start()

        # Add time.sleep patch
        self.sleep_patcher = patch('time.sleep')
        self.mock_sleep = self.sleep_patcher.start()

    def tearDown(self):
        """Restore original k8s_apis."""
        self.patcher.stop()
        self.sleep_patcher.stop()

    @patch.dict('os.environ', {'RANK': '-1', 'WORLD_SIZE': '0'})
    def test_wait_rank_valid_response(self):
        """Test wait_rank with valid response."""
        mock_resp = ("job_rank_info", {"worker_envs": {"TEST_ENV": "test_value"}})
        self.mock_k8s_apis.hyperpod_wait_rank_info.return_value = mock_resp

        wait_rank()

        self.mock_k8s_apis.hyperpod_wait_rank_info.assert_called_once()
        self.assertEqual(os.environ["TEST_ENV"], "test_value")
        self.mock_sleep.assert_not_called()  # Should not sleep for valid response

class TestHPAgentK8sAPIFactory(unittest.TestCase):
    def setUp(self):
        """Setup test environment."""
        # Create mock for k8s_apis
        self.mock_k8s_apis = MagicMock()
        # Patch the k8s_apis in the module
        self.patcher = patch("hyperpod_checkpointless_training.inprocess.train_utils.k8s_apis", self.mock_k8s_apis)
        self.patcher.start()

    def tearDown(self):
        """Restore original patches."""
        self.patcher.stop()

    def test_factory_returns_singleton(self):
        """Test that factory returns the singleton k8s_apis instance."""

        factory = HPAgentK8sAPIFactory()
        result = factory()
        self.assertEqual(result, self.mock_k8s_apis)
