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
