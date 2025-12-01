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

from hyperpod_checkpointless_training.inprocess.abort import AbortTransformerEngine
from hyperpod_checkpointless_training.inprocess.utils import HPState


class TestAbortTransformerEngine(unittest.TestCase):
    """Test AbortTransformerEngine functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.abort_te = AbortTransformerEngine()
        self.state = HPState()

    def test_call_with_transformer_engine_available(self):
        """Test __call__ when transformer_engine is available"""
        mock_te = MagicMock()
        mock_te_fp8 = MagicMock()

        def mock_import(name, *args, **kwargs):
            if name == 'transformer_engine.pytorch':
                return mock_te
            elif name == 'transformer_engine.pytorch.fp8':
                return mock_te_fp8
            raise ImportError(f"No module named '{name}'")

        with patch('builtins.__import__', side_effect=mock_import):
            result = self.abort_te(self.state)
            self.assertEqual(result, self.state)

    def test_call_without_transformer_engine(self):
        """Test __call__ when transformer_engine is not available"""
        with patch('builtins.__import__', side_effect=ImportError):
            result = self.abort_te(self.state)
            self.assertEqual(result, self.state)


if __name__ == '__main__':
    unittest.main()
