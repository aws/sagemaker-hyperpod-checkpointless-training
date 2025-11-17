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
