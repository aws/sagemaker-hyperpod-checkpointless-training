import unittest
from unittest.mock import patch, MagicMock, call
import torch
import gc
import megatron

try:
    import transformer_engine
    HAS_TRANSFORMER_ENGINE = True
except ImportError:
    HAS_TRANSFORMER_ENGINE = False


class TestCleanupFunctions(unittest.TestCase):

    @patch('megatron.core.rerun_state_machine.destroy_rerun_state_machine')
    @patch('megatron.core.parallel_state.destroy_model_parallel')
    @patch('megatron.core.parallel_state.destroy_global_memory_buffer')
    @patch('megatron.core.num_microbatches_calculator.destroy_num_microbatches_calculator')
    def test_abort_megatron_new_version(self, mock_calc, mock_buffer, mock_parallel, mock_rerun):
        from hyperpod_checkpointless_training.nemo_plugins.abort import abort_megatron

        abort_megatron()

        mock_calc.assert_called_once()
        mock_buffer.assert_called_once()
        mock_parallel.assert_called_once()
        mock_rerun.assert_called_once()

    @patch('torch.compiler.reset')
    def test_abort_torch_compile(self, mock_reset):
        from hyperpod_checkpointless_training.nemo_plugins.abort import abort_torch_compile

        abort_torch_compile()
        mock_reset.assert_called_once()

    @unittest.skipIf(not HAS_TRANSFORMER_ENGINE, "transformer_engine not installed")
    @patch('transformer_engine.pytorch.fp8.FP8GlobalStateManager.reset')
    @patch('transformer_engine.pytorch.module.base')
    def test_abort_te(self, mock_base, mock_fp8_reset):
        from hyperpod_checkpointless_training.nemo_plugins.abort import abort_te

        abort_te()

        self.assertEqual(mock_base._multi_stream_cublas_workspace, [])
        self.assertEqual(mock_base._dummy_wgrads, {})
        self.assertIsNone(mock_base._cublas_workspace)
        self.assertIsNone(mock_base._ub_communicators)
        self.assertIsNone(mock_base._MIN_STREAM_PRIORITY)
        self.assertEqual(mock_base.layers_atomic_ring_exchange, [])
        mock_fp8_reset.assert_called_once()

    @unittest.skipIf(not HAS_TRANSFORMER_ENGINE, "transformer_engine not installed")
    @patch('transformer_engine.pytorch.module.base')
    def test_abort_te_without_fp8(self, mock_base):
        with patch('transformer_engine.pytorch.fp8', side_effect=ImportError):
            from hyperpod_checkpointless_training.nemo_plugins.abort import abort_te

            abort_te()

        self.assertEqual(mock_base._multi_stream_cublas_workspace, [])
        self.assertEqual(mock_base._dummy_wgrads, {})
        self.assertIsNone(mock_base._cublas_workspace)

    @unittest.skipIf(not HAS_TRANSFORMER_ENGINE, "transformer_engine not installed")
    @patch('importlib.reload')
    def test_reload_megatron_and_te(self, mock_reload):
        from hyperpod_checkpointless_training.nemo_plugins.abort import reload_megatron_and_te

        reload_megatron_and_te()
        self.assertEqual(mock_reload.call_count, 2)

    @patch('gc.get_objects')
    @patch('hyperpod_checkpointless_training.inprocess.tools.memory_tracker.is_meaningful_tensor')
    def test_cleanup_ddp_basic(self, mock_is_meaningful, mock_get_objects):
        mock_trainer = MagicMock()
        mock_model_chunk = MagicMock()
        mock_ddp = MagicMock()
        mock_param = MagicMock()
        mock_buffer = MagicMock()
        mock_bucket = MagicMock()

        # Setup trainer and model structure
        mock_trainer.strategy.megatron_parallel = [mock_model_chunk]
        mock_model_chunk.module = mock_ddp
        mock_model_chunk.buffers = [mock_buffer]
        mock_buffer.buckets = [mock_bucket]
        mock_bucket.params_list = [mock_param]

        # Setup DDP module
        mock_ddp.module.parameters.return_value = [mock_param]
        mock_param.requires_grad = True
        mock_param.main_grad = MagicMock()
        mock_param.main_param = MagicMock()

        # Setup parameter expansion and grad function
        mock_expanded = MagicMock()
        mock_param.expand_as.return_value = mock_expanded
        mock_grad_acc = MagicMock()
        mock_expanded.grad_fn.next_functions = [(mock_grad_acc, None)]
        mock_handle = MagicMock()
        mock_grad_acc.register_hook.return_value = mock_handle
        mock_handle.hooks_dict_ref.return_value = {}
        mock_handle.extra_dict_ref = []

        from hyperpod_checkpointless_training.nemo_plugins.abort import cleanup_ddp

        cleanup_ddp(mock_trainer)

        mock_ddp.disable_forward_pre_hook.assert_called_once_with(param_sync=False)
        self.assertIsNone(mock_ddp.grad_accs)
        self.assertIsNone(mock_param.main_grad)
        self.assertIsNone(mock_param.main_param)

    @patch('gc.get_objects')
    @patch('hyperpod_checkpointless_training.inprocess.tools.memory_tracker.is_meaningful_tensor')
    def test_cleanup_ddp_with_tensor_hook(self, mock_is_meaningful, mock_get_objects):
        mock_trainer = MagicMock()
        mock_model_chunk = MagicMock()
        mock_ddp = MagicMock()
        mock_tensor = MagicMock()

        # Setup basic structure
        mock_trainer.strategy.megatron_parallel = [mock_model_chunk]
        mock_model_chunk.module = mock_ddp
        mock_model_chunk.buffers = []
        mock_ddp.module.parameters.return_value = []

        # Setup tensor cleanup
        mock_get_objects.return_value = [mock_tensor]
        mock_is_meaningful.return_value = True
        mock_handle = MagicMock()
        mock_tensor.register_hook.return_value = mock_handle
        mock_handle.hooks_dict_ref.return_value = {}
        mock_handle.extra_dict_ref = []

        from hyperpod_checkpointless_training.nemo_plugins.abort import cleanup_ddp

        cleanup_ddp(mock_trainer, clean_tensor_hook=True)

        mock_get_objects.assert_called_once()
        mock_is_meaningful.assert_called_once_with(mock_tensor)
        mock_tensor.register_hook.assert_called_once()

    def test_cleanup_ddp_disable_hook_failure(self):
        mock_trainer = MagicMock()
        mock_model_chunk = MagicMock()
        mock_ddp = MagicMock()

        mock_trainer.strategy.megatron_parallel = [mock_model_chunk]
        mock_model_chunk.module = mock_ddp
        mock_model_chunk.buffers = []
        mock_ddp.module.parameters.return_value = []
        mock_ddp.disable_forward_pre_hook.side_effect = Exception("Hook disable failed")

        from hyperpod_checkpointless_training.nemo_plugins.abort import cleanup_ddp

        # Should not raise exception
        cleanup_ddp(mock_trainer)

    def test_cleanup_rope_with_module_attribute(self):
        mock_pl_module = MagicMock()
        mock_rope_module = MagicMock()

        # Setup the mock module hierarchy
        mock_pl_module.module = mock_rope_module
        mock_rope_module.__class__ = megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding
        mock_rope_module.children.return_value = []

        from hyperpod_checkpointless_training.nemo_plugins.abort import cleanup_rope

        cleanup_rope(mock_pl_module)

        mock_rope_module.forward.cache_clear.assert_called_once()

    def test_cleanup_rope_without_module_attribute(self):
        mock_pl_module = MagicMock()
        # Remove module attribute to test direct search
        del mock_pl_module.module
        mock_pl_module.__class__ = megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding
        mock_pl_module.children.return_value = []

        from hyperpod_checkpointless_training.nemo_plugins.abort import cleanup_rope

        cleanup_rope(mock_pl_module)

        mock_pl_module.forward.cache_clear.assert_called_once()

    def test_cleanup_rope_nested_modules(self):
        mock_pl_module = MagicMock()
        mock_child_module = MagicMock()
        mock_rope_module = MagicMock()

        # Setup nested module hierarchy
        mock_pl_module.module = mock_child_module
        mock_child_module.__class__ = MagicMock()  # Not a RoPE module
        mock_child_module.children.return_value = [mock_rope_module]
        mock_rope_module.__class__ = megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding
        mock_rope_module.children.return_value = []

        from hyperpod_checkpointless_training.nemo_plugins.abort import cleanup_rope

        cleanup_rope(mock_pl_module)

        mock_rope_module.forward.cache_clear.assert_called_once()


if __name__ == '__main__':
    unittest.main()
