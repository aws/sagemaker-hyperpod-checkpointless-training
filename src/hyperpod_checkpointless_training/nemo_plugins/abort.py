import importlib

import megatron
import torch
import gc

from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg
from hyperpod_checkpointless_training.inprocess.tools import memory_tracker

hp_logger = get_logger()


def abort_megatron():
    """Destroy Megatron global state for clean reinitialization after failure."""
    try:
        from megatron.training.training import destroy_global_state

        destroy_global_state()
    except ImportError:
        from megatron.core import rerun_state_machine
        from megatron.core.num_microbatches_calculator import (
            destroy_num_microbatches_calculator,
        )
        from megatron.core.parallel_state import (
            destroy_global_memory_buffer,
            destroy_model_parallel,
        )

        destroy_num_microbatches_calculator()
        destroy_global_memory_buffer()
        destroy_model_parallel()
        rerun_state_machine.destroy_rerun_state_machine()


def abort_torch_compile():
    torch.compiler.reset()


def abort_te():
    """Clear Transformer Engine global state including workspaces and FP8 state."""
    from transformer_engine.pytorch.module import base

    base._multi_stream_cublas_workspace = []
    base._dummy_wgrads = {}
    base._cublas_workspace = None
    base._ub_communicators = None
    base._MIN_STREAM_PRIORITY, _MAX_STREAM_PRIORITY = None, None
    base.layers_atomic_ring_exchange = []

    try:
        import transformer_engine.pytorch.fp8 as te_fp8
    except Exception:
        pass
    else:
        # Clear a class-member containing a process group
        te_fp8.FP8GlobalStateManager.reset()


def reload_megatron_and_te():
    """Reload Megatron and Transformer Engine modules to reset module-level state."""
    import megatron
    import transformer_engine

    megatron = importlib.reload(megatron)
    transformer_engine = importlib.reload(transformer_engine)


def cleanup_rope(pl_module):
    """Clear RoPE (Rotary Position Embedding) forward cache to prevent memory leaks."""
    # Check if pl_module has a module attribute
    if hasattr(pl_module, "module"):
        module_to_search = pl_module.module
    else:
        module_to_search = pl_module

    def _find_and_clean_rope(module):
        if isinstance(
            module,
            megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding,
        ):
            module.forward.cache_clear()
            hp_logger.debug(debug_msg("Clean up RoPE cache"))

        for child in module.children():
            _find_and_clean_rope(child)

    _find_and_clean_rope(module_to_search)


def cleanup_ddp(trainer, clean_tensor_hook=False):
    """
    Clean up Megatron DDP hooks and parameter references to prevent memory leaks.

    Args:
        trainer: PyTorch Lightning trainer instance
        clean_tensor_hook: If True, clear hooks from all GPU tensors (expensive)
    """
    hp_logger.debug(debug_msg("starting to release main_grad/main_param"))
    megatron_parallel = trainer.strategy.megatron_parallel

    def dummy_fn():
        pass

    def clear_hooks(handle):
        handle.remove()
        hooks_dict = handle.hooks_dict_ref()
        if hooks_dict is not None:
            hooks_dict.clear()
        for ref in handle.extra_dict_ref:
            extra_dict = ref()
            if extra_dict is not None:
                extra_dict.clear()

    for model_chunk_idx, model_chunk in enumerate(megatron_parallel):
        ddp = model_chunk.module
        try:
            ddp.disable_forward_pre_hook(param_sync=False)
            hp_logger.debug(debug_msg("disable_forward_pre_hook"))
        except:  # noqa
            hp_logger.debug(debug_msg("Failed to disable_forward_pre_hook"))

        # We can only get the handle by register a new hook
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/distributed_data_parallel.py#L410

        for param in ddp.module.parameters():
            if param.requires_grad:
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                clear_hooks(grad_acc.register_hook(dummy_fn))
        ddp.grad_accs = None

        # Unlink model params with master param/grad
        for buffer in model_chunk.buffers:
            for bucket in buffer.buckets:
                for param in bucket.params_list:
                    if hasattr(param, "main_grad") and param.main_grad is not None:
                        param.main_grad = None
                    if hasattr(param, "main_param") and param.main_param is not None:
                        param.main_param = None
    if clean_tensor_hook:
        objects = gc.get_objects()
        gpu_tensors = [obj for obj in objects if memory_tracker.is_meaningful_tensor(obj)]
        for gpu_tensor in gpu_tensors:
            clear_hooks(gpu_tensor.register_hook(dummy_fn))
