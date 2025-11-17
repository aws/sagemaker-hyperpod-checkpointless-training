import torch
from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer

from hyperpod_checkpointless_training.inprocess.parameter_update_lock import ParameterUpdateLock


def patch_megatron_optimizer():
    """
    Patch the megatron optimizer to enter the param_update_lock before the actual optimizer step
    This has to be done in a monkey patch way as PTL/NeMo does not offer this level of support
    """
    original_step = MixedPrecisionOptimizer.step_with_ready_grads

    @torch.no_grad()
    def patched_step_with_ready_grads(self):
        param_update_lock = ParameterUpdateLock()
        # Enter lock only once
        if not param_update_lock.acquired:
            param_update_lock.__enter__()
        return original_step(self)

    MixedPrecisionOptimizer.step_with_ready_grads = patched_step_with_ready_grads

def suppress_no_sync_warning():
    import sys

    def patched_unraisablehook(unraisable):
        """
        Suppress unraisable exceptions from Megatron DDP no_sync
        generator cleanup, fallback to default for other objects
        """
        obj_repr = repr(unraisable.object)

        if "DistributedDataParallel.no_sync" in obj_repr:
            return

        # Fallback to default
        sys.__unraisablehook__(unraisable)

    sys.unraisablehook = patched_unraisablehook
