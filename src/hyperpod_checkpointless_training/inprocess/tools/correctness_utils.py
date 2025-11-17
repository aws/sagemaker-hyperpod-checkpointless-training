import torch

from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg

hp_logger = get_logger()


def dump_training_state_for_debug(self, trainer, step, filepath=""):
    """
    Dump relevant training state with all tensor numerical values for debugging loss mismatch.
    """
    rank = torch.distributed.get_rank()
    print(f"dump_training_state_for_debug rank: {rank}, step: {step}, tag: {filepath}")

    with open(f"{filepath}.step{step}.rank{rank}", "w") as f:
        f.write(f"=== Training State at Step {step} (Rank {rank}) ===\n")
        f.write(f"Global Step: {trainer.global_step}\n")
        f.write(f"Epoch: {trainer.current_epoch}\n\n")

        # Optimizer state with all params
        if trainer.optimizers:
            opt = trainer.optimizers[0]
            f.write(f"=== Optimizer Info ===\n")
            f.write(f"Optimizer type: {type(opt).__name__}\n")
            if hasattr(opt, 'state_dict'):
                opt_state = opt.state_dict()
                if 'state' in opt_state and opt_state['state']:
                    f.write(f"Total params with state: {len(opt_state['state'])}\n\n")
                    # Dump ALL params
                    for idx, (param_id, param_state) in enumerate(opt_state['state'].items()):
                        f.write(f"--- Optimizer Param {idx} (id={param_id}) ---\n")
                        if 'step' in param_state:
                            f.write(f"  step: {param_state['step']}\n")
                        if 'exp_avg' in param_state:
                            tensor = param_state['exp_avg'].float()
                            f.write(f"  exp_avg shape: {tensor.shape}\n")
                            f.write(f"  exp_avg mean: {tensor.mean().item():.10e}\n")
                            f.write(f"  exp_avg std: {tensor.std().item():.10e}\n")
                            f.write(f"  exp_avg first 10: {tensor.flatten()[:10].tolist()}\n")
                        if 'exp_avg_sq' in param_state:
                            tensor = param_state['exp_avg_sq'].float()
                            f.write(f"  exp_avg_sq mean: {tensor.mean().item():.10e}\n")
                            f.write(f"  exp_avg_sq first 10: {tensor.flatten()[:10].tolist()}\n")
                        f.write("\n")

        # Model parameters - ALL Adapter params
        f.write(f"=== Model Parameters (Adapter) ===\n")
        adapter_params = [(name, param) for name, param in trainer.model.named_parameters()
                        if self.is_adapter_key(name) and param.requires_grad]
        f.write(f"Total adapter params: {len(adapter_params)}\n\n")
        for idx, (name, param) in enumerate(adapter_params):
            f.write(f"--- Adapter Param {idx}: {name} ---\n")
            f.write(f"  shape: {param.shape}\n")
            f.write(f"  mean: {param.float().mean().item():.10e}\n")
            f.write(f"  std: {param.float().std().item():.10e}\n")
            f.write(f"  first 10: {param.flatten()[:10].float().tolist()}\n\n")

        # Model parameters - ALL Base Model params
        f.write(f"=== Model Parameters (Base Model) ===\n")
        base_params = [(name, param) for name, param in trainer.model.named_parameters()
                    if not self.is_adapter_key(name)]
        f.write(f"Total base model params: {len(base_params)}\n\n")
        for idx, (name, param) in enumerate(base_params):
            f.write(f"--- Base Param {idx}: {name} ---\n")
            f.write(f"  shape: {param.shape}\n")
            f.write(f"  mean: {param.float().mean().item():.10e}\n")
            f.write(f"  std: {param.float().std().item():.10e}\n")
            f.write(f"  first 10: {param.flatten()[:10].float().tolist()}\n\n")

        # LR Scheduler state
        f.write(f"=== LR Scheduler Info ===\n")
        if trainer.lr_scheduler_configs:
            for i, config in enumerate(trainer.lr_scheduler_configs):
                sched = config.scheduler
                f.write(f"Scheduler {i}: {type(sched).__name__}\n")
                if hasattr(sched, 'last_epoch'):
                    f.write(f"  last_epoch: {sched.last_epoch}\n")
                if hasattr(sched, '_step_count'):
                    f.write(f"  _step_count: {sched._step_count}\n")
                if hasattr(sched, 'get_last_lr'):
                    f.write(f"  last_lr: {sched.get_last_lr()}\n")

        # Checkpoint manager state
        f.write(f"\n=== Checkpoint Manager State ===\n")
        f.write(f"self.global_step: {self.global_step}\n")
        f.write(f"Has _checkpoint: {self._checkpoint is not None}\n")
        f.write(f"Has base_model_weights: {self.base_model_weights is not None}\n")

    hp_logger.info(debug_msg(f"Dumped training state to {filepath}.step{step}.rank{rank}"))


def print_trainer_base_model(trainer, tag=None):

    hp_logger.info(f"=== Model Parameters (Base Model and Adapter) === {tag} at step {trainer.global_step}")
    base_params = [(name, param) for name, param in trainer.model.named_parameters()]
    hp_logger.info(f"Total base model params: {len(base_params)}")
    for idx, (name, param) in enumerate(base_params):
        hp_logger.info(f"--- Base Param {idx}: {name} ---")
        hp_logger.info(f"  shape: {param.shape}")
        hp_logger.info(f"  device: {param.device}")
        hp_logger.info(f"  mean: {param.float().mean().item():.10e}")
        hp_logger.info(f"  std: {param.float().std().item():.10e}")
        hp_logger.info(f"  first 10: {param.flatten()[:10].float().tolist()}\n\n")
