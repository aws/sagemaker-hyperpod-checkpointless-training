import os
import random
import signal
import logging
from dataclasses import dataclass, field
import threading
import time
from typing import Set
import torch.nn as nn
from hyperpod_checkpointless_training.inprocess.utils import debug_msg
from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.tools.inject_fault import dispatch_fault_injection, Fault

NVRX_FAULT_TYPES = [fault.name for fault in Fault]
UNSUPPORTED_RANDOM_FAULT_INJECTION = [
    Fault.WORKLOAD_EXC.name,
    Fault.ASYNC_EXC.name,
    Fault.SIGNAL_EXC.name,
]
hp_logger = get_logger()


@dataclass
class TestFaultConfig:
    fault_type: str = "ipr"
    fault_prob_after_bwd: float = 0
    fault_prob_between_lock: float = 0
    fault_prob_during_fwd: float = 0
    fault_prob_during_bwd: float = 0
    fault_prob_random: float = 0
    fault_time_interval_sec: int = -1  # value < 0 disables time-based injector, disabled by default
    fault_ranks: Set[int] = field(default_factory=set)
    steps_before_fault: int = 10000

    def __post_init__(self):
        if self.fault_type not in ["ipr", "plr"] and self.fault_type not in list(Fault):
            hp_logger.warning(
                f"Invalid fault_type: {self.fault_type}. Must be 'ipr' or 'plr' or {NVRX_FAULT_TYPES}. "
                f"Note: random fault injector does not support the following fault types: {UNSUPPORTED_RANDOM_FAULT_INJECTION}."
            )
        if not 0.0 <= self.fault_prob_after_bwd <= 1.0:
            hp_logger.warning(
                f"prob_after_bwd must be between 0 and 1, got {self.fault_prob_after_bwd}. Setting to 0"
            )
            self.fault_prob_after_bwd = 0
        if not 0.0 <= self.fault_prob_between_lock <= 1.0:
            hp_logger.warning(
                f"prob_between_lock must be between 0 and 1, got {self.fault_prob_between_lock}. Setting to 0"
            )
            self.fault_prob_between_lock = 0
        if not 0.0 <= self.fault_prob_during_fwd <= 1.0:
            hp_logger.warning(
                f"fault_prob_during_fwd must be between 0 and 1, got {self.fault_prob_during_fwd}. Setting to 0"
            )
            self.fault_prob_during_fwd = 0
        if not 0.0 <= self.fault_prob_during_bwd <= 1.0:
            hp_logger.warning(
                f"fault_prob_during_bwd must be between 0 and 1, got {self.fault_prob_during_bwd}. Setting to 0"
            )
            self.fault_prob_during_bwd = 0
        if not 0.0 <= self.fault_prob_random <= 1.0:
            hp_logger.warning(
                f"fault_prob_random must be between 0 and 1, got {self.fault_prob_random}. Setting to 0"
            )
            self.fault_prob_random = 0
        if self.fault_time_interval_sec >= 0:
            if self.steps_before_fault > -1:
                hp_logger.warning(
                    f"Time-based fault injection mode enabled. "
                    f"Resetting steps_before_fault to -1 to disable step-based fault injection mode."
                )
                self.steps_before_fault = -1


class ErrorInjectionHook:
    """Base class hook for injecting test faults"""
    def __init__(self, test_fault_config, trainer, fault_injector):
        self.test_fault_config = test_fault_config
        self.trainer = trainer
        self.rank = trainer.strategy.cluster_environment.global_rank()
        self.current_step = -1
        self.fault_injector = fault_injector


class ErrorInjectionForwardHook(ErrorInjectionHook):
    """
    Hook for injecting test faults during the fwd pass.
    This hook will be registered using orch.nn.Module.register_forward_hook.
    Doc: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
    """

    def __call__(self, module, args, output):
        current_step = self.trainer.global_step
        if current_step != self.current_step:
            self.current_step = current_step
            self.injected_this_step = False
        if not self.injected_this_step:
            self.fault_injector.maybe_inject(
                fault_prob=self.test_fault_config.fault_prob_during_fwd,
                msg="Simulating failure during forward pass",
            )
            self.injected_this_step = True


class ErrorInjectionBackwardHook(ErrorInjectionHook):
    """
    Hook for injecting test faults during the bwd pass.
    This hook will be registered using orch.nn.Module.register_full_backward_hook.
    Doc: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
    """

    def __call__(self, module, grad_input, grad_output):
        current_step = self.trainer.global_step
        if current_step != self.current_step:
            self.current_step = current_step
            self.injected_this_step = False
        if not self.injected_this_step:
            self.fault_injector.maybe_inject(
                fault_prob=self.test_fault_config.fault_prob_during_bwd,
                msg="Simulating failure during backward pass",
            )
            self.injected_this_step = True


class RandomErrorInjector(threading.Thread):
    """
    Background thread to do random error injections
    """
    def __init__(self, test_fault_config, trainer):
        super().__init__()
        self.test_fault_config = test_fault_config
        self.trainer = trainer
        self.rank = trainer.strategy.cluster_environment.global_rank()
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self):
        while not self.stop_event.is_set():
            # Wait for some random amount of time between 0 and 20 if time-based injection is not enabled
            sleep_duration = (
                random.uniform(0, 20)
                if getattr(self.test_fault_config, "fault_time_interval_sec", -1) < 0
                else self.test_fault_config.fault_time_interval_sec
            )
            hp_logger.debug(
                f"Sleeping for {sleep_duration}s before attempting to inject random error"
            )
            time.sleep(sleep_duration)
            hp_wrapper = self.trainer.strategy.get_wrapper() if hasattr(self.trainer.strategy, "get_wrapper") else None
            if (
                hp_wrapper.step_upon_restart > self.test_fault_config.steps_before_fault
                and random.random() < self.test_fault_config.fault_prob_random
                and self.rank in self.test_fault_config.fault_ranks
            ):
                # Send a signal to the main process
                if self.test_fault_config.fault_type == "ipr":
                    hp_logger.debug("RandomErrorInjector sending ipr fault")
                    self.stop()
                    os.kill(
                        os.getpid(), signal.SIGUSR1
                    )  # user defined error - main thread will handle it as a RuntimeError
                elif self.test_fault_config.fault_type == "plr":
                    hp_logger.debug("RandomErrorInjector sending plr fault")
                    self.stop()
                    os.kill(os.getpid(), signal.SIGKILL)
                elif self.test_fault_config.fault_type not in UNSUPPORTED_RANDOM_FAULT_INJECTION:
                    hp_logger.debug(f"RandomErrorInjector attempting to inject fault.")
                    inject_fault(self.test_fault_config.fault_type, self.stop)
                else:
                    hp_logger.warning(
                        f"Unsupported fault_type: {self.test_fault_config.fault_type}"
                    )

    def stop(self):
        self.stop_event.set()


class FaultInjector:
    """
    Fault injector used to simulate faults in the main training thread.

    Supports two triggering modes, controlled by the settings in TestFaultConfig:
      - **Time-based**: A fault can be injected only if at least `fault_time_interval_sec`
        seconds have passed since the last injection (or since `on_rcb_start` if none have occurred yet).
      - **Step-based**: A fault can be injected once the trainer's
        `step_upon_restart` exceeds `steps_before_fault`. This ensures faults
        are only injected after a minimum number of training steps have been completed.
    """

    def __init__(self, test_fault_config, trainer):
        self.test_fault_config = test_fault_config
        self.trainer = trainer
        self.last_injection_time = time.time()
        self.rank = trainer.strategy.cluster_environment.global_rank()

    def maybe_inject(self, fault_prob, msg="Fault injection triggered."):
        """
        Attempt to inject a fault at the current point in training.

        Args:
            fault_prob (float):
                Probability [0, 1] of injecting a fault if conditions are met.
            msg (str):
                Message to include in logs and exceptions when a fault is injected.

        Conditions:
            - The current rank must be in `fault_ranks`.
            - Either:
                * Time since last injection >= `fault_time_interval_sec` (time-based mode), OR
                * `step_upon_restart` > `steps_before_fault` (step-based mode).
            - The generated random probability is less than `fault_prob`.

        Behavior:
            - If `fault_type == "ipr"` → Raises `RuntimeError` with the debug message.
            - If `fault_type == "plr"` → Sends SIGKILL to the process.
            - Else → Logs and calls `inject_fault()` for NVRx faults.

        Returns:
            None
        """
        now = time.time()
        random.seed(now)
        prob = random.random()

        time_lapsed = now - self.last_injection_time

        hp_wrapper = self.trainer.strategy.get_wrapper() if hasattr(self.trainer.strategy, "get_wrapper") else None
        if getattr(self.test_fault_config, "fault_time_interval_sec", -1) >= 0:
            step_mode_trigger = False
            time_mode_trigger = time_lapsed >= self.test_fault_config.fault_time_interval_sec
        else:
            step_mode_trigger = (
                hp_wrapper is not None
                and hp_wrapper.step_upon_restart > self.test_fault_config.steps_before_fault
            )
            time_mode_trigger = False

        if (
            prob < fault_prob
            and self.rank in self.test_fault_config.fault_ranks
            and (time_mode_trigger or step_mode_trigger)
        ):

            hp_logger.info(
                f"Fault injection time lapsed: {time_lapsed}s. "
                f"Updating last_injection_time from {self.last_injection_time}s to {now}s."
            )
            self.last_injection_time = now

            msg = debug_msg(
                f"[{'STEP_MODE' if step_mode_trigger else 'TIME_MODE'}]: {msg}",
                rank=self.trainer.strategy.cluster_environment.global_rank(),
                seq=0 if hp_wrapper is None else hp_wrapper.seq.get(),
                steps=self.trainer.global_step,
            )

            if self.test_fault_config.fault_type == "ipr":
                raise RuntimeError(msg)
            elif self.test_fault_config.fault_type == "plr":
                os.kill(os.getpid(), signal.SIGKILL)
            else:
                hp_logger.debug(msg)
                inject_fault(self.test_fault_config.fault_type)


def inject_fault(fault_type, callable=None):
    try:
        fault = Fault[fault_type]
        if callable is not None:
            callable()
        dispatch_fault_injection(
            fault=fault,
            delay=0.0,
            callback=hp_logger.debug(f"Fault successfully injected: {fault}."),
        )
    except KeyError:
        hp_logger.warning(f"Undefined fault_type: {fault_type}")


def find_middle_transformer_layer(module):
    """
    Finds the middle TransformerLayer in a module.

    Note that there are a few assumptions being made that are AGI/Megatron specific
    Assumptions (these may change depending on the model being used):
      - Transformer layers are created as nn.ModuleList
        - https://code.amazon.com/packages/AGI3P-Megatron-LM/blobs/12fee178ee4534cf08749db4d6915807766febf1/--/megatron/core/transformer/transformer_block.py#L278
      - Each transformer layer has class name == 'TransformerLayer'
        - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_layer.py#L255
    """

    def _find_transformer_layers_list(module):
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            if all(child.__class__.__name__ == "TransformerLayer" for child in module):
                return module

        for child in module.children():
            result = _find_transformer_layers_list(child)
            if result is not None:
                return result
        return None

    try:
        layers = _find_transformer_layers_list(module)
        if layers is not None:
            middle_idx = len(layers) // 2
            return layers[middle_idx]
    except Exception as e:
        hp_logger.warning(f"Error while searching for middle TransformerLayer: {e}")

    return None
