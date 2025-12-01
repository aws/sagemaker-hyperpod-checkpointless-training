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

import signal
from typing import Any

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override
from hyperpod_checkpointless_training.inprocess.utils import debug_msg

from hyperpod_checkpointless_training.nemo_plugins.error_injection_utils import (
    TestFaultConfig,
    ErrorInjectionForwardHook,
    ErrorInjectionBackwardHook,
    RandomErrorInjector,
    find_middle_transformer_layer,
    FaultInjector,
)
from hyperpod_checkpointless_training.inprocess.utils import debug_msg
from hyperpod_checkpointless_training.inprocess.logger import get_logger

hp_logger = get_logger()

class HPFaultInjectionCallback(Callback):
    """
    A PyTorch Lightning callback for injecting/simulating faults.



    This callback allows for controlled fault injection at various points in the training loop:
    - During forward pass
    - During backward pass
    - During lock
    - After backward
    - Randomly during training

    The fault injection is configurable through the TestFaultConfig class and registered BEFORE
    the RestartlessCallback.

    Current state:
        :
        |
        | restartless_cb::on_after_grad_clipping () 1. inject fault 2. lock.acquire()  <--injects fault after backward.
        |
        |
        | restartless_cb::on_train_batch_end() 1. inject fault 2. lock.release()  <-- inject fault with lock.
        |
        V

    Proposed state:
    Here is the sequence of invocations
        :
        |
        | fault_inj_cb::on_after_grad_clipping() 1. inject fault <-- injects fault after backward
        | restartless_cb::on_after_grad_clipping() 1. lock.acquire()  <--injects error after backward.
        |
        | fault_inj_cb::on_train_batch_end() 1. inject fault <-- inject fault with lock.
        | restartless_cb::on_train_batch_end() 1. lock.release()
        |
        V

    """

    def __init__(self, test_fault_config: TestFaultConfig = TestFaultConfig()):
        """
        Initialize the fault injection callback.

        Args:
            test_fault_config: Configuration for fault injection
        """
        self.test_fault_config = test_fault_config
        self.hook_handles = None
        self.random_error_injector = None
        self.error_signal_handler_registered = False
        self.fault_injector = None

    def on_rcb_start(self, trainer, pl_module,):
        """
        Starts random error injection thread on_rcb_start. This is required
        since we stop the error injection thread whenever it injects error
        so we do not repeatedly trigger errors while restart is going on.

        We also register the error hooks here. This is required in the spare
        node case because replacement spare nodes will not have registered the
        error hooks prior to rcb since they will not yet have been assigned
        their rank.
        """
        rank = trainer.strategy.cluster_environment.global_rank()
        if rank in self.test_fault_config.fault_ranks:
            self.fault_injector = FaultInjector(self.test_fault_config, trainer)
            self._register_error_injection_hooks(trainer, pl_module)
            self._start_random_error_injector(trainer)
            hp_logger.debug(debug_msg(f"Fault injection hooks registered for rank {rank}"))

    @override
    def on_after_grad_clipping(self, trainer, pl_module, *args, **kwargs) -> None:
        """Inject faults after gradient clipping (after backward pass)."""
        if self.fault_injector is not None:
            self.fault_injector.maybe_inject(
                fault_prob=self.test_fault_config.fault_prob_after_bwd,
                msg="Simulating failure after backward",
            )

    @override
    def on_train_batch_end(
        self, trainer, pl_module, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Inject faults after batch processing."""
        if self.fault_injector is not None:
            self.fault_injector.maybe_inject(
                fault_prob=self.test_fault_config.fault_prob_between_lock,
                msg="Simulating failure between lock",
            )

    def _register_error_injection_hooks(self, trainer, pl_module):
        """Setup error injection hooks for forward and backward passes."""

        # Return if already initialized
        if self.hook_handles is not None:
            return
        self.hook_handles = []

        # Check if pl_module has a module attribute
        if hasattr(pl_module, 'module'):
            module_to_search = pl_module.module
        else:
            module_to_search = pl_module

        # Targeting one of the transformer layers to do error injection
        target_layer = find_middle_transformer_layer(module_to_search)
        if target_layer is None:
            hp_logger.warning(debug_msg("Could not find middle TransformerLayer. Falling back to module level hook."))
            target_layer = module_to_search

        if self.test_fault_config.fault_prob_during_fwd > 0:
            self.error_injection_fwd_hook = ErrorInjectionForwardHook(
                self.test_fault_config,
                trainer,
                self.fault_injector,
            )
            fwd_error_handle = target_layer.register_forward_hook(
                self.error_injection_fwd_hook,
                always_call=True
            )
            self.hook_handles.append(fwd_error_handle)
            hp_logger.debug(debug_msg("Registered forward pass error injection hook"))

        if self.test_fault_config.fault_prob_during_bwd > 0:
            self.error_injection_bwd_hook = ErrorInjectionBackwardHook(
                self.test_fault_config,
                trainer,
                self.fault_injector,
            )
            bwd_error_handle = target_layer.register_full_backward_hook(
                self.error_injection_bwd_hook
            )
            self.hook_handles.append(bwd_error_handle)
            hp_logger.debug(debug_msg("Registered backward pass error injection hook"))

    def _start_random_error_injector(self, trainer):
        """Start the random error injector thread."""
        def error_signal_handler(signum, frame):
            hp_logger.debug("Entering error_signal_handler")
            msg = debug_msg(f"Simulating random failure",
                        rank=trainer.strategy.cluster_environment.global_rank(),
                        seq=getattr(trainer.fit_loop, 'hp_wrapper', None).seq.get() if hasattr(trainer.fit_loop, 'hp_wrapper') else 0,
                        steps=trainer.global_step)
            raise RuntimeError(msg)

        if self.test_fault_config.fault_prob_random > 0:
            # Register the signal handler if not already done
            if not self.error_signal_handler_registered:
                signal.signal(signal.SIGUSR1, error_signal_handler)
                self.error_signal_handler_registered = True

            # Clean up existing injector if it exists
            if self.random_error_injector is not None:
                if self.random_error_injector.is_alive():
                    self.random_error_injector.stop()
                    self.random_error_injector.join()
                self.random_error_injector = None

            # Create and start new injector
            self.random_error_injector = RandomErrorInjector(self.test_fault_config, trainer)
            self.random_error_injector.start()
            hp_logger.debug(f"Started random error injector on rank {trainer.strategy.cluster_environment.global_rank()}")

    @override
    def on_exception(self, trainer, pl_module, exception):
        """Handle exceptions during training."""
        hp_logger.debug(debug_msg(f"Exception observed in fault_inj callback: {exception}"))
        # No special handling needed for fault injection callback

    @override
    def on_fit_end(self, trainer, pl_module):
        """Clean up hooks and threads."""
        if isinstance(self.hook_handles, list):
            # Clean up error injection hooks
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles = None

        # Clean up error injection thread
        if self.random_error_injector:
            self.random_error_injector.stop()
            self.random_error_injector.join()
            self.random_error_injector = None

        hp_logger.info(debug_msg("Fault injection hooks and threads cleaned up"))
