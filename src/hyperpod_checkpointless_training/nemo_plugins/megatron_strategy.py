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

import atexit
from typing import Any, Mapping
import lightning.pytorch as pl

from nemo.lightning.pytorch.strategies import MegatronStrategy
from typing_extensions import override
from hyperpod_checkpointless_training.nemo_plugins.utils import create_store
from lightning.fabric.plugins import ClusterEnvironment
from hyperpod_checkpointless_training.inprocess.utils import debug_msg
from hyperpod_checkpointless_training.nemo_plugins.fault_injection import HPFaultInjectionCallback
from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.nemo_plugins.callbacks import CheckpointlessCallback
from nemo.lightning.pytorch.callbacks import PEFT

hp_logger = get_logger()


class CheckpointlessMegatronStrategy(MegatronStrategy):
    """
    NeMo Megatron strategy with integrated checkpointless recovery capabilities.

    CORE RESPONSIBILITIES:

    1. **Distributed Setup**: Initializes process groups using TCPStore with restart-aware
       prefixes or rootless connection for fault-tolerant distributed coordination.

    2. **Recovery Strategy Selection**: Intelligently chooses between checkpointless
       peer-to-peer recovery and traditional disk-based checkpoint loading based
       on system health and availability.

    3. **Checkpoint Loading**: Handles model state dict loading with checkpointless
       recovery compatibility, skipping state dict when not present.

    4. **Fault Tolerance Integration**: Provides access to HPCallWrapper for coordinating
       fault tolerance mechanisms and registers abort handlers with trainer.

    5. **PEFT Detection**: Identifies Parameter-Efficient Fine-Tuning configurations
       to enable specialized checkpoint handling for adapter-based training.

    6. **Teardown Override**: Skips PyTorch Lightning native teardown, delegating
       cleanup to abort handlers for proper fault recovery.
    """

    def __init__(self, *args, **kwargs):
        """Initialize strategy with distributed store support for fault tolerance."""
        super().__init__(*args, **kwargs)
        self.base_store = None

    def setup(self, trainer: "pl.Trainer") -> None:
        super().setup(trainer)

        for callback in trainer.callbacks:
            if isinstance(callback, HPFaultInjectionCallback):
                hp_logger.debug(debug_msg("Register fault injections hooks from HPFaultInjectionCallback"))
                callback.on_rcb_start(trainer, trainer.lightning_module)

        self.get_wrapper().finalize.register_attributes(trainer)

        abortList = self.get_wrapper().abort.instances
        for abort in abortList:
            if hasattr(abort, "register_trainer"):
                abort.register_trainer(trainer)

    @override
    def setup_distributed(self) -> None:
        """Init process group useing either tcpstore with Prefix or rootless."""
        create_store(self)
        super().setup_distributed()

    def get_wrapper(self):
        """Get the HPCallWrapper instance for fault tolerance coordination."""
        return self.trainer.wrapper

    def load_model_state_dict(
        self, checkpoint: Mapping[str, Any], strict: bool = True
    ) -> None:
        """
        Load model state dict with checkpointless recovery compatibility.

        Handles both checkpointless and traditional checkpoint formats,
        skipping state dict loading if not present (common in checkpointless recovery).

        Args:
            checkpoint: Checkpoint dictionary containing model state
            strict: Whether to strictly enforce state dict key matching
        """
        if "state_dict" not in checkpoint:
            return

        return super().load_model_state_dict(checkpoint, strict=strict)

    def is_peft(self):
        """Check if PEFT (Parameter-Efficient Fine-Tuning) is enabled."""
        return any(isinstance(cb, PEFT) for cb in self.trainer.callbacks)

    def teardown(self) -> None:
        """
        We want to skip the PTL native teardown as our abort will handle the clean up
        """
        pass
