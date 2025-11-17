# type: ignore

import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch.distributed
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override
from megatron.core import parallel_state

from hyperpod_checkpointless_training.dataloader.batch_hashing import (
    compute_batch_with_hash,
)
from hyperpod_checkpointless_training.inprocess.logger import get_logger

hp_logger = get_logger()


class DataCorrectnessCallback(Callback):
    """
    Simple Lightning callback for data correctness validation during fault tolerance testing.

    Hashes each batch, logs it, and saves to files for easy comparison across runs.
    """

    def __init__(
        self,
        save_dir: str = "/path_to_your_dir",
        run_type: str = "baseline",
    ):
        """
        Initialize the data correctness callback.

        Args:
            save_dir: Directory to save hash files
            run_type: Type of run - "baseline", "ipr", "plr"
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.run_type = run_type

        self.hash_file = self.save_dir / self._get_hash_filename()

        # Only create and write header if file doesn't exist
        if not self.hash_file.exists():
            with open(self.hash_file, "w") as f:
                f.write(f"# Batch hashes for {run_type} run\n")
                f.write(f"# Format: Epoch <epoch> | Step <step_number>: <hash>\n")
            hp_logger.info(
                f"DataCorrectnessCallback initialized - created new file {self.hash_file}"
            )
        else:
            hp_logger.info(
                f"DataCorrectnessCallback initialized - appending to existing file {self.hash_file}"
            )

    def _get_hash_filename(self) -> str:
        """Generate appropriate filename based on run type and fault step."""
        return f"{self.run_type}_hashes.txt"

    def _save_hash_to_file(self, epoch: int, global_step: int, batch_hash: str) -> None:
        try:

            global_rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            cp_size = parallel_state.get_context_parallel_world_size()

            dp_size = parallel_state.get_data_parallel_world_size()
            dp_rank = parallel_state.get_data_parallel_rank()

            with open(self.hash_file, "a") as f:
                f.write(
                    f"Epoch {epoch} | Step {global_step}: hash={batch_hash} | "
                    f"global_rank={global_rank} | local_rank={local_rank} | "
                    f"dp_size={dp_size} | dp_rank={dp_rank} | "
                    f"tp_size={tp_size} | pp_size={pp_size} | cp_size={cp_size}\n"
                )
                f.flush()

        except Exception as e:
            hp_logger.error(f"Failed to save hash to file: {e}")

    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Hash the batch, log it, and save to file.
        """
        try:
            step = trainer.global_step
            batch_hash = compute_batch_with_hash(batch, step)

            self._save_hash_to_file(trainer.current_epoch, step, batch_hash)

        except Exception as e:
            hp_logger.error(
                f"Failed to compute batch hash at step {step}: {e}"
            )