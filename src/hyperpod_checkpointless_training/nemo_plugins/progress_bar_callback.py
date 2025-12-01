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

import time
import datetime
from typing import Optional


from lightning.pytorch.callbacks import ProgressBar
from typing_extensions import override


class HCTProgressBar(ProgressBar):
    def __init__(self, refresh_rate: int = 25, total_steps: int = 250000):
        super().__init__()
        self._active = True
        self._train_batch_idx = 0
        self._train_epoch_idx = 0
        self._refresh_rate = refresh_rate
        self._last_batch_end_logged: Optional[float] = None
        self._total_steps = total_steps

    def disable(self):
        self._active = False

    def enable(self):
        self._active = True

    def print(self, *args, **kwargs):
        if self._active:
            print(*args, **kwargs)

    def state_dict(self):
        return {
            "train_batch_idx": self._train_batch_idx,
            "train_epoch_idx": self._train_epoch_idx,
            "last_batch_end_logged": self._last_batch_end_logged,
        }

    def load_state_dict(self, state_dict):
        self._train_batch_idx = state_dict.get("train_batch_idx")
        self._train_epoch_idx = state_dict.get("train_epoch_idx", 0)
        self._last_batch_end_logged = state_dict.get("last_batch_end_logged")

    @override
    def on_train_epoch_start(self, trainer, pl_module):
        self._last_batch_end_logged = time.time()

    @override
    def on_train_epoch_end(self, trainer, pl_module):
        self._train_epoch_idx += 1

    @override
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int):
        if self._train_batch_idx is None:
            print(f"train_batch_idx is not available from checkpoint, Using batch_idx={batch_idx} as best guess")
            # Subtract by 1 here because it gets added in on_train_batch_end
            self._train_batch_idx = batch_idx - 1

    @override
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._train_batch_idx += 1
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self._train_batch_idx % self._refresh_rate == 0:
            current_time = time.time()
            timestamp = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
            
            if self._last_batch_end_logged is not None:
                time_diff = current_time - self._last_batch_end_logged
                it_per_seconds = self._refresh_rate / time_diff
                self.print(
                    f"[{timestamp}] [Epoch {self._train_epoch_idx} Batch {self._train_batch_idx} "
                    f"It/s {it_per_seconds:.3f} Hours left {(self._total_steps - self._train_batch_idx) / it_per_seconds / 3600:.2f}]:",
                    self.get_metrics(trainer, pl_module),
                )
            else:
                self.print(
                    f"[{timestamp}] [Epoch {self._train_epoch_idx} Batch {self._train_batch_idx}]:",
                    self.get_metrics(trainer, pl_module),
                )
            
            self._last_batch_end_logged = current_time

    @override
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.print(f"Val batch {batch_idx} / {self.total_val_batches}")