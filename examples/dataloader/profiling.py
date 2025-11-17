# type: ignore
import numpy as np
from tabulate import tabulate
import torch.distributed as dist
import torch
import os
from megatron.core import parallel_state as mcore_parallel_state

class Profiling:

    def __init__(self):
        self.metrics = {}
        self._parallel_state = mcore_parallel_state

    @property
    def is_terminal_rank(self):
        return (
            self._parallel_state.is_pipeline_first_stage()
            or self._parallel_state.is_pipeline_last_stage()
        )

    @property
    def is_tp_0(
        self,
    ):
        return (
            torch.distributed.get_rank()
            % self._parallel_state.get_tensor_model_parallel_world_size()
            == 0
        )

    @property
    def is_tp_01(self):
        return (
            torch.distributed.get_rank()
            % self._parallel_state.get_tensor_model_parallel_world_size()
            <= 1
        )

    @property
    def should_print(self):
        # Print metrics on terminal tp0 and tp1 or non terminal tp0
        return (self.is_terminal_rank and self.is_tp_01) or (self.is_tp_0)

    def add_metric(self, name, value, once=False):
        if once and name in self.metrics:
            return
        self.metrics[name] = self.metrics.get(name, [])
        self.metrics[name].append(value)

    def compute_metrics_and_print(self, print_all=False):
        computed_metrics = []
        computed_metrics.append([f"Name_rank_{dist.get_rank()}_pid_{os. getpid()}", "metrics"])
        if print_all or self.should_print:
            for k, v in self.metrics.items():
                computed_metrics.append([k, np.average(v)])
            for k, v in self.metrics.items():
                if k not in ["Broadcast Batch Time", "BDL Between Batch", "Time between batch"]:
                    continue
                all_vals = [float(f"{val:.5f}") for val in v]
                print(f"{k} ===> {all_vals[:10]}")
            print(tabulate(computed_metrics, headers="firstrow", tablefmt="fancy_grid"))
