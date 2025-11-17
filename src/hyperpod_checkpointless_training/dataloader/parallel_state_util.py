# type: ignore
"""
Megatron-Core utilities for parallel processing and rank management.
"""

import os
import datetime
from functools import cached_property
from typing import List
from torch import distributed as dist

from tenacity import retry, stop_after_attempt, wait_exponential
from megatron.core import parallel_state as mcore_parallel_state
from hyperpod_checkpointless_training.inprocess.logger import get_logger

logger = get_logger(__name__)


class MegatronParallelStateUtil:
    """Utility class for Megatron-Core parallel processing operations."""

    def __init__(self, parallel_state=None):
        """
        Initialize MegatronParallelStateUtil.

        Args:
            parallel_state: Optional parallel state object. If None, uses mcore_parallel_state.
        """
        self.parallel_state = mcore_parallel_state
        self.model_parallel_group = None
        if self.parallel_state is None:
            raise ImportError("Megatron-Core parallel_state is not available")
 
    def is_tp_0(self):
        """Check if the current rank has TP rank 0."""
        return (
            dist.get_rank() % self.parallel_state.get_tensor_model_parallel_world_size()
            == 0
        )

    @cached_property
    def pdl_ranks_in_mp_group(self) -> List[List[int]]:
        """
        Get all pipeline ranks merged into a single list.

        Returns:
            List of lists containing all pdl ranks with the same dp rank
        """
        tp_size = self.parallel_state.get_tensor_model_parallel_world_size()
        cp_size = self.parallel_state.get_context_parallel_world_size()
        pp_size = self.parallel_state.get_pipeline_model_parallel_world_size()
        dp_size = self.parallel_state.get_data_parallel_world_size(
            with_context_parallel=False
        )
        tcp_size = tp_size * cp_size

        rank_generator = self.parallel_state.RankGenerator(
            tp=tp_size,
            ep=1,
            dp=dp_size,
            pp=pp_size,
            cp=cp_size,
            order=os.environ.get("PARALLELISM_ORDER", "tp-cp-ep-dp-pp"),
            rank_offset=0,  # assuming no encoder
        )
        all_pp_group_ranks = []
        for group_ranks in rank_generator.get_ranks("pp"):
            all_pp_group_ranks.append(group_ranks)

        pdl_global_ranks = []

        for i in range(dp_size):
            local_pp_rank_groups = all_pp_group_ranks[tcp_size * i : tcp_size * (i + 1)]
            # transpose the pp rank groups, so that first row is all PP0, second row is all PP1, etc.
            pp_rank_values = [list(col) for col in list(zip(*local_pp_rank_groups))]

            # Every PDL on First stage
            local_pdl_global_ranks = pp_rank_values[0][::tp_size]

            if pp_size > 1:
                # Every PDL on non_terminal stage
                for col in pp_rank_values[1:-1]:
                    local_pdl_global_ranks.extend(col[::tp_size])

                last_pipeline_start = 0
                if cp_size == 1:
                    local_pdl_global_ranks += [pp_rank_values[-1][0]]
                    last_pipeline_start = tp_size
                # Every PDL on Last stage, might exclude the first one.
                local_pdl_global_ranks.extend(
                    pp_rank_values[-1][last_pipeline_start::tp_size]
                )
            pdl_global_ranks.append(local_pdl_global_ranks)

        return pdl_global_ranks

    def create_model_parallel_group(self):
        """
        Creates the model parallel group if needed.
        The process group will be use before the training to gather the cache status
        within the ranks.

        Returns:
            The model parallel group.
        """
        if self.parallel_state.get_context_parallel_world_size() > 1:
            self.model_parallel_group = self._create_model_parallel_group()
            return self.model_parallel_group
        else:
            return self.parallel_state.get_model_parallel_group()
    
    @cached_property
    def get_model_parallel_ranks(self):
        parallel_state = self.parallel_state

        rank_generator = self._get_rank_generator_no_ep(parallel_state)

        all_mp_group_ranks = self._get_all_group_ranks_from_group(
            rank_generator, "tp-pp-cp"
        )
        return all_mp_group_ranks

    def _get_all_group_ranks_from_group(self, rank_generator, group: str):
        """
        Get all group ranks for a given parallelism group.
        """
        all_group_ranks = []
        for group_ranks in rank_generator.get_ranks(group):
            all_group_ranks.append(group_ranks)
        return all_group_ranks

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1)
    )  # Maximum 3 attempts
    def _create_torch_distributed_group_with_retries(self, ranks, backend, timeout):
        group = dist.new_group(ranks, backend=backend, timeout=timeout)
        return group

    def _create_model_parallel_group(self, backend="gloo"):
        """Creates the model parallel group.

        Returns:
            The model parallel group.
        """
        rank = dist.get_rank()
        all_mp_group_ranks = self.get_model_parallel_ranks

        # Initialize current rank's model parallel group
        current_rank_mp_group = None
        for mp_rank_group in all_mp_group_ranks:
            group = self._create_torch_distributed_group_with_retries(
                mp_rank_group, backend=backend, timeout=datetime.timedelta(3600)
            )
            if rank in mp_rank_group:
                current_rank_mp_group = group

        return current_rank_mp_group

    def _get_rank_generator_no_ep(
        self, parallel_state, parallelism_order: str | None = None
    ):
        """Gets the mcore rank generator with no EP."""
        tp_size, cp_size, pp_size, dp_size = self._get_tp_cp_pp_dp_sizes(parallel_state)
        rank_generator = self._get_rank_generator(
            parallel_state,
            tp_size=tp_size,
            cp_size=cp_size,
            pp_size=pp_size,
            dp_size=dp_size,
            parallelism_order=parallelism_order,
        )
        return rank_generator

    def _get_tp_cp_pp_dp_sizes(self, parallel_state):
        """Gets the parallelism sizes excluding EP."""
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        cp_size = parallel_state.get_context_parallel_world_size()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        dp_size = parallel_state.get_data_parallel_world_size(
            with_context_parallel=False
        )
        return tp_size, cp_size, pp_size, dp_size

    def _get_rank_generator(
        self,
        parallel_state,
        tp_size: int = 1,
        cp_size: int = 1,
        pp_size: int = 1,
        dp_size: int = 1,
        ep_size: int = 1,
        parallelism_order: str | None = None,
    ):
        """
        Gets the mcore rank generator for a given parallelism order.
        """
        if parallelism_order is None:
            parallelism_order = os.environ.get("PARALLELISM_ORDER", "tp-cp-ep-dp-pp")
        # Note: megatron core rank generator does not support EP > 1 and CP > 1 together
        rank_generator = parallel_state.RankGenerator(
            tp=tp_size,
            ep=ep_size,
            dp=dp_size,
            pp=pp_size,
            cp=cp_size,
            order=parallelism_order,
            rank_offset=0,  # assuming no encoder
        )
        return rank_generator
