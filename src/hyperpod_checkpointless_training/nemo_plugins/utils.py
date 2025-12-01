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

import os

import torch
from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg

hp_logger = get_logger()

def get_sharded_tensor_states(optimizer):
    """
    Get sharded state dict of param tensors where each buffer is mapped to corresponding model param.

    Args:
        optimizer: The optimizer to extract sharded tensor states from

    Returns:
        A dictionary mapping parameter indices to tensor states
    """
    param_state = {}
    optim_state = {}

    param_idx = 0
    for gbuf_range_maps in optimizer.gbuf_ranges:
        for gbuf_range_map_for_all_buckets in gbuf_range_maps.values():
            for gbuf_range_map in gbuf_range_map_for_all_buckets:
                for model_param in gbuf_range_map["param_map"].keys():
                    group_index, group_order = optimizer.model_param_group_index_map[
                        model_param
                    ]
                    main_param = optimizer.optimizer.param_groups[group_index][
                        "params"
                    ][group_order]
                    optim_state[param_idx] = optimizer.optimizer.state[main_param]
                    param_state[param_idx] = main_param
                    param_idx += 1
    return param_state, optim_state


def use_tcpstore():
    skip_tcpstore = int(os.environ.get("HPCT_USE_ROOTLESS", "0"))
    return skip_tcpstore == 0


def init_process_group(strategy):
    """Initialize PyTorch distributed process group with or without TCPStore."""
    if not use_tcpstore():
        return init_process_group_without_tcpstore(strategy)
    return init_process_group_with_tcpstore(strategy)


def create_store(strategy):
    """Create TCPStore with restart-aware prefix for fault tolerance."""
    if not use_tcpstore():
        hp_logger.info(debug_msg("Bootstrap using ROOTLESS connection"))
        return
    if not strategy.base_store:
        strategy.base_store = torch.distributed.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=int(os.environ["MASTER_PORT"]) + 1,
            world_size=int(os.environ["WORLD_SIZE"]),
            is_master=(int(os.environ["RANK"]) == 0),
            multi_tenant=True,
            wait_for_workers=True,
            use_libuv=True,
        )

    restart_num = os.environ["JOB_RESTART_COUNT"]
    strategy.store = torch.distributed.PrefixStore(str(restart_num), strategy.base_store)


def init_process_group_with_tcpstore(strategy):
    """Initialize process group using TCPStore with restart count prefix."""
    create_store(strategy)
    torch.distributed.init_process_group(
        strategy._get_process_group_backend(),
        rank=strategy.cluster_environment.global_rank(),
        world_size=strategy.cluster_environment.world_size(),
        timeout=strategy._timeout,
        store=strategy.store,
    )


def init_process_group_without_tcpstore(strategy):
    """Initialize process group using rootless connection without TCPStore."""
    torch.distributed.init_process_group(
        strategy._get_process_group_backend(),
        rank=strategy.cluster_environment.global_rank(),
        world_size=strategy.cluster_environment.world_size(),
        timeout=strategy._timeout,
    )
