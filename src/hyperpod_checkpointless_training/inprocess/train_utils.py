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
import torch.multiprocessing as mp

from hyperpod_checkpointless_training.inprocess.elastic.hp_agent_api import HPAgentK8sAPI
from hyperpod_elastic_agent.ipc import InProcessRestartSocketClient

from hyperpod_checkpointless_training.inprocess.logger import get_logger
from hyperpod_checkpointless_training.inprocess.utils import debug_msg

from hyperpod_checkpointless_training.nemo_plugins.patches import suppress_no_sync_warning

hp_logger = get_logger()


k8s_apis = HPAgentK8sAPI(InProcessRestartSocketClient())


class HPAgentK8sAPIFactory:
    def __call__(self):
        return k8s_apis


def wait_rank():
    if mp.current_process().name != "MainProcess":
        # skip wait_rank for subprocess
        return

    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 0))

    hp_logger.info(debug_msg(f"Calling wait rank"))

    resp = k8s_apis.hyperpod_wait_rank_info()

    hp_logger.info(debug_msg(f"hyperpod_wait_rank_info response: {resp}"))

    assert isinstance(resp, tuple)
    typ, rank_info = resp
    assert typ == "job_rank_info"
    env = rank_info.get("worker_envs", None)
    assert env is not None

    for k, v in env.items():
        hp_logger.debug(debug_msg(f"setting {k} to {v}"))
        os.environ[k] = v

    suppress_no_sync_warning()

    return
