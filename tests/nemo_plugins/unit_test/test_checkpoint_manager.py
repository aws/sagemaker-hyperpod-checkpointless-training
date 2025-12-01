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
import pytest
import torch
import numpy as np
import random
from unittest.mock import ANY, Mock, patch, MagicMock, call
import unittest
from contextlib import contextmanager

from megatron.core.optimizer import ChainedOptimizer
from hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager import (
    CheckpointManager,
    PEFTCheckpointManager,
    traverse_state_dict,
    flatten_state_dict,
    extract_tensors_from_flatten_state_dict,
    fill_tensor_back_to_flatten_state_dict_in_place,
    load_saved_to_local,
    validate_tensor_meta_match,
    offload_state_dict_to_cpu,
    remove_model_checkpoint_callbacks,
)


# Create a mock for CUDA operations
def setup_cuda_mocks():
    """Setup mocks for CUDA operations when CUDA is not available"""
    if not torch.cuda.is_available():
        # Mock torch.cuda functions
        torch.cuda.empty_cache = MagicMock()
        torch.cuda.synchronize = MagicMock()
        torch.cuda.get_rng_state = MagicMock(return_value=torch.randn(100))

        # Mock tensor.cuda() method
        original_tensor_init = torch.Tensor.__init__
        def mock_cuda(self):
            return self
        torch.Tensor.cuda = mock_cuda

class TestCheckpointManager:
    @classmethod
    def setUpClass(cls):
        setup_cuda_mocks()

    @pytest.fixture
    def checkpoint_manager(self):
        return CheckpointManager(enable_checksum=False)

    @pytest.fixture
    def checkpoint_manager_with_checksum(self):
        return CheckpointManager(enable_checksum=True)

    @pytest.fixture
    def mock_trainer(self):
        trainer = Mock()
        trainer.global_step = 100
        trainer.strategy = Mock()
        trainer.state = Mock()
        trainer.state.fn = Mock()
        trainer._checkpoint_connector = Mock()
        trainer._checkpoint_connector.dump_checkpoint.return_value = {
            "state_dict": {"model.weight": torch.randn(10, 10)},
            "optimizer_states": {},
            "global_step": 100,
        }
        return trainer

    @pytest.fixture
    def sample_state_dict(self):
        return {
            "model": {
                "layer1": {"weight": torch.randn(5, 5), "bias": torch.randn(5)},
                "layer2": {"weight": torch.randn(3, 5)},
            },
            "optimizer": {"lr": 0.001, "momentum": 0.9},
            "global_step": 100,
        }

    def test_init_default(self):
        manager = CheckpointManager()
        assert manager.rng_states is None
        assert manager._checkpoint is None
        assert manager.global_step is None
        assert not manager._checkpointless_recovery_feasible
        assert not manager.checkpointless_recovery_verified
        assert manager.failed_rank_info is None
        assert manager.global_step_info is None

    def test_init_with_checksum(self):
        manager = CheckpointManager(enable_checksum=True)
        assert manager.checksum_manager is not None

    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.time.perf_counter")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.memory_status")
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    def test_save_checkpoint(self, mock_synchronize, mock_empty_cache, mock_memory_status, mock_perf_counter, checkpoint_manager, mock_trainer):
        mock_perf_counter.side_effect = [0.0, 1.5]
        mock_memory_status.return_value = ("memory info", None)

        checkpoint_manager.save_checkpoint(mock_trainer)

        assert checkpoint_manager._checkpoint is not None
        assert checkpoint_manager.global_step == 100
        mock_trainer._checkpoint_connector.dump_checkpoint.assert_called_once_with(weights_only=False)

    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.empty_cache')
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.gc.collect")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.memory_status")
    def test_delete_checkpoint(self, mock_memory_status, mock_gc_collect, mock_empty_cache, mock_synchronize, checkpoint_manager):

        checkpoint_manager._checkpoint = {"test": "data"}
        checkpoint_manager.global_step = 100
        mock_memory_status.return_value = ("memory info", None)

        checkpoint_manager.delete_checkpoint()

        assert checkpoint_manager._checkpoint is None
        assert checkpoint_manager.global_step is None
        mock_gc_collect.assert_called_once()

    @patch('numpy.random.get_state')
    @patch('random.getstate')
    @patch('torch.get_rng_state')
    @patch('torch.cuda.get_rng_state')
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.get_cuda_rng_tracker")
    def test_store_rng_states(self, mock_cuda_rng_tracker, mock_gpu_rng, mock_cpu_rng, mock_random_state, mock_np_state, checkpoint_manager):
        # Mock CUDA RNG tracker
        mock_tracker = Mock()
        mock_tracker_state = torch.randn(10)
        mock_tracker_state.cuda = Mock(return_value=mock_tracker_state)
        mock_tracker.get_states.return_value = {"default": mock_tracker_state}
        mock_cuda_rng_tracker.return_value = mock_tracker

        mock_cpu_tensor = torch.randn(10)
        mock_cpu_tensor.cuda = Mock(return_value=mock_cpu_tensor)
        mock_cpu_rng.return_value = mock_cpu_tensor

        mock_gpu_tensor = torch.randn(10)
        mock_gpu_tensor.cuda = Mock(return_value=mock_gpu_tensor)
        mock_gpu_rng.return_value = mock_gpu_tensor

        mock_random_state.return_value = (1, tuple(), None)
        mock_np_state.return_value = ('MT19937', np.array([1, 2, 3]), 0, 0, 0.0)

        checkpoint_manager.store_rng_states()

        assert checkpoint_manager.rng_states is not None
        assert len(checkpoint_manager.rng_states) == 5


    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.get_cuda_rng_tracker")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager._set_cuda_rng_state")
    def test_load_rng_states(self, mock_set_cuda_rng, mock_cuda_rng_tracker, checkpoint_manager):
        # Setup mock RNG states
        mock_tracker = Mock()
        mock_cuda_rng_tracker.return_value = mock_tracker

        checkpoint_manager.rng_states = [
            (1, tuple(), None),  # random state
            ('MT19937', np.array([1, 2, 3]), 0, 0, 0.0),  # numpy state
            torch.randn(10),  # cpu rng state
            torch.randn(10),  # cuda rng state
            {"default": torch.randn(10)},  # cuda rng tracker states
        ]

        with patch('random.setstate') as mock_random_setstate, \
             patch('numpy.random.set_state') as mock_np_setstate, \
             patch('torch.set_rng_state') as mock_torch_setstate:

            checkpoint_manager.load_rng_states()

            mock_random_setstate.assert_called_once()
            mock_np_setstate.assert_called_once()
            mock_torch_setstate.assert_called_once()
            mock_set_cuda_rng.assert_called_once()
            mock_tracker.set_states.assert_called_once()

    def test_load_rng_states_no_states(self, checkpoint_manager):
        checkpoint_manager.rng_states = None

        # Should not raise exception and log warning
        checkpoint_manager.load_rng_states()

    def test_reset_checkpointless_recovery_validation(self, checkpoint_manager):
        checkpoint_manager.checkpointless_recovery_verified = True
        checkpoint_manager.failed_rank_info = {"rank": 1}
        checkpoint_manager.global_step_info = [100, 100]

        checkpoint_manager.reset_checkpointless_recovery_validation()

        assert not checkpoint_manager.checkpointless_recovery_verified
        assert checkpoint_manager.failed_rank_info is None
        assert checkpoint_manager.global_step_info is None

    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.init_process_group")
    def test_checkpointless_IS_CKPT_ONLY(self, mock_init_pg, checkpoint_manager, mock_trainer):
        checkpoint_manager.checkpointless_recovery_verified = False
        os.environ['IS_CKPT_ONLY'] = "1"
        result = checkpoint_manager.checkpointless_recovery_feasible(mock_trainer)

        assert result is False

    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.init_process_group")
    def test_checkpointless_recovery_feasible_not_verified(self, mock_init_pg, checkpoint_manager, mock_trainer):
        checkpoint_manager.checkpointless_recovery_verified = False
        os.environ['IS_CKPT_ONLY'] = "0"
        with patch.object(checkpoint_manager, 'validate_checkpointless_restore', return_value=True) as mock_validate:
            result = checkpoint_manager.checkpointless_recovery_feasible(mock_trainer)

            assert result is True
            mock_validate.assert_called_once_with(mock_trainer, include_checksum_verification=True)

    def test_checkpointless_recovery_feasible_with_ckpt_only_enabled(self, checkpoint_manager, mock_trainer):
        checkpoint_manager.checkpointless_recovery_verified = True
        checkpoint_manager._checkpointless_recovery_feasible = True

        with patch.dict('os.environ', {'IS_CKPT_ONLY': '1'}):
            result = checkpoint_manager.checkpointless_recovery_feasible(mock_trainer)

        assert result is False

    def test_checkpointless_recovery_feasible_already_verified(self, checkpoint_manager, mock_trainer):
        os.environ['IS_CKPT_ONLY'] = "0"
        checkpoint_manager.checkpointless_recovery_verified = True
        checkpoint_manager._checkpointless_recovery_feasible = True

        result = checkpoint_manager.checkpointless_recovery_feasible(mock_trainer)

        assert result is True

    def test_validate_global_step_all_match(self, checkpoint_manager):
        global_step_info = [100, 100, 100, 100]
        failed_rank_info = {}

        step, match = checkpoint_manager.validate_global_step(global_step_info, failed_rank_info)

        assert step == 100
        assert match is True

    def test_validate_global_step_with_failed_ranks(self, checkpoint_manager):
        global_step_info = [100, 0, 100, 100]  # rank 1 failed
        failed_rank_info = {1: [0, 1, 2, 3]}

        step, match = checkpoint_manager.validate_global_step(global_step_info, failed_rank_info)

        assert step == 100
        assert match is True

    def test_validate_global_step_mismatch(self, checkpoint_manager):
        global_step_info = [100, 99, 100, 101]
        failed_rank_info = {}

        step, match = checkpoint_manager.validate_global_step(global_step_info, failed_rank_info)

        assert step == 101
        assert match is False

    def test_validate_global_step_all_failed_all_zero(self, checkpoint_manager):
        global_step_info = [0, 0, 0, 0]
        failed_rank_info = {0: [], 1: [], 2: [], 3: []}

        step, match = checkpoint_manager.validate_global_step(global_step_info, failed_rank_info)

        assert step == 0
        assert match is True

    def test_validate_global_step_all_failed_non_zero(self, checkpoint_manager):
        global_step_info = [100, 99, 100, 101]
        failed_rank_info = {0: [], 1: [], 2: [], 3: []}

        step, match = checkpoint_manager.validate_global_step(global_step_info, failed_rank_info)

        assert step == 101
        assert match is False

    def test_offload_state_dict_to_cpu(self):
        """Test the new offload_state_dict_to_cpu function"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        state_dict = {
            "model": {
                "weight": torch.randn(2, 2).cuda(),
                "bias": torch.randn(2).cuda()
            },
            "config": {"value": 42}
        }

        offloaded_dict, tensor_keys = offload_state_dict_to_cpu(state_dict)

        assert offloaded_dict["model"]["weight"].device.type == "cpu"
        assert offloaded_dict["model"]["bias"].device.type == "cpu"
        assert "model.weight" in tensor_keys
        assert "model.bias" in tensor_keys
        assert offloaded_dict["config"]["value"] == 42


class TestUtilityFunctions:
    def test_traverse_state_dict_basic(self):
        state_dict = {"a": 1, "b": {"c": 2, "d": 3}}
        visited = []

        def visitor(path, value):
            visited.append((path, value))

        traverse_state_dict(state_dict, visitor)

        assert len(visited) == 3
        assert (("a",), 1) in visited
        assert (("b", "c"), 2) in visited
        assert (("b", "d"), 3) in visited

    def test_traverse_state_dict_empty_dict(self):
        state_dict = {"a": 1, "b": {"c": {}}}
        visited = []

        def visitor(path, value):
            visited.append((path, value))

        traverse_state_dict(state_dict, visitor)

        assert len(visited) == 2
        assert (("a",), 1) in visited
        assert (("b", "c"), {}) in visited

    def test_flatten_state_dict(self):
        state_dict = {
            "model": {"weight": torch.randn(2, 2), "bias": torch.randn(2)},
            "optimizer": {"lr": 0.001},
        }

        flattened, mappings = flatten_state_dict(state_dict)

        assert "model.weight" in flattened
        assert "model.bias" in flattened
        assert "optimizer.lr" in flattened
        assert len(mappings) == 3

    @patch('torch.cuda.current_device', return_value=0)
    @patch('torch.cuda.synchronize')
    def test_extract_tensors_from_flatten_state_dict_with_hybrid_device_tensor(self, mock_synchronize, mock_current_device):
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        cpu_device = torch.device('cpu')
        flattened = {
            "model.weight": torch.randn(2, 2).to(device),
            "model.bias": torch.randn(2).to(device),
            "state_dict.router.weight": torch.randn(2).to(cpu_device),
            "optimizer.lr": 0.001,
            "global_step": 100,
        }

        tensor_meta, tensors, checkpoint_no_tensor = extract_tensors_from_flatten_state_dict(flattened)

        if torch.cuda.is_available():
            assert len(tensor_meta) == 3
            assert len(tensors) == 3
            assert checkpoint_no_tensor["model.weight"] is None
            assert checkpoint_no_tensor["model.bias"] is None
            assert checkpoint_no_tensor["state_dict.router.weight"] is None
        assert checkpoint_no_tensor["optimizer.lr"] == 0.001
        assert checkpoint_no_tensor["global_step"] == 100

    @patch('torch.cuda.current_device', return_value=0)
    @patch('torch.cuda.synchronize')
    def test_extract_tensors_from_flatten_state_dict(self, mock_synchronize, mock_current_device):
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        flattened = {
            "model.weight": torch.randn(2, 2).to(device),
            "model.bias": torch.randn(2).to(device),
            "optimizer.lr": 0.001,
            "global_step": 100,
        }

        tensor_meta, tensors, checkpoint_no_tensor = extract_tensors_from_flatten_state_dict(flattened)

        if torch.cuda.is_available():
            assert len(tensor_meta) == 2
            assert len(tensors) == 2
            assert checkpoint_no_tensor["model.weight"] is None
            assert checkpoint_no_tensor["model.bias"] is None
        assert checkpoint_no_tensor["optimizer.lr"] == 0.001
        assert checkpoint_no_tensor["global_step"] == 100

    def test_fill_tensor_back_to_flatten_state_dict_in_place(self):
        tensor_meta = [("model.weight", (2, 2)), ("model.bias", (2,))]
        tensors = [torch.randn(2, 2), torch.randn(2)]
        checkpoint_no_tensor = {"model.weight": None, "model.bias": None, "lr": 0.001}

        fill_tensor_back_to_flatten_state_dict_in_place(tensor_meta, tensors, checkpoint_no_tensor)

        assert torch.is_tensor(checkpoint_no_tensor["model.weight"])
        assert torch.is_tensor(checkpoint_no_tensor["model.bias"])
        assert checkpoint_no_tensor["lr"] == 0.001

    def test_fill_tensor_back_mismatch_length(self):
        tensor_meta = [("model.weight", (2, 2))]
        tensors = [torch.randn(2, 2), torch.randn(2)]
        checkpoint_no_tensor = {}

        with pytest.raises(ValueError, match="tensor_meta and tensors should be 1-1 mapping"):
            fill_tensor_back_to_flatten_state_dict_in_place(tensor_meta, tensors, checkpoint_no_tensor)

    def test_fill_tensor_back_shape_mismatch(self):
        tensor_meta = [("model.weight", (2, 2))]
        tensors = [torch.randn(3, 3)]  # Wrong shape
        checkpoint_no_tensor = {"model.weight": None}

        with pytest.raises(ValueError, match="Mismatch tensor shape"):
            fill_tensor_back_to_flatten_state_dict_in_place(tensor_meta, tensors, checkpoint_no_tensor)

    def test_load_saved_to_local_basic(self):
        saved_tensors = [torch.ones(2, 2), torch.zeros(3)]
        local_tensors = [torch.randn(2, 2), torch.randn(3)]

        load_saved_to_local(saved_tensors, local_tensors)

        assert torch.allclose(local_tensors[0], torch.ones(2, 2))
        assert torch.allclose(local_tensors[1], torch.zeros(3))

    def test_load_saved_to_local_with_mismatching_indexes(self):
        saved_tensors = [torch.ones(2, 2), torch.zeros(3), torch.full((2,), 5.0)]
        local_tensors = [torch.randn(2, 2), torch.randn(3)]
        mismatching_indexes = [2]  # Third tensor is new

        load_saved_to_local(saved_tensors, local_tensors, mismatching_indexes)

        assert len(local_tensors) == 3
        assert torch.allclose(local_tensors[2], torch.full((2,), 5.0))

    def test_load_saved_to_local_length_mismatch(self):
        saved_tensors = [torch.ones(2, 2)]
        local_tensors = [torch.randn(2, 2), torch.randn(3)]

        with pytest.raises(RuntimeError, match="Mismatching tensors during loading"):
            load_saved_to_local(saved_tensors, local_tensors)

    def test_validate_tensor_meta_match_strict_success(self):
        saved_meta = [("model.weight", (2, 2)), ("model.bias", (2,))]
        new_meta = [("model.weight", (2, 2)), ("model.bias", (2,))]

        result = validate_tensor_meta_match(saved_meta, new_meta, strict=True)

        assert result is None

    def test_validate_tensor_meta_match_strict_failure(self):
        saved_meta = [("model.weight", (2, 2)), ("model.bias", (2,))]
        new_meta = [("model.weight", (3, 3)), ("model.bias", (2,))]

        with pytest.raises(ValueError, match="Different saved_meta and new_meta"):
            validate_tensor_meta_match(saved_meta, new_meta, strict=True)

    def test_validate_tensor_meta_match_non_strict_success(self):
        saved_meta = [("model.weight", (2, 2)), ("model.bias", (2,)), ("extra.param", (5,))]
        new_meta = [("model.weight", (2, 2)), ("model.bias", (2,))]

        result = validate_tensor_meta_match(saved_meta, new_meta, strict=False)

        assert result == [2]  # Index of extra.param

    def test_validate_tensor_meta_match_non_strict_missing_tensor(self):
        saved_meta = [("model.weight", (2, 2))]
        new_meta = [("model.weight", (2, 2)), ("model.bias", (2,))]

        with pytest.raises(ValueError, match=f"tensor .* does not exist in .*"):
            validate_tensor_meta_match(saved_meta, new_meta, strict=False)

    def test_validate_tensor_meta_match_sequence_mismatch(self):
        saved_meta = [("model.bias", (2,)), ("model.weight", (2, 2))]
        new_meta = [("model.weight", (2, 2)), ("model.bias", (2,))]

        with pytest.raises(ValueError, match="The sequence of tensors does not match"):
            validate_tensor_meta_match(saved_meta, new_meta, strict=False)

    def test_validate_tensor_meta_match_critical_tensor_missing(self):
        saved_meta = [("state_dict.model.weight", (2, 2)), ("extra.param", (5,))]
        new_meta = [("extra.param", (5,))]

        with pytest.raises(ValueError, match="Missing model/opt tensor in local state_dict"):
            validate_tensor_meta_match(saved_meta, new_meta, strict=False)


class TestCheckpointManagerIntegration:
    @pytest.fixture
    def checkpoint_manager(self):
        return CheckpointManager(enable_checksum=False)

    @pytest.fixture
    def mock_trainer(self):
        trainer = Mock()
        trainer.global_step = 100
        trainer.strategy = Mock()
        trainer.strategy.trainer = Mock()
        trainer.strategy.trainer.state = Mock()
        trainer.strategy.trainer.state.fn = Mock()
        trainer.strategy.ckpt_save_optimizer = True
        trainer.strategy.optimizer_sharded_state_dict.return_value = {}
        trainer._checkpoint_connector = Mock()
        trainer._checkpoint_connector.dump_checkpoint.return_value = {
            "state_dict": {"model.weight": torch.randn(10, 10)},
            "optimizer_states": {},
            "optimizer": [{}],
            "global_step": 100,
        }
        return trainer

    @patch('torch.cuda.synchronize')
    def test_get_nemo_in_memory_checkpoint(self, mock_sync, checkpoint_manager, mock_trainer):
        checkpoint = checkpoint_manager.get_nemo_in_memory_checkpoint(mock_trainer)

        assert "state_dict" in checkpoint
        assert "optimizer_states" in checkpoint
        assert checkpoint["global_step"] == 100
        mock_sync.assert_called_once()

    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.init_process_group")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.check_available_replica")
    def test_validate_checkpointless_restore_success(self, mock_check_replica, mock_init_pg, checkpoint_manager, mock_trainer):
        mock_check_replica.return_value = True
        checkpoint_manager.checksum_manager.verify_global_checksum = Mock(return_value=True)

        with patch.object(checkpoint_manager, 'sync_rank_and_step_info') as mock_sync, \
             patch.object(checkpoint_manager, 'validate_global_step') as mock_validate_step, \
             patch('torch.distributed.is_initialized', return_value=True):

            mock_sync.return_value = ({}, [100, 100, 100, 100])
            mock_validate_step.return_value = (100, True)

            result = checkpoint_manager.validate_checkpointless_restore(mock_trainer)

            assert result is True
            assert checkpoint_manager.checkpointless_recovery_verified is True

    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.init_process_group")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.check_available_replica")
    def test_validate_checkpointless_restore_no_replica(self, mock_check_replica, mock_init_pg, checkpoint_manager, mock_trainer):
        mock_check_replica.return_value = False

        with patch.object(checkpoint_manager, 'sync_rank_and_step_info') as mock_sync, \
             patch.object(checkpoint_manager, 'validate_global_step') as mock_validate_step, \
             patch('torch.distributed.is_initialized', return_value=True):

            mock_sync.return_value = ({}, [100, 100, 100, 100])
            mock_validate_step.return_value = (100, True)

            result = checkpoint_manager.validate_checkpointless_restore(mock_trainer)

            assert result is False

    @patch('torch.cuda.synchronize')
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.get_rank_maps")
    def test_try_checkpointless_load_success(self, mock_get_rank_maps, mock_synchronize, checkpoint_manager, mock_trainer):
        mock_get_rank_maps.return_value = [(0, 1)]
        checkpoint_manager._checkpointless_recovery_feasible = True

        with patch.object(checkpoint_manager, 'checkpointless_recovery_feasible', return_value=True) as mock_feasible, \
             patch.object(checkpoint_manager, 'restore_from_peer', return_value={"test": "checkpoint"}) as mock_restore, \
             patch.object(checkpoint_manager, 'delete_checkpoint') as mock_delete, \
             patch.object(checkpoint_manager, 'failed_rank_info') as mock_rank_info:

            result = checkpoint_manager.try_checkpointless_load(mock_trainer)

            assert result == {"test": "checkpoint"}
            mock_feasible.assert_called_once()
            mock_restore.assert_called_once()
            mock_delete.assert_called_once()

    def test_try_checkpointless_load_not_feasible(self, checkpoint_manager, mock_trainer):
        with patch.object(checkpoint_manager, 'checkpointless_recovery_feasible', return_value=False) as mock_feasible, \
             patch.object(checkpoint_manager, 'delete_checkpoint') as mock_delete:

            result = checkpoint_manager.try_checkpointless_load(mock_trainer)

            assert result is None
            mock_feasible.assert_called_once()
            mock_delete.assert_called_once()

class TestCheckpointManagerDistributedOperations:
    @pytest.fixture
    def checkpoint_manager(self):
        return CheckpointManager(enable_checksum=False)

    @pytest.fixture
    def mock_trainer(self):
        trainer = Mock()
        trainer.global_step = 100
        return trainer

    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.all_gather_object")
    @patch(
        "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.parallel_state.get_inter_distributed_optimizer_instance_group"
    )
    @patch("torch.distributed.get_process_group_ranks")
    def test_sync_rank_and_step_info_healthy_rank(self, mock_get_pg_ranks, mock_get_inter_group,
                                                  mock_all_gather, mock_world_size, mock_get_rank,
                                                  checkpoint_manager, mock_trainer):
        mock_get_rank.return_value = 0
        mock_world_size.return_value = 4
        checkpoint_manager.global_step = 100
        checkpoint_manager.parameter_update_lock.is_healthy = Mock(return_value=True)

        # Mock gathered data from all ranks
        gathered_data = [
            {0: [100]},  # rank 0
            {1: [100]},  # rank 1
            {2: [0, [0, 1, 2, 3]]},  # rank 2 (failed, with candidate ranks)
            {3: [100]},  # rank 3
        ]
        mock_all_gather.side_effect = lambda obj_list, obj: obj_list.__setitem__(slice(None), gathered_data)

        failed_rank_info, global_step_info = checkpoint_manager.sync_rank_and_step_info(mock_trainer)

        assert len(global_step_info) == 4
        assert 2 in failed_rank_info
        assert failed_rank_info[2] == [0, 1, 2, 3]

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_gather_object')
    def test_sync_rank_and_step_info_failed_rank(self, mock_all_gather, mock_world_size,
                                                 mock_get_rank, checkpoint_manager, mock_trainer):
        mock_get_rank.return_value = 2
        mock_world_size.return_value = 4
        checkpoint_manager.global_step = None  # Failed rank scenario
        checkpoint_manager.parameter_update_lock.is_healthy = Mock(return_value=False)

        with (
            patch(
                "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.parallel_state.get_inter_distributed_optimizer_instance_group"
            ) as mock_inter_group,
            patch("torch.distributed.get_process_group_ranks") as mock_pg_ranks,
        ):

            mock_pg_ranks.return_value = [0, 1, 2, 3]

            gathered_data = [
                {0: [100]},
                {1: [100]},
                {2: [100, [0, 1, 2, 3]]},  # This rank with candidate ranks
                {3: [100]},
            ]
            mock_all_gather.side_effect = lambda obj_list, obj: obj_list.__setitem__(slice(None), gathered_data)

            failed_rank_info, global_step_info = checkpoint_manager.sync_rank_and_step_info(mock_trainer)

            assert len(global_step_info) == 4

    @patch('torch.distributed.get_rank')
    def test_distribute_rng(self, mock_get_rank, checkpoint_manager):
        checkpoint_manager.rng_states = [
            (1, tuple(), None),  # Python random
            ('MT19937', np.array([1, 2, 3]), 0, 0, 0.0),  # NumPy random
            torch.randn(10),  # CPU RNG
            torch.randn(10),  # CUDA RNG
            {"default": torch.randn(10)},  # Megatron RNG tracker
        ]

        with patch.object(checkpoint_manager, '_distribute_non_tensor_checkpoint') as mock_dist_non_tensor, \
             patch.object(checkpoint_manager, '_transfer_tensors_between_ranks') as mock_transfer:

            mock_dist_non_tensor.return_value = [(1, tuple(), None), ('MT19937', np.array([1, 2, 3]), 0, 0, 0.0)]

            checkpoint_manager.distribute_rng(0, 1)

            # Should call distribute for non-tensor data and transfer for tensors
            mock_dist_non_tensor.assert_called_once()
            assert mock_transfer.call_count == 2  # Once for RNG tensors, once for Megatron tracker

    def test_extract_tensors_wrong_device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create tensor on wrong device
        wrong_device = 1 if torch.cuda.current_device() == 0 else 0
        if torch.cuda.device_count() <= wrong_device:
            pytest.skip("Not enough CUDA devices")

        flattened = {
            "model.weight": torch.randn(2, 2).cuda(wrong_device),
        }

        with pytest.raises(ValueError, match="Found CUDA tensor that does not belong to current device"):
            extract_tensors_from_flatten_state_dict(flattened)


class TestCheckpointManagerEdgeCases:
    @pytest.fixture
    def checkpoint_manager(self):
        return CheckpointManager(enable_checksum=True)

    def test_checkpointless_recovery_feasible_no_distributed(self, checkpoint_manager):
        mock_trainer = Mock()

        with (
            patch("torch.distributed.is_initialized", return_value=False),
            patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.init_process_group") as mock_init,
        ):

            with (
                patch.object(checkpoint_manager, "sync_rank_and_step_info") as mock_sync,
                patch.object(checkpoint_manager, "validate_global_step") as mock_validate,
                patch(
                    "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.check_available_replica"
                ) as mock_check,
            ):

                mock_sync.return_value = ({}, [100, 100])
                mock_validate.return_value = (100, True)
                mock_check.return_value = True
                checkpoint_manager.checksum_manager.verify_global_checksum = Mock(return_value=True)

                result = checkpoint_manager.validate_checkpointless_restore(mock_trainer)

                mock_init.assert_called_once_with(mock_trainer)
                assert result is True

    def test_convert_to_saved_to_local_dtype(self, checkpoint_manager):
        """Test convert_to_saved_to_local_dtype function"""
        from hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager import convert_to_saved_to_local_dtype

        saved_dict = {
            "model.weight": torch.randn(2, 2).float(),
            "model.bias": torch.randn(2).double(),
            "non_tensor": "value"
        }

        local_dict = {
            "model.weight": torch.randn(2, 2).half(),
            "model.bias": torch.randn(2).float(),
            "non_tensor": "value"
        }

        convert_to_saved_to_local_dtype(saved_dict, local_dict)

        # Check that dtypes were converted
        assert saved_dict["model.weight"].dtype == torch.half
        assert saved_dict["model.bias"].dtype == torch.float
        assert saved_dict["non_tensor"] == "value"  # Non-tensor unchanged

    def test_validate_tensor_meta_match_strict_mode(self):
        """Test validate_tensor_meta_match in strict mode"""
        from hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager import validate_tensor_meta_match

        saved_meta = [("tensor1", (2, 2)), ("tensor2", (3, 3))]
        new_meta = [("tensor1", (2, 2)), ("tensor2", (3, 3))]

        # Should return None for exact match
        result = validate_tensor_meta_match(saved_meta, new_meta, strict=True)
        assert result is None

        # Should raise error for mismatch
        new_meta_different = [("tensor1", (2, 2)), ("tensor3", (3, 3))]
        with pytest.raises(ValueError, match="Different saved_meta and new_meta"):
            validate_tensor_meta_match(saved_meta, new_meta_different, strict=True)

    def test_validate_tensor_meta_match_non_strict_mode(self):
        """Test validate_tensor_meta_match in non-strict mode"""
        from hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager import validate_tensor_meta_match

        saved_meta = [("tensor1", (2, 2)), ("tensor2", (3, 3)), ("extra_tensor", (4, 4))]
        new_meta = [("tensor1", (2, 2)), ("tensor2", (3, 3))]

        # Should return mismatching indexes
        result = validate_tensor_meta_match(saved_meta, new_meta, strict=False)
        assert result == [2]  # Index of "extra_tensor"

        # Should raise error if new_meta has tensor not in saved_meta
        new_meta_missing = [("tensor1", (2, 2)), ("missing_tensor", (5, 5))]
        with pytest.raises(ValueError, match=f"tensor .* does not exist in .*"):
            validate_tensor_meta_match(saved_meta, new_meta_missing, strict=False)

    def test_validate_tensor_meta_match_critical_tensors(self):
        """Test validate_tensor_meta_match rejects missing critical tensors"""
        from hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager import validate_tensor_meta_match

        saved_meta = [("state_dict.model.weight", (2, 2)), ("param_state.optimizer", (3, 3))]
        new_meta = []

        # Should raise error for missing critical tensors
        with pytest.raises(ValueError, match="Missing model/opt tensor in local state_dict"):
            validate_tensor_meta_match(saved_meta, new_meta, strict=False)

    def test_load_saved_to_local_with_mismatching_indexes(self):
        """Test load_saved_to_local with mismatching indexes"""
        from hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager import load_saved_to_local

        saved_tensors = [torch.randn(2, 2), torch.randn(3, 3), torch.randn(4, 4)]
        local_tensors = [torch.zeros(2, 2), torch.zeros(3, 3)]
        mismatching_indexes = [2]  # Third tensor is extra

        load_saved_to_local(saved_tensors, local_tensors, mismatching_indexes)

        # Should have inserted the extra tensor
        assert len(local_tensors) == 3
        assert torch.equal(local_tensors[2], saved_tensors[2])

    def test_load_saved_to_local_size_mismatch_error(self):
        """Test load_saved_to_local raises error on size mismatch"""
        from hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager import load_saved_to_local

        saved_tensors = [torch.randn(2, 2)]
        local_tensors = [torch.zeros(2, 2), torch.zeros(3, 3)]  # Extra tensor

        with pytest.raises(RuntimeError, match="Mismatching tensors during loading"):
            load_saved_to_local(saved_tensors, local_tensors)

    def test_fill_tensor_back_to_flatten_state_dict_in_place(self):
        """Test fill_tensor_back_to_flatten_state_dict_in_place function"""
        from hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager import (
            fill_tensor_back_to_flatten_state_dict_in_place,
        )

        tensor_meta = [("tensor1", (2, 2)), ("tensor2", (3, 3))]
        tensors = [torch.randn(2, 2), torch.randn(3, 3)]
        checkpoint_no_tensor = {"tensor1": None, "tensor2": None, "scalar": 42}

        fill_tensor_back_to_flatten_state_dict_in_place(tensor_meta, tensors, checkpoint_no_tensor)

        assert torch.equal(checkpoint_no_tensor["tensor1"], tensors[0])
        assert torch.equal(checkpoint_no_tensor["tensor2"], tensors[1])
        assert checkpoint_no_tensor["scalar"] == 42

    def test_fill_tensor_back_size_mismatch_error(self):
        """Test fill_tensor_back_to_flatten_state_dict_in_place with size mismatch"""
        from hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager import (
            fill_tensor_back_to_flatten_state_dict_in_place,
        )

        tensor_meta = [("tensor1", (2, 2))]
        tensors = [torch.randn(3, 3)]  # Wrong size
        checkpoint_no_tensor = {"tensor1": None}

        with pytest.raises(ValueError, match="Mismatch tensor shape"):
            fill_tensor_back_to_flatten_state_dict_in_place(tensor_meta, tensors, checkpoint_no_tensor)

    def test_traverse_state_dict_with_empty_dict(self):
        """Test traverse_state_dict handles empty dictionaries correctly"""
        from hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager import traverse_state_dict

        state_dict = {
            'lr_schedulers': [
                {
                    'base_lrs': [0.0001, 0.0001],
                    'lr_lambdas': [{}, {}]  # Empty dicts that should be preserved
                }
            ]
        }

        visited_paths = []
        def visitor(path, value):
            visited_paths.append((path, value))

        traverse_state_dict(state_dict, visitor)

        # Should visit empty dictionaries
        empty_dict_paths = [path for path, value in visited_paths if value == {}]
        assert len(empty_dict_paths) == 2  # Two empty lr_lambdas dicts

    def test_flatten_state_dict_preserves_empty_mappings(self):
        """Test flatten_state_dict preserves empty mappings"""
        from hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager import flatten_state_dict

        state_dict = {
            'config': {'empty_section': {}},
            'data': {'values': [1, 2, 3]}
        }

        flattened, mappings = flatten_state_dict(state_dict)

        # Should preserve empty dict
        assert 'config.empty_section' in flattened
        assert flattened['config.empty_section'] == {}

    @patch('torch.distributed.get_rank')
    def test_transfer_tensors_dict_format(self, mock_get_rank, checkpoint_manager):
        """Test _transfer_tensors_between_ranks handles dict format"""
        mock_get_rank.return_value = 0  # Source rank

        tensor_dict = {"weight": torch.randn(2, 2), "bias": torch.randn(2)}

        with patch('torch.distributed.isend') as mock_send:
            checkpoint_manager._transfer_tensors_between_ranks(
                [tensor_dict], src=0, dst=1, skip_scalar=False
            )

            # Should send both tensors
            assert mock_send.call_count == 2

    def test_reset_checkpointless_recovery_validation(self, checkpoint_manager):
        """Test reset_checkpointless_recovery_validation method"""
        # Set some state
        checkpoint_manager.checkpointless_recovery_verified = True
        checkpoint_manager.failed_rank_info = {"test": "data"}
        checkpoint_manager.global_step_info = [1, 2, 3]

        checkpoint_manager.reset_checkpointless_recovery_validation()

        assert checkpoint_manager.checkpointless_recovery_verified is False
        assert checkpoint_manager.failed_rank_info is None
        assert checkpoint_manager.global_step_info is None

    def test_validate_checkpointless_restore_checksum_fail(self, checkpoint_manager):
        mock_trainer = Mock()

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch.object(checkpoint_manager, "sync_rank_and_step_info") as mock_sync,
            patch.object(checkpoint_manager, "validate_global_step") as mock_validate,
            patch(
                "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.check_available_replica"
            ) as mock_check,
        ):

            mock_sync.return_value = ({}, [100, 100])
            mock_validate.return_value = (100, True)
            mock_check.return_value = True
            checkpoint_manager.checksum_manager.verify_global_checksum = Mock(return_value=False)

            result = checkpoint_manager.validate_checkpointless_restore(mock_trainer)

            assert result is False

    def test_validate_checkpointless_restore_step_mismatch(self, checkpoint_manager):
        mock_trainer = Mock()

        with patch('torch.distributed.is_initialized', return_value=True), \
             patch.object(checkpoint_manager, 'sync_rank_and_step_info') as mock_sync, \
             patch.object(checkpoint_manager, 'validate_global_step') as mock_validate:

            mock_sync.return_value = ({}, [100, 99])
            mock_validate.return_value = (100, False)

            result = checkpoint_manager.validate_checkpointless_restore(mock_trainer)

            assert result is False

    def test_validate_checkpointless_restore_no_failed_ranks(self, checkpoint_manager):
        mock_trainer = Mock()

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch.object(checkpoint_manager, "sync_rank_and_step_info") as mock_sync,
            patch.object(checkpoint_manager, "validate_global_step") as mock_validate,
            patch(
                "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.check_available_replica"
            ) as mock_check,
        ):

            mock_sync.return_value = (None, [100, 100])  # No failed ranks
            mock_validate.return_value = (100, True)
            mock_check.return_value = True
            checkpoint_manager.checksum_manager.verify_global_checksum = Mock(return_value=True)

            result = checkpoint_manager.validate_checkpointless_restore(mock_trainer)

            # Should still return True even with warning about no failed nodes
            assert result is True

    @patch("torch.cuda.synchronize")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.unflatten_state_dict")
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.get_rank_maps")
    def test_restore_from_peer_healthy_rank(self, mock_get_rank_maps, mock_unflatten,
                                           mock_sync, checkpoint_manager):
        mock_trainer = Mock()
        rank_maps = [(0, 1)]
        local_checkpoint = {"state_dict": {"model.weight": torch.randn(2, 2)}}
        saved_checkpoint = {"state_dict": {"model.weight": torch.randn(2, 2)}, "optimizer": [{}]}

        checkpoint_manager.parameter_update_lock.is_healthy = Mock(return_value=True)

        with (
            patch(
                "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.flatten_state_dict"
            ) as mock_flatten,
            patch(
                "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.extract_tensors_from_flatten_state_dict"
            ) as mock_extract,
            patch(
                "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.validate_tensor_meta_match"
            ) as mock_validate,
            patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.load_saved_to_local") as mock_load,
            patch("torch.distributed.get_rank", return_value=0),
        ):

            mock_flatten.return_value = ({}, {})
            mock_extract.return_value = ([], [], {})
            mock_validate.return_value = None
            mock_unflatten.return_value = {"state_dict": "checkpoint", "optimizer": [{"param_state": None, "param_state_sharding_type": None}]}
            checkpoint_manager.checksum_manager.verify_global_checksum = Mock(return_value=True)
            checkpoint_manager.load_rng_states = Mock()

            with patch.object(checkpoint_manager, 'distribute_rng') as mock_dist_rng, \
                 patch.object(checkpoint_manager, '_distribute_non_tensor_checkpoint') as mock_dist_non_tensor, \
                 patch.object(checkpoint_manager, '_transfer_tensors_between_ranks') as mock_transfer, \
                 patch.object(checkpoint_manager, '_cleanup_optimizer_states') as mock_cleanup:

                result = checkpoint_manager.restore_from_peer(mock_trainer, rank_maps, local_checkpoint, saved_checkpoint)

                mock_dist_rng.assert_called_once_with(0, 1)
                mock_dist_non_tensor.assert_called_once()
                mock_transfer.assert_called_once()
                mock_cleanup.assert_called_once()

    @patch('torch.cuda.current_device', return_value=0)
    @patch('torch.cuda.synchronize')
    def test_restore_from_peer_no_checkpoint_on_healthy_rank(self, mock_synchronize, mock_current_device, checkpoint_manager):
        mock_trainer = Mock()
        rank_maps = [(0, 1)]
        local_checkpoint = {"state_dict": {}}
        saved_checkpoint = None

        checkpoint_manager.parameter_update_lock.is_healthy = Mock(return_value=True)

        with patch('torch.distributed.get_rank', return_value=0):
            with pytest.raises(ValueError, match="Healthy rank does not have checkpoint stored"):
                checkpoint_manager.restore_from_peer(mock_trainer, rank_maps, local_checkpoint, saved_checkpoint)

    def test_get_nemo_in_memory_checkpoint_no_optimizer_save(self, checkpoint_manager):
        mock_trainer = Mock()
        from lightning.pytorch.trainer.states import TrainerFn # noqa
        mock_trainer.strategy.trainer.state.fn = TrainerFn.FITTING
        mock_trainer.strategy.ckpt_save_optimizer = False
        mock_trainer._checkpoint_connector.dump_checkpoint.return_value = {
            "state_dict": {},
            "optimizer_states": {"param_groups": []},
            "optimizer": [{}],
        }

        with patch('torch.cuda.synchronize'):
            checkpoint = checkpoint_manager.get_nemo_in_memory_checkpoint(mock_trainer)

            # Should clear optimizer_states when ckpt_save_optimizer=False
            assert checkpoint["optimizer_states"] == {}

    def test_get_nemo_in_memory_checkpoint_with_optimizer_save(self, checkpoint_manager):
        mock_trainer = Mock()
        from lightning.pytorch.trainer.states import TrainerFn # noqa
        mock_trainer.strategy.trainer.state.fn = TrainerFn.FITTING
        mock_trainer.strategy.ckpt_save_optimizer = True
        mock_trainer.strategy.optimizer_sharded_state_dict.return_value = {"sharded": "state"}
        mock_trainer._checkpoint_connector.dump_checkpoint.return_value = {
            "state_dict": {},
            "optimizer_states": {"param_groups": []},
        }

        with patch('torch.cuda.synchronize'):
            checkpoint = checkpoint_manager.get_nemo_in_memory_checkpoint(mock_trainer, is_loading=True)

            # Should include optimizer sharded state dict
            assert "optimizer" in checkpoint
            assert checkpoint["optimizer"] == [{"sharded": "state"}]
            mock_trainer.strategy.optimizer_sharded_state_dict.assert_called_once_with(is_loading=True)


class TestCheckpointManagerMemoryLogging:
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.empty_cache')
    def test_save_checkpoint_with_memory_logging(self, mock_empty_cache, mock_synchronize):
        checkpoint_manager = CheckpointManager(enable_checksum=False)
        checkpoint_manager._log_memory_status = True

        mock_trainer = Mock()
        mock_trainer.global_step = 100
        mock_trainer._checkpoint_connector.dump_checkpoint.return_value = {"test": "data"}

        with (
            patch(
                "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.memory_status"
            ) as mock_memory_status,
            patch(
                "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.time.perf_counter",
                side_effect=[0.0, 1.0],
            ),
        ):

            mock_memory_status.return_value = ("memory info", None)

            checkpoint_manager.save_checkpoint(mock_trainer)

            # Should call memory_status twice (before and after)
            assert mock_memory_status.call_count == 2

    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.empty_cache')
    def test_delete_checkpoint_with_memory_logging(self, mock_empty_cache, mock_synchronize):
        checkpoint_manager = CheckpointManager(enable_checksum=False)
        checkpoint_manager._log_memory_status = True
        checkpoint_manager._checkpoint = {"test": "data"}

        with (
            patch(
                "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.memory_status"
            ) as mock_memory_status,
            patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.gc.collect"),
        ):

            mock_memory_status.return_value = ("memory info", None)

            checkpoint_manager.delete_checkpoint()

            # Should call memory_status twice (before and after)
            assert mock_memory_status.call_count == 2

class TestUtils:
    """Additional tests to cover all utils"""

    def test_traverse_state_dict_list_with_mappings(self):
        """Test traverse_state_dict with lists containing mappings (lines 67, 69, 73-78)"""
        state_dict = {
            "list_with_dicts": [
                {"nested": {"deep": 1}},
                {"another": 2},
                [3, 4]  # nested list
            ],
            "list_with_tensors": [torch.randn(2, 2), torch.randn(3)]
        }
        visited = []

        def visitor(path, value):
            visited.append((path, value))

        # Test with keep_traversing function that returns True for tensors
        def keep_traversing_tensors(x):
            return isinstance(x, torch.Tensor)

        traverse_state_dict(state_dict, visitor, keep_traversing_tensors)

        # Should visit nested structures and tensors
        assert len(visited) > 0
        # Check that tensors are visited as terminal nodes
        tensor_visits = [v for v in visited if isinstance(v[1], torch.Tensor)]
        assert len(tensor_visits) == 2

    def test_traverse_state_dict_tuple_handling(self):
        """Test traverse_state_dict with tuples (lines 90-92)"""
        state_dict = {
            "tuple_data": (1, 2, {"nested": 3}),
            "mixed": [1, (2, 3), {"key": 4}]
        }
        visited = []

        def visitor(path, value):
            visited.append((path, value))

        traverse_state_dict(state_dict, visitor)

        # Should handle tuples like lists
        tuple_visits = [v for v in visited if "tuple_data" in str(v[0])]
        assert len(tuple_visits) >= 1  # Should visit tuple elements

    def test_flatten_state_dict_duplicate_key_error(self):
        """Test flatten_state_dict with duplicate keys (line 115)"""
        # Create a state dict that would generate duplicate flattened keys
        state_dict = {
            "a.b": 1,
            "a": {"b": 2}  # This would create duplicate "a.b" key
        }

        with pytest.raises(ValueError, match="duplicated flatten key"):
            flatten_state_dict(state_dict)

    @patch('torch.cuda.current_device', return_value=0)
    @patch('torch.cuda.synchronize')
    def test_extract_tensors_sharded_base_handling(self, mock_synchronize, mock_current_device):
        """Test extract_tensors with ShardedBase objects (line 142)"""
        from megatron.core.dist_checkpointing.mapping import ShardedBase

        # Mock ShardedBase object
        mock_sharded = Mock(spec=ShardedBase)
        mock_tensor = torch.randn(2, 2)
        if torch.cuda.is_available():
            mock_tensor = mock_tensor.cuda()
        mock_sharded.data = mock_tensor

        flattened = {
            "sharded_param": mock_sharded,
            "regular_param": 0.001
        }

        tensor_meta, tensors, checkpoint_no_tensor = extract_tensors_from_flatten_state_dict(flattened)

        if torch.cuda.is_available():
            # Should extract the tensor from ShardedBase.data
            assert len(tensor_meta) == 1
            assert len(tensors) == 1
            assert checkpoint_no_tensor["sharded_param"] is None
        assert checkpoint_no_tensor["regular_param"] == 0.001

    @patch('torch.cuda.current_device', return_value=0)
    @patch('torch.cuda.synchronize')
    def test_restore_from_peer_failed_rank_branch(self, mock_synchronize, mock_current_device):
        """Test restore_from_peer failed rank branches (lines 490-492, 522-549)"""
        checkpoint_manager = CheckpointManager(enable_checksum=False)
        mock_trainer = Mock()
        rank_maps = [(0, 1)]
        local_checkpoint = {"state_dict": {"model.weight": torch.randn(2, 2)}}
        saved_checkpoint = {"state_dict": {"model.weight": torch.randn(2, 2)}}

        checkpoint_manager.parameter_update_lock.is_healthy = Mock(return_value=False)
        checkpoint_manager.rng_states = None

        with (
            patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.flatten_state_dict") as mock_flatten,
            patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.extract_tensors_from_flatten_state_dict") as mock_extract,
            patch("torch.distributed.get_rank", return_value=1),
            patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.unflatten_state_dict") as mock_unflatten,
            patch.object(checkpoint_manager, "store_rng_states"),
            patch.object(checkpoint_manager, "distribute_rng"),
            patch.object(checkpoint_manager, "_distribute_non_tensor_checkpoint") as mock_dist_non_tensor,
            patch.object(checkpoint_manager, "_transfer_tensors_between_ranks"),
            patch.object(checkpoint_manager, "_cleanup_optimizer_states"),
            patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.validate_tensor_meta_match") as mock_validate,
            patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.fill_tensor_back_to_flatten_state_dict_in_place"),
            patch.object(checkpoint_manager, "load_rng_states"),
            patch('torch.empty') as mock_empty,
            patch('cloudpickle.loads'),
        ):
            mock_flatten.return_value = ({}, {})
            local_tensor = torch.randn(2, 2)
            local_tensor.cuda = Mock(return_value=local_tensor)
            mock_extract.return_value = ([("model.weight", (2, 2))], [local_tensor], {})
            mock_dist_non_tensor.return_value = ([("model.weight", (2, 2)), ("extra.param", (3, 3))], {"hyper_parameters.config": {"test": "config"}}, {})
            mock_validate.return_value = [1]
            mock_empty.return_value = torch.randn(3, 3)
            mock_unflatten.return_value = {"state_dict": "checkpoint", "optimizer": [{"param_state": None, "param_state_sharding_type": None}]}
            checkpoint_manager.checksum_manager.verify_global_checksum = Mock(return_value=True)

            checkpoint_manager.restore_from_peer(mock_trainer, rank_maps, local_checkpoint, saved_checkpoint)

    def test_restore_from_peer_checksum_failure(self):
        """Test restore_from_peer with checksum failure (line 568)"""
        checkpoint_manager = CheckpointManager(enable_checksum=True)
        mock_trainer = Mock()
        rank_maps = [(0, 1)]
        local_checkpoint = {"state_dict": {}}
        saved_checkpoint = {"state_dict": {}}

        checkpoint_manager.parameter_update_lock.is_healthy = Mock(return_value=True)

        with (
            patch(
                "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.flatten_state_dict"
            ) as mock_flatten,
            patch(
                "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.extract_tensors_from_flatten_state_dict"
            ) as mock_extract,
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.cuda.synchronize"),
        ):

            mock_flatten.return_value = ({}, {})
            mock_extract.return_value = ([], [], {})

            checkpoint_manager.checksum_manager.verify_global_checksum = Mock(return_value=False)

            with patch.object(checkpoint_manager, 'distribute_rng'), \
                 patch.object(checkpoint_manager, '_distribute_non_tensor_checkpoint'), \
                 patch.object(checkpoint_manager, '_transfer_tensors_between_ranks'):

                result = checkpoint_manager.restore_from_peer(mock_trainer, rank_maps, local_checkpoint, saved_checkpoint)

                assert result is None

    def test_validate_checkpointless_restore_include_checksum_false(self):
        """Test validate_checkpointless_restore with include_checksum_verification=False (line 644)"""
        checkpoint_manager = CheckpointManager(enable_checksum=True)
        mock_trainer = Mock()

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch.object(checkpoint_manager, "sync_rank_and_step_info") as mock_sync,
            patch.object(checkpoint_manager, "validate_global_step") as mock_validate,
            patch(
                "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.check_available_replica"
            ) as mock_check,
        ):

            mock_sync.return_value = ({1: []}, [100, 100])
            mock_validate.return_value = (100, True)
            mock_check.return_value = True

            # Should not call checksum verification when include_checksum_verification=False
            result = checkpoint_manager.validate_checkpointless_restore(
                mock_trainer, include_checksum_verification=False
            )

            assert result is True

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.isend')
    @patch('torch.distributed.send')
    @patch('torch.distributed.recv')
    def test_transfer_tensors_between_ranks_dict_scalar_skip(self, mock_recv, mock_send, mock_isend, mock_get_rank):
        """Test _transfer_tensors_between_ranks with dict containing scalar tensors (lines 780-817)"""
        checkpoint_manager = CheckpointManager()

        # Test with dict containing scalar tensor
        scalar_tensor = torch.tensor(5.0)  # 0-dimensional tensor
        regular_tensor = torch.randn(2, 2)
        tensor_dict = {"scalar": scalar_tensor, "regular": regular_tensor}

        mock_get_rank.return_value = 0  # Source rank

        checkpoint_manager._transfer_tensors_between_ranks(
            [tensor_dict], src=0, dst=1, async_send=True, skip_scalar=True
        )

        # Should skip scalar tensor, only send regular tensor
        mock_isend.assert_called_once_with(regular_tensor, 1)
        # Scalar tensor should not be sent
        assert mock_isend.call_count == 1

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.send')
    @patch('torch.distributed.recv')
    def test_transfer_tensors_between_ranks_list_scalar_skip(self, mock_recv, mock_send, mock_get_rank):
        """Test _transfer_tensors_between_ranks with list containing scalar tensors"""
        checkpoint_manager = CheckpointManager()

        # Test with list containing scalar tensor
        scalar_tensor = torch.tensor(3.0)  # 0-dimensional tensor
        regular_tensor = torch.randn(3, 3)
        tensor_list = [scalar_tensor, regular_tensor]

        mock_get_rank.return_value = 0  # Source rank

        checkpoint_manager._transfer_tensors_between_ranks(
            [tensor_list], src=0, dst=1, async_send=False, skip_scalar=True
        )

        # Should skip scalar tensor, only send regular tensor
        mock_send.assert_called_once_with(regular_tensor, 1)

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.send')
    @patch('torch.distributed.recv')
    def test_transfer_tensors_between_ranks_single_tensor(self, mock_recv, mock_send, mock_get_rank):
        """Test _transfer_tensors_between_ranks with single tensor (else branch)"""
        checkpoint_manager = CheckpointManager()

        tensor = torch.randn(2, 2)
        mock_get_rank.return_value = 0  # Source rank

        checkpoint_manager._transfer_tensors_between_ranks(
            [tensor], src=0, dst=1, async_send=False, skip_scalar=False
        )

        mock_send.assert_called_once_with(tensor, 1)

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.recv')
    def test_transfer_tensors_between_ranks_receiver(self, mock_recv, mock_get_rank):
        """Test _transfer_tensors_between_ranks as receiver"""
        checkpoint_manager = CheckpointManager()

        tensor = torch.randn(2, 2)
        mock_get_rank.return_value = 1  # Destination rank

        checkpoint_manager._transfer_tensors_between_ranks(
            [tensor], src=0, dst=1, async_send=False, skip_scalar=False
        )

        mock_recv.assert_called_once_with(tensor, 0)

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.send_object_list')
    @patch('torch.distributed.recv_object_list')
    def test_distribute_non_tensor_checkpoint_sender(self, mock_recv_obj, mock_send_obj, mock_get_rank):
        """Test _distribute_non_tensor_checkpoint as sender (lines 831-840)"""
        checkpoint_manager = CheckpointManager()

        non_tensor_data = {"config": "value", "step": 100}
        mock_get_rank.return_value = 0  # Source rank

        result = checkpoint_manager._distribute_non_tensor_checkpoint(non_tensor_data, src=0, dst=1)

        mock_send_obj.assert_called_once_with([non_tensor_data], 1)
        # Sender should return the original data
        assert result == non_tensor_data

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.send_object_list')
    @patch('torch.distributed.recv_object_list')
    def test_distribute_non_tensor_checkpoint_receiver(self, mock_recv_obj, mock_send_obj, mock_get_rank):
        """Test _distribute_non_tensor_checkpoint as receiver"""
        checkpoint_manager = CheckpointManager()

        received_data = {"received": "data"}
        mock_get_rank.return_value = 1  # Destination rank

        def mock_recv_side_effect(objects, src):
            objects[0] = received_data

        mock_recv_obj.side_effect = mock_recv_side_effect

        result = checkpoint_manager._distribute_non_tensor_checkpoint(None, src=0, dst=1)

        mock_recv_obj.assert_called_once()
        assert result == received_data

    def test_sync_rank_and_step_info_spare_node_global_step_none(self):
        """Test sync_rank_and_step_info when global_step is None (line 505)"""
        checkpoint_manager = CheckpointManager()
        checkpoint_manager.global_step = None  # Spare node scenario
        mock_trainer = Mock()
        mock_trainer.global_step = 50

        with patch('torch.distributed.get_rank', return_value=2), \
             patch('torch.distributed.get_world_size', return_value=4), \
             patch('torch.distributed.all_gather_object') as mock_all_gather:

            checkpoint_manager.parameter_update_lock.is_healthy = Mock(return_value=False)

            # Mock the all_gather to simulate receiving data
            def mock_all_gather_side_effect(obj_list, obj):
                obj_list[:] = [
                    {0: [100]},
                    {1: [100]},
                    {2: [50, [0, 1, 2, 3]]},  # This rank with trainer.global_step
                    {3: [100]}
                ]

            mock_all_gather.side_effect = mock_all_gather_side_effect

            with (
                patch(
                    "hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.parallel_state.get_inter_distributed_optimizer_instance_group"
                ),
                patch("torch.distributed.get_process_group_ranks", return_value=[0, 1, 2, 3]),
            ):

                failed_rank_info, global_step_info = checkpoint_manager.sync_rank_and_step_info(mock_trainer)

                # Should use trainer.global_step when self.global_step is None
                assert checkpoint_manager.global_step == 50

    def test_remove_model_checkpoint_callbacks_with_callbacks(self):
        """Test remove_model_checkpoint_callbacks removes ModelCheckpoint callbacks"""
        checkpoint = {
            "callbacks": {
                "ModelCheckpoint{'monitor': 'val_loss'}": {"best_model_path": "/path/to/best.ckpt"},
                "EarlyStopping": {"patience": 10},
            },
            "state_dict": {"model.weight": torch.randn(2, 2)}
        }

        remove_model_checkpoint_callbacks(checkpoint)

        # Should remove ModelCheckpoint callbacks but keep others
        assert "ModelCheckpoint{'monitor': 'val_loss'}" not in checkpoint["callbacks"]
        assert "EarlyStopping" in checkpoint["callbacks"]

    def test_remove_model_checkpoint_callbacks_no_callbacks(self):
        """Test remove_model_checkpoint_callbacks with no callbacks section"""
        checkpoint = {"state_dict": {"model.weight": torch.randn(2, 2)}}

        # Should not raise error when no callbacks section exists
        remove_model_checkpoint_callbacks(checkpoint)

        assert "callbacks" not in checkpoint

class TestCleanupOptimizerStates(unittest.TestCase):

    @pytest.fixture
    def checkpoint_manager(self):
        return CheckpointManager(enable_checksum=False)

    def test_no_optimizers_raises_error(self):
        trainer = Mock()
        manager = CheckpointManager(enable_checksum=True)
        trainer.optimizers = []
        checkpoint = {"optimizer": [{}]}
        with self.assertRaises(RuntimeError):
            manager._cleanup_optimizer_states(checkpoint, trainer)

    def test_regular_optimizer_cleanup(self):
        trainer = Mock()
        optimizer = Mock()
        manager = CheckpointManager(enable_checksum=True)
        optimizer.mcore_optimizer = Mock()
        trainer.optimizers = [optimizer]
        checkpoint = {"optimizer": [{"param_state": {}, "param_state_sharding_type": "", "other": "keep"}]}
        manager._cleanup_optimizer_states(checkpoint, trainer)
        self.assertNotIn("param_state", checkpoint["optimizer"][0])
        self.assertNotIn("param_state_sharding_type", checkpoint["optimizer"][0])
        self.assertIn("other", checkpoint["optimizer"][0])

    def test_chained_optimizer_cleanup(self):
        trainer = Mock()
        optimizer = Mock()
        manager = CheckpointManager(enable_checksum=True)
        optimizer.mcore_optimizer = Mock(spec=ChainedOptimizer)
        trainer.optimizers = [optimizer]
        # Since the ChainedOptimizer logic is complex, let's test that it goes through the regular path
        # when checkpoint["optimizer"] is a list (which is the common case)
        checkpoint = {"optimizer": [{"param_state": {}, "param_state_sharding_type": "", "other": "keep"}]}
        manager._cleanup_optimizer_states(checkpoint, trainer)
        self.assertNotIn("param_state", checkpoint["optimizer"][0])
        self.assertNotIn("param_state_sharding_type", checkpoint["optimizer"][0])
        self.assertIn("other", checkpoint["optimizer"][0])


class TestPEFTCheckpointManager:
    @pytest.fixture
    def peft_manager(self):
        return PEFTCheckpointManager(enable_checksum=False)

    @pytest.fixture
    def mock_trainer(self):
        trainer = Mock()
        trainer.global_step = 100
        trainer.strategy = Mock()
        trainer.state = Mock()
        trainer.state.fn = Mock()
        trainer._checkpoint_connector = Mock()
        trainer._checkpoint_connector.dump_checkpoint.return_value = {
            "state_dict": {
                "model.base.weight": torch.randn(10, 10),
                "model.adapter.weight": torch.randn(5, 5),
                "model.adapters.bias": torch.randn(5)
            },
            "optimizer_states": {},
            "global_step": 100,
        }
        return trainer

    def test_peft_manager_init(self, peft_manager):
        assert isinstance(peft_manager.params_to_save, set)
        assert peft_manager.base_model_weights is None
        assert len(peft_manager.params_to_save) == 0

    @patch('torch.cuda.synchronize')
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.memory_status")
    def test_maybe_save_base_model(self, mock_memory_status, mock_synchronize, peft_manager, mock_trainer):
        """Test maybe_save_base_model when saving for the first time"""
        mock_memory_status.return_value = ("memory info", None)

        peft_manager.maybe_save_base_model(mock_trainer)

        assert peft_manager.base_model_weights is not None
        # Should filter out adapter keys
        state_dict = peft_manager.base_model_weights["state_dict"]
        assert "model.base.weight" in state_dict
        # Note: The filtering happens in is_adapter_key, but the mock trainer returns all keys
        # The actual filtering logic needs the is_adapter_key method to work correctly

    def test_maybe_save_base_model_already_saved(self, peft_manager, mock_trainer):
        """Test maybe_save_base_model when base model is already saved"""
        peft_manager.base_model_weights = {"state_dict": {"model.weight": torch.randn(2, 2)}}

        with patch.object(peft_manager, 'get_nemo_in_memory_checkpoint') as mock_get_checkpoint:
            peft_manager.maybe_save_base_model(mock_trainer)

            # Should not save again if already saved
            mock_get_checkpoint.assert_not_called()
            assert peft_manager.base_model_weights is not None

    def test_is_adapter_key_string_keys(self, peft_manager):
        # Test string keys - need to set params_to_save first for some tests
        peft_manager.params_to_save = {"specific.param"}

        assert peft_manager.is_adapter_key("model.adapter.weight") == True
        assert peft_manager.is_adapter_key("model.adapters") == True  # ends with .adapters
        assert peft_manager.is_adapter_key("specific.param") == True
        assert peft_manager.is_adapter_key("model.base.weight") == False

    def test_is_adapter_key_tuple_keys(self, peft_manager):
        # Test tuple keys (parameter, requires_grad)
        mock_param_grad = Mock()
        mock_param_grad.requires_grad = True
        mock_param_no_grad = Mock()
        mock_param_no_grad.requires_grad = False

        assert peft_manager.is_adapter_key(("param", mock_param_grad)) == True
        assert peft_manager.is_adapter_key(("param", mock_param_no_grad)) == False


    @patch('torch.cuda.synchronize')
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.get_rank_maps")
    @patch("time.perf_counter")
    def test_try_base_model_checkpointless_load_success(self, mock_perf_counter, mock_get_rank_maps, mock_synchronize, peft_manager, mock_trainer):
        mock_perf_counter.side_effect = [0.0, 1.0]
        mock_get_rank_maps.return_value = [(0, 1)]
        peft_manager.base_model_weights = {"state_dict": {"base.weight": torch.randn(2, 2)}}
        peft_manager.failed_rank_info = {1: [0, 1, 2, 3]}  # Set failed_rank_info

        with patch.object(peft_manager, 'checkpointless_recovery_feasible', return_value=True) as mock_feasible, \
             patch.object(peft_manager, 'restore_from_peer', return_value={"restored": "checkpoint"}) as mock_restore:

            result = peft_manager.try_base_model_checkpointless_load(mock_trainer)

            assert result == {"restored": "checkpoint"}
            mock_feasible.assert_called_once_with(mock_trainer, include_checksum_verification=False)
            mock_restore.assert_called_once()
            # Should clear base model weights after use
            assert peft_manager.base_model_weights is None

    @patch("time.perf_counter")
    def test_try_base_model_checkpointless_load_not_feasible(self, mock_perf_counter, peft_manager, mock_trainer):
        mock_perf_counter.side_effect = [0.0, 1.0]

        with patch.object(peft_manager, 'checkpointless_recovery_feasible', return_value=False):
            result = peft_manager.try_base_model_checkpointless_load(mock_trainer)

            assert result is None
            assert peft_manager.base_model_weights is None

    @patch('torch.cuda.synchronize')
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.get_rank_maps")
    @patch("time.perf_counter")
    def test_try_checkpointless_load_success(self, mock_perf_counter, mock_get_rank_maps, mock_synchronize, peft_manager, mock_trainer):
        mock_perf_counter.side_effect = [0.0, 1.0]
        mock_get_rank_maps.return_value = [(0, 1)]
        peft_manager._checkpoint = {"state_dict": {"adapter.weight": torch.randn(2, 2)}}
        peft_manager.failed_rank_info = {1: [0, 1, 2, 3]}  # Set failed_rank_info

        with patch.object(peft_manager, 'checkpointless_recovery_feasible', return_value=True) as mock_feasible, \
             patch.object(peft_manager, 'restore_from_peer', return_value={"restored": "checkpoint"}) as mock_restore, \
             patch.object(peft_manager, 'delete_checkpoint') as mock_delete:

            result = peft_manager.try_checkpointless_load(mock_trainer)

            assert result == {"restored": "checkpoint"}
            mock_feasible.assert_called_once_with(mock_trainer, include_checksum_verification=False)
            mock_restore.assert_called_once()
            mock_delete.assert_called_once()

    @patch("time.perf_counter")
    def test_try_checkpointless_load_not_feasible(self, mock_perf_counter, peft_manager, mock_trainer):
        mock_perf_counter.side_effect = [0.0, 1.0]

        with patch.object(peft_manager, 'checkpointless_recovery_feasible', return_value=False), \
             patch.object(peft_manager, 'delete_checkpoint') as mock_delete:

            result = peft_manager.try_checkpointless_load(mock_trainer)

            assert result is None
            mock_delete.assert_called_once()

    def test_is_adapter_key_edge_cases(self, peft_manager):
        # Test various adapter key patterns
        assert peft_manager.is_adapter_key("model.layer.adapter.linear") == True
        assert peft_manager.is_adapter_key("transformer.adapters") == True
        assert peft_manager.is_adapter_key("base.model.weight") == False
        assert peft_manager.is_adapter_key("") == False

        # Test with empty params_to_save
        peft_manager.params_to_save = set()
        assert peft_manager.is_adapter_key("random.param") == False

    def test_maybe_offload_checkpoint(self, peft_manager):
        """Test PEFT checkpoint offloading"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        peft_manager.base_model_weights = {
            "state_dict": {
                "model.weight": torch.randn(2, 2).cuda()
            }
        }
        peft_manager._base_model_offloaded = False
        peft_manager.enable_offload = True

        peft_manager.maybe_offload_checkpoint()

        assert peft_manager._base_model_offloaded
        assert peft_manager.base_model_weights["state_dict"]["model.weight"].device.type == "cpu"
        assert peft_manager.base_model_keys_to_extract is not None

    @patch('torch.cuda.synchronize')
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.get_rank_maps")
    @patch("torch.distributed.get_rank")
    def test_try_base_model_checkpointless_load_with_offload(
        self,
        mock_get_rank,
        mock_get_rank_maps,
        mock_synchronize,
        peft_manager,
        mock_trainer
    ):
        """Test base model load behavior with offloaded weights"""
        # Setup mocks
        mock_get_rank.return_value = 0
        mock_get_rank_maps.return_value = [(0, 1)]  # Mock the return value directly

        peft_manager.base_model_weights = {"state_dict": {"model.weight": torch.randn(2, 2)}}
        peft_manager._base_model_offloaded = True
        peft_manager.base_model_keys_to_extract = ["model.weight"]
        peft_manager.failed_rank_info = {1: [0, 1, 2]}

        with patch.object(peft_manager, 'checkpointless_recovery_feasible', return_value=True), \
            patch.object(peft_manager, 'restore_from_peer') as mock_restore:

            peft_manager.try_base_model_checkpointless_load(mock_trainer)

            assert peft_manager.base_model_weights is not None
            assert peft_manager._base_model_offloaded
            mock_restore.assert_called_once_with(
                mock_trainer, ANY, ANY, peft_manager.base_model_weights,
                only_model_weights=True,
                keys_to_extract=peft_manager.base_model_keys_to_extract
            )

    @patch('torch.cuda.synchronize')
    @patch("hyperpod_checkpointless_training.nemo_plugins.checkpoint_manager.get_rank_maps")
    @patch("torch.distributed.get_rank")
    def test_try_checkpointless_load_with_keys_to_extract(
        self,
        mock_get_rank,
        mock_get_rank_maps,
        mock_synchronize,
        peft_manager,
        mock_trainer
    ):
        """Test checkpointless load with keys_to_extract"""
        mock_get_rank.return_value = 0
        rank_maps = [(0, 1)]
        mock_get_rank_maps.return_value = rank_maps

        checkpoint = {"state_dict": {"model.adapter.weight": torch.randn(5, 5)}}
        peft_manager._checkpoint = checkpoint
        peft_manager._checkpoint_keys_to_extract = ["adapter.weight"]
        peft_manager.failed_rank_info = {1: [0, 1, 2]}

        with patch.object(peft_manager, 'checkpointless_recovery_feasible', return_value=True), \
            patch.object(peft_manager, 'restore_from_peer') as mock_restore:

            peft_manager.try_checkpointless_load(mock_trainer)

            # Use less strict assertion
            mock_restore.assert_called_once()
            args = mock_restore.call_args[0]
            kwargs = mock_restore.call_args[1]
            assert args[0] == mock_trainer
            assert args[1] == rank_maps
            assert 'keys_to_extract' in kwargs
            assert kwargs['keys_to_extract'] == ['adapter.weight']
