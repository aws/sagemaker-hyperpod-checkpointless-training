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

# type: ignore
import unittest
from unittest.mock import Mock, patch, MagicMock
import datetime
import os


class TestMegatronParallelStateUtil(unittest.TestCase):

    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state", None)
    def test_no_megatron_dependency(self):
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        with self.assertRaises(ImportError) as context:
            util = MegatronParallelStateUtil()
        
        self.assertIn("Megatron-Core parallel_state is not available", str(context.exception))

    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.dist.get_rank")
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_is_tp_0(self, mock_parallel_state, mock_get_rank):
        """Test is_tp_0 method returns True when rank % tp_world_size == 0."""
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        util = MegatronParallelStateUtil()
        
        mock_get_rank.return_value = 0
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 2
        self.assertTrue(util.is_tp_0())
        
        mock_get_rank.return_value = 1
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 2
        self.assertFalse(util.is_tp_0())
        
        mock_get_rank.return_value = 3
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 4
        self.assertFalse(util.is_tp_0())

    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_create_model_parallel_group_cp_size_1(self, mock_parallel_state):
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        util = MegatronParallelStateUtil()
        mock_parallel_state.get_context_parallel_world_size.return_value = 1
        mock_group = Mock()
        mock_parallel_state.get_model_parallel_group.return_value = mock_group

        result = util.create_model_parallel_group()
        self.assertEqual(result, mock_group)

    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.dist.get_rank")
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_create_model_parallel_group_cp_size_greater_than_1(self, mock_parallel_state, mock_get_rank):
        """Test create_model_parallel_group when context_parallel_world_size > 1."""
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        util = MegatronParallelStateUtil()
        
        # Setup: CP > 1 triggers the conditional branch
        mock_parallel_state.get_context_parallel_world_size.return_value = 2
        mock_get_rank.return_value = 0
        
        # Mock the _create_model_parallel_group method
        mock_created_group = Mock()
        util._create_model_parallel_group = Mock(return_value=mock_created_group)
        
        result = util.create_model_parallel_group()
        
        # Verify _create_model_parallel_group was called
        util._create_model_parallel_group.assert_called_once()
        
        # Verify the result is the created group
        self.assertEqual(result, mock_created_group)
        
        # Verify model_parallel_group was set
        self.assertEqual(util.model_parallel_group, mock_created_group)

    @patch.dict(os.environ, {}, clear=True)
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_get_tp_cp_pp_dp_sizes(self, mock_parallel_state):
        """Test _get_tp_cp_pp_dp_sizes returns correct parallelism sizes."""
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        util = MegatronParallelStateUtil()
        
        # Setup mock return values
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 4
        mock_parallel_state.get_context_parallel_world_size.return_value = 2
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 3
        mock_parallel_state.get_data_parallel_world_size.return_value = 5
        
        tp_size, cp_size, pp_size, dp_size = util._get_tp_cp_pp_dp_sizes(mock_parallel_state)
        
        self.assertEqual(tp_size, 4)
        self.assertEqual(cp_size, 2)
        self.assertEqual(pp_size, 3)
        self.assertEqual(dp_size, 5)
        mock_parallel_state.get_data_parallel_world_size.assert_called_once_with(with_context_parallel=False)

    @patch.dict(os.environ, {"PARALLELISM_ORDER": "tp-cp-ep-dp-pp"}, clear=True)
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_get_rank_generator(self, mock_parallel_state):
        """Test _get_rank_generator creates RankGenerator with correct parameters."""
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        util = MegatronParallelStateUtil()
        
        mock_rank_generator = MagicMock()
        mock_parallel_state.RankGenerator.return_value = mock_rank_generator
        
        result = util._get_rank_generator(
            mock_parallel_state,
            tp_size=2,
            cp_size=1,
            pp_size=4,
            dp_size=3,
            ep_size=1,
            parallelism_order="tp-cp-ep-dp-pp"
        )
        
        mock_parallel_state.RankGenerator.assert_called_once_with(
            tp=2,
            ep=1,
            dp=3,
            pp=4,
            cp=1,
            order="tp-cp-ep-dp-pp",
            rank_offset=0
        )
        self.assertEqual(result, mock_rank_generator)

    @patch.dict(os.environ, {}, clear=True)
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_get_rank_generator_no_ep(self, mock_parallel_state):
        """Test _get_rank_generator_no_ep creates RankGenerator without EP."""
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        util = MegatronParallelStateUtil()
        
        # Setup parallelism sizes
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 2
        mock_parallel_state.get_context_parallel_world_size.return_value = 1
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 4
        mock_parallel_state.get_data_parallel_world_size.return_value = 3
        
        mock_rank_generator = MagicMock()
        mock_parallel_state.RankGenerator.return_value = mock_rank_generator
        
        result = util._get_rank_generator_no_ep(mock_parallel_state)
        
        mock_parallel_state.RankGenerator.assert_called_once_with(
            tp=2,
            ep=1,
            dp=3,
            pp=4,
            cp=1,
            order="tp-cp-ep-dp-pp",
            rank_offset=0
        )
        self.assertEqual(result, mock_rank_generator)

    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_get_all_group_ranks_from_group(self, mock_parallel_state):
        """Test _get_all_group_ranks_from_group extracts all ranks from a group."""
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        util = MegatronParallelStateUtil()
        
        mock_rank_generator = MagicMock()
        mock_rank_generator.get_ranks.return_value = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ]
        
        result = util._get_all_group_ranks_from_group(mock_rank_generator, "tp-pp-cp")
        
        mock_rank_generator.get_ranks.assert_called_once_with("tp-pp-cp")
        self.assertEqual(result, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.dist.new_group")
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_create_torch_distributed_group_with_retries(self, mock_parallel_state, mock_new_group):
        """Test _create_torch_distributed_group_with_retries creates distributed group."""
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        util = MegatronParallelStateUtil()
        
        mock_group = Mock()
        mock_new_group.return_value = mock_group
        
        ranks = [0, 1, 2, 3]
        backend = "gloo"
        timeout = datetime.timedelta(seconds=3600)
        
        result = util._create_torch_distributed_group_with_retries(ranks, backend, timeout)
        
        mock_new_group.assert_called_once_with(ranks, backend=backend, timeout=timeout)
        self.assertEqual(result, mock_group)

    @patch.dict(os.environ, {}, clear=True)
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.dist.get_rank")
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_get_model_parallel_ranks(self, mock_parallel_state, mock_get_rank):
        """Test get_model_parallel_ranks property returns correct rank groups."""
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        util = MegatronParallelStateUtil()
        
        # Setup parallelism sizes
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 2
        mock_parallel_state.get_context_parallel_world_size.return_value = 2
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 2
        mock_parallel_state.get_data_parallel_world_size.return_value = 1
        
        mock_rank_generator = MagicMock()
        mock_parallel_state.RankGenerator.return_value = mock_rank_generator
        mock_rank_generator.get_ranks.return_value = [
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]
        
        # Clear cached property if it exists
        if 'get_model_parallel_ranks' in util.__dict__:
            del util.__dict__['get_model_parallel_ranks']
        
        result = util.get_model_parallel_ranks
        
        self.assertEqual(result, [[0, 1, 2, 3], [4, 5, 6, 7]])
        mock_rank_generator.get_ranks.assert_called_once_with("tp-pp-cp")

    @patch.dict(os.environ, {}, clear=True)
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.dist.new_group")
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.dist.get_rank")
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_create_model_parallel_group_full_flow(self, mock_parallel_state, mock_get_rank, mock_new_group):
        """Test _create_model_parallel_group creates groups and returns current rank's group."""
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        util = MegatronParallelStateUtil()
        
        # Setup: rank 2 is in the second group
        mock_get_rank.return_value = 2
        
        # Setup parallelism sizes
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 2
        mock_parallel_state.get_context_parallel_world_size.return_value = 1
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 2
        mock_parallel_state.get_data_parallel_world_size.return_value = 1
        
        mock_rank_generator = MagicMock()
        mock_parallel_state.RankGenerator.return_value = mock_rank_generator
        mock_rank_generator.get_ranks.return_value = [
            [0, 1],  # First model parallel group
            [2, 3]   # Second model parallel group (contains rank 2)
        ]
        
        # Mock distributed groups
        mock_group_1 = Mock()
        mock_group_2 = Mock()
        mock_new_group.side_effect = [mock_group_1, mock_group_2]
        
        # Clear cached property if it exists
        if 'get_model_parallel_ranks' in util.__dict__:
            del util.__dict__['get_model_parallel_ranks']
        
        result = util._create_model_parallel_group(backend="gloo")
        
        # Verify both groups were created
        self.assertEqual(mock_new_group.call_count, 2)
        
        # Verify the correct group was returned (rank 2 is in second group)
        self.assertEqual(result, mock_group_2)
        
        # Verify groups were created with correct parameters
        calls = mock_new_group.call_args_list
        self.assertEqual(calls[0][0][0], [0, 1])
        self.assertEqual(calls[0][1]['backend'], 'gloo')
        self.assertEqual(calls[1][0][0], [2, 3])
        self.assertEqual(calls[1][1]['backend'], 'gloo')

    @patch.dict(os.environ, {}, clear=True)
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_pdl_ranks_in_mp_group_simple_case(self, mock_parallel_state):
        """Test pdl_ranks_in_mp_group with simple configuration: TP=2, CP=1, PP=2, DP=2."""
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        # Setup: TP=2, CP=1, PP=2, DP=2 (total 8 ranks)
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 2
        mock_parallel_state.get_context_parallel_world_size.return_value = 1
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 2
        mock_parallel_state.get_data_parallel_world_size.return_value = 2

        # Mock RankGenerator
        mock_rank_generator = MagicMock()
        mock_parallel_state.RankGenerator.return_value = mock_rank_generator
        
        # For TP=2, CP=1, PP=2, DP=2 with order "tp-cp-ep-dp-pp"
        # get_ranks("pp") returns groups where each group shares the same PP rank
        # With tcp_size=2, we have 2 groups per DP, total 4 groups
        mock_rank_generator.get_ranks.return_value = [
            [0, 2],  # PP0, TP0 across DP
            [1, 3],  # PP0, TP1 across DP
            [4, 6],  # PP1, TP0 across DP
            [5, 7],  # PP1, TP1 across DP
        ]

        util = MegatronParallelStateUtil()
        
        # Clear cached property if it exists
        if 'pdl_ranks_in_mp_group' in util.__dict__:
            del util.__dict__['pdl_ranks_in_mp_group']
        
        result = util.pdl_ranks_in_mp_group

        # Expected: 2 DP groups, each with PDL ranks
        # DP0 gets first tcp_size=2 groups: [[0,2], [1,3]]
        # Transpose: [[0,1], [2,3]] -> PP0=[0,1], PP1=[2,3]
        # PDL ranks: PP0[::2]=[0], PP1[0]=[2] (cp_size==1), then PP1[2::2]=[]
        # Result: [0, 2]
        # DP1 gets next tcp_size=2 groups: [[4,6], [5,7]]
        # Transpose: [[4,5], [6,7]] -> PP0=[4,5], PP1=[6,7]
        # PDL ranks: PP0[::2]=[4], PP1[0]=[6], then PP1[2::2]=[]
        # Result: [4, 6]
        expected = [
            [0, 2],  # DP0 PDL ranks
            [4, 6],  # DP1 PDL ranks
        ]
        
        self.assertEqual(result, expected)
        self.assertEqual(len(result), 2)  # 2 DP groups
        self.assertEqual(len(result[0]), 2)  # 2 PP stages

    

    @patch.dict(os.environ, {}, clear=True)
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_pdl_ranks_in_mp_group_single_pp_stage(self, mock_parallel_state):
        """Test pdl_ranks_in_mp_group with PP=1 (single pipeline stage)."""
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        # Setup: TP=4, CP=1, PP=1, DP=2 (total 8 ranks)
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 4
        mock_parallel_state.get_context_parallel_world_size.return_value = 1
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 1
        mock_parallel_state.get_data_parallel_world_size.return_value = 2

        # Mock RankGenerator
        mock_rank_generator = MagicMock()
        mock_parallel_state.RankGenerator.return_value = mock_rank_generator
        
        # For TP=4, CP=1, PP=1, DP=2, tcp_size=4
        # get_ranks("pp") returns tcp_size * pp_size * dp_size = 4 * 1 * 2 = 8 groups
        mock_rank_generator.get_ranks.return_value = [
            [0],  # DP0, PP0, TP0
            [1],  # DP0, PP0, TP1
            [2],  # DP0, PP0, TP2
            [3],  # DP0, PP0, TP3
            [4],  # DP1, PP0, TP0
            [5],  # DP1, PP0, TP1
            [6],  # DP1, PP0, TP2
            [7],  # DP1, PP0, TP3
        ]

        util = MegatronParallelStateUtil()
        
        # Clear cached property if it exists
        if 'pdl_ranks_in_mp_group' in util.__dict__:
            del util.__dict__['pdl_ranks_in_mp_group']
        
        result = util.pdl_ranks_in_mp_group

        # Expected: 2 DP groups, each with only first stage PDL ranks
        # DP0 gets first tcp_size=4 groups: [[0], [1], [2], [3]]
        # Transpose: [[0,1,2,3]] -> PP0=[0,1,2,3]
        # PDL ranks: PP0[::4]=[0]
        # DP1 gets next tcp_size=4 groups: [[4], [5], [6], [7]]
        # Transpose: [[4,5,6,7]] -> PP0=[4,5,6,7]
        # PDL ranks: PP0[::4]=[4]
        expected = [
            [0],  # DP0 PDL ranks
            [4],  # DP1 PDL ranks
        ]
        
        self.assertEqual(result, expected)
        self.assertEqual(len(result), 2)  # 2 DP groups
        self.assertEqual(len(result[0]), 1)  # Only 1 PP stage


    @patch.dict(os.environ, {}, clear=True)
    @patch("hyperpod_checkpointless_training.dataloader.parallel_state_util.mcore_parallel_state")
    def test_pdl_ranks_in_mp_group_cached_property(self, mock_parallel_state):
        """Test that pdl_ranks_in_mp_group is properly cached."""
        from hyperpod_checkpointless_training.dataloader.parallel_state_util import MegatronParallelStateUtil

        # Setup simple configuration
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 2
        mock_parallel_state.get_context_parallel_world_size.return_value = 1
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 1
        mock_parallel_state.get_data_parallel_world_size.return_value = 1

        mock_rank_generator = MagicMock()
        mock_parallel_state.RankGenerator.return_value = mock_rank_generator
        mock_rank_generator.get_ranks.return_value = [[0, 1]]

        util = MegatronParallelStateUtil()
        
        # Clear cached property if it exists
        if 'pdl_ranks_in_mp_group' in util.__dict__:
            del util.__dict__['pdl_ranks_in_mp_group']
        
        # First access
        result1 = util.pdl_ranks_in_mp_group
        
        # Second access should use cached value
        result2 = util.pdl_ranks_in_mp_group
        
        # Should be the same object (cached)
        self.assertIs(result1, result2)
        
        # RankGenerator should only be called once
        self.assertEqual(mock_parallel_state.RankGenerator.call_count, 1)
