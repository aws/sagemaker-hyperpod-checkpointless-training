import os
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from hyperpod_checkpointless_training.nemo_plugins.utils import (
    get_sharded_tensor_states,
    init_process_group,
    init_process_group_with_tcpstore,
    init_process_group_without_tcpstore,
    use_tcpstore,
    create_store,
)


class TestGetShardedTensorStates:
    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer with complex nested structure"""
        optimizer = Mock()

        # Create mock parameters
        mock_param1 = Mock()
        mock_param2 = Mock()
        mock_param3 = Mock()

        # Create mock main parameters
        mock_main_param1 = Mock()
        mock_main_param2 = Mock()
        mock_main_param3 = Mock()

        # Create mock optimizer states
        mock_state1 = {"momentum_buffer": torch.randn(10), "step": 1}
        mock_state2 = {"momentum_buffer": torch.randn(5), "step": 2}
        mock_state3 = {"momentum_buffer": torch.randn(3), "step": 3}

        # Setup optimizer structure
        mock_optimizer = Mock()
        mock_optimizer.param_groups = [
            {"params": [mock_main_param1, mock_main_param2]},  # group 0
            {"params": [mock_main_param3]},  # group 1
        ]
        mock_optimizer.state = {
            mock_main_param1: mock_state1,
            mock_main_param2: mock_state2,
            mock_main_param3: mock_state3,
        }

        # Setup gbuf_ranges structure
        gbuf_range_map1 = {
            "param_map": {
                mock_param1: "range1",
                mock_param2: "range2",
            }
        }
        gbuf_range_map2 = {
            "param_map": {
                mock_param3: "range3",
            }
        }

        gbuf_range_maps = {
            "bucket1": [gbuf_range_map1],
            "bucket2": [gbuf_range_map2],
        }

        optimizer.gbuf_ranges = [gbuf_range_maps]
        optimizer.optimizer = mock_optimizer

        # Setup model_param_group_index_map
        optimizer.model_param_group_index_map = {
            mock_param1: (0, 0),  # group 0, order 0
            mock_param2: (0, 1),  # group 0, order 1
            mock_param3: (1, 0),  # group 1, order 0
        }

        return optimizer, {
            "params": [mock_param1, mock_param2, mock_param3],
            "main_params": [mock_main_param1, mock_main_param2, mock_main_param3],
            "states": [mock_state1, mock_state2, mock_state3],
        }

    def test_get_sharded_tensor_states_basic(self, mock_optimizer):
        """Test basic functionality of get_sharded_tensor_states"""
        optimizer, expected = mock_optimizer

        param_state, optim_state = get_sharded_tensor_states(optimizer)

        # Should have 3 parameters
        assert len(param_state) == 3
        assert len(optim_state) == 3

        # Verify parameter mapping
        assert param_state[0] == expected["main_params"][0]
        assert param_state[1] == expected["main_params"][1]
        assert param_state[2] == expected["main_params"][2]

        # Verify optimizer state mapping
        assert optim_state[0] == expected["states"][0]
        assert optim_state[1] == expected["states"][1]
        assert optim_state[2] == expected["states"][2]

    def test_get_sharded_tensor_states_empty_optimizer(self):
        """Test get_sharded_tensor_states with empty optimizer"""
        optimizer = Mock()
        optimizer.gbuf_ranges = []

        param_state, optim_state = get_sharded_tensor_states(optimizer)

        assert param_state == {}
        assert optim_state == {}

    def test_get_sharded_tensor_states_multiple_gbuf_ranges(self):
        """Test get_sharded_tensor_states with multiple gbuf_ranges"""
        optimizer = Mock()

        # Create mock parameters
        mock_param1 = Mock()
        mock_param2 = Mock()
        mock_main_param1 = Mock()
        mock_main_param2 = Mock()

        # Setup optimizer
        mock_optimizer = Mock()
        mock_optimizer.param_groups = [{"params": [mock_main_param1, mock_main_param2]}]
        mock_optimizer.state = {
            mock_main_param1: {"state1": "value1"},
            mock_main_param2: {"state2": "value2"},
        }

        # Setup multiple gbuf_ranges
        gbuf_range_map1 = {"param_map": {mock_param1: "range1"}}
        gbuf_range_map2 = {"param_map": {mock_param2: "range2"}}

        gbuf_ranges1 = {"bucket1": [gbuf_range_map1]}
        gbuf_ranges2 = {"bucket2": [gbuf_range_map2]}

        optimizer.gbuf_ranges = [gbuf_ranges1, gbuf_ranges2]
        optimizer.optimizer = mock_optimizer
        optimizer.model_param_group_index_map = {
            mock_param1: (0, 0),
            mock_param2: (0, 1),
        }

        param_state, optim_state = get_sharded_tensor_states(optimizer)

        assert len(param_state) == 2
        assert len(optim_state) == 2

    def test_get_sharded_tensor_states_multiple_buckets(self):
        """Test get_sharded_tensor_states with multiple buckets per gbuf_range"""
        optimizer = Mock()

        # Create mock parameters
        mock_param1 = Mock()
        mock_param2 = Mock()
        mock_main_param1 = Mock()
        mock_main_param2 = Mock()

        # Setup optimizer
        mock_optimizer = Mock()
        mock_optimizer.param_groups = [{"params": [mock_main_param1, mock_main_param2]}]
        mock_optimizer.state = {
            mock_main_param1: {"state1": "value1"},
            mock_main_param2: {"state2": "value2"},
        }

        # Setup multiple buckets
        gbuf_range_map1 = {"param_map": {mock_param1: "range1"}}
        gbuf_range_map2 = {"param_map": {mock_param2: "range2"}}

        gbuf_ranges = {
            "bucket1": [gbuf_range_map1],
            "bucket2": [gbuf_range_map2],
        }

        optimizer.gbuf_ranges = [gbuf_ranges]
        optimizer.optimizer = mock_optimizer
        optimizer.model_param_group_index_map = {
            mock_param1: (0, 0),
            mock_param2: (0, 1),
        }

        param_state, optim_state = get_sharded_tensor_states(optimizer)

        assert len(param_state) == 2
        assert len(optim_state) == 2

    def test_get_sharded_tensor_states_multiple_range_maps(self):
        """Test get_sharded_tensor_states with multiple range maps per bucket"""
        optimizer = Mock()

        # Create mock parameters
        mock_param1 = Mock()
        mock_param2 = Mock()
        mock_main_param1 = Mock()
        mock_main_param2 = Mock()

        # Setup optimizer
        mock_optimizer = Mock()
        mock_optimizer.param_groups = [{"params": [mock_main_param1, mock_main_param2]}]
        mock_optimizer.state = {
            mock_main_param1: {"state1": "value1"},
            mock_main_param2: {"state2": "value2"},
        }

        # Setup multiple range maps per bucket
        gbuf_range_map1 = {"param_map": {mock_param1: "range1"}}
        gbuf_range_map2 = {"param_map": {mock_param2: "range2"}}

        gbuf_ranges = {
            "bucket1": [gbuf_range_map1, gbuf_range_map2],  # Multiple range maps
        }

        optimizer.gbuf_ranges = [gbuf_ranges]
        optimizer.optimizer = mock_optimizer
        optimizer.model_param_group_index_map = {
            mock_param1: (0, 0),
            mock_param2: (0, 1),
        }

        param_state, optim_state = get_sharded_tensor_states(optimizer)

        assert len(param_state) == 2
        assert len(optim_state) == 2


class TestUseTcpstore:
    def test_use_tcpstore_default(self):
        """Test use_tcpstore with default environment (no HPCT_USE_ROOTLESS)"""
        with patch.dict(os.environ, {}, clear=True):
            result = use_tcpstore()
            assert result is True

    def test_use_tcpstore_skip_false(self):
        """Test use_tcpstore when HPCT_USE_ROOTLESS=0"""
        with patch.dict(os.environ, {"HPCT_USE_ROOTLESS": "0"}):
            result = use_tcpstore()
            assert result is True

    def test_use_tcpstore_skip_true(self):
        """Test use_tcpstore when HPCT_USE_ROOTLESS=1"""
        with patch.dict(os.environ, {"HPCT_USE_ROOTLESS": "1"}):
            result = use_tcpstore()
            assert result is False

    def test_use_tcpstore_skip_other_values(self):
        """Test use_tcpstore with other values"""
        with patch.dict(os.environ, {"HPCT_USE_ROOTLESS": "2"}):
            result = use_tcpstore()
            assert result is False

        # Test that invalid values raise ValueError when converted to int
        with patch.dict(os.environ, {"HPCT_USE_ROOTLESS": "invalid"}):
            with pytest.raises(ValueError):
                use_tcpstore()


class TestInitProcessGroup:

    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.init_process_group_with_tcpstore")
    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.init_process_group_without_tcpstore")
    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.use_tcpstore")
    def test_init_process_group_with_tcpstore(
        self, mock_use_tcpstore, mock_without_tcp, mock_with_tcp
    ):
        """Test init_process_group when use_tcpstore returns True"""
        mock_use_tcpstore.return_value = True
        mock_strategy = Mock()

        init_process_group(mock_strategy)

        mock_with_tcp.assert_called_once_with(mock_strategy)
        mock_without_tcp.assert_not_called()

    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.init_process_group_with_tcpstore")
    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.init_process_group_without_tcpstore")
    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.use_tcpstore")
    def test_init_process_group_without_tcpstore(
        self, mock_use_tcpstore, mock_without_tcp, mock_with_tcp
    ):
        """Test init_process_group when use_tcpstore returns False"""
        mock_use_tcpstore.return_value = False
        mock_strategy = Mock()

        init_process_group(mock_strategy)

        mock_without_tcp.assert_called_once_with(mock_strategy)
        mock_with_tcp.assert_not_called()


class TestCreateStore:
    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy for testing"""
        strategy = Mock()
        strategy.base_store = None
        return strategy

    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.use_tcpstore")
    def test_create_store_when_tcpstore_disabled(self, mock_use_tcpstore, mock_strategy):
        """Test create_store returns early when use_tcpstore is False"""
        mock_use_tcpstore.return_value = False

        create_store(mock_strategy)

        # Should not modify strategy when tcpstore is disabled
        assert mock_strategy.base_store is None

    @patch.dict(
        os.environ,
        {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500", "WORLD_SIZE": "4", "RANK": "0", "JOB_RESTART_COUNT": "2"},
    )
    @patch("torch.distributed.TCPStore")
    @patch("torch.distributed.PrefixStore")
    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.use_tcpstore")
    def test_create_store_creates_base_store(self, mock_use_tcpstore, mock_prefix_store, mock_tcp_store, mock_strategy):
        """Test create_store creates base_store when it doesn't exist"""
        mock_use_tcpstore.return_value = True
        mock_tcp_instance = Mock()
        mock_tcp_store.return_value = mock_tcp_instance
        mock_prefix_instance = Mock()
        mock_prefix_store.return_value = mock_prefix_instance

        create_store(mock_strategy)

        # Verify TCPStore creation
        mock_tcp_store.assert_called_once_with(
            host_name="127.0.0.1",
            port=29501,  # MASTER_PORT + 1
            world_size=4,
            is_master=True,  # RANK == 0
            multi_tenant=True,
            wait_for_workers=True,
            use_libuv=True,
        )

        # Verify PrefixStore creation
        mock_prefix_store.assert_called_once_with("2", mock_tcp_instance)
        assert mock_strategy.base_store == mock_tcp_instance
        assert mock_strategy.store == mock_prefix_instance

    @patch.dict(
        os.environ,
        {"MASTER_ADDR": "192.168.1.1", "MASTER_PORT": "12345", "WORLD_SIZE": "8", "RANK": "3", "JOB_RESTART_COUNT": "5"},
    )
    @patch("torch.distributed.PrefixStore")
    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.use_tcpstore")
    def test_create_store_reuses_existing_base_store(self, mock_use_tcpstore, mock_prefix_store, mock_strategy):
        """Test create_store reuses existing base_store"""
        mock_use_tcpstore.return_value = True
        existing_store = Mock()
        mock_strategy.base_store = existing_store
        mock_prefix_instance = Mock()
        mock_prefix_store.return_value = mock_prefix_instance

        create_store(mock_strategy)

        # Should reuse existing base_store
        assert mock_strategy.base_store == existing_store
        mock_prefix_store.assert_called_once_with("5", existing_store)
        assert mock_strategy.store == mock_prefix_instance

    @patch.dict(
        os.environ,
        {"MASTER_ADDR": "10.0.0.1", "MASTER_PORT": "8000", "WORLD_SIZE": "2", "RANK": "1", "JOB_RESTART_COUNT": "0"},
    )
    @patch("torch.distributed.TCPStore")
    @patch("torch.distributed.PrefixStore")
    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.use_tcpstore")
    def test_create_store_non_master_rank(self, mock_use_tcpstore, mock_prefix_store, mock_tcp_store, mock_strategy):
        """Test create_store with non-master rank"""
        mock_use_tcpstore.return_value = True
        mock_tcp_instance = Mock()
        mock_tcp_store.return_value = mock_tcp_instance

        create_store(mock_strategy)

        # Verify is_master=False for non-zero rank
        mock_tcp_store.assert_called_once_with(
            host_name="10.0.0.1",
            port=8001,
            world_size=2,
            is_master=False,  # RANK != 0
            multi_tenant=True,
            wait_for_workers=True,
            use_libuv=True,
        )


class TestInitProcessGroupWithTcpstore:
    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy with required attributes"""
        strategy = Mock()
        strategy.base_store = None
        strategy._get_process_group_backend.return_value = "nccl"
        strategy._timeout = 30

        # Mock cluster environment
        strategy.cluster_environment = Mock()
        strategy.cluster_environment.global_rank.return_value = 1
        strategy.cluster_environment.world_size.return_value = 4

        return strategy

    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.create_store")
    @patch("torch.distributed.init_process_group")
    def test_init_process_group_with_tcpstore(self, mock_init_pg, mock_create_store, mock_strategy):
        """Test init_process_group_with_tcpstore calls create_store and init_process_group"""
        mock_store = Mock()
        mock_strategy.store = mock_store

        init_process_group_with_tcpstore(mock_strategy)

        # Verify create_store is called
        mock_create_store.assert_called_once_with(mock_strategy)

        # Verify process group initialization
        mock_init_pg.assert_called_once_with(
            "nccl",
            rank=1,
            world_size=4,
            timeout=30,
            store=mock_store,
        )


class TestInitProcessGroupWithoutTcpstore:
    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy for testing"""
        strategy = Mock()
        strategy._get_process_group_backend.return_value = "gloo"
        strategy._timeout = 60

        # Mock cluster environment
        strategy.cluster_environment = Mock()
        strategy.cluster_environment.global_rank.return_value = 2
        strategy.cluster_environment.world_size.return_value = 8

        return strategy

    @patch("torch.distributed.init_process_group")
    def test_init_process_group_without_tcpstore(self, mock_init_pg, mock_strategy):
        """Test init_process_group_without_tcpstore"""
        init_process_group_without_tcpstore(mock_strategy)

        mock_init_pg.assert_called_once_with(
            "gloo",
            rank=2,
            world_size=8,
            timeout=60,
        )

    @patch("torch.distributed.init_process_group")
    def test_init_process_group_without_tcpstore_different_backend(
        self, mock_init_pg, mock_strategy
    ):
        """Test init_process_group_without_tcpstore with different backend"""
        mock_strategy._get_process_group_backend.return_value = "mpi"
        mock_strategy.cluster_environment.global_rank.return_value = 0
        mock_strategy.cluster_environment.world_size.return_value = 1
        mock_strategy._timeout = 120

        init_process_group_without_tcpstore(mock_strategy)

        mock_init_pg.assert_called_once_with(
            "mpi",
            rank=0,
            world_size=1,
            timeout=120,
        )


class TestUtilsIntegration:
    """Integration tests for utils functions"""

    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.use_tcpstore")
    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.create_store")
    @patch("torch.distributed.init_process_group")
    def test_full_init_process_group_flow_with_tcpstore(self, mock_init_pg, mock_create_store, mock_use_tcpstore):
        """Test complete flow of init_process_group with TCPStore"""
        mock_use_tcpstore.return_value = True

        strategy = Mock()
        strategy._get_process_group_backend.return_value = "nccl"
        strategy._timeout = 30
        strategy.cluster_environment = Mock()
        strategy.cluster_environment.global_rank.return_value = 1
        strategy.cluster_environment.world_size.return_value = 4
        strategy.store = Mock()

        init_process_group(strategy)

        mock_create_store.assert_called_once_with(strategy)
        mock_init_pg.assert_called_once_with(
            "nccl",
            rank=1,
            world_size=4,
            timeout=30,
            store=strategy.store,
        )

    @patch("hyperpod_checkpointless_training.nemo_plugins.utils.use_tcpstore")
    @patch("torch.distributed.init_process_group")
    def test_full_init_process_group_flow_without_tcpstore(self, mock_init_pg, mock_use_tcpstore):
        """Test complete flow of init_process_group without TCPStore"""
        mock_use_tcpstore.return_value = False

        strategy = Mock()
        strategy._get_process_group_backend.return_value = "gloo"
        strategy._timeout = 60
        strategy.cluster_environment = Mock()
        strategy.cluster_environment.global_rank.return_value = 0
        strategy.cluster_environment.world_size.return_value = 2

        init_process_group(strategy)

        mock_init_pg.assert_called_once_with(
            "gloo",
            rank=0,
            world_size=2,
            timeout=60,
        )

    def test_environment_variable_handling(self):
        """Test various environment variable scenarios"""
        # Test missing environment variable
        with patch.dict(os.environ, {}, clear=True):
            assert use_tcpstore() is True

        # Test various HPCT_USE_ROOTLESS values
        test_cases = [
            ("0", True),
            ("1", False),
            ("2", False),
            ("10", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"HPCT_USE_ROOTLESS": env_value}):
                assert use_tcpstore() == expected
