import unittest
import pytest
from unittest.mock import patch, MagicMock

from hyperpod_checkpointless_training.nemo_plugins.load_balancer import get_rank_maps, check_available_replica

@pytest.fixture(autouse=True)
def mock_parallel_state():
    with patch("hyperpod_checkpointless_training.nemo_plugins.load_balancer.parallel_state") as mock_ps:
        mock_ps._INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = MagicMock()
        mock_ps.get_inter_partial_data_parallel_group.return_value = 'mocked_group'
        yield mock_ps

@pytest.fixture(autouse=True)
def mock_dist():
    with patch("hyperpod_checkpointless_training.nemo_plugins.load_balancer.dist") as mock_dist:
        mock_dist.get_process_group_ranks.return_value = [0, 1, 2, 3]
        yield mock_dist

class TestLoadBalancer(unittest.TestCase):
    def test_get_rank_maps_normal_case(self):
        faulty_ranks = [1, 2]
        result = get_rank_maps(faulty_ranks)
        self.assertEqual(result, [[0, 1], [3, 2]])

    def test_get_rank_maps_no_faulty_ranks(self):
        faulty_ranks = []
        result = get_rank_maps(faulty_ranks)
        self.assertEqual(result, [])

    def test_get_rank_maps_all_ranks_faulty(self):
        faulty_ranks = [0, 1, 2, 3]
        result = get_rank_maps(faulty_ranks)
        self.assertEqual(result, [])

    def test_get_rank_maps_some_faulty_ranks_outside_group(self):
        faulty_ranks = [1, 4, 5]
        result = get_rank_maps(faulty_ranks)
        self.assertEqual(result, [[0, 1]])

    def test_check_available_replica_all_available(self):
        rank_info = {
            0: [1, 2],  # Rank 0 is faulty, but has ranks 1 and 2 available
            1: [2, 3]   # Rank 1 is faulty, but has ranks 2 and 3 available
        }
        self.assertTrue(check_available_replica(rank_info))

    def test_check_available_replica_none_available(self):
        rank_info = {
            0: [1, 2],  # Rank 0 is faulty, has ranks 1 and 2 available
            1: [0, 2],  # Rank 1 is faulty, but all its replicas (0, 2) are either faulty or already used
            2: [0, 1]   # Rank 2 is faulty, but all its replicas (0, 1) are faulty
        }
        self.assertFalse(check_available_replica(rank_info))

    def test_check_available_replica_empty_input(self):
        rank_info = {}
        self.assertTrue(check_available_replica(rank_info))

    def test_check_available_replica_single_rank(self):
        rank_info = {
            0: [1, 2, 3]  # Rank 0 is faulty, has ranks 1, 2, 3 available
        }
        self.assertTrue(check_available_replica(rank_info))

    def test_check_available_replica_single_rank_no_replicas(self):
        rank_info = {
            0: [1],  # Rank 0 is faulty
            1: [0]   # Its only replica (rank 1) is also faulty
        }
        self.assertFalse(check_available_replica(rank_info))

if __name__ == "__main__":
    unittest.main()
