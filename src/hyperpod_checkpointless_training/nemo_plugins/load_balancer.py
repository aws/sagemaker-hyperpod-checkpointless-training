import torch.distributed as dist
from megatron.core import parallel_state


def get_rank_maps(faulty_ranks):
    """
    Generate load-balanced rank mappings for P2P checkpoint recovery.

    Args:
        faulty_ranks: Set of failed rank IDs requiring recovery

    Returns:
        list: List of [src_rank, dst_rank] pairs for checkpoint transfer
    """
    # Get the inter DP group (hybrd shard group) for the current rank
    inter_dp_group = parallel_state.get_inter_distributed_optimizer_instance_group()
    candidate_ranks = dist.get_process_group_ranks(inter_dp_group)

    # Gather faulty ranks that are associated with the current rank
    faulty_list = []
    for faulty_rank in faulty_ranks:
        # If the faulty rank is associated with the current rank, add it to the faulty_list
        if faulty_rank in candidate_ranks:
            faulty_list.append(faulty_rank)
            # Exclude faulty ranks from the list as they are not eligible candidates.
            candidate_ranks.remove(faulty_rank)

    # If there is no candidate ranks available or all the faulty ranks are not in the current dp group,
    # return an empty list
    if len(candidate_ranks) == 0 or len(faulty_list) == 0:
        return []

    # Keep track of each rank's load
    rank_count = {rank: 0 for rank in candidate_ranks}
    rank_maps = []

    # Retrieve the rank map for each faulty rank associated with the current rank
    for rank in faulty_list:
        # Pick the candidate rank to restore from
        src_rank = min(candidate_ranks, key=rank_count.get)
        rank_count[src_rank] += 1
        rank_map = [src_rank, rank]
        rank_maps.append(rank_map)

    return rank_maps


def check_available_replica(rank_info):
    """
    Validate that all failed ranks have at least one healthy replica.

    Args:
        rank_info: Dict mapping failed ranks to their replica groups

    Returns:
        bool: True if all failed ranks have healthy replicas, False otherwise
    """
    for group in rank_info.values():
        filtered_groups = [rank for rank in group if rank not in rank_info.keys()]
        if not filtered_groups:  # Check if any groups are empty
            return False
    return True
