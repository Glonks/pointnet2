import torch


def farthest_point_sampling(
    points: torch.Tensor,
    num_centroids: int
) -> torch.Tensor:
    return torch.ops.point_ops.farthest_point_sampling(points, num_centroids)
