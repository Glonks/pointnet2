import torch

from . import _C  # type: ignore

from .farthest_point_sampling import farthest_point_sampling


__all__ = [
    'farthest_point_sampling'
]
