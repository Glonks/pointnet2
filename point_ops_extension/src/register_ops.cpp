#include <torch/library.h>

#include "farthest_point_sampling/farthest_point_sampling.hpp"

// farthest point sampling
TORCH_LIBRARY(point_ops, m) {
    m.def("farthest_point_sampling(Tensor points, int num_centroids) -> Tensor");
}
TORCH_LIBRARY_IMPL(point_ops, CPU, m) {
    m.impl("farthest_point_sampling", &FarthestPointSamplingCPU);
}
TORCH_LIBRARY_IMPL(point_ops, CUDA, m) {
    m.impl("farthest_point_sampling", &FarthestPointSamplingCUDA);
}

// ball query
