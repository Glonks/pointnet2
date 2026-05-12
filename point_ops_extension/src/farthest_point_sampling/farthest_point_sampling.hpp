#pragma once

#include <torch/extension.h>

torch::Tensor FarthestPointSamplingCPU(
    const torch::Tensor& points,
    const int64_t num_centroids
);

torch::Tensor FarthestPointSamplingCUDA(
    const torch::Tensor& points,
    const int64_t num_centroids
);
