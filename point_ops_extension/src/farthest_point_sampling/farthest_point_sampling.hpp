#pragma once

#include <torch/extension.h>

at::Tensor FarthestPointSamplingCPU(
    const at::Tensor& points,
    const int64_t num_centroids
);

at::Tensor FarthestPointSamplingCUDA(
    const at::Tensor& points,
    const int64_t num_centroids
);
