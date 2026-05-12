#include "farthest_point_sampling.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <cmath>

constexpr int MAX_THREADS_PER_BLOCK = 1024;

template <unsigned int block_size>
__global__ void farthest_point_sampling_kernel(
    const at::PackedTensorAccessor32<float, 3, at::RestrictPtrTraits> points_accessor,
    const int64_t num_centroids,
    at::PackedTensorAccessor32<int64_t, 2, at::RestrictPtrTraits> centroid_indices_accessor,
    at::PackedTensorAccessor32<float, 2, at::RestrictPtrTraits> min_distances_sq_accessor
) {
    __shared__ int64_t _s_previous_centroid_index;
    __shared__ int64_t _s_max_indices[block_size];
    __shared__ float _s_max_distances[block_size];

    const int64_t batch_index = blockIdx.x;
    const size_t thread_index = threadIdx.x;

    const int64_t num_points = points_accessor.size(1);
    const int64_t num_features = points_accessor.size(2);

    int64_t previous_centroid_index = centroid_indices_accessor[batch_index][0];
    for (int64_t i = 1; i < num_centroids; ++i) {
        float local_max_distance = std::numeric_limits<float>::lowest();
        int64_t local_max_index = 0;

        for (int64_t point_index = thread_index; point_index < num_points; point_index += block_size) {
            float distance_sq = 0.0F;
            for (int64_t feature_index = 0; feature_index < num_features; ++feature_index) {
                float difference = (
                    points_accessor[batch_index][previous_centroid_index][feature_index] -
                    points_accessor[batch_index][point_index][feature_index]
                );
                distance_sq += difference * difference;
            }

            const float min_distance_sq = min(
                distance_sq,
                min_distances_sq_accessor[batch_index][point_index]
            );
            min_distances_sq_accessor[batch_index][point_index] = min_distance_sq;

            if (min_distance_sq > local_max_distance) {
                local_max_distance = min_distance_sq;
                local_max_index  = point_index;
            }
        }

        // Tree reduction to find argmax across all threads
        _s_max_distances[thread_index] = local_max_distance;
        _s_max_indices[thread_index]  = local_max_index;
        __syncthreads();

        for (int stride = block_size / 2; stride > 0; stride >>= 1) {
            if (thread_index < stride) {
                if (_s_max_distances[thread_index + stride] > _s_max_distances[thread_index]) {
                    _s_max_distances[thread_index] = _s_max_distances[thread_index + stride];
                    _s_max_indices[thread_index] = _s_max_indices[thread_index + stride];
                }
            }
            __syncthreads();
        }

        if (thread_index == 0) {
            centroid_indices_accessor[batch_index][i] = _s_max_indices[0];
            _s_previous_centroid_index = _s_max_indices[0];
        }

        __syncthreads();
        previous_centroid_index = _s_previous_centroid_index;
    }
}

at::Tensor FarthestPointSamplingCUDA(
    const at::Tensor& points,
    const int64_t num_centroids
) {
    const int64_t batch_size = points.size(0);
    const int64_t num_points = points.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Randomly sample indices for the first centroid
    const auto seed = at::randint(num_points, {batch_size}, at::kLong);

    auto centroid_indices = at::full(
        {batch_size, num_centroids},
        -1,
        at::TensorOptions().dtype(at::kLong).device(points.device())
    );
    centroid_indices.select(1, 0).copy_(seed);

    auto min_distances_sq = at::full(
        {batch_size, num_points},
        std::numeric_limits<float>::infinity(),
        at::TensorOptions().dtype(at::kFloat).device(points.device())
    );

    auto points_accessor = points.packed_accessor32<float, 3, at::RestrictPtrTraits>();
    auto centroid_indices_accessor = centroid_indices.packed_accessor32<int64_t, 2, at::RestrictPtrTraits>();
    auto min_distances_sq_accessor = min_distances_sq.packed_accessor32<float, 2, at::RestrictPtrTraits>();

    const size_t num_blocks = batch_size;
    const size_t num_threads = std::max(
        2,
        std::min(
            1 << static_cast<int>(std::log2(static_cast<float>(num_points))),
            MAX_THREADS_PER_BLOCK
        )
    );

    switch(num_threads) {
        case 1024:
            farthest_point_sampling_kernel<1024><<<num_blocks, num_threads, 0, stream>>>(
                points_accessor, num_centroids, centroid_indices_accessor, min_distances_sq_accessor
            );
            break;
        case 512:
            farthest_point_sampling_kernel<512><<<num_blocks, num_threads, 0, stream>>>(
                points_accessor, num_centroids, centroid_indices_accessor, min_distances_sq_accessor
            );
            break;
        case 256:
            farthest_point_sampling_kernel<256><<<num_blocks, num_threads, 0, stream>>>(
                points_accessor, num_centroids, centroid_indices_accessor,  min_distances_sq_accessor
            );
            break;
        case 128:
            farthest_point_sampling_kernel<128><<<num_blocks, num_threads, 0, stream>>>(
                points_accessor, num_centroids, centroid_indices_accessor,  min_distances_sq_accessor
            );
            break;
        case 64:
            farthest_point_sampling_kernel<64><<<num_blocks, num_threads, 0, stream>>>(
                points_accessor, num_centroids, centroid_indices_accessor,  min_distances_sq_accessor
            );
            break;
        case 32:
            farthest_point_sampling_kernel<32><<<num_blocks, num_threads, 0, stream>>>(
                points_accessor, num_centroids, centroid_indices_accessor,  min_distances_sq_accessor
            );
            break;
        case 16:
            farthest_point_sampling_kernel<16><<<num_blocks, num_threads, 0, stream>>>(
                points_accessor, num_centroids, centroid_indices_accessor,  min_distances_sq_accessor
            );
            break;
        case 8:
            farthest_point_sampling_kernel<8><<<num_blocks, num_threads, 0, stream>>>(
                points_accessor, num_centroids, centroid_indices_accessor,  min_distances_sq_accessor
            );
            break;
        case 4:
            farthest_point_sampling_kernel<4><<<num_blocks, num_threads, 0, stream>>>(
                points_accessor, num_centroids, centroid_indices_accessor, min_distances_sq_accessor
            );
            break;
        case 2:
            farthest_point_sampling_kernel<2><<<num_blocks, num_threads, 0, stream>>>(
                points_accessor, num_centroids, centroid_indices_accessor, min_distances_sq_accessor
            );
            break;
        default:
            farthest_point_sampling_kernel<1024><<<num_blocks, num_threads, 0, stream>>>(
                points_accessor, num_centroids, centroid_indices_accessor, min_distances_sq_accessor
            );
        }

    return centroid_indices;
}
