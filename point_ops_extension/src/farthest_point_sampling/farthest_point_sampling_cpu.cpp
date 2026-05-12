#include "farthest_point_sampling.hpp"

#include <limits>

torch::Tensor FarthestPointSamplingCPU(
    const torch::Tensor& points,
    const int64_t num_centroids
) {
    const int64_t batch_size = points.size(0);
    const int64_t num_points = points.size(1);
    const int64_t num_features = points.size(2);

    auto centroid_indices = torch::full(
        {batch_size, num_centroids},
        -1,
        torch::TensorOptions().dtype(torch::kLong)
    );

    auto min_distances_sq = torch::full(
        {batch_size, num_points},
        std::numeric_limits<float>::infinity(),
        torch::TensorOptions().dtype(torch::kFloat32)
    );

    auto seed = torch::randint(num_points, {batch_size}, torch::kLong);
    centroid_indices.select(1, 0).copy_(seed);

    auto points_accessor = points.accessor<float, 3>();
    auto centroid_indices_accessor = centroid_indices.accessor<int64_t, 2>();
    auto min_distances_sq_accessor = min_distances_sq.accessor<float, 2>();

    for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        int64_t previous_centroid_index = centroid_indices_accessor[batch_index][0];

        for (int64_t i = 1; i < num_centroids; ++i) {
            float local_max_dist = std::numeric_limits<float>::lowest();
            int64_t local_max_index = 0;

            for (int64_t point_index = 0; point_index < num_points; ++point_index) {
                float distance_sq = 0.0f;
                for (int64_t d = 0; d < num_features; ++d) {
                    float difference = (
                        points_accessor[batch_index][previous_centroid_index][d] -
                        points_accessor[batch_index][point_index][d]
                    );
                    distance_sq += difference * difference;
                }

                const float min_dist_sq = std::min(
                    distance_sq,
                    min_distances_sq_accessor[batch_index][point_index]
                );
                min_distances_sq_accessor[batch_index][point_index] = min_dist_sq;

                if (min_dist_sq > local_max_dist) {
                    local_max_dist = min_dist_sq;
                    local_max_index  = point_index;
                }
            }

            centroid_indices_accessor[batch_index][i] = local_max_index;
            previous_centroid_index = local_max_index;
        }
    }

    return centroid_indices;
}
