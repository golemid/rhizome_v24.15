#pragma once

#include "verification/harness.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include <algorithm>

namespace rhizome {

struct DeviationSampleResult {
    size_t sample_count = 0;
    float vertex_max = 0.0f;
    float midpoint_max = 0.0f;
    float combined_max = 0.0f;
    float mean = 0.0f;
    float rms = 0.0f;
};

inline DeviationSampleResult evaluate_mesh_deviations_on_gpu(
    const TriangleMesh& mesh,
    int shape_type,
    float p0,
    float p1 = 0.0f,
    glm::vec3 box_half_extents = glm::vec3(0.0f)
) {
    DeviationSampleResult res;
    CudaRasterizer::evaluate_mesh_deviations_gpu(
        mesh, shape_type, p0, p1, box_half_extents,
        res.sample_count, res.vertex_max, res.midpoint_max, res.mean, res.rms
    );
    res.combined_max = std::max(res.vertex_max, res.midpoint_max);
    return res;
}

} // namespace rhizome
