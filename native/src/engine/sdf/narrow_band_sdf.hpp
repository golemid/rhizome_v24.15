#pragma once

#include "engine/io/mesh.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include <memory>
#include <string>
#include <cmath>

namespace rhizome {

struct SdfConfig {
    float voxel_size = 0.5f;
    float half_width_voxels = 3.0f;
    float winding_number_theta = 0.5f;
    float min_curvature_radius = 0.5f;
    float refined_voxel_size = 0.25f;
    bool enable_winding_normalizer = true;
};

class NarrowBandSdf {
public:
    NarrowBandSdf() = default;
    ~NarrowBandSdf() = default;

    /**
     * @brief Computes narrow-band 3D Distance Grid from surface mesh via CUDA GPU kernels.
     */
    std::shared_ptr<CudaVolumeGrid> build_from_mesh(
        const TriangleMesh& mesh,
        const SdfConfig& config,
        std::string& out_log
    );

    /**
     * @brief Extracts watertight, 2-manifold triangle mesh from 3D Distance Grid at isovalue = 0.0.
     */
    static TriangleMesh extract_mesh(
        std::shared_ptr<CudaVolumeGrid> grid,
        float isovalue = 0.0f,
        float adaptivity = 0.0f
    );

    static constexpr float calculate_chord_error(float radius, float cell_size) noexcept {
        if (radius <= 0.0f) return 0.0f;
        const float half_chord = (cell_size * 1.41421356f) * 0.5f;
        if (half_chord >= radius) return radius;
        return radius - std::sqrt(radius * radius - half_chord * half_chord);
    }
};

} // namespace rhizome
