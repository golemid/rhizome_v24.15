#pragma once

#include "engine/colonization/colonization.hpp"
#include "engine/io/mesh.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <cmath>

class RootImplicitField {
public:
    // Rasterizes root graph into a 3D Distance Grid using each node's stored radius
    static std::shared_ptr<rhizome::CudaVolumeGrid> rasterize_root_graph(
        const std::vector<ColonizationNode>& nodes,
        float voxel_size,
        float half_width_voxels = 3.0f
    );

    // Rasterizes root graph into a 3D Distance Grid using vascular caliber tapering parameters
    static std::shared_ptr<rhizome::CudaVolumeGrid> rasterize_root_graph(
        const std::vector<ColonizationNode>& nodes,
        float voxel_size,
        float trunk_caliber,
        float tendril_caliber,
        float taper_curve,
        float half_width_voxels = 3.0f,
        bool preserve_beaded = false,
        float branch_smooth_k = 0.35f
    );

    // Confinement boolean: Root intersected with Vessel interior: max(root_sdf, vessel_sdf)
    // v24.10 Track E R2: preserve_input_grid=false intersects root_grid IN
    // PLACE (for callers that rasterized a fresh temp grid and drop it right
    // after - saves one full-size host grid copy per bake). The default
    // (true) keeps the historical copy semantics; all verification-harness
    // and panel callers are unchanged.
    static std::shared_ptr<rhizome::CudaVolumeGrid> intersect_with_vessel(
        std::shared_ptr<rhizome::CudaVolumeGrid> root_grid,
        std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
        float wall_cut_smooth = 0.0f,
        bool preserve_input_grid = true
    );
};
