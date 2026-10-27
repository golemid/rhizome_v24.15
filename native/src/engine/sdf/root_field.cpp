#include "engine/sdf/root_field.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include <iostream>

std::shared_ptr<rhizome::CudaVolumeGrid> RootImplicitField::rasterize_root_graph(
    const std::vector<ColonizationNode>& nodes,
    float voxel_size,
    float half_width_voxels
) {
    auto volume = std::make_shared<rhizome::CudaVolumeGrid>();
    volume->voxel_size = voxel_size;
    if (nodes.empty()) {
        return volume;
    }

    float gpu_ms = 0.0f;
    rhizome::CudaRasterizer::rasterize_root_nodes_gpu(
        nodes.data(), nodes.size(),
        voxel_size, half_width_voxels,
        false, 0.0f, 0.0f, 1.0f,
        *volume, gpu_ms
    );
    return volume;
}

std::shared_ptr<rhizome::CudaVolumeGrid> RootImplicitField::rasterize_root_graph(
    const std::vector<ColonizationNode>& nodes,
    float voxel_size,
    float trunk_caliber,
    float tendril_caliber,
    float taper_curve,
    float half_width_voxels,
    bool preserve_beaded,
    float branch_smooth_k
) {
    auto volume = std::make_shared<rhizome::CudaVolumeGrid>();
    volume->voxel_size = voxel_size;
    if (nodes.empty()) {
        return volume;
    }

    float gpu_ms = 0.0f;
    rhizome::CudaRasterizer::rasterize_root_nodes_gpu(
        nodes.data(), nodes.size(),
        voxel_size, half_width_voxels,
        true, trunk_caliber, tendril_caliber, taper_curve,
        *volume, gpu_ms, preserve_beaded, branch_smooth_k
    );
    return volume;
}

std::shared_ptr<rhizome::CudaVolumeGrid> RootImplicitField::intersect_with_vessel(
    std::shared_ptr<rhizome::CudaVolumeGrid> root_grid,
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    float wall_cut_smooth,
    bool preserve_input_grid
) {
    if (!root_grid || !root_grid->is_valid()) return root_grid;
    if (!vessel_grid || !vessel_grid->is_valid()) return root_grid;

    // v24.10 Track E R2: preserve_input_grid=false skips the full-grid host
    // copy for growth bakes, where root_grid is a fresh local rasterization
    // the caller drops immediately after this call. The default (true) keeps
    // the historical copy semantics for every other caller (verification
    // harness and panels included).
    std::shared_ptr<rhizome::CudaVolumeGrid> result = preserve_input_grid
        ? std::make_shared<rhizome::CudaVolumeGrid>(*root_grid)
        : root_grid;
    if (!rhizome::CudaRasterizer::intersect_volumes_gpu(*result, *vessel_grid, wall_cut_smooth)) {
        std::cerr << "[CUDA Error] GPU volumetric intersection failed in intersect_with_vessel.\n";
    }

    return result;
}
