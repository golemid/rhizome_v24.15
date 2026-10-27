#include "engine/sdf/narrow_band_sdf.hpp"
#include "engine/io/parsers.hpp"
#include <iostream>
#include <sstream>
#include <vector>

namespace rhizome {

std::shared_ptr<CudaVolumeGrid> NarrowBandSdf::build_from_mesh(
    const TriangleMesh& mesh,
    const SdfConfig& config,
    std::string& out_log
) {
    std::ostringstream ss;
    ss << "[CUDA NarrowBandSdf] Starting exact GPU Euclidean SDF construction...\n";
    ss << "  Target Voxel Size: " << config.voxel_size << " mm\n";

    TriangleMesh working_mesh = mesh;
    if (config.enable_winding_normalizer && working_mesh.triangle_count() > 0) {
        MeshWindingPass::orient_mesh(working_mesh);
    }

    float time_ms = 0.0f;
    auto volume = CudaRasterizer::build_from_mesh(working_mesh, config.voxel_size, time_ms);
    if (!volume || !volume->is_valid()) {
        out_log = "CUDA Mesh-to-SDF voxel generation failed";
        return nullptr;
    }

    ss << "  CUDA GPU 3D Volume built successfully in " << time_ms << " ms ("
       << volume->dims.x << "x" << volume->dims.y << "x" << volume->dims.z << " = "
       << volume->data.size() << " voxels)\n";
    out_log = ss.str();
    return volume;
}

TriangleMesh NarrowBandSdf::extract_mesh(
    std::shared_ptr<CudaVolumeGrid> grid,
    float isovalue,
    float adaptivity
) {
    (void)adaptivity;
    if (!grid || !grid->is_valid()) {
        return TriangleMesh();
    }
    return CudaRasterizer::extract_iso_surface(*grid, isovalue);
}

} // namespace rhizome
