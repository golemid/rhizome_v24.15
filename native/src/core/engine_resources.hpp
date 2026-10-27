#pragma once
// =========================================================================
// RHIZOME ENGINE RESOURCES — core module (v23.29 Phase 2)
// =========================================================================
// The live GPU / OpenGL resource bundle (Phase 1 proposal Sec 5.1):
//   * AppState  (core/app_state.hpp)  — every persistable / multi-subsystem
//     value. Single Source of Truth. Never duplicated in main.cpp locals.
//   * EngineResources (this file)     — every LIVE, non-persistable
//     simulation & render resource (mesh storages, GL buffers, CUDA volume
//     grids, camera, colonizer session, cached metrics). Owned by main.cpp
//     and passed by reference into controllers and panels.
//
// NOTE: the viewport include was relocated to ui/viewport/viewport_renderer.hpp
// in Phase 4. The struct itself is final.
// =========================================================================

#include <memory>
#include <string>
#include <vector>

#include "engine/io/mesh.hpp"                  // TriangleMesh
#include "engine/colonization/colonization.hpp"  // SpaceColonization, ColonizationNode
#include "engine/sdf/cuda_types.hpp"             // CudaVolumeGrid
#include "verification/harness.hpp"       // VerificationResult
#include "ui/viewport/viewport_renderer.hpp"     // Camera3D, GL buffer structs, TestMeshObject
                                             // (relocated from app/ in Phase 4)

namespace rhizome {

// -------------------------------------------------------------------------
// Mesh quality metrics (GPU-computed via CudaRasterizer::compute_mesh_metrics_gpu
// — zero CPU fallback). Migrated from app/ui_panels.hpp in Phase 2 so that
// core code (growth controller) has a single authoritative definition.
// Final home: cached STATE belongs with EngineResources in core/, not with
// the engine/mesh/ kernels that compute it.
// -------------------------------------------------------------------------
struct CachedMeshMetrics {
    size_t total_tris = 0;
    size_t total_edges = 0;
    float avg_edge_mm = 0.0f;
    float min_edge_mm = 0.0f;
    float max_edge_mm = 0.0f;
};

CachedMeshMetrics compute_mesh_metrics(const rhizome::TriangleMesh& mesh);

// -------------------------------------------------------------------------
// EngineResources — live, non-persistable simulation & render resources.
// -------------------------------------------------------------------------
struct EngineResources {
    // Host-side mesh storages (GPU-mirrored)
    TriangleMesh original_mesh;
    TriangleMesh converted_body_mesh;
    TriangleMesh roots_mesh;
    TriangleMesh generated_merged_mesh;

    // GPU volume grids (CUDA device memory)
    std::shared_ptr<CudaVolumeGrid> current_sdf_grid;
    std::shared_ptr<CudaVolumeGrid> current_roots_grid;

    // OpenGL buffers
    GLMeshBuffers orig_gl_buffers;
    GLMeshBuffers body_gl_buffers;
    GLMeshBuffers roots_gl_buffers;
    GLMeshBuffers merged_gl_buffers;
    GLMeshBuffers seed_sphere_buffer;   // unit-sphere marker used for seed rendering
    BoundingBoxBuffers bbox_buffers;
    FloorGridBuffers floor_grid_buffers;
    BranchSkeletonBuffers branch_gl_buffers;
    SporeCloudBuffers spore_gl_buffers;

    // Camera & live growth session
    Camera3D camera;
    SpaceColonization live_colonizer;

    // Verification / telemetry
    std::vector<TestMeshObject> test_objects;
    std::vector<VerificationResult> last_test_results;
    std::vector<ColonizationNode> current_branch_nodes;

    // Cached GPU mesh metrics
    CachedMeshMetrics cached_roots_metrics;
    CachedMeshMetrics cached_body_metrics;
    CachedMeshMetrics cached_merged_metrics;

    // Display name of the loaded specimen (resource metadata, not UI state)
    std::string loaded_model_name = "Default Shape";
};

} // namespace rhizome
