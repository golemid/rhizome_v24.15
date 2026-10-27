#pragma once

#include "engine/io/mesh.hpp"
#include <string>
#include <vector>
#include <glm/glm.hpp>

struct VerificationResult {
    std::string primitive_name;
    bool passed = false;
    rhizome::TriangleMesh result_mesh;
    size_t input_triangles = 0;
    size_t extracted_vertices = 0;
    size_t extracted_triangles = 0;
    size_t sample_count = 0;
    float max_deviation_mm = 0.0f;
    float midpoint_max_deviation_mm = 0.0f;
    float mean_deviation_mm = 0.0f;
    float rms_deviation_mm = 0.0f;
    std::string details;
};

class VerificationHarness {
public:
    static std::vector<VerificationResult> run_analytical_suite();
    static std::vector<VerificationResult> run_all() { return run_analytical_suite(); }

    // Phase 1 analytical primitives with vertex + midpoint sampling
    static VerificationResult run_sphere_test(float radius, float voxel_size);
    static VerificationResult run_torus_test(float major_r, float minor_r, float voxel_size);
    static VerificationResult run_filleted_box_test(glm::vec3 half_extents, float fillet_r, float voxel_size);

    // Phase 2 additions
    static VerificationResult run_capsule_chain_test(float voxel_size = 0.35f);
    static VerificationResult run_dirty_mesh_gwn_test();
    static VerificationResult run_root_containment_and_manifold_test();
    static void run_profiling_benchmark_60k();

    // Phase 3 additions: Imported vessels, precision, and GPU benchmark
    static VerificationResult run_end_to_end_vessel_pipeline_test();
    static VerificationResult run_round_trip_precision_test();
    static VerificationResult run_cuda_rasterization_benchmark();

    // Phase 4 additions: Eikonal metric integrity, stencil self-test, and exact offsetting
    static VerificationResult run_gradient_stencil_self_test();
    static VerificationResult run_eikonal_property_test();
    static VerificationResult run_offset_exactness_test();

    // Phase 5 additions: CUDA 5-Point Full Verification Suite
    static VerificationResult run_cuda_mesh_to_sdf_test();
    static VerificationResult run_cuda_spore_search_test();
    static VerificationResult run_cuda_capsule_csg_test();
    static VerificationResult run_cuda_marching_cubes_test();
    static VerificationResult run_cuda_async_worker_test();
    static std::vector<VerificationResult> run_cuda_5point_suite();

    // Phase 6 additions: Natural Root-Bound (Volumetric) 3D Colonization Verification
    static VerificationResult run_natural_root_bound_sphere_test();
    static VerificationResult run_natural_root_bound_box_test();

    // Phase 0 additions: Headless GPU Viewport Raymarching, VBO Packing, Seed Picking & STL/Mesh Ops Smoke Test
    static VerificationResult run_gpu_viewport_and_stl_smoke_test();

    // Phase 6 additions: Headless Growth Algorithm .json and .cu Export/Import Round-Trip Test (Test 20)
    static VerificationResult run_algorithm_io_round_trip_test();

    // Procedural test mesh and vessel generators
    static rhizome::TriangleMesh generate_sphere(float radius, int rings = 96, int sectors = 96);
    static rhizome::TriangleMesh generate_torus(float major_r, float minor_r, int rings = 96, int sectors = 96);
    static rhizome::TriangleMesh generate_filleted_box(glm::vec3 half_extents, float fillet_r);
    static void repair_mesh_topology(rhizome::TriangleMesh& mesh);
    static void ensure_test_vessels_exist();
};

namespace rhizome {
    using VerificationHarness = ::VerificationHarness;
    using VerificationResult = ::VerificationResult;
}

