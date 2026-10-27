#include "verification/harness.hpp"
#include "harness_internal.hpp"
#include "engine/sdf/narrow_band_sdf.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace rhizome;

void VerificationHarness::repair_mesh_topology(TriangleMesh& mesh) {
    if (mesh.indices.empty()) return;
    float time_ms = 0.0f;
    CudaRasterizer::repair_mesh_manifold_gpu(mesh, time_ms);
}

TriangleMesh VerificationHarness::generate_sphere(float radius, int rings, int sectors) {
    TriangleMesh mesh = CudaRasterizer::generate_sphere_gpu(radius, rings, sectors);
    repair_mesh_topology(mesh);
    return mesh;
}

TriangleMesh VerificationHarness::generate_torus(float major_r, float minor_r, int rings, int sectors) {
    TriangleMesh mesh = CudaRasterizer::generate_torus_gpu(major_r, minor_r, rings, sectors);
    repair_mesh_topology(mesh);
    return mesh;
}

TriangleMesh VerificationHarness::generate_filleted_box(glm::vec3 half_extents, float fillet_r) {
    TriangleMesh mesh = CudaRasterizer::generate_filleted_box_gpu(half_extents, fillet_r, 48);
    repair_mesh_topology(mesh);
    return mesh;
}

VerificationResult VerificationHarness::run_sphere_test(float radius, float voxel_size) {
    VerificationResult res;
    res.primitive_name = "Analytical Sphere (R = " + std::to_string(radius) + "mm)";
    TriangleMesh input_mesh = generate_sphere(radius, 96, 96);
    res.input_triangles = input_mesh.triangle_count();
    NarrowBandSdf pipeline;
    SdfConfig cfg;
    cfg.voxel_size = voxel_size;
    cfg.half_width_voxels = 4.0f;
    std::string log;
    auto grid = pipeline.build_from_mesh(input_mesh, cfg, log);
    if (!grid) {
        res.passed = false;
        res.details = "SDF pipeline failed: " + log;
        return res;
    }
    TriangleMesh extracted = NarrowBandSdf::extract_mesh(grid, 0.0f, 0.0f);
    res.result_mesh = extracted;
    res.extracted_vertices = extracted.vertex_count();
    res.extracted_triangles = extracted.triangle_count();
    auto stats = evaluate_mesh_deviations_on_gpu(extracted, 0, radius);
    res.sample_count = stats.sample_count;
    res.max_deviation_mm = stats.vertex_max;
    res.midpoint_max_deviation_mm = stats.midpoint_max;
    res.mean_deviation_mm = stats.mean;
    res.rms_deviation_mm = stats.rms;
    res.passed = (res.sample_count > 0 &&
        std::isfinite(res.mean_deviation_mm) &&
        std::isfinite(res.rms_deviation_mm) &&
        res.max_deviation_mm <= 0.1f &&
        res.midpoint_max_deviation_mm <= 0.1f);
    std::ostringstream ss;
    ss << "Sphere R=" << radius << "mm | Extracted Verts: " << res.extracted_vertices
       << " | Samples: " << res.sample_count
       << " | Vertex Max: " << res.max_deviation_mm << "mm"
       << " | Midpoint Max: " << res.midpoint_max_deviation_mm << "mm"
       << " | Mean: " << res.mean_deviation_mm << "mm"
       << " | RMS: " << res.rms_deviation_mm << "mm"
       << " | Status: " << (res.passed ? "PASS" : "FAIL");
    res.details = ss.str();
    return res;
}

VerificationResult VerificationHarness::run_torus_test(float major_r, float minor_r, float voxel_size) {
    VerificationResult res;
    res.primitive_name = "Analytical Torus (R = " + std::to_string(major_r) + "mm, r = " + std::to_string(minor_r) + "mm)";
    TriangleMesh input_mesh = generate_torus(major_r, minor_r, 96, 96);
    res.input_triangles = input_mesh.triangle_count();
    NarrowBandSdf pipeline;
    SdfConfig cfg;
    cfg.voxel_size = voxel_size;
    cfg.half_width_voxels = 4.0f;
    std::string log;
    auto grid = pipeline.build_from_mesh(input_mesh, cfg, log);
    if (!grid) {
        res.passed = false;
        res.details = "SDF pipeline failed: " + log;
        return res;
    }
    TriangleMesh extracted = NarrowBandSdf::extract_mesh(grid, 0.0f, 0.0f);
    res.result_mesh = extracted;
    res.extracted_vertices = extracted.vertex_count();
    res.extracted_triangles = extracted.triangle_count();
    auto stats = evaluate_mesh_deviations_on_gpu(extracted, 1, major_r, minor_r);
    res.sample_count = stats.sample_count;
    res.max_deviation_mm = stats.vertex_max;
    res.midpoint_max_deviation_mm = stats.midpoint_max;
    res.mean_deviation_mm = stats.mean;
    res.rms_deviation_mm = stats.rms;
    res.passed = (res.sample_count > 0 &&
        std::isfinite(res.mean_deviation_mm) &&
        std::isfinite(res.rms_deviation_mm) &&
        res.max_deviation_mm <= 0.1f &&
        res.midpoint_max_deviation_mm <= 0.1f);
    std::ostringstream ss;
    ss << "Torus R=" << major_r << ", r=" << minor_r << "mm | Extracted Verts: " << res.extracted_vertices
       << " | Samples: " << res.sample_count
       << " | Vertex Max: " << res.max_deviation_mm << "mm"
       << " | Midpoint Max: " << res.midpoint_max_deviation_mm << "mm"
       << " | Mean: " << res.mean_deviation_mm << "mm"
       << " | RMS: " << res.rms_deviation_mm << "mm"
       << " | Status: " << (res.passed ? "PASS" : "FAIL");
    res.details = ss.str();
    return res;
}

VerificationResult VerificationHarness::run_filleted_box_test(glm::vec3 half_extents, float fillet_r, float voxel_size) {
    VerificationResult res;
    res.primitive_name = "Filleted Box (40x40x20mm, r = 2.000000mm)";
    TriangleMesh input_mesh = generate_filleted_box(half_extents, fillet_r);
    res.input_triangles = input_mesh.triangle_count();
    NarrowBandSdf pipeline;
    SdfConfig cfg;
    cfg.voxel_size = voxel_size;
    cfg.half_width_voxels = 4.0f;
    std::string log;
    auto grid = pipeline.build_from_mesh(input_mesh, cfg, log);
    if (!grid) {
        res.passed = false;
        res.details = "SDF pipeline failed: " + log;
        return res;
    }
    TriangleMesh extracted = NarrowBandSdf::extract_mesh(grid, 0.0f, 0.0f);
    res.result_mesh = extracted;
    res.extracted_vertices = extracted.vertex_count();
    res.extracted_triangles = extracted.triangle_count();
    auto stats = evaluate_mesh_deviations_on_gpu(extracted, 2, fillet_r, 0.0f, half_extents);
    res.sample_count = stats.sample_count;
    res.max_deviation_mm = stats.vertex_max;
    res.midpoint_max_deviation_mm = stats.midpoint_max;
    res.mean_deviation_mm = stats.mean;
    res.rms_deviation_mm = stats.rms;
    res.passed = (res.sample_count > 0 &&
        std::isfinite(res.mean_deviation_mm) &&
        std::isfinite(res.rms_deviation_mm) &&
        res.max_deviation_mm <= 0.1f &&
        res.midpoint_max_deviation_mm <= 0.1f);
    std::ostringstream ss;
    ss << "Filleted Box | Extracted Verts: " << res.extracted_vertices
       << " | Samples: " << res.sample_count
       << " | Vertex Max: " << res.max_deviation_mm << "mm"
       << " | Midpoint Max: " << res.midpoint_max_deviation_mm << "mm"
       << " | Mean: " << res.mean_deviation_mm << "mm"
       << " | RMS: " << res.rms_deviation_mm << "mm"
       << " | Status: " << (res.passed ? "PASS" : "FAIL");
    res.details = ss.str();
    return res;
}

std::vector<VerificationResult> VerificationHarness::run_analytical_suite() {
    std::vector<VerificationResult> results;
    std::cout << "\n=================================================================\n";
    std::cout << "  RHIZOME NATIVE: COMPLETE VERIFICATION & ACCELERATION SUITE\n";
    std::cout << "  Acceptance Criteria: Vertex & Midpoint Max Deviation <= 0.100 mm\n";
    std::cout << "=================================================================\n";
    // Test 1: Sphere R = 20.0mm
    std::cout << "[Test 1/20] Running Sphere (R = 20mm)..." << std::flush;
    results.push_back(run_sphere_test(20.0f, 0.40f));
    std::cout << " DONE.\n" << results.back().details << "\n";
    // Test 2: Torus R = 25.0mm, r = 5.0mm
    std::cout << "[Test 2/20] Running Torus (Major=25mm, Minor=5mm)..." << std::flush;
    results.push_back(run_torus_test(25.0f, 5.0f, 0.35f));
    std::cout << " DONE.\n" << results.back().details << "\n";
    // Test 3: Filleted Box
    std::cout << "[Test 3/20] Running Filleted Box..." << std::flush;
    results.push_back(run_filleted_box_test({20.0f, 20.0f, 10.0f}, 2.0f, 0.35f));
    std::cout << " DONE.\n" << results.back().details << "\n";
    // Test 4: Capsule Chain
    std::cout << "[Test 4/20] Running Capsule Chain (Metric Integrity)..." << std::flush;
    results.push_back(run_capsule_chain_test(0.25f));
    std::cout << " DONE.\n" << results.back().details << "\n";
    // Test 5: Dirty Mesh GWN
    std::cout << "[Test 5/20] Running Dirty-Mesh GWN Sign Classification..." << std::flush;
    results.push_back(run_dirty_mesh_gwn_test());
    std::cout << " DONE.\n" << results.back().details << "\n";
    // Test 6: Root Containment & Manifold
    std::cout << "[Test 6/20] Running Root Containment & 2-Manifold Check..." << std::flush;
    results.push_back(run_root_containment_and_manifold_test());
    std::cout << " DONE.\n" << results.back().details << "\n";
    // Test 7: End-to-End Imported-Vessel Pipeline (3 Vessels)
    std::cout << "[Test 7/20] Running End-to-End Imported-Vessel Pipeline (3 Vessels)..." << std::flush;
    results.push_back(run_end_to_end_vessel_pipeline_test());
    std::cout << " DONE.\n" << results.back().details << "\n";
    // Test 8: Round-Trip Precision Test
    std::cout << "[Test 8/20] Running Round-Trip Precision Test (Re-imported OBJ vs In-Memory SDF)..." << std::flush;
    results.push_back(run_round_trip_precision_test());
    std::cout << " DONE.\n" << results.back().details << "\n";
    // Test 9: GPU CUDA Acceleration Benchmark
    std::cout << "[Test 9/20] Running CUDA GPU Narrow-Band Rasterization Benchmark..." << std::flush;
    results.push_back(run_cuda_rasterization_benchmark());
    std::cout << " DONE.\n" << results.back().details << "\n";
    // Test 10: Eikonal Property (|∇f| = 1) of Stored Narrow-Band Field
    std::cout << "[Test 10/20] Running Eikonal Property (|∇f| = 1) of Stored Narrow-Band Field..." << std::flush;
    results.push_back(run_eikonal_property_test());
    std::cout << " DONE." << results.back().details << "\n";
    // Test 11: Offset Exactness (Sphere & Torus at t = ±2.0mm)
    std::cout << "[Test 11/20] Running Offset Exactness (Sphere & Torus at t = ±2.0mm)..." << std::flush;
    results.push_back(run_offset_exactness_test());
    std::cout << " DONE.\n" << results.back().details << "\n";

    // Test 12-16: CUDA 5-Point Full Verification Suite
    std::cout << "[Test 12/20] Running CUDA Mesh-to-SDF Voxel Field..." << std::flush;
    results.push_back(run_cuda_mesh_to_sdf_test());
    std::cout << " DONE.\n" << results.back().details << "\n";

    std::cout << "[Test 13/20] Running CUDA Spore Search (25,000 Spores)..." << std::flush;
    results.push_back(run_cuda_spore_search_test());
    std::cout << " DONE.\n" << results.back().details << "\n";

    std::cout << "[Test 14/20] Running CUDA Root Capsule CSG Blending..." << std::flush;
    results.push_back(run_cuda_capsule_csg_test());
    std::cout << " DONE.\n" << results.back().details << "\n";

    std::cout << "[Test 15/20] Running CUDA Marching Cubes Iso-Surface Extractor..." << std::flush;
    results.push_back(run_cuda_marching_cubes_test());
    std::cout << " DONE.\n" << results.back().details << "\n";

    std::cout << "[Test 16/20] Running Async CUDA Stream Background Worker..." << std::flush;
    results.push_back(run_cuda_async_worker_test());
    std::cout << " DONE.\n" << results.back().details << "\n";

    // Test 17 & 18: Roots (Volumetric Root-Bound) 3D Colonization Verification
    std::cout << "[Test 17/20] Running Roots in Sphere (R = 20mm)..." << std::flush;
    results.push_back(run_natural_root_bound_sphere_test());
    std::cout << " DONE.\n" << results.back().details << "\n";

    std::cout << "[Test 18/20] Running Roots in Box (40x40x40mm)..." << std::flush;
    results.push_back(run_natural_root_bound_box_test());
    std::cout << " DONE.\n" << results.back().details << "\n";

    std::cout << "[Test 19/20] Running GPU Viewport Raymarch, VBO Packers & Binary STL Smoke Gate..." << std::flush;
    results.push_back(run_gpu_viewport_and_stl_smoke_test());
    std::cout << " DONE.\n" << results.back().details << "\n";

    // Test 20: Algorithm I/O Headless Serialization Round-Trip (.json & .cu)
    std::cout << "[Test 20/20] Running Algorithm I/O Headless Serialization Round-Trip (.json & .cu)..." << std::flush;
    results.push_back(run_algorithm_io_round_trip_test());
    std::cout << " DONE.\n" << results.back().details << "\n";

    std::cout << "\n=================================================================\n";
    bool all_passed = true;
    for (const auto& r : results) {
        if (!r.passed) all_passed = false;
    }
    std::cout << "  OVERALL SUITE RESULT: " << (all_passed ? "PASSED ALL 20 TESTS" : "FAILED") << "\n";
    std::cout << "=================================================================\n";
    run_profiling_benchmark_60k();
    return results;
}
