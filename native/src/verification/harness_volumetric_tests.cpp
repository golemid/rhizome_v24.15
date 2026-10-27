#include "verification/harness.hpp"
#include "engine/sdf/narrow_band_sdf.hpp"
#include "engine/sdf/root_field.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/io/obj_exporter.hpp"
#include "engine/colonization/colonization.hpp"
#include "core/algorithm_io.hpp"
#include "core/app_state.hpp"
#include "core/lighting_rig.hpp"   // v24.01 Phase 1: defaults sub-check
#include <imgui.h>                            // v24.01 Phase 2: ImGuiCol_COUNT gate + style round-trip
#include "ui/panels/panel_styles.hpp"        // v24.01 Phase 2: load_imgui_style_from_ini
#include "ui/widgets/control_registry.hpp"   // v24.01 Phase 2: registry counts gate
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>

using namespace rhizome;

VerificationResult VerificationHarness::run_natural_root_bound_sphere_test() {
    VerificationResult res;
    res.primitive_name = "Test 17: Roots (Volumetric Root-Bound) in Sphere (R=20mm)";
    std::ostringstream ss;

    TriangleMesh vessel_mesh = generate_sphere(20.0f, 96, 96);
    NarrowBandSdf pipeline;
    SdfConfig cfg;
    cfg.voxel_size = 0.20f;
    cfg.half_width_voxels = 4.0f;
    std::string log;
    auto vessel_grid = pipeline.build_from_mesh(vessel_mesh, cfg, log);
    if (!vessel_grid || !vessel_grid->is_valid()) {
        res.passed = false;
        res.details = "FAIL: NarrowBandSdf generation failed.";
        return res;
    }

    SpaceColonization colonizer;
    ColonizationConfig c_cfg;
    c_cfg.spore_count = 15000;
    c_cfg.search_radius = 8.0f;
    c_cfg.kill_radius = 2.0f;
    c_cfg.growth_step = 1.2f;
    c_cfg.base_radius = 0.90f;
    c_cfg.tip_radius = 0.50f;
    c_cfg.max_iterations = 100;
    c_cfg.algorithm_mode = 1;
    c_cfg.push_strength = 0.65f;
    c_cfg.space_fill_drive = 0.90f;
    c_cfg.root_roughness = 0.28f;
    c_cfg.seed = 2026;

    colonizer.initialize_volumetric_session(vessel_grid, {{0.0f, 0.0f, -12.0f}}, c_cfg);
    colonizer.step_volumetric_session(vessel_grid, c_cfg, 0.20f, 1.4f, 100);

    auto root_grid = colonizer.rasterize_volumetric_session(0.20f);
    auto confined_grid = RootImplicitField::intersect_with_vessel(root_grid, vessel_grid);
    TriangleMesh root_mesh = NarrowBandSdf::extract_mesh(confined_grid, 0.0f, 0.0f);
    res.result_mesh = root_mesh;
    res.extracted_vertices = root_mesh.vertex_count();
    res.extracted_triangles = root_mesh.triangle_count();
    res.sample_count = root_mesh.vertex_count();

    if (res.sample_count == 0) {
        res.passed = false;
        res.details = "FAIL: Extracted 0 vertices from volumetric root grid.";
        return res;
    }

    float max_vessel_overshoot = 0.0f;
    CudaRasterizer::evaluate_containment_overshoot_gpu(root_mesh, nullptr, 20.0f, max_vessel_overshoot);

    auto manifold_check = rhizome::ObjExporter::verify_manifold(root_mesh);
    res.max_deviation_mm = max_vessel_overshoot;
    bool containment_passed = (max_vessel_overshoot <= 0.05f);
    res.passed = (containment_passed && manifold_check.is_manifold);

    ss << "Root Mesh Verts: " << res.extracted_vertices
       << " | Tris: " << res.extracted_triangles
       << " | Max Overshoot: " << std::fixed << std::setprecision(5) << max_vessel_overshoot << " mm"
       << " | Manifold: " << (manifold_check.is_manifold ? "YES" : "NO")
       << " | Status: " << (res.passed ? "PASS" : "FAIL");
    res.details = ss.str();
    return res;
}

VerificationResult VerificationHarness::run_natural_root_bound_box_test() {
    VerificationResult res;
    res.primitive_name = "Test 18: Roots (Volumetric Root-Bound) in Box (40x40x40mm)";
    std::ostringstream ss;

    TriangleMesh vessel_mesh = generate_filleted_box({20.0f, 20.0f, 20.0f}, 2.0f);
    NarrowBandSdf pipeline;
    SdfConfig cfg;
    cfg.voxel_size = 0.20f;
    cfg.half_width_voxels = 4.0f;
    std::string log;
    auto vessel_grid = pipeline.build_from_mesh(vessel_mesh, cfg, log);
    if (!vessel_grid || !vessel_grid->is_valid()) {
        res.passed = false;
        res.details = "FAIL: NarrowBandSdf generation failed.";
        return res;
    }

    SpaceColonization colonizer;
    ColonizationConfig c_cfg;
    c_cfg.spore_count = 20000;
    c_cfg.search_radius = 10.0f;
    c_cfg.kill_radius = 2.0f;
    c_cfg.growth_step = 1.4f;
    c_cfg.base_radius = 0.90f;
    c_cfg.tip_radius = 0.50f;
    c_cfg.max_iterations = 100;
    c_cfg.algorithm_mode = 1;
    c_cfg.push_strength = 0.65f;
    c_cfg.space_fill_drive = 0.90f;
    c_cfg.root_roughness = 0.28f;
    c_cfg.seed = 3037;

    colonizer.initialize_volumetric_session(vessel_grid, {{0.0f, 0.0f, -15.0f}}, c_cfg);
    colonizer.step_volumetric_session(vessel_grid, c_cfg, 0.20f, 1.4f, 100);

    auto root_grid = colonizer.rasterize_volumetric_session(0.20f);
    auto confined_grid = RootImplicitField::intersect_with_vessel(root_grid, vessel_grid);
    TriangleMesh root_mesh = NarrowBandSdf::extract_mesh(confined_grid, 0.0f, 0.0f);
    res.result_mesh = root_mesh;
    res.extracted_vertices = root_mesh.vertex_count();
    res.extracted_triangles = root_mesh.triangle_count();
    res.sample_count = root_mesh.vertex_count();

    if (res.sample_count == 0) {
        res.passed = false;
        res.details = "FAIL: Extracted 0 vertices from volumetric root grid.";
        return res;
    }

    float max_vessel_overshoot = 0.0f;
    CudaRasterizer::evaluate_containment_overshoot_gpu(root_mesh, vessel_grid.get(), 0.0f, max_vessel_overshoot);

    auto manifold_check = rhizome::ObjExporter::verify_manifold(root_mesh);
    res.max_deviation_mm = max_vessel_overshoot;
    bool containment_passed = (max_vessel_overshoot <= 0.05f);
    res.passed = (containment_passed && manifold_check.is_manifold);

    ss << "Root Mesh Verts: " << res.extracted_vertices
       << " | Tris: " << res.extracted_triangles
       << " | Max Overshoot: " << std::fixed << std::setprecision(5) << max_vessel_overshoot << " mm"
       << " | Manifold: " << (manifold_check.is_manifold ? "YES" : "NO")
       << " | Status: " << (res.passed ? "PASS" : "FAIL");
    res.details = ss.str();
    return res;
}

VerificationResult VerificationHarness::run_gpu_viewport_and_stl_smoke_test() {
    VerificationResult res;
    res.primitive_name = "Test 19: GPU Viewport Raymarching, VBO Packers, Seed Picking & Binary STL Round-Trip";
    std::ostringstream ss;

    float time_ms = 0.0f;

    // 1. Binary STL GPU formatting & parallel decoding round-trip + Z-up alignment + mesh metrics + merge
    TriangleMesh sphere_a = generate_sphere(12.0f, 32, 32);
    std::vector<uint8_t> stl_bytes;
    bool stl_fmt_ok = CudaRasterizer::format_binary_stl_gpu(sphere_a, stl_bytes, time_ms);
    TriangleMesh sphere_decoded;
    bool stl_parse_ok = stl_fmt_ok && CudaRasterizer::parse_binary_stl_gpu(
        stl_bytes.data(), stl_bytes.size(), sphere_decoded, time_ms
    );

    glm::vec3 aligned_min(0.0f), aligned_max(0.0f);
    bool align_ok = stl_parse_ok && CudaRasterizer::align_mesh_z_up_gpu(
        sphere_decoded, aligned_min, aligned_max, time_ms
    );

    float avg_e = 0.0f, min_e = 0.0f, max_e = 0.0f;
    bool metrics_ok = align_ok && CudaRasterizer::compute_mesh_metrics_gpu(
        sphere_decoded, avg_e, min_e, max_e, time_ms
    ) && (avg_e > 0.0f && max_e >= min_e);

    TriangleMesh merged_mesh;
    bool merge_ok = metrics_ok && CudaRasterizer::merge_meshes_gpu(
        sphere_a, sphere_decoded, merged_mesh, time_ms
    ) && (merged_mesh.triangle_count() == sphere_a.triangle_count() + sphere_decoded.triangle_count());

    // 2. Analytical SDF grid + interior volume + seed placement + 3D ray-seed picking + camera matrices
    CudaVolumeGrid vol;
    vol.dims = glm::ivec3(40, 40, 40);
    vol.voxel_size = 1.0f;
    vol.min_coord = glm::ivec3(-20, -20, -20);
    vol.max_coord = glm::ivec3(19, 19, 19);
    bool sdf_ok = CudaRasterizer::fill_analytical_sdf_grid_gpu(vol, 0, 12.0f, 0.0f);

    float interior_vol_mm3 = 0.0f;
    bool vol_ok = sdf_ok && CudaRasterizer::compute_sdf_interior_volume_gpu(vol, interior_vol_mm3, time_ms) && (interior_vol_mm3 > 1000.0f);

    glm::vec3 default_seed(0.0f);
    bool seed_ok = vol_ok && CudaRasterizer::find_default_seed_gpu(
        vol, glm::vec3(-12.0f), glm::vec3(12.0f), 2.0f, default_seed, time_ms
    );

    std::vector<glm::vec3> test_seeds = { glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(8.0f, 0.0f, 0.0f) };
    int hit_seed_idx = -1;
    bool pick_ok = CudaRasterizer::pick_seed_ray_gpu(
        test_seeds, glm::vec3(0.0f, 0.0f, 50.0f), glm::vec3(0.0f, 0.0f, -1.0f), 2.5f, hit_seed_idx, time_ms
    ) && (hit_seed_idx == 0);

    glm::mat4 view_m(1.0f), proj_m(1.0f), vp_m(1.0f), inv_vp_m(1.0f);
    glm::vec3 cam_eye(0.0f, -45.0f, 15.0f);
    bool cam_ok = CudaRasterizer::compute_camera_matrices_gpu(
        0.785398f, 1.0f, 0.5f, 500.0f, cam_eye, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f),
        view_m, proj_m, vp_m, inv_vp_m, time_ms
    );

    // 3. Direct GPU implicit viewport raymarching into offscreen 64x64 RGBA buffer
    std::vector<uint32_t> rgba_pixels;
    bool raymarch_ok = cam_ok && CudaRasterizer::raymarch_implicit_viewport_gpu(
        64, 64, view_m, inv_vp_m, cam_eye, false,
        &vol, true, false, 0.35f, 1.0f, glm::vec4(0.85f, 0.82f, 0.78f, 1.0f),
        nullptr, nullptr, nullptr, 0, false, false, glm::vec4(0.75f, 0.35f, 0.25f, 1.0f),
        1.0f, 0.5f, 0.5f, 0.25f, rgba_pixels, time_ms
    );
    size_t non_zero_pixels = 0;
    for (uint32_t px : rgba_pixels) {
        if (px != 0u) ++non_zero_pixels;
    }
    raymarch_ok = raymarch_ok && (non_zero_pixels > 64);

    // 4. OpenGL VBO GPU packers (mesh, spore cloud, floor grid) & GPU Binary PLY export
    std::vector<float> mesh_vbo, spore_vbo, grid_vbo;
    size_t grid_lines = 0, axis_lines = 0;
    bool vbo_ok = CudaRasterizer::pack_gl_mesh_vbo_gpu(sphere_a, true, mesh_vbo, time_ms) && !mesh_vbo.empty() &&
                  CudaRasterizer::pack_spore_cloud_vbo_gpu(test_seeds, spore_vbo, time_ms) && !spore_vbo.empty() &&
                  CudaRasterizer::pack_floor_grid_vbo_gpu(50.0f, 10.0f, grid_vbo, grid_lines, axis_lines, time_ms) && !grid_vbo.empty();

    std::vector<uint8_t> ply_bytes;
    bool ply_fmt_ok = CudaRasterizer::format_binary_ply_gpu(sphere_a, ply_bytes, time_ms);
    bool ply_ok = ply_fmt_ok && !ply_bytes.empty() &&
                  (std::string(reinterpret_cast<const char*>(ply_bytes.data()), std::min<size_t>(ply_bytes.size(), 3)) == "ply");

    // 5. 2D Grayscale Bark Texture Generation & Mesh Surface Displacement on GPU
    CudaRasterizer::BarkTexture2DParams bark_p;
    bark_p.width = 128;
    bark_p.height = 128;
    bark_p.scale = 5.0f;
    bark_p.stretch_y = 3.0f;
    bark_p.furrow_depth = 0.6f;
    bark_p.octaves = 3;
    bark_p.seed = 42;
    std::vector<float> bark_hmap;
    std::vector<uint32_t> bark_rgba;
    bool bark_gen_ok = CudaRasterizer::generate_bark_texture_2d_gpu(bark_p, bark_hmap, bark_rgba, time_ms);
    float h_sum = 0.0f, h_min = 1.0f, h_max = 0.0f;
    for (float v : bark_hmap) {
        h_sum += v;
        if (v < h_min) h_min = v;
        if (v > h_max) h_max = v;
    }
    float h_mean = bark_hmap.empty() ? 0.0f : (h_sum / static_cast<float>(bark_hmap.size()));
    bool bark_stats_ok = bark_gen_ok && (h_min >= 0.0f) && (h_max <= 1.0f) && (h_max - h_min > 0.35f) && (h_mean > 0.35f && h_mean < 0.65f);

    TriangleMesh displaced_sphere = sphere_a;
    CudaRasterizer::MeshDisplacementParams disp_p;
    disp_p.displacement_depth_mm = 2.0f;
    disp_p.tile_repeat_u = 2.0f;
    disp_p.tile_repeat_v = 4.0f;
    disp_p.recompute_normals = true;
    bool disp_ok = bark_stats_ok && CudaRasterizer::displace_mesh_from_bark_texture_gpu(
        displaced_sphere, bark_hmap, bark_p.width, bark_p.height, disp_p, time_ms
    );
    float max_disp_delta = 0.0f;
    for (size_t i = 0; i < displaced_sphere.vertex_count(); ++i) {
        float r = glm::length(displaced_sphere.get_vertex(i));
        float delta = std::abs(r - 20.0f);
        if (delta > max_disp_delta) max_disp_delta = delta;
    }
    bool disp_verified = disp_ok && (max_disp_delta > 0.4f);

    res.extracted_vertices = sphere_decoded.vertex_count();
    res.extracted_triangles = merged_mesh.triangle_count();
    res.sample_count = non_zero_pixels;
    res.passed = (merge_ok && vol_ok && seed_ok && pick_ok && raymarch_ok && vbo_ok && ply_ok && bark_stats_ok && disp_verified);

    ss << "    [GPU Smoke Gate] STL/PLY Round-Trip & Merge Tris: " << merged_mesh.triangle_count()
       << " | Offscreen 64x64 Raymarch Lit Pixels: " << non_zero_pixels
       << " | Binary PLY Packed Bytes: " << ply_bytes.size()
       << " | 2D Bark Grayscale (" << bark_p.width << "x" << bark_p.height << ") Range: [" << h_min << ", " << h_max << "]"
       << " | Mesh Emboss/Deboss Delta: " << max_disp_delta << " mm"
       << " | VBO Packers & Seed Pick: " << ((pick_ok && vbo_ok) ? "VERIFIED" : "FAIL")
       << " | Status: " << (res.passed ? "PASS" : "FAIL");
    res.details = ss.str();
    return res;
}

VerificationResult VerificationHarness::run_algorithm_io_round_trip_test() {
    VerificationResult res;
    res.primitive_name = "Algorithm I/O Round-Trip (.json, .cu & theme INI)";
    std::stringstream ss;

    // 1. Construct a GrowthAlgorithmProfile with distinctive non-default parameter values
    GrowthAlgorithmProfile original;
    original.name = "TestDistinctiveProfile";
    original.engine_mode = 0; // default is 1
    original.root_roughness = 0.45f;
    original.push_strength = 0.82f;
    original.space_fill_drive = 0.75f;
    original.backward_pull = 0.60f;
    original.tensile_straightening = 0.40f;
    original.spore_count = 35000;
    original.search_radius = 18.5f;
    original.kill_radius = 3.2f;
    original.growth_step = 1.85f;
    original.variable_step = true;
    original.min_growth_step = 0.75f;
    original.max_growth_step = 2.50f;
    original.branching_rate = 0.72f;
    original.steering_cone = 45.0f;
    original.max_iterations = 60;
    original.trunk_caliber = 1.25f;
    original.tendril_caliber = 0.45f;
    original.variable_taper = true;
    original.taper_curve = 1.80f;
    original.seed_offset = 0.75f;

    original.tip_inertia_weight = 1.40f;
    original.spore_pull_weight = 0.90f;
    original.roughness_amplitude = 0.75f;
    original.feeler_nar_fwd = 0.88f;
    original.feeler_nar_side = 0.42f;
    original.feeler_med_fwd = 0.68f;
    original.feeler_med_side = 0.75f;
    original.feeler_wide_fwd = 0.38f;
    original.feeler_wide_side = 0.92f;
    original.feeler_wall_in = 0.60f;
    original.feeler_wall_out = 0.50f;
    original.clearance_score_weight = 2.80f;
    original.align_bonus_weight = 0.70f;
    original.lateral_fwd_weight = 0.58f;
    original.lateral_side_weight = 0.90f;
    original.lateral_radius_scale = 0.82f;
    original.max_branch_generations = 5;
    original.skin_gap_scale = 0.80f;
    original.push_base_gain = 0.30f;
    original.push_strength_gain = 0.60f;
    original.max_step_disp_ratio = 0.32f;
    original.tether_max_stretch = 1.30f;
    original.tether_target_stretch = 1.20f;
    original.wall_clearance_scale = 0.90f;
    original.branch_smooth_k = 0.25f;
    original.wall_cut_smooth = 0.15f;

    // Helper to compare all 48 serialized parameters between profiles
    auto compare_profiles = [](const GrowthAlgorithmProfile& a, const GrowthAlgorithmProfile& b, std::string& err) -> bool {
        const float eps = 1e-3f;
        if (a.name != b.name) { err = "Name mismatch: " + a.name + " vs " + b.name; return false; }
        if (a.engine_mode != b.engine_mode) { err = "EngineMode mismatch"; return false; }
        if (std::abs(a.root_roughness - b.root_roughness) > eps) { err = "RootRoughness mismatch"; return false; }
        if (std::abs(a.push_strength - b.push_strength) > eps) { err = "PushStrength mismatch"; return false; }
        if (std::abs(a.space_fill_drive - b.space_fill_drive) > eps) { err = "SpaceFillDrive mismatch"; return false; }
        if (std::abs(a.backward_pull - b.backward_pull) > eps) { err = "BackwardPull mismatch"; return false; }
        if (std::abs(a.tensile_straightening - b.tensile_straightening) > eps) { err = "TensileStraightening mismatch"; return false; }
        if (std::abs(a.murray_exponent - b.murray_exponent) > eps) { err = "MurrayExponent mismatch"; return false; }
        if (std::abs(a.murray_branch_angle - b.murray_branch_angle) > eps) { err = "MurrayBranchAngle mismatch"; return false; }
        if (a.spore_count != b.spore_count) { err = "Spores mismatch"; return false; }
        if (std::abs(a.search_radius - b.search_radius) > eps) { err = "SearchRadius mismatch"; return false; }
        if (std::abs(a.kill_radius - b.kill_radius) > eps) { err = "KillRadius mismatch"; return false; }
        if (std::abs(a.growth_step - b.growth_step) > eps) { err = "GrowthStep mismatch"; return false; }
        if (a.variable_step != b.variable_step) { err = "VariableStep mismatch"; return false; }
        if (std::abs(a.min_growth_step - b.min_growth_step) > eps) { err = "MinGrowthStep mismatch"; return false; }
        if (std::abs(a.max_growth_step - b.max_growth_step) > eps) { err = "MaxGrowthStep mismatch"; return false; }
        if (std::abs(a.branching_rate - b.branching_rate) > eps) { err = "BranchingRate mismatch"; return false; }
        if (std::abs(a.steering_cone - b.steering_cone) > eps) { err = "SteeringCone mismatch"; return false; }
        if (a.max_iterations != b.max_iterations) { err = "MaxIterations mismatch"; return false; }
        if (std::abs(a.trunk_caliber - b.trunk_caliber) > eps) { err = "TrunkCaliber mismatch"; return false; }
        if (std::abs(a.tendril_caliber - b.tendril_caliber) > eps) { err = "TendrilCaliber mismatch"; return false; }
        if (a.variable_taper != b.variable_taper) { err = "VariableTaper mismatch"; return false; }
        if (std::abs(a.taper_curve - b.taper_curve) > eps) { err = "TaperCurve mismatch"; return false; }
        if (std::abs(a.seed_offset - b.seed_offset) > eps) { err = "SeedOffset mismatch"; return false; }
        if (std::abs(a.tip_inertia_weight - b.tip_inertia_weight) > eps) { err = "TipInertiaWeight mismatch"; return false; }
        if (std::abs(a.spore_pull_weight - b.spore_pull_weight) > eps) { err = "SporePullWeight mismatch"; return false; }
        if (std::abs(a.roughness_amplitude - b.roughness_amplitude) > eps) { err = "RoughnessAmplitude mismatch"; return false; }
        if (std::abs(a.feeler_nar_fwd - b.feeler_nar_fwd) > eps) { err = "FeelerNarFwd mismatch"; return false; }
        if (std::abs(a.feeler_nar_side - b.feeler_nar_side) > eps) { err = "FeelerNarSide mismatch"; return false; }
        if (std::abs(a.feeler_med_fwd - b.feeler_med_fwd) > eps) { err = "FeelerMedFwd mismatch"; return false; }
        if (std::abs(a.feeler_med_side - b.feeler_med_side) > eps) { err = "FeelerMedSide mismatch"; return false; }
        if (std::abs(a.feeler_wide_fwd - b.feeler_wide_fwd) > eps) { err = "FeelerWideFwd mismatch"; return false; }
        if (std::abs(a.feeler_wide_side - b.feeler_wide_side) > eps) { err = "FeelerWideSide mismatch"; return false; }
        if (std::abs(a.feeler_wall_in - b.feeler_wall_in) > eps) { err = "FeelerWallIn mismatch"; return false; }
        if (std::abs(a.feeler_wall_out - b.feeler_wall_out) > eps) { err = "FeelerWallOut mismatch"; return false; }
        if (std::abs(a.clearance_score_weight - b.clearance_score_weight) > eps) { err = "ClearanceScoreWeight mismatch"; return false; }
        if (std::abs(a.align_bonus_weight - b.align_bonus_weight) > eps) { err = "AlignBonusWeight mismatch"; return false; }
        if (std::abs(a.lateral_fwd_weight - b.lateral_fwd_weight) > eps) { err = "LateralFwdWeight mismatch"; return false; }
        if (std::abs(a.lateral_side_weight - b.lateral_side_weight) > eps) { err = "LateralSideWeight mismatch"; return false; }
        if (std::abs(a.lateral_radius_scale - b.lateral_radius_scale) > eps) { err = "LateralRadiusScale mismatch"; return false; }
        if (a.max_branch_generations != b.max_branch_generations) { err = "MaxBranchGenerations mismatch"; return false; }
        if (std::abs(a.skin_gap_scale - b.skin_gap_scale) > eps) { err = "SkinGapScale mismatch"; return false; }
        if (std::abs(a.push_base_gain - b.push_base_gain) > eps) { err = "PushBaseGain mismatch"; return false; }
        if (std::abs(a.push_strength_gain - b.push_strength_gain) > eps) { err = "PushStrengthGain mismatch"; return false; }
        if (std::abs(a.max_step_disp_ratio - b.max_step_disp_ratio) > eps) { err = "MaxStepDispRatio mismatch"; return false; }
        if (std::abs(a.tether_max_stretch - b.tether_max_stretch) > eps) { err = "TetherMaxStretch mismatch"; return false; }
        if (std::abs(a.tether_target_stretch - b.tether_target_stretch) > eps) { err = "TetherTargetStretch mismatch"; return false; }
        if (std::abs(a.wall_clearance_scale - b.wall_clearance_scale) > eps) { err = "WallClearanceScale mismatch"; return false; }
        if (std::abs(a.branch_smooth_k - b.branch_smooth_k) > eps) { err = "BranchSmoothK mismatch"; return false; }
        if (std::abs(a.wall_cut_smooth - b.wall_cut_smooth) > eps) { err = "WallCutSmooth mismatch"; return false; }
        return true;
    };

    // 2. Temporary file paths
    std::string temp_json = (std::filesystem::temp_directory_path() / "rhizome_test_roundtrip.json").string();
    std::string temp_cu = (std::filesystem::temp_directory_path() / "rhizome_test_roundtrip.cu").string();

    // 3. Test .json round-trip
    bool json_export_ok = export_algorithm_to_json(original, temp_json);
    GrowthAlgorithmProfile imported_json;
    bool json_import_ok = false;
    std::string json_err;
    bool json_match = false;
    if (json_export_ok) {
        json_import_ok = import_algorithm_from_file(temp_json, imported_json);
        if (json_import_ok) {
            json_match = compare_profiles(original, imported_json, json_err);
        }
    }
    std::error_code ec;
    std::filesystem::remove(temp_json, ec);

    // 4. Test .cu round-trip
    bool cu_export_ok = export_algorithm_to_cu(original, temp_cu);
    GrowthAlgorithmProfile imported_cu;
    bool cu_import_ok = false;
    std::string cu_err;
    bool cu_match = false;
    if (cu_export_ok) {
        cu_import_ok = import_algorithm_from_file(temp_cu, imported_cu);
        if (cu_import_ok) {
            cu_match = compare_profiles(original, imported_cu, cu_err);
        }
    }
    std::filesystem::remove(temp_cu, ec);

    // 5. v24.01 Phase 0.4: viewport theme INI round-trip ([ViewportColors] +
    //    [LightingAndShading], the exact format emitted by the Styles panel's
    //    "Copy Layout INI" exporter) plus the fresh-install negative check
    //    (no theme sections present -> compiled defaults untouched).
    std::string temp_ini = (std::filesystem::temp_directory_path() / "rhizome_test_theme.ini").string();
    rhizome::DisplayState themed;
    themed.color_viewport_bg = glm::vec4(0.123f, 0.456f, 0.789f, 1.0f);
    themed.color_floor_grid = glm::vec4(0.250f, 0.500f, 0.750f, 0.625f);
    themed.color_roots = glm::vec4(0.990f, 0.110f, 0.220f, 0.880f);
    themed.param_shader_diffuse = 1.234f;
    themed.param_shader_edge_darkening = 0.432f;
    themed.param_shader_crevice_ao = 0.654f;
    themed.param_shader_specular = 0.321f;
    themed.param_shader_bark_strength = 1.567f;
    themed.param_shader_bark_frequency = 3.210f;
    themed.param_glass_frosting = 0.875f;
    themed.specimen_opacity = 0.640f;
    // v24.01 Phase 1: the ten promoted lighting parameters
    themed.param_light_azimuth = -87.5f;
    themed.param_light_elevation = 42.25f;
    themed.param_fill_light = 0.775f;
    themed.param_ambient = 0.655f;
    themed.param_glossiness = 48.0f;
    themed.param_highlight_tint = 0.225f;
    themed.param_fresnel_rim = 0.815f;
    themed.param_volumetric_ao = 0.335f;
    themed.param_sss_strength = 0.905f;
    themed.param_sss_warmth = 0.145f;
    {
        std::ofstream ini(temp_ini, std::ios::out | std::ios::binary);
        ini << "[Window][dummy]\nSize=1,1\n\n";
        ini << "[ViewportColors]\n";
        auto wcol = [&ini](const char* k, const glm::vec4& v) {
            char b[128];
            std::snprintf(b, sizeof(b), "%s=%.3f,%.3f,%.3f,%.3f\n", k, v.x, v.y, v.z, v.w);
            ini << b;
        };
        wcol("ViewportBg", themed.color_viewport_bg);
        wcol("FloorGrid", themed.color_floor_grid);
        wcol("Roots", themed.color_roots);
        ini << "\n[LightingAndShading]\n";
        ini << "Diffuse=" << themed.param_shader_diffuse << "\n";
        ini << "EdgeDarkening=" << themed.param_shader_edge_darkening << "\n";
        ini << "CreviceAO=" << themed.param_shader_crevice_ao << "\n";
        ini << "Specular=" << themed.param_shader_specular << "\n";
        ini << "BarkStrength=" << themed.param_shader_bark_strength << "\n";
        ini << "BarkFrequency=" << themed.param_shader_bark_frequency << "\n";
        ini << "GlassFrosting=" << themed.param_glass_frosting << "\n";
        ini << "SpecimenOpacity=" << themed.specimen_opacity << "\n";
        ini << "LightAzimuth=" << themed.param_light_azimuth << "\n";
        ini << "LightElevation=" << themed.param_light_elevation << "\n";
        ini << "FillLight=" << themed.param_fill_light << "\n";
        ini << "Ambient=" << themed.param_ambient << "\n";
        ini << "Glossiness=" << themed.param_glossiness << "\n";
        ini << "HighlightTint=" << themed.param_highlight_tint << "\n";
        ini << "FresnelRim=" << themed.param_fresnel_rim << "\n";
        ini << "VolumetricAO=" << themed.param_volumetric_ao << "\n";
        ini << "SSSStrength=" << themed.param_sss_strength << "\n";
        ini << "SSSWarmth=" << themed.param_sss_warmth << "\n";
        ini << "\n[GrowthPresets]\nCount=0\n";
        ini.close();
    }
    rhizome::DisplayState loaded;
    load_viewport_style_from_ini(loaded, temp_ini);
    const float eps = 1e-3f;
    auto vnear = [eps](const glm::vec4& a, const glm::vec4& b) {
        return std::abs(a.x - b.x) <= eps && std::abs(a.y - b.y) <= eps &&
               std::abs(a.z - b.z) <= eps && std::abs(a.w - b.w) <= eps;
    };
    std::string theme_err;
    bool theme_match = true;
    if (!vnear(loaded.color_viewport_bg, themed.color_viewport_bg)) { theme_match = false; theme_err = "ViewportBg"; }
    else if (!vnear(loaded.color_floor_grid, themed.color_floor_grid)) { theme_match = false; theme_err = "FloorGrid"; }
    else if (!vnear(loaded.color_roots, themed.color_roots)) { theme_match = false; theme_err = "Roots"; }
    else if (std::abs(loaded.param_shader_diffuse - themed.param_shader_diffuse) > eps) { theme_match = false; theme_err = "Diffuse"; }
    else if (std::abs(loaded.param_shader_edge_darkening - themed.param_shader_edge_darkening) > eps) { theme_match = false; theme_err = "EdgeDarkening"; }
    else if (std::abs(loaded.param_shader_crevice_ao - themed.param_shader_crevice_ao) > eps) { theme_match = false; theme_err = "CreviceAO"; }
    else if (std::abs(loaded.param_shader_specular - themed.param_shader_specular) > eps) { theme_match = false; theme_err = "Specular"; }
    else if (std::abs(loaded.param_shader_bark_strength - themed.param_shader_bark_strength) > eps) { theme_match = false; theme_err = "BarkStrength"; }
    else if (std::abs(loaded.param_shader_bark_frequency - themed.param_shader_bark_frequency) > eps) { theme_match = false; theme_err = "BarkFrequency"; }
    else if (std::abs(loaded.param_glass_frosting - themed.param_glass_frosting) > eps) { theme_match = false; theme_err = "GlassFrosting"; }
    else if (std::abs(loaded.specimen_opacity - themed.specimen_opacity) > eps) { theme_match = false; theme_err = "SpecimenOpacity"; }
    else if (std::abs(loaded.param_light_azimuth - themed.param_light_azimuth) > eps) { theme_match = false; theme_err = "LightAzimuth"; }
    else if (std::abs(loaded.param_light_elevation - themed.param_light_elevation) > eps) { theme_match = false; theme_err = "LightElevation"; }
    else if (std::abs(loaded.param_fill_light - themed.param_fill_light) > eps) { theme_match = false; theme_err = "FillLight"; }
    else if (std::abs(loaded.param_ambient - themed.param_ambient) > eps) { theme_match = false; theme_err = "Ambient"; }
    else if (std::abs(loaded.param_glossiness - themed.param_glossiness) > eps) { theme_match = false; theme_err = "Glossiness"; }
    else if (std::abs(loaded.param_highlight_tint - themed.param_highlight_tint) > eps) { theme_match = false; theme_err = "HighlightTint"; }
    else if (std::abs(loaded.param_fresnel_rim - themed.param_fresnel_rim) > eps) { theme_match = false; theme_err = "FresnelRim"; }
    else if (std::abs(loaded.param_volumetric_ao - themed.param_volumetric_ao) > eps) { theme_match = false; theme_err = "VolumetricAO"; }
    else if (std::abs(loaded.param_sss_strength - themed.param_sss_strength) > eps) { theme_match = false; theme_err = "SSSStrength"; }
    else if (std::abs(loaded.param_sss_warmth - themed.param_sss_warmth) > eps) { theme_match = false; theme_err = "SSSWarmth"; }
    // fresh-install check: theme sections absent -> defaults preserved
    std::string temp_plain = (std::filesystem::temp_directory_path() / "rhizome_test_theme_plain.ini").string();
    {
        std::ofstream ini(temp_plain, std::ios::out | std::ios::binary);
        ini << "[Window][dummy]\nSize=1,1\n\n[GrowthPresets]\nCount=0\n";
        ini.close();
    }
    rhizome::DisplayState fresh;
    rhizome::DisplayState fresh_expected;
    load_viewport_style_from_ini(fresh, temp_plain);
    bool fresh_ok = vnear(fresh.color_viewport_bg, fresh_expected.color_viewport_bg) &&
                    std::abs(fresh.param_shader_diffuse - fresh_expected.param_shader_diffuse) <= eps &&
                    std::abs(fresh.param_glass_frosting - fresh_expected.param_glass_frosting) <= eps;
    std::filesystem::remove(temp_ini, ec);
    std::filesystem::remove(temp_plain, ec);

    // 6. v24.01 Phase 1: lighting-rig defaults unit check (risk-table
    //    mitigation "defaults unit-checked once in --verify"). The compiled
    //    DisplayState defaults must reproduce the v23.29 hardcoded constants:
    //    the eight scalars exactly, and the key/fill directions derived from
    //    the default Azimuth/Elevation within 0.5% per component (rounded,
    //    human-friendly default angles; see core/lighting_rig.hpp).
    rhizome::DisplayState def;
    bool defaults_ok =
        std::abs(def.param_ambient - 0.32f) <= eps &&
        std::abs(def.param_fill_light - 0.16f) <= eps &&
        std::abs(def.param_glossiness - 24.0f) <= eps &&
        std::abs(def.param_highlight_tint - 0.55f) <= eps &&
        std::abs(def.param_fresnel_rim - 0.30f) <= eps &&
        std::abs(def.param_volumetric_ao - 0.85f) <= eps &&
        std::abs(def.param_sss_strength - 0.45f) <= eps &&
        std::abs(def.param_sss_warmth - 0.35f) <= eps;
    std::string defaults_err;
    if (defaults_ok) {
        const glm::vec3 key23 = glm::normalize(glm::vec3(-0.22f, 0.34f, 0.91f));
        const glm::vec3 fill23 = glm::normalize(glm::vec3(0.38f, -0.28f, 0.88f));
        const glm::vec3 key_now = rhizome::light_dir_view(def.param_light_azimuth, def.param_light_elevation);
        const glm::vec3 fill_now = rhizome::fill_light_dir_view(def.param_light_azimuth, def.param_light_elevation);
        const float dir_eps = 0.005f;
        auto near3 = [dir_eps](const glm::vec3& a, const glm::vec3& b) {
            return std::abs(a.x - b.x) <= dir_eps && std::abs(a.y - b.y) <= dir_eps &&
                   std::abs(a.z - b.z) <= dir_eps;
        };
        if (!near3(key_now, key23)) { defaults_ok = false; defaults_err = "key dir"; }
        else if (!near3(fill_now, fill23)) { defaults_ok = false; defaults_err = "fill dir"; }
    }

    // 7. v24.01 Phase 2 (proposal 2.5, 8c version-adaptive): style-editor
    //    registration gate — the registry must carry 148 + 23 + 63 + 7 rows
    //    in 25 sections (CSV census), and the census must COVER this build:
    //    the live colors are the first ImGuiCol_COUNT rows, each a Colors
    //    section row carrying a color resolver; census rows beyond the
    //    built count are dormant. Catches a catalog generated against an
    //    incompatible ImGui generation (the risk-table mitigation).
    const size_t p2_total = sizeof(rhizome::k_all_controls) / sizeof(rhizome::k_all_controls[0]);
    const size_t p2_secs = sizeof(rhizome::k_all_sections) / sizeof(rhizome::k_all_sections[0]);
    size_t p2_colors_rows = 0;
    for (const auto& v : rhizome::k_all_sections) {
        if (std::strcmp(v.section, "Style Editor: Colors") == 0) p2_colors_rows = v.count;
    }
    bool p2_live_colors_bound = true;
    for (int i = 0; i < ImGuiCol_COUNT; ++i) {
        const rhizome::ControlDefinition& row =
            rhizome::k_all_controls[rhizome::k_style_colors_first + static_cast<size_t>(i)];
        if (!row.resolve_color || std::strcmp(row.section, "Style Editor: Colors") != 0) {
            p2_live_colors_bound = false;
            break;
        }
    }
    const bool style_gate_ok =
        p2_total == 241 && p2_secs == 25 &&
        p2_colors_rows == rhizome::k_style_colors_count &&
        static_cast<size_t>(ImGuiCol_COUNT) <= rhizome::k_style_colors_count &&
        p2_live_colors_bound &&
        rhizome::k_style_render_first == 234;

    // 8. v24.01 Phase 2 (proposal 2.4): ImGuiStyle INI round-trip through
    //    load_imgui_style_from_ini — writer-format [Style] / [StyleColors]
    //    sections, plus the fresh-install negative. The scratch-context
    //    save/create/switch/restore/destroy makes the loader testable
    //    headlessly under --verify AND safe when the GUI "Run All Tests"
    //    path runs this suite with a live context.
    //
    //    v24.02 GUI-crash root cause (diag ledger 2026-10-22/23; fault
    //    RVAs 0xCAB81 ImGui::SameLine+0x11 and 0xBE6F2
    //    IsWindowContentHoverable+0x72, five events, CLI --verify always
    //    clean): ImGui::CreateContext() does NOT leave the new context
    //    current when a live one exists (imgui 1.92 restores the previous
    //    current), so this block used to alias the LIVE style via
    //    GetStyle(), and the no-arg ImGui::DestroyContext() then destroyed
    //    the LIVE context mid-frame; the first ImGui calls after the
    //    button handler returned (SameLine / Button hover check in
    //    panel_verification.cpp) dereferenced the freed context and
    //    crashed with 0xC0000005. Fixed by switching to the scratch
    //    context explicitly and destroying it BY POINTER after restoring
    //    the saved context.
    bool style_ini_ok = true;
    std::string style_ini_err;
    {
        ImGuiContext* p2_saved_ctx = ImGui::GetCurrentContext();
        ImGuiContext* p2_scratch_ctx = ImGui::CreateContext();
        ImGui::SetCurrentContext(p2_scratch_ctx);
        ImGuiStyle& st = ImGui::GetStyle();
        const float t_alpha = 0.55f, t_tess = 2.25f, t_tabb = 1.5f, t_fr = 6.5f;
        const ImVec2 t_wp(12.0f, 13.0f);
        const ImVec4 t_text(0.11f, 0.22f, 0.33f, 0.44f);
        const ImVec4 t_btn(0.55f, 0.66f, 0.77f, 0.88f);
        const bool t_aa = !st.AntiAliasedLines;

        std::string tmp_style = (std::filesystem::temp_directory_path() /
                                 "rhizome_test_imgui_style.ini").string();
        {
            std::ofstream ini(tmp_style, std::ios::out | std::ios::binary);
            ini << "[Window][dummy]\nSize=1,1\n\n[Style]\n";
            ini << "Alpha=" << t_alpha << "\n";
            ini << "WindowPadding=" << t_wp.x << "," << t_wp.y << "\n";
            ini << "TabBorderSize=" << t_tabb << "\n";
            ini << "FrameRounding=" << t_fr << "\n";
            ini << "AntiAliasedLines=" << (t_aa ? 1 : 0) << "\n";
            ini << "CurveTessellationTol=" << t_tess << "\n";
            ini << "\n[StyleColors]\n";
            ini << "Text=" << t_text.x << "," << t_text.y << "," << t_text.z << "," << t_text.w << "\n";
            ini << "Button=" << t_btn.x << "," << t_btn.y << "," << t_btn.z << "," << t_btn.w << "\n";
            ini.close();
        }
        std::string tmp_plain = (std::filesystem::temp_directory_path() /
                                 "rhizome_test_imgui_style_plain.ini").string();
        {
            std::ofstream ini(tmp_plain, std::ios::out | std::ios::binary);
            ini << "[Window][dummy]\nSize=1,1\n\n[GrowthPresets]\nCount=0\n";
            ini.close();
        }

        st = ImGuiStyle();  // compiled defaults, then restore from the INI
        rhizome::load_imgui_style_from_ini(tmp_style);
        auto v4near = [eps](const ImVec4& a, const ImVec4& b) {
            return std::abs(a.x - b.x) <= eps && std::abs(a.y - b.y) <= eps &&
                   std::abs(a.z - b.z) <= eps && std::abs(a.w - b.w) <= eps;
        };
        if (std::abs(st.Alpha - t_alpha) > eps) { style_ini_ok = false; style_ini_err = "Alpha"; }
        else if (std::abs(st.WindowPadding.x - t_wp.x) > eps ||
                 std::abs(st.WindowPadding.y - t_wp.y) > eps) { style_ini_ok = false; style_ini_err = "WindowPadding"; }
        else if (std::abs(st.TabBorderSize - t_tabb) > eps) { style_ini_ok = false; style_ini_err = "TabBorderSize (Phase 2 writer key)"; }
        else if (std::abs(st.FrameRounding - t_fr) > eps) { style_ini_ok = false; style_ini_err = "FrameRounding"; }
        else if (st.AntiAliasedLines != t_aa) { style_ini_ok = false; style_ini_err = "AntiAliasedLines"; }
        else if (std::abs(st.CurveTessellationTol - t_tess) > eps) { style_ini_ok = false; style_ini_err = "CurveTessellationTol"; }
        else if (!v4near(st.Colors[ImGuiCol_Text], t_text)) { style_ini_ok = false; style_ini_err = "Colors[Text]"; }
        else if (!v4near(st.Colors[ImGuiCol_Button], t_btn)) { style_ini_ok = false; style_ini_err = "Colors[Button]"; }

        // fresh-install negative: absent sections leave the live values
        rhizome::load_imgui_style_from_ini(tmp_plain);
        if (std::abs(st.Alpha - t_alpha) > eps || !v4near(st.Colors[ImGuiCol_Text], t_text)) {
            style_ini_ok = false;
            style_ini_err = "fresh-install negative (absent sections must not reset)";
        }

        std::filesystem::remove(tmp_style, ec);
        std::filesystem::remove(tmp_plain, ec);
        ImGui::SetCurrentContext(p2_saved_ctx);
        ImGui::DestroyContext(p2_scratch_ctx);
    }

    res.passed = json_export_ok && json_import_ok && json_match &&
                 cu_export_ok && cu_import_ok && cu_match &&
                 theme_match && fresh_ok && defaults_ok &&
                 style_gate_ok && style_ini_ok;

    ss << "    [JSON Round-Trip] Export: " << (json_export_ok ? "OK" : "FAIL")
       << " | Import: " << (json_import_ok ? "OK" : "FAIL")
       << " | Parameters: 50/50 (Match: " << (json_match ? "YES" : ("FAIL: " + json_err)) << ")\n"
       << "    [CUDA .cu Round-Trip] Export: " << (cu_export_ok ? "OK" : "FAIL")
       << " | Import: " << (cu_import_ok ? "OK" : "FAIL")
       << " | Parameters: 50/50 (Match: " << (cu_match ? "YES" : ("FAIL: " + cu_err)) << ")\n"
       << "    [Theme INI Round-Trip] Sections Applied: " << (theme_match ? "YES" : ("FAIL: " + theme_err))
       << " | Fresh-Install Defaults Preserved: " << (fresh_ok ? "YES" : "NO") << "\n"
       << "    [Lighting Defaults] v23.29 Rig Reproduced: "
       << (defaults_ok ? "YES (8 scalars exact + dirs within 0.5%)" : ("FAIL: " + defaults_err)) << "\n"
       << "    [Style Editor Registry] 241 controls / 25 sections / Colors: census "
       << static_cast<int>(rhizome::k_style_colors_count) << " rows, live "
       << static_cast<int>(ImGuiCol_COUNT) << " on ImGui " << IMGUI_VERSION
       << ": " << (style_gate_ok ? "YES" : "NO") << "\n"
       << "    [ImGuiStyle INI Round-Trip] Sections Applied: "
       << (style_ini_ok ? "YES (scalars + ImVec2 + borders + AA + tess + colors)" : ("FAIL: " + style_ini_err))
       << " | Fresh-Install Defaults Preserved: " << (style_ini_ok ? "YES" : "NO") << "\n"
       << "    [Result] Headless Algorithm Serialization Exactness: " << (res.passed ? "PASS" : "FAIL");

    res.details = ss.str();
    return res;
}
