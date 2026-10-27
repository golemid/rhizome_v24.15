#include "core/growth_controller.hpp"
#include "core/unified_logger.hpp"
#include "engine/io/parsers.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/root_field.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

namespace rhizome {

void init_default_growth_algorithms(std::vector<GrowthAlgorithmProfile>& growth_algorithms, int& selected_algorithm_idx) {
    GrowthAlgorithmProfile default_corals;
    default_corals.name = "Corals (Space Colonization)";
    default_corals.engine_mode = 0;

    GrowthAlgorithmProfile default_roots;
    default_roots.name = "Roots (Volumetric Root-Bound)";
    default_roots.engine_mode = 1;
    default_roots.backward_pull = 1.0f;
    default_roots.tensile_straightening = 0.0f;

    GrowthAlgorithmProfile default_kelp;
    default_kelp.name = "Kelp (Tensile Streamlines)";
    default_kelp.engine_mode = 1;
    default_kelp.root_roughness = 0.28f;
    default_kelp.push_strength = 0.65f;
    default_kelp.space_fill_drive = 0.90f;
    default_kelp.backward_pull = 0.50f;
    default_kelp.tensile_straightening = 0.90f;
    default_kelp.spore_count = 25000;
    default_kelp.search_radius = 16.0f;
    default_kelp.kill_radius = 2.6f;
    default_kelp.growth_step = 1.45f;
    default_kelp.variable_step = false;
    default_kelp.branching_rate = 0.58f;
    default_kelp.steering_cone = 35.0f;
    default_kelp.max_iterations = 45;
    default_kelp.trunk_caliber = 1.00f;
    default_kelp.tendril_caliber = 0.35f;

    GrowthAlgorithmProfile default_murray;
    default_murray.name = "Vascular (Murray Bifurcation)";
    default_murray.engine_mode = 2;
    default_murray.root_roughness = 0.15f;
    default_murray.push_strength = 0.55f;
    default_murray.space_fill_drive = 0.88f;
    default_murray.backward_pull = 0.80f;
    default_murray.tensile_straightening = 0.40f;
    default_murray.murray_exponent = 3.0f;
    default_murray.murray_branch_angle = 37.5f;
    default_murray.spore_count = 25000;
    default_murray.search_radius = 14.0f;
    default_murray.kill_radius = 2.2f;
    default_murray.growth_step = 1.35f;
    default_murray.branching_rate = 0.70f;
    default_murray.trunk_caliber = 1.20f;
    default_murray.tendril_caliber = 0.40f;

    GrowthAlgorithmProfile default_creeping;
    default_creeping.name = "Creeping (Root Surface Binder)";
    default_creeping.engine_mode = 3;
    default_creeping.surface_adherence = 0.90f;
    default_creeping.root_roughness = 0.22f;
    default_creeping.push_strength = 0.40f;
    default_creeping.space_fill_drive = 0.95f;
    default_creeping.backward_pull = 0.70f;
    default_creeping.tensile_straightening = 0.35f;
    default_creeping.spore_count = 25000;
    default_creeping.search_radius = 14.0f;
    default_creeping.kill_radius = 2.0f;
    default_creeping.growth_step = 1.25f;
    default_creeping.branching_rate = 0.65f;
    default_creeping.steering_cone = 40.0f;
    default_creeping.max_iterations = 45;
    default_creeping.trunk_caliber = 0.45f;
    default_creeping.tendril_caliber = 0.20f;

    // v24.12 Track F R1 / v24.13 Track F R2: Developmental FSPM (Root Typ
    // lineage). Per-order gravitropic setpoints, acropetal laterals, Murray
    // caliber at bake, SLA print floor. Corpse physics
    // (roughness/push/pull) neutralized; branch_smooth_k 0.35 = junction
    // fillet for print strength. The per-order table is derived in
    // build_colonization_config_from_state from GrowthState's dev_* user
    // surface (CTRL242-251); v24.13 raises the crown to 7 framework arms and
    // scales length caps + iteration budget from the vessel reference
    // (dev_vessel_ref_mm) measured at Start.
    GrowthAlgorithmProfile default_dev;
    default_dev.name = "Root Typ (Developmental FSPM)";
    default_dev.engine_mode = 4;
    default_dev.primary_roots = 7;
    default_dev.growth_step = 1.30f;
    default_dev.max_iterations = 110;
    default_dev.trunk_caliber = 1.20f;
    default_dev.tendril_caliber = 0.30f;
    default_dev.root_roughness = 0.0f;
    default_dev.push_strength = 0.0f;
    default_dev.space_fill_drive = 0.50f;
    default_dev.backward_pull = 0.0f;
    default_dev.tensile_straightening = 0.0f;
    default_dev.surface_adherence = 0.0f;
    default_dev.spore_count = 1000;
    default_dev.search_radius = 8.0f;
    default_dev.kill_radius = 1.0f;
    default_dev.branching_rate = 0.5f;
    default_dev.steering_cone = 60.0f;
    default_dev.branch_smooth_k = 0.35f;
    default_dev.wall_clearance_scale = 0.90f;

    growth_algorithms = { default_corals, default_roots, default_kelp, default_murray, default_creeping, default_dev };
    load_growth_algorithms_from_ini(growth_algorithms, selected_algorithm_idx, "imgui.ini");

    bool has_kelp = false;
    bool has_murray = false;
    bool has_creeping = false;
    bool has_dev = false;
    for (const auto& a : growth_algorithms) {
        if (a.name.find("Kelp") != std::string::npos) has_kelp = true;
        if (a.name.find("Murray") != std::string::npos || a.name.find("Vascular") != std::string::npos) has_murray = true;
        if (a.name.find("Creeping") != std::string::npos || a.name.find("Binder") != std::string::npos) has_creeping = true;
        if (a.name.find("Root Typ") != std::string::npos || a.name.find("Developmental") != std::string::npos) has_dev = true;
    }
    if (!has_kelp) growth_algorithms.push_back(default_kelp);
    if (!has_murray) growth_algorithms.push_back(default_murray);
    if (!has_creeping) growth_algorithms.push_back(default_creeping);
    if (!has_dev) growth_algorithms.push_back(default_dev);
}

void apply_algorithm_profile_to_state(AppState& app_state, const std::vector<GrowthAlgorithmProfile>& growth_algorithms, int idx) {
    if (idx < 0 || idx >= static_cast<int>(growth_algorithms.size())) return;
    app_state.growth.selected_algorithm_idx = idx;
    const auto& a = growth_algorithms[idx];
    app_state.growth.algorithm_mode = a.engine_mode;
    app_state.growth.primary_roots = a.primary_roots;
    app_state.growth.root_roughness = a.root_roughness;
    app_state.growth.push_strength = a.push_strength;
    app_state.growth.space_fill_drive = a.space_fill_drive;
    app_state.growth.backward_pull = a.backward_pull;
    app_state.growth.tensile_straightening = a.tensile_straightening;
    app_state.growth.murray_exponent = a.murray_exponent;
    app_state.growth.murray_branch_angle = a.murray_branch_angle;
    app_state.growth.surface_adherence = a.surface_adherence;
    app_state.growth.spore_count = a.spore_count;
    app_state.growth.search_radius = a.search_radius;
    app_state.growth.kill_radius = a.kill_radius;
    app_state.growth.growth_step = a.growth_step;
    app_state.growth.variable_step = a.variable_step;
    app_state.growth.min_growth_step = a.min_growth_step;
    app_state.growth.max_growth_step = a.max_growth_step;
    app_state.growth.branching_rate = a.branching_rate;
    app_state.growth.steering_cone = a.steering_cone;
    app_state.growth.max_iterations = a.max_iterations;
    app_state.growth.trunk_caliber = a.trunk_caliber;
    app_state.growth.tendril_caliber = a.tendril_caliber;
    app_state.growth.variable_taper = a.variable_taper;
    app_state.growth.taper_curve = a.taper_curve;
    app_state.growth.wall_clearance_scale = a.wall_clearance_scale;
    app_state.growth.branch_smooth_k = a.branch_smooth_k;
    app_state.growth.wall_cut_smooth = a.wall_cut_smooth;
    app_state.growth.seed_offset = a.seed_offset;
}

void capture_state_into_algorithm_profile(const AppState& app_state, GrowthAlgorithmProfile& a) {
    a.engine_mode = app_state.growth.algorithm_mode;
    a.primary_roots = app_state.growth.primary_roots;
    a.root_roughness = app_state.growth.root_roughness;
    a.push_strength = app_state.growth.push_strength;
    a.space_fill_drive = app_state.growth.space_fill_drive;
    a.backward_pull = app_state.growth.backward_pull;
    a.tensile_straightening = app_state.growth.tensile_straightening;
    a.murray_exponent = app_state.growth.murray_exponent;
    a.murray_branch_angle = app_state.growth.murray_branch_angle;
    a.surface_adherence = app_state.growth.surface_adherence;
    a.spore_count = app_state.growth.spore_count;
    a.search_radius = app_state.growth.search_radius;
    a.kill_radius = app_state.growth.kill_radius;
    a.growth_step = app_state.growth.growth_step;
    a.variable_step = app_state.growth.variable_step;
    a.min_growth_step = app_state.growth.min_growth_step;
    a.max_growth_step = app_state.growth.max_growth_step;
    a.branching_rate = app_state.growth.branching_rate;
    a.steering_cone = app_state.growth.steering_cone;
    a.max_iterations = app_state.growth.max_iterations;
    a.trunk_caliber = app_state.growth.trunk_caliber;
    a.tendril_caliber = app_state.growth.tendril_caliber;
    a.variable_taper = app_state.growth.variable_taper;
    a.taper_curve = app_state.growth.taper_curve;
    a.wall_clearance_scale = app_state.growth.wall_clearance_scale;
    a.branch_smooth_k = app_state.growth.branch_smooth_k;
    a.wall_cut_smooth = app_state.growth.wall_cut_smooth;
    a.seed_offset = app_state.growth.seed_offset;
}

ColonizationConfig build_colonization_config_from_state(
    const AppState& app_state,
    const std::vector<GrowthAlgorithmProfile>& growth_algorithms,
    const std::vector<uint8_t>& seed_frozen
) {
    ColonizationConfig c_cfg;
    c_cfg.spore_count = app_state.growth.spore_count;
    c_cfg.search_radius = app_state.growth.search_radius;
    c_cfg.kill_radius = app_state.growth.kill_radius;
    c_cfg.growth_step = app_state.growth.growth_step;
    c_cfg.variable_step = app_state.growth.variable_step;
    c_cfg.min_growth_step = app_state.growth.min_growth_step;
    c_cfg.max_growth_step = app_state.growth.max_growth_step;
    c_cfg.max_iterations = app_state.growth.max_iterations;
    c_cfg.branching_rate = app_state.growth.branching_rate;
    c_cfg.steering_cone = app_state.growth.steering_cone;
    c_cfg.base_radius = app_state.growth.trunk_caliber;
    c_cfg.tip_radius = app_state.growth.tendril_caliber;
    c_cfg.variable_taper = app_state.growth.variable_taper;
    c_cfg.algorithm_mode = app_state.growth.algorithm_mode;
    c_cfg.primary_roots = app_state.growth.primary_roots;
    c_cfg.root_roughness = app_state.growth.root_roughness;
    c_cfg.push_strength = app_state.growth.push_strength;
    c_cfg.space_fill_drive = app_state.growth.space_fill_drive;
    c_cfg.backward_pull = app_state.growth.backward_pull;
    c_cfg.tensile_straightening = app_state.growth.tensile_straightening;
    c_cfg.murray_exponent = app_state.growth.murray_exponent;
    c_cfg.murray_branch_angle = app_state.growth.murray_branch_angle;
    c_cfg.surface_adherence = app_state.growth.surface_adherence;
    c_cfg.wall_clearance_scale = app_state.growth.wall_clearance_scale;
    c_cfg.branch_smooth_k = app_state.growth.branch_smooth_k;
    std::vector<uint8_t> effective_frozen = seed_frozen;
    for (size_t s = 0; s < app_state.seed.has_lateral.size(); ++s) {
        if (app_state.seed.has_lateral[s] != 0) {
            if (s >= effective_frozen.size()) {
                effective_frozen.resize(s + 1, 0);
            }
            effective_frozen[s] = 1;
        }
    }
    c_cfg.frozen_seeds = effective_frozen;
    c_cfg.seed_has_lateral = app_state.seed.has_lateral;
    c_cfg.seed_snapped_to_root = app_state.seed.snapped_to_root;
    c_cfg.lateral_density = app_state.growth.lateral_density;
    c_cfg.lateral_trunk_caliber = app_state.growth.lateral_trunk_caliber;
    c_cfg.lateral_tip_caliber = app_state.growth.lateral_tip_caliber;
    c_cfg.lateral_max_length = app_state.growth.lateral_max_length;
    c_cfg.lateral_wander = app_state.growth.lateral_wander;
    c_cfg.lateral_branch_angle = app_state.growth.lateral_branch_angle;
    c_cfg.seed = 42;

    // v24.12 Track F R1 / v24.13 Track F R2: derive the developmental
    // per-order table from the user surface. GSA[2]/[3] ride order 1
    // (+10/+18 deg, clamped 89); dev_vigor scales every rate;
    // dev_lateral_freq divides every spacing. Base tables (order radius /
    // tip radius) ride the ColonizationConfig struct defaults. Per-root
    // length caps are VESSEL-RELATIVE when the controller measured a vessel
    // reference at Start (dev_vessel_ref_mm): max_steps[o] carries the
    // target length dev_len_frac[o] x vessel_ref at this order's step size,
    // so one profile scales across specimen sizes; without a reference the
    // static fallback table applies. The mode-4 iteration budget is raised
    // to arm_steps + 45 when the vessel demands it (the cascade needs room
    // below the longest arm to finish).
    c_cfg.dev_gsa_deg[0] = app_state.growth.dev_gsa_framework;
    c_cfg.dev_gsa_deg[1] = app_state.growth.dev_gsa_lateral;
    c_cfg.dev_gsa_deg[2] = std::min(89.0f, app_state.growth.dev_gsa_lateral + 10.0f);
    c_cfg.dev_gsa_deg[3] = std::min(89.0f, app_state.growth.dev_gsa_lateral + 18.0f);
    c_cfg.dev_vessel_ref_mm = app_state.growth.dev_vessel_ref_mm;
    {
        const float base_rate[4] = {1.00f, 0.72f, 0.50f, 0.34f};
        const int base_spacing[4] = {16, 20, 28, 36};
        const float freq = std::max(0.25f, app_state.growth.dev_lateral_freq);
        for (int o = 0; o < 4; ++o) {
            c_cfg.dev_rate[o] = base_rate[o] * std::max(0.10f, app_state.growth.dev_vigor);
            c_cfg.dev_spacing[o] = std::max(2, static_cast<int>(std::lroundf(static_cast<float>(base_spacing[o]) / freq)));
        }
        if (c_cfg.dev_vessel_ref_mm > 1.0f) {
            for (int o = 0; o < 4; ++o) {
                const float target_len = c_cfg.dev_len_frac[o] * c_cfg.dev_vessel_ref_mm;
                const float step_len = std::max(0.25f, c_cfg.growth_step * c_cfg.dev_rate[o]);
                // v24.14 hotfix: std::lroundf returns long — mixing it with the
                // 6.0f float literal broke std::max/std::min template deduction
                // on MSVC (C2672). Clamp in the int domain exactly like the
                // dev_spacing derivation above.
                const int raw_steps = static_cast<int>(std::lroundf(target_len / step_len));
                c_cfg.dev_max_steps[o] = std::min(300, std::max(6, raw_steps));
            }
        }
        if (c_cfg.algorithm_mode == 4) {
            c_cfg.max_iterations = std::max(c_cfg.max_iterations, c_cfg.dev_max_steps[0] + 45);
        }
    }
    c_cfg.dev_grav_gain = app_state.growth.dev_grav_gain;
    c_cfg.dev_noise = app_state.growth.dev_noise;
    c_cfg.dev_branch_angle = app_state.growth.dev_branch_angle;
    c_cfg.dev_murray_gamma = app_state.growth.dev_murray_gamma;
    c_cfg.dev_min_print_radius = app_state.growth.dev_min_print_radius;
    c_cfg.dev_section_budget = app_state.growth.dev_section_budget;
    if (!growth_algorithms.empty() && app_state.growth.selected_algorithm_idx >= 0 && app_state.growth.selected_algorithm_idx < static_cast<int>(growth_algorithms.size())) {
        const auto& a = growth_algorithms[app_state.growth.selected_algorithm_idx];
        c_cfg.tip_inertia_weight = a.tip_inertia_weight;
        c_cfg.spore_pull_weight = a.spore_pull_weight;
        c_cfg.roughness_amplitude = a.roughness_amplitude;
        c_cfg.feeler_nar_fwd = a.feeler_nar_fwd;
        c_cfg.feeler_nar_side = a.feeler_nar_side;
        c_cfg.feeler_med_fwd = a.feeler_med_fwd;
        c_cfg.feeler_med_side = a.feeler_med_side;
        c_cfg.feeler_wide_fwd = a.feeler_wide_fwd;
        c_cfg.feeler_wide_side = a.feeler_wide_side;
        c_cfg.feeler_wall_in = a.feeler_wall_in;
        c_cfg.feeler_wall_out = a.feeler_wall_out;
        c_cfg.clearance_score_weight = a.clearance_score_weight;
        c_cfg.align_bonus_weight = a.align_bonus_weight;
        c_cfg.lateral_fwd_weight = a.lateral_fwd_weight;
        c_cfg.lateral_side_weight = a.lateral_side_weight;
        c_cfg.lateral_radius_scale = a.lateral_radius_scale;
        c_cfg.max_branch_generations = a.max_branch_generations;
        c_cfg.skin_gap_scale = a.skin_gap_scale;
        c_cfg.push_base_gain = a.push_base_gain;
        c_cfg.push_strength_gain = a.push_strength_gain;
        c_cfg.max_step_disp_ratio = a.max_step_disp_ratio;
        c_cfg.tether_max_stretch = a.tether_max_stretch;
        c_cfg.tether_target_stretch = a.tether_target_stretch;
    }
    return c_cfg;
}

void run_all_tests_controller(GrowthControllerContext& ctx) {
    ctx.app_state.verification.tests_running = true;

    ctx.res.original_mesh = rhizome::TriangleMesh{};
    ctx.res.converted_body_mesh = rhizome::TriangleMesh{};
    ctx.res.roots_mesh = rhizome::TriangleMesh{};
    ctx.res.orig_gl_buffers.cleanup();
    ctx.res.body_gl_buffers.cleanup();
    ctx.res.roots_gl_buffers.cleanup();
    ctx.res.merged_gl_buffers.cleanup();

    for (auto& obj : ctx.res.test_objects) {
        obj.buffers.cleanup();
    }
    ctx.res.test_objects.clear();

    ctx.res.last_test_results = rhizome::VerificationHarness::run_analytical_suite();

    int passed_count = 0;
    int failed_count = 0;
    float current_x = 0.0f;
    const float spacing = 50.0f;

    glm::vec3 global_min(1e9f);
    glm::vec3 global_max(-1e9f);
    bool has_any_mesh = false;

    for (const auto& res : ctx.res.last_test_results) {
        if (res.passed) {
            passed_count++;
        } else {
            failed_count++;
        }

        if (res.result_mesh.vertex_count() > 0 && res.result_mesh.triangle_count() > 0) {
            rhizome::TriangleMesh mesh = res.result_mesh;

            glm::vec3 min_p(0.0f), max_p(0.0f), shifted_min(0.0f), shifted_max(0.0f);
            float pos_ms = 0.0f;
            rhizome::CudaRasterizer::position_mesh_at_x_gpu(
                mesh, current_x, min_p, max_p, shifted_min, shifted_max, pos_ms
            );

            global_min = glm::min(global_min, shifted_min);
            global_max = glm::max(global_max, shifted_max);
            has_any_mesh = true;

            TestMeshObject obj;
            obj.name = res.primitive_name;
            obj.passed = res.passed;
            obj.color = res.passed ? ctx.app_state.display.color_test_pass : ctx.app_state.display.color_test_fail;
            obj.buffers.upload(mesh, true);
            ctx.res.test_objects.push_back(std::move(obj));

            float width = std::max(max_p.x - min_p.x, 20.0f);
            current_x += width + spacing;
        }
    }

    if (has_any_mesh) {
        ctx.res.bbox_buffers.update(global_min, global_max);
        glm::vec3 all_center = (global_min + global_max) * 0.5f;
        float all_size = glm::length(global_max - global_min);
        ctx.res.camera.target = all_center;
        ctx.res.camera.distance = std::max(all_size * 1.3f, 80.0f);

        float extent = std::max({global_max.x - global_min.x, global_max.y - global_min.y, 100.0f});
        ctx.res.floor_grid_buffers.init(extent * 1.5f, 25.0f);
    }

    ctx.app_state.display.show_converted_body = false;
    ctx.app_state.display.show_original_mesh = false;
    ctx.app_state.display.show_roots = false;

    ctx.app_state.specimen.specimen_source_name = "test_suite";
    ctx.app_state.specimen.specimen_source_format = "TESTS";
    if (has_any_mesh) {
        ctx.app_state.specimen.specimen_dimensions = global_max - global_min;
    } else {
        ctx.app_state.specimen.specimen_dimensions = glm::vec3(0.0f);
    }
    size_t total_triangles = 0;
    for (const auto& res : ctx.res.last_test_results) {
        total_triangles += res.result_mesh.triangle_count();
    }
    ctx.app_state.specimen.specimen_triangle_count = total_triangles;
    ctx.app_state.specimen.specimen_implicit_status = std::to_string(passed_count) + "/" + std::to_string(ctx.res.last_test_results.size()) + " Passed";
    ctx.app_state.specimen.specimen_shell_valid = (failed_count == 0);
    ctx.app_state.specimen.specimen_shell_valid_msg = (failed_count == 0) ? "Pass (All)" : std::to_string(failed_count) + " Failed";
    ctx.app_state.specimen.specimen_interior_volume_ml = 0.0f;

    ctx.app_state.verification.tests_running = false;
    ctx.app_state.verification.show_tests_window = true;
    ctx.app_state.ui.status_bar_message = "Tests Complete: " + std::to_string(passed_count) + " Passed, " + std::to_string(failed_count) + " Failed";
}

void update_bounding_box_controller(GrowthControllerContext& ctx) {
    if (ctx.res.original_mesh.vertex_count() == 0) return;
    glm::vec3 min_p(0.0f), max_p(0.0f);
    float bbox_ms = 0.0f;
    rhizome::CudaRasterizer::compute_bounding_box_gpu(ctx.res.original_mesh, min_p, max_p, bbox_ms);
    ctx.res.bbox_buffers.update(min_p, max_p);

    glm::vec3 center = (min_p + max_p) * 0.5f;
    float size = glm::length(max_p - min_p);
    ctx.res.camera.target = center;
    ctx.res.camera.distance = std::max(size * 1.5f, 30.0f);

    float extent = std::max({max_p.x - min_p.x, max_p.y - min_p.y, 60.0f});
    float grid_sz = std::max(80.0f, extent * 1.3f);
    float grid_step = std::max(5.0f, std::round(grid_sz / 10.0f));
    ctx.res.floor_grid_buffers.init(grid_sz, grid_step);
}

void convert_and_update_model_controller(GrowthControllerContext& ctx, const TriangleMesh& mesh, const std::string& name) {
    ctx.res.loaded_model_name = name;
    ctx.res.original_mesh = mesh;

    glm::vec3 min_p(0.0f), max_p(0.0f);
    if (ctx.res.original_mesh.vertex_count() > 0) {
        float align_ms = 0.0f;
        rhizome::CudaRasterizer::align_mesh_z_up_gpu(ctx.res.original_mesh, min_p, max_p, align_ms);
    }

    ctx.app_state.specimen.specimen_source_name = name;
    size_t dot_idx = name.find_last_of('.');
    if (dot_idx != std::string::npos) {
        std::string ext_str = name.substr(dot_idx);
        std::transform(ext_str.begin(), ext_str.end(), ext_str.begin(), ::tolower);
        if (ext_str == ".obj") ctx.app_state.specimen.specimen_source_format = "OBJ";
        else if (ext_str == ".stl") ctx.app_state.specimen.specimen_source_format = "STL";
        else ctx.app_state.specimen.specimen_source_format = name.substr(dot_idx + 1);
    } else {
        ctx.app_state.specimen.specimen_source_format = "PROCEDURAL";
    }

    ctx.app_state.specimen.specimen_triangle_count = ctx.res.original_mesh.triangle_count();

    if (ctx.res.original_mesh.vertex_count() > 0) {
        ctx.app_state.specimen.specimen_dimensions = max_p - min_p;
        ctx.app_state.specimen.specimen_extent = std::max({ctx.app_state.specimen.specimen_dimensions.x, ctx.app_state.specimen.specimen_dimensions.y, ctx.app_state.specimen.specimen_dimensions.z, 20.0f});
    } else {
        ctx.app_state.specimen.specimen_dimensions = glm::vec3(0.0f);
        ctx.app_state.specimen.specimen_extent = 100.0f;
    }

    {
        rhizome::TriangleMesh check_mesh = ctx.res.original_mesh;
        auto prepass = rhizome::MeshWindingPass::orient_mesh(check_mesh);
        ctx.app_state.specimen.specimen_shell_valid = prepass.is_orientable && !prepass.had_inconsistent_winding;
        ctx.app_state.specimen.specimen_shell_valid_msg = ctx.app_state.specimen.specimen_shell_valid ? "Pass (Manifold)" :
            (prepass.is_orientable ? "Warn (Winding Unified)" : "Warn (Non-Manifold)");
    }

    ctx.app_state.specimen.specimen_implicit_status = "Building";
    float cuda_time_ms = 0.0f;
    auto cuda_volume = rhizome::CudaRasterizer::build_from_mesh(ctx.res.original_mesh, ctx.app_state.growth.detail_level, cuda_time_ms);
    ctx.res.current_sdf_grid = cuda_volume;
    ctx.app_state.specimen.specimen_build_time_ms = cuda_time_ms;
    ctx.app_state.specimen.specimen_implicit_status = (cuda_volume && cuda_volume->is_valid()) ? "Built" : "Failed";

    ctx.res.converted_body_mesh.vertices.clear();
    ctx.res.converted_body_mesh.indices.clear();
    ctx.res.body_gl_buffers.cleanup();
    ctx.res.generated_merged_mesh.vertices.clear();
    ctx.res.generated_merged_mesh.indices.clear();
    ctx.res.merged_gl_buffers.cleanup();
    ctx.app_state.mesh_export.has_generated_roots_mesh = false;
    ctx.app_state.mesh_export.has_generated_vessel_mesh = false;
    ctx.app_state.mesh_export.has_generated_merged_mesh = false;
    ctx.res.cached_merged_metrics = CachedMeshMetrics();
    ctx.app_state.display.show_wireframe = false;

    {
        float vol_mm3 = 0.0f, vol_ms = 0.0f;
        if (cuda_volume && cuda_volume->is_valid()) {
            rhizome::CudaRasterizer::compute_sdf_interior_volume_gpu(*cuda_volume, vol_mm3, vol_ms);
        } else {
            rhizome::CudaRasterizer::compute_signed_volume_gpu(ctx.res.original_mesh, vol_mm3, vol_ms);
        }
        ctx.app_state.specimen.specimen_interior_volume_ml = std::abs(vol_mm3) / 1000.0f;
    }

    ctx.res.orig_gl_buffers.upload(ctx.res.original_mesh, true);
    update_bounding_box_controller(ctx);
    ctx.res.cached_body_metrics = compute_mesh_metrics(ctx.res.original_mesh);
    ctx.res.cached_roots_metrics = CachedMeshMetrics();

    glm::vec3 center = (min_p + max_p) * 0.5f;
    glm::vec3 default_seed(center.x, center.y, min_p.z + 2.0f * ctx.app_state.growth.detail_level);
    if (cuda_volume && cuda_volume->is_valid()) {
        float desired_margin = -(ctx.app_state.growth.trunk_caliber + ctx.app_state.growth.seed_offset);
        float seed_ms = 0.0f;
        rhizome::CudaRasterizer::find_default_seed_gpu(
            *cuda_volume, min_p, max_p, desired_margin, default_seed, seed_ms
        );
    }
    ctx.app_state.seed.points.clear();
    ctx.app_state.seed.points.push_back(default_seed);
    ctx.app_state.seed.frozen.clear();
    ctx.app_state.seed.frozen.push_back(0);
    ctx.app_state.seed.snapped_to_root.clear();
    ctx.app_state.seed.snapped_to_root.push_back(0);
    ctx.app_state.seed.radii.clear();
    ctx.app_state.seed.radii.push_back(ctx.app_state.growth.trunk_caliber);
    ctx.app_state.seed.selected_seed_index = 0;

    ctx.res.current_roots_grid = nullptr;
    ctx.res.roots_mesh.vertices.clear();
    ctx.res.roots_mesh.indices.clear();
    ctx.res.roots_gl_buffers.cleanup();
    ctx.app_state.display.show_roots = false;
    ctx.app_state.growth.is_growing_live = false;
    ctx.app_state.growth.is_growth_paused = false;
    ctx.app_state.growth.has_live_session = false;

    for (auto& obj : ctx.res.test_objects) {
        obj.buffers.cleanup();
    }
    ctx.res.test_objects.clear();

    ctx.app_state.ui.status_bar_message = "Loaded " + name + " (Direct GPU SDF Volume in " + std::to_string(static_cast<int>(cuda_time_ms)) + " ms)";
    RhizomeLog("REACTION", ctx.app_state.ui.status_bar_message.c_str());
    RhizomeLogVramSnapshot("Model Loaded & SDF Rasterized");
}

void ensure_seed_points_controller(GrowthControllerContext& ctx) {
    if (ctx.app_state.seed.points.empty() && ctx.res.original_mesh.vertex_count() > 0) {
        glm::vec3 min_p(0.0f), max_p(0.0f);
        float bbox_ms = 0.0f;
        rhizome::CudaRasterizer::compute_bounding_box_gpu(ctx.res.original_mesh, min_p, max_p, bbox_ms);
        glm::vec3 center = (min_p + max_p) * 0.5f;
        glm::vec3 default_seed(center.x, center.y, min_p.z + 2.0f * ctx.app_state.growth.detail_level);
        if (ctx.res.current_sdf_grid && ctx.res.current_sdf_grid->is_valid()) {
            float desired_margin = -(ctx.app_state.growth.trunk_caliber + ctx.app_state.growth.seed_offset);
            float seed_ms = 0.0f;
            rhizome::CudaRasterizer::find_default_seed_gpu(
                *ctx.res.current_sdf_grid, min_p, max_p, desired_margin, default_seed, seed_ms
            );
            // v24.13 Track F R2: mode 4 auto-seeds at the CROWN — the top of
            // the vessel interior on the center column — instead of the
            // floor-seeking default above (find_default_seed_gpu prioritizes
            // the LOWEST interior voxel, which anchored the whole v24.12
            // structure at the vessel floor: the reported "concentrated in
            // the bottom" blob). A crown anchor grows the classical root
            // ball: arms descend from the top while the GSA hierarchy fills
            // the volume. Host-side scan down the center column of the
            // host-resident SDF grid; falls back to the default seed when
            // the column never clears the margin.
            if (ctx.app_state.growth.algorithm_mode == 4 && !ctx.res.current_sdf_grid->data.empty()) {
                const auto& grid = *ctx.res.current_sdf_grid;
                const float vs = grid.voxel_size;
                const int xi = static_cast<int>(std::lround((grid.min_coord.x + grid.dims.x * 0.5f))) - grid.min_coord.x;
                const int yi = static_cast<int>(std::lround((grid.min_coord.y + grid.dims.y * 0.5f))) - grid.min_coord.y;
                if (xi >= 0 && yi >= 0 && xi < grid.dims.x && yi < grid.dims.y) {
                    for (int zi = grid.dims.z - 1; zi >= 0; --zi) {
                        const size_t base = (static_cast<size_t>(zi) * grid.dims.y + yi) * grid.dims.x + xi;
                        if (grid.data[base] <= desired_margin) {
                            default_seed = glm::vec3(
                                (grid.min_coord.x + xi) * vs,
                                (grid.min_coord.y + yi) * vs,
                                (grid.min_coord.z + zi) * vs);
                            break;
                        }
                    }
                }
            }
        }
        ctx.app_state.seed.points.push_back(default_seed);
        ctx.app_state.seed.frozen.push_back(0);
        ctx.app_state.seed.snapped_to_root.push_back(0);
        ctx.app_state.seed.has_lateral.push_back(0);
        ctx.app_state.seed.radii.push_back(ctx.app_state.growth.trunk_caliber);
        ctx.app_state.seed.selected_seed_index = 0;
    }
    if (ctx.app_state.seed.frozen.size() < ctx.app_state.seed.points.size()) {
        ctx.app_state.seed.frozen.resize(ctx.app_state.seed.points.size(), 0);
    }
    if (ctx.app_state.seed.snapped_to_root.size() < ctx.app_state.seed.points.size()) {
        ctx.app_state.seed.snapped_to_root.resize(ctx.app_state.seed.points.size(), 0);
    }
    if (ctx.app_state.seed.has_lateral.size() < ctx.app_state.seed.points.size()) {
        ctx.app_state.seed.has_lateral.resize(ctx.app_state.seed.points.size(), 0);
    }
    if (ctx.app_state.seed.radii.size() < ctx.app_state.seed.points.size()) {
        ctx.app_state.seed.radii.resize(ctx.app_state.seed.points.size(), ctx.app_state.growth.trunk_caliber);
    }
}

// v24.13 Track F R2: structure dump writer — the "artificial eye" feed.
// Every finalized volumetric bake (batch Start, live-growth end, Stop-bake)
// writes logs/structure_dump_mode<M>_<YYYYMMDD_HHMMSS>.txt: a header with
// the growth parameters in force plus one row per section
// (idx x y z radius parent order root_id). Section index order IS growth
// chronology (volumetric sessions are append-only), so the dump replays the
// growth process offline. Rotation keeps the newest five dumps. The file is
// analyzed off-box by scripts/eye/analyze_dump.py (multi-view renders,
// morphology metrics, fusion/blob indices, growth animation MP4, SLA
// printability report) — the closed loop that drives algorithm tuning
// without guessing from screenshots.
static void write_structure_dump_controller(GrowthControllerContext& ctx, const ColonizationConfig& c_cfg) {
    try {
        const auto& vol_secs = ctx.res.live_colonizer.get_volumetric_sections();
        if (vol_secs.empty()) return;

        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories("logs", ec);

        std::time_t now = std::time(nullptr);
        std::tm tmv{};
#ifdef _WIN32
        localtime_s(&tmv, &now);
#else
        localtime_r(&now, &tmv);
#endif
        char stamp[32];
        std::snprintf(stamp, sizeof(stamp), "%04d%02d%02d_%02d%02d%02d",
                      tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                      tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
        const int mode = ctx.app_state.growth.algorithm_mode;
        const std::string path = "logs/structure_dump_mode" + std::to_string(mode) + "_" + stamp + ".txt";

        std::FILE* f = std::fopen(path.c_str(), "w");
        if (!f) return;

        char vmin[96] = "?";
        char vmax[96] = "?";
        if (ctx.res.current_sdf_grid && ctx.res.current_sdf_grid->is_valid()) {
            const auto& grid = *ctx.res.current_sdf_grid;
            const float vs = grid.voxel_size;
            std::snprintf(vmin, sizeof(vmin), "%.2f %.2f %.2f",
                          grid.min_coord.x * vs, grid.min_coord.y * vs, grid.min_coord.z * vs);
            std::snprintf(vmax, sizeof(vmax), "%.2f %.2f %.2f",
                          (grid.min_coord.x + grid.dims.x) * vs,
                          (grid.min_coord.y + grid.dims.y) * vs,
                          (grid.min_coord.z + grid.dims.z) * vs);
        }

        std::fprintf(f, "# RHIZOME STRUCTURE DUMP v1\n");
        std::fprintf(f, "# mode=%d\n", mode);
        std::fprintf(f, "# iterations=%d\n", ctx.res.live_colonizer.get_current_iteration());
        std::fprintf(f, "# sections=%zu\n", vol_secs.size());
        std::fprintf(f, "# seeds=%zu\n", ctx.app_state.seed.points.size());
        std::fprintf(f, "# vessel_min=%s\n", vmin);
        std::fprintf(f, "# vessel_max=%s\n", vmax);
        std::fprintf(f, "# growth_step=%.2f\n", c_cfg.growth_step);
        std::fprintf(f, "# gsa_framework=%.1f\n", c_cfg.dev_gsa_deg[0]);
        std::fprintf(f, "# gsa_lateral=%.1f\n", c_cfg.dev_gsa_deg[1]);
        std::fprintf(f, "# gsa_order2=%.1f\n", c_cfg.dev_gsa_deg[2]);
        std::fprintf(f, "# gsa_order3=%.1f\n", c_cfg.dev_gsa_deg[3]);
        std::fprintf(f, "# grav_gain=%.3f\n", c_cfg.dev_grav_gain);
        std::fprintf(f, "# noise=%.3f\n", c_cfg.dev_noise);
        std::fprintf(f, "# branch_angle=%.1f\n", c_cfg.dev_branch_angle);
        std::fprintf(f, "# murray_gamma=%.2f\n", c_cfg.dev_murray_gamma);
        std::fprintf(f, "# min_print_radius=%.2f\n", c_cfg.dev_min_print_radius);
        std::fprintf(f, "# section_budget=%d\n", c_cfg.dev_section_budget);
        std::fprintf(f, "# vessel_ref_mm=%.2f\n", c_cfg.dev_vessel_ref_mm);
        std::fprintf(f, "# rate=%.2f,%.2f,%.2f,%.2f\n", c_cfg.dev_rate[0], c_cfg.dev_rate[1], c_cfg.dev_rate[2], c_cfg.dev_rate[3]);
        std::fprintf(f, "# spacing=%d,%d,%d,%d\n", c_cfg.dev_spacing[0], c_cfg.dev_spacing[1], c_cfg.dev_spacing[2], c_cfg.dev_spacing[3]);
        std::fprintf(f, "# max_steps=%d,%d,%d,%d\n", c_cfg.dev_max_steps[0], c_cfg.dev_max_steps[1], c_cfg.dev_max_steps[2], c_cfg.dev_max_steps[3]);
        std::fprintf(f, "# order_radius=%.2f,%.2f,%.2f,%.2f\n", c_cfg.dev_order_radius[0], c_cfg.dev_order_radius[1], c_cfg.dev_order_radius[2], c_cfg.dev_order_radius[3]);
        std::fprintf(f, "# tip_radius=%.2f,%.2f,%.2f,%.2f\n", c_cfg.dev_tip_radius[0], c_cfg.dev_tip_radius[1], c_cfg.dev_tip_radius[2], c_cfg.dev_tip_radius[3]);
        std::fprintf(f, "# COLUMNS idx x y z radius parent order root_id\n");
        for (size_t i = 0; i < vol_secs.size(); ++i) {
            const auto& s = vol_secs[i];
            const int order = std::min(std::max(s.pinned >> 1, 0), 3);
            std::fprintf(f, "%zu %.2f %.2f %.2f %.3f %d %d %d\n",
                         i, s.pos.x, s.pos.y, s.pos.z, s.radius, s.parent_sec, order, s.root_id);
        }
        std::fclose(f);

        // Rotation: keep the newest five structure dumps.
        std::vector<fs::path> dumps;
        for (const auto& entry : fs::directory_iterator("logs", ec)) {
            if (ec) break;
            const std::string name = entry.path().filename().string();
            if (name.rfind("structure_dump_mode", 0) == 0) {
                dumps.push_back(entry.path());
            }
        }
        if (dumps.size() > 5) {
            std::sort(dumps.begin(), dumps.end()); // timestamped names sort chronologically
            for (size_t i = 0; i + 5 < dumps.size(); ++i) {
                fs::remove(dumps[i], ec);
            }
        }

        char msg[512];
        std::snprintf(msg, sizeof(msg),
            "[Structure Dump] wrote %zu sections (mode %d, iter %d) -> %s — send this file back (or bundle.zip) for shape analysis",
            vol_secs.size(), mode, ctx.res.live_colonizer.get_current_iteration(), path.c_str());
        rhizome::RhizomeLog("EYE", msg);
    } catch (...) {
        // Diagnostics must never take the growth path down.
        rhizome::RhizomeLog("EYE", "[Structure Dump] write failed (non-fatal)");
    }
}

void update_implicit_session_controller(GrowthControllerContext& ctx, bool finalize_sdf_volume) {
    if (!ctx.res.current_sdf_grid) return;

    // v24.10 Track E R2: the viewport caches a device-side copy of the last
    // baked roots grid keyed on its host pointer, and only frees it when a
    // NEW grid appears on a later frame. Every path through this function
    // either drops (finalize_sdf_volume=false) or rebuilds (=true) the roots
    // grid, so release the stale device copy NOW, before the rasterize /
    // intersect uploads below, instead of carrying it through the rebuild
    // (the measured second-Start VRAM spike: old + new roots grids
    // coexisting on device). No-op after the first call in a live session.
    rhizome::CudaRasterizer::release_roots_viewport_cache_gpu();

    float vox = ctx.app_state.growth.detail_level;
    // v24.12 Track F R1: mode 4 rides the volumetric branch (sections + bake)
    const bool vol_mode = (ctx.app_state.growth.algorithm_mode == 1 ||
                           ctx.app_state.growth.algorithm_mode == 2 ||
                           ctx.app_state.growth.algorithm_mode == 4);
    if (vol_mode) {
        auto c_cfg = build_colonization_config_from_state(ctx.app_state, ctx.app_state.growth.growth_algorithms, ctx.app_state.seed.frozen);
        std::string sync_msg = "update_implicit_session: Before sync, vol_sections_.size()=" + std::to_string(ctx.res.live_colonizer.get_volumetric_sections().size());
        RhizomeLog("DEBUG", sync_msg.c_str());
        // v24.12 Track F R1: mode 4 has no decoration laterals
        if (ctx.app_state.growth.algorithm_mode != 4) {
            ctx.res.live_colonizer.sync_lateral_roots_session(c_cfg);
        }

        const auto& vol_secs = ctx.res.live_colonizer.get_volumetric_sections();
        if (!vol_secs.empty()) {
            ctx.res.current_branch_nodes.resize(vol_secs.size());
            rhizome::CudaRasterizer::convert_vol_sections_to_nodes_gpu(
                reinterpret_cast<const rhizome::CudaVolSection*>(vol_secs.data()),
                vol_secs.size(),
                ctx.res.current_branch_nodes.data()
            );
            ctx.res.branch_gl_buffers.upload(ctx.res.current_branch_nodes, ctx.app_state.display.colorize_roots || ctx.app_state.display.colorize_strands);
        } else {
            ctx.res.current_branch_nodes.clear();
            ctx.res.branch_gl_buffers.cleanup();
        }
        if (finalize_sdf_volume) {
            // v24.10 Track E R2: drop the previous bake BEFORE rasterizing the
            // replacement. A continuing Start otherwise holds the old baked
            // grid, the fresh rasterization, and the intersection result
            // simultaneously (three full-size grids + the vessel on the host
            // side) - measured as the second-Start memory spike and OOM
            // crash. The live path below has always dropped the grid; the
            // bake path now does the same, first. If a rasterization fails
            // the display falls back to no roots grid (same as live mode)
            // until the next successful bake.
            ctx.res.current_roots_grid = nullptr;
            // v24.12 Track F R1: in mode 4 the bake consumes the Murray
            // census — rebuild structural caliber right before rasterizing,
            // then report the SLA print audit for the geometry being frozen.
            if (ctx.app_state.growth.algorithm_mode == 4) {
                ctx.res.live_colonizer.apply_dev_murray_caliber(c_cfg);
                ctx.res.live_colonizer.emit_print_audit(c_cfg);
            }
            auto root_grid = ctx.res.live_colonizer.rasterize_volumetric_session(vox, false, ctx.app_state.growth.branch_smooth_k);
            if (root_grid && root_grid->is_valid()) {
                // v24.10 Track E R2: preserve_input_grid=false - root_grid is
                // a fresh local rasterization that is dropped right after
                // this call, so intersect it in place instead of copying the
                // full grid first (one full-size host allocation per bake).
                ctx.res.current_roots_grid = RootImplicitField::intersect_with_vessel(root_grid, ctx.res.current_sdf_grid, ctx.app_state.growth.wall_cut_smooth, false);
            }
            // v24.13 Track F R2: every finalized volumetric bake writes the
            // structure dump (the artificial-eye feed; see the writer above).
            write_structure_dump_controller(ctx, c_cfg);
        } else {
            ctx.res.current_roots_grid = nullptr;
        }
    } else {
        ctx.res.current_branch_nodes = ctx.res.live_colonizer.get_nodes();
        ctx.res.branch_gl_buffers.upload(ctx.res.current_branch_nodes, ctx.app_state.display.colorize_roots || ctx.app_state.display.colorize_strands);
        if (finalize_sdf_volume) {
            // v24.10 Track E R2: same release-before-rebake as the volumetric
            // branch above (see the rationale comment there).
            ctx.res.current_roots_grid = nullptr;
            auto root_grid = RootImplicitField::rasterize_root_graph(
                ctx.res.live_colonizer.get_nodes(),
                vox,
                ctx.app_state.growth.trunk_caliber,
                ctx.app_state.growth.tendril_caliber,
                ctx.app_state.growth.variable_taper ? 0.0f : ctx.app_state.growth.taper_curve,
                4.0f,
                false,
                ctx.app_state.growth.branch_smooth_k
            );
            if (root_grid && root_grid->is_valid()) {
                // v24.10 Track E R2: in-place intersect (fresh local grid,
                // dropped right after) - see the volumetric branch comment.
                ctx.res.current_roots_grid = RootImplicitField::intersect_with_vessel(root_grid, ctx.res.current_sdf_grid, ctx.app_state.growth.wall_cut_smooth, false);
            }
        } else {
            ctx.res.current_roots_grid = nullptr;
        }
    }
    if (ctx.app_state.display.show_spores) {
        ctx.res.spore_gl_buffers.upload(ctx.res.live_colonizer.get_spores());
    }
}

static void stop_live_growth_controller_impl(GrowthControllerContext& ctx) {
    ctx.app_state.growth.is_growing_live = false;
    ctx.app_state.growth.is_growth_paused = true;
    if (ctx.app_state.growth.has_live_session && ctx.res.current_sdf_grid) {
        update_implicit_session_controller(ctx, true);
        ctx.app_state.ui.status_bar_message = "Growth stopped & baked at step " + std::to_string(ctx.res.live_colonizer.get_current_iteration()) +
                                 " - Click Start to continue or Clear to reset";
    } else {
        ctx.app_state.ui.status_bar_message = "Growth stopped";
    }
    RhizomeLog("REACTION", ctx.app_state.ui.status_bar_message.c_str());
    RhizomeLogVramSnapshot("Growth Paused");
}

// v24.11 Track E R3 (growth exception guard): public wrappers. The app has
// no other exception handling on the growth path; before this change any
// throw inside a Start/Stop/live-step (e.g. std::bad_alloc from a bake
// allocation, or a length_error from an impossible grid size) escalated
// std::terminate -> abort() -> UCRT __fastfail -> exit code 0xC0000409
// with no diagnostic (the reproducible quick-second-Start crash captured
// by the watchdog, 3/3 runs). The wrappers convert any exception into a
// logged status-bar message and leave the session in the same resting
// state as Stop: growth halted, last successful bake kept (or none), app
// fully usable. The _impl bodies above/below are unchanged.
static void growth_guard_report_exception(GrowthControllerContext& ctx, const char* phase, const char* what) {
    ctx.app_state.growth.is_growing_live = false;
    ctx.app_state.growth.is_growth_paused = true;
    ctx.app_state.ui.status_bar_message = std::string("Growth ") + phase + " failed: " + what +
        " - app kept running; click Start to retry or Clear to reset";
    RhizomeLog("REACTION", ctx.app_state.ui.status_bar_message.c_str());
}

void stop_live_growth_controller(GrowthControllerContext& ctx) {
    try {
        stop_live_growth_controller_impl(ctx);
    } catch (const std::exception& e) {
        growth_guard_report_exception(ctx, "stop/bake", e.what());
    } catch (...) {
        growth_guard_report_exception(ctx, "stop/bake", "unknown exception");
    }
}

static void run_growth_engine_controller_impl(GrowthControllerContext& ctx) {
    if (!ctx.res.current_sdf_grid) {
        ctx.app_state.ui.status_bar_message = "Please import a 3D model before growing.";
        RhizomeLog("REACTION", ctx.app_state.ui.status_bar_message.c_str());
        return;
    }

    ensure_seed_points_controller(ctx);
    // v24.13 Track F R2: measure the vessel reference (largest bbox extent)
    // for the vessel-relative mode-4 length caps. Refreshed at every Start;
    // a stale reference is harmless (the next Start overwrites it after any
    // model load / Clear).
    if (ctx.res.current_sdf_grid && ctx.res.current_sdf_grid->is_valid()) {
        const auto& vgrid = *ctx.res.current_sdf_grid;
        const float ex = vgrid.dims.x * vgrid.voxel_size;
        const float ey = vgrid.dims.y * vgrid.voxel_size;
        const float ez = vgrid.dims.z * vgrid.voxel_size;
        ctx.app_state.growth.dev_vessel_ref_mm = std::max(ex, std::max(ey, ez));
    }
    ColonizationConfig c_cfg = build_colonization_config_from_state(ctx.app_state, ctx.app_state.growth.growth_algorithms, ctx.app_state.seed.frozen);

    bool continuing_existing = ctx.app_state.growth.has_live_session &&
        (((ctx.app_state.growth.algorithm_mode == 1 || ctx.app_state.growth.algorithm_mode == 2 || ctx.app_state.growth.algorithm_mode == 4) && !ctx.res.live_colonizer.get_volumetric_sections().empty()) ||
         (ctx.app_state.growth.algorithm_mode == 0 && !ctx.res.live_colonizer.get_nodes().empty()));

    if (!continuing_existing) {
        if (ctx.app_state.growth.algorithm_mode == 1 || ctx.app_state.growth.algorithm_mode == 2 || ctx.app_state.growth.algorithm_mode == 4) {
            ctx.res.live_colonizer.initialize_volumetric_session(ctx.res.current_sdf_grid, ctx.app_state.seed.points, c_cfg);
        } else {
            ctx.res.live_colonizer.initialize(ctx.res.current_sdf_grid, ctx.app_state.seed.points, c_cfg);
        }
        ctx.app_state.growth.has_live_session = true;
    } else {
        ctx.res.live_colonizer.sync_seeds_session(ctx.res.current_sdf_grid, ctx.app_state.seed.points, c_cfg);
        ctx.res.live_colonizer.prepare_resume_session(ctx.res.current_sdf_grid, c_cfg);
    }

    ctx.app_state.display.show_roots = true;
    RhizomeLog("REACTION", "Roots visualization automatically enabled for growth session");

    if (ctx.app_state.growth.live_growth) {
        ctx.app_state.growth.is_growth_paused = false;
        ctx.app_state.growth.is_growing_live = true;
        // v24.13 Track F R2: during live growth the only progressive visual
        // is the branch skeleton overlay (the roots SDF volume is baked once
        // at the end). The skeleton layer defaults OFF, which is why live
        // mode previously showed "no animation, just the final blob".
        // Auto-enable it for the live session, exactly like show_roots above.
        if (!ctx.app_state.display.show_branch_skeleton) {
            ctx.app_state.display.show_branch_skeleton = true;
            RhizomeLog("REACTION", "Branch skeleton layer auto-enabled for live animated growth");
        }
        update_implicit_session_controller(ctx, false);
        ctx.app_state.ui.status_bar_message = continuing_existing
            ? ("Continuing root growth live from step " + std::to_string(ctx.res.live_colonizer.get_current_iteration()) + "...")
            : "Growing roots live (Direct GPU Raymarching)... Adjust sliders or click Stop anytime.";
        RhizomeLog("REACTION", ctx.app_state.ui.status_bar_message.c_str());
        RhizomeLogVramSnapshot("Growth Started");
        return;
    }

    ctx.app_state.growth.is_growing_live = false;
    ctx.app_state.growth.is_growth_paused = false;
    ctx.app_state.ui.status_bar_message = continuing_existing ? "Continuing root growth..." : "Growing roots...";
    RhizomeLog("REACTION", ctx.app_state.ui.status_bar_message.c_str());

    const int batch_steps = (ctx.app_state.growth.algorithm_mode == 1 || ctx.app_state.growth.algorithm_mode == 2)
        ? std::max(c_cfg.max_iterations, static_cast<int>(c_cfg.max_iterations * (1.0f + 0.85f * c_cfg.space_fill_drive)))
        : c_cfg.max_iterations;

    if (ctx.app_state.growth.algorithm_mode == 1 || ctx.app_state.growth.algorithm_mode == 2 || ctx.app_state.growth.algorithm_mode == 4) {
        ctx.res.live_colonizer.step_volumetric_session(
            ctx.res.current_sdf_grid,
            c_cfg,
            ctx.app_state.growth.detail_level,
            ctx.app_state.growth.taper_curve,
            batch_steps
        );
    } else {
        ctx.res.live_colonizer.step_classic_session(
            ctx.res.current_sdf_grid,
            c_cfg,
            batch_steps
        );
    }

    update_implicit_session_controller(ctx, true);
    ctx.res.roots_mesh.vertices.clear();
    ctx.res.roots_mesh.indices.clear();
    ctx.res.roots_gl_buffers.cleanup();

    ctx.app_state.ui.status_bar_message = "Growth finished & baked at step " + std::to_string(ctx.res.live_colonizer.get_current_iteration()) +
                             " (Click Start to continue or Clear to reset)";
    RhizomeLog("REACTION", ctx.app_state.ui.status_bar_message.c_str());
    RhizomeLogVramSnapshot("Batch Growth Baked");
    RhizomeLogMemory("Root Sections", ctx.res.live_colonizer.get_volumetric_sections().size(), sizeof(SpaceColonization::VolCrossSection));
}

void run_growth_engine_controller(GrowthControllerContext& ctx) {
    try {
        run_growth_engine_controller_impl(ctx);
    } catch (const std::exception& e) {
        growth_guard_report_exception(ctx, "run", e.what());
    } catch (...) {
        growth_guard_report_exception(ctx, "run", "unknown exception");
    }
}

static void step_live_growth_frame_controller_impl(GrowthControllerContext& ctx, double now_time) {
    if (!ctx.app_state.growth.is_growing_live || ctx.app_state.growth.is_growth_paused || !ctx.res.current_sdf_grid) return;

    double step_interval = 1.0 / static_cast<double>(std::max(1, ctx.app_state.growth.live_fps));
    if (now_time - ctx.app_state.growth.last_live_step_time >= step_interval) {
        ctx.app_state.growth.last_live_step_time = now_time;
        ColonizationConfig live_cfg = build_colonization_config_from_state(ctx.app_state, ctx.app_state.growth.growth_algorithms, ctx.app_state.seed.frozen);
        bool still_growing = false;
        if (ctx.app_state.growth.algorithm_mode == 1 || ctx.app_state.growth.algorithm_mode == 2 || ctx.app_state.growth.algorithm_mode == 4) {
            still_growing = ctx.res.live_colonizer.step_volumetric_session(
                ctx.res.current_sdf_grid,
                live_cfg,
                ctx.app_state.growth.detail_level,
                ctx.app_state.growth.taper_curve,
                ctx.app_state.growth.live_steps_per_frame
            );
        } else {
            still_growing = ctx.res.live_colonizer.step_classic_session(
                ctx.res.current_sdf_grid,
                live_cfg,
                ctx.app_state.growth.live_steps_per_frame
            );
        }

        if (still_growing) {
            update_implicit_session_controller(ctx, false);
            ctx.app_state.ui.status_bar_message = "Growing live... Step " + std::to_string(ctx.res.live_colonizer.get_current_iteration()) +
                                     " (Direct GPU Raymarching) - Adjust sliders or click Stop";
        } else {
            ctx.app_state.growth.is_growing_live = false;
            update_implicit_session_controller(ctx, true);
            ctx.app_state.ui.status_bar_message = "Growth finished & baked at step " + std::to_string(ctx.res.live_colonizer.get_current_iteration()) +
                                     " (Direct GPU Volume)";
            RhizomeLog("REACTION", ctx.app_state.ui.status_bar_message.c_str());
            RhizomeLogVramSnapshot("Live Growth Baked");
            RhizomeLogMemory("Root Sections", ctx.res.live_colonizer.get_volumetric_sections().size(), sizeof(SpaceColonization::VolCrossSection));
        }
    }
}

void step_live_growth_frame_controller(GrowthControllerContext& ctx, double now_time) {
    try {
        step_live_growth_frame_controller_impl(ctx, now_time);
    } catch (const std::exception& e) {
        growth_guard_report_exception(ctx, "live step", e.what());
    } catch (...) {
        growth_guard_report_exception(ctx, "live step", "unknown exception");
    }
}

void load_default_specimen_controller(GrowthControllerContext& ctx) {
    rhizome::VerificationHarness::ensure_test_vessels_exist();
    rhizome::TriangleMesh init_mesh;
    std::string parse_err;
    bool mug_loaded = false;

    const std::vector<std::string> search_paths = {
        "assets/mug.obj",
        "native/assets/mug.obj",
        "../assets/mug.obj",
        "../../assets/mug.obj",
        "../native/assets/mug.obj"
    };

    for (const auto& path : search_paths) {
        std::error_code ec;
        if (std::filesystem::exists(path, ec) && !ec) {
            if (rhizome::ObjParser::parse(path, init_mesh, parse_err)) {
                mug_loaded = true;
                convert_and_update_model_controller(ctx, init_mesh, "mug.obj");
                break;
            }
        }
    }

    if (!mug_loaded) {
        init_mesh = rhizome::VerificationHarness::generate_sphere(20.0f, 64, 64);
        convert_and_update_model_controller(ctx, init_mesh, "Sphere");
    }
}


// =========================================================================
// MESH METRICS (migrated from app/ui_panels.cpp in Phase 2; relocates to
// engine/mesh in Phase 3). All metric computation executes on the CUDA GPU
// via CudaRasterizer::compute_mesh_metrics_gpu — zero CPU fallback.
// =========================================================================
CachedMeshMetrics compute_mesh_metrics(const rhizome::TriangleMesh& mesh) {
    CachedMeshMetrics m;
    m.total_tris = mesh.triangle_count();
    m.total_edges = m.total_tris * 3 / 2;
    if (m.total_tris > 0 && mesh.indices.size() >= 3) {
        float gpu_ms = 0.0f;
        rhizome::CudaRasterizer::compute_mesh_metrics_gpu(
            mesh, m.avg_edge_mm, m.min_edge_mm, m.max_edge_mm, gpu_ms
        );
    }
    return m;
}

} // namespace rhizome
