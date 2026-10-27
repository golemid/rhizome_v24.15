#include "engine/colonization/colonization.hpp"
#include "core/unified_logger.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <cstdio>

void SpaceColonization::initialize(std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
                                   const std::vector<glm::vec3>& seed_positions,
                                   const ColonizationConfig& config) {
    nodes_.clear();
    spores_.clear();
    current_iter_ = 0;
    bonus_iterations_ = 0;

    // 1. Initialize seed root nodes on GPU
    if (!seed_positions.empty()) {
        nodes_.resize(seed_positions.size());
        rhizome::CudaRasterizer::init_seed_nodes_gpu(
            seed_positions, config.base_radius, nodes_.data(), nodes_.size()
        );
        if (!config.frozen_seeds.empty()) {
            bool spawned_dummy = false;
            size_t cur_cnt = nodes_.size();
            rhizome::CudaRasterizer::sync_classic_seed_freeze_and_spawn_gpu(
                seed_positions, config.frozen_seeds, config.base_radius,
                nodes_.data(), cur_cnt, nodes_.size(), spawned_dummy
            );
        }
    } else {
        ColonizationNode root;
        root.position = glm::vec3(0.0f);
        root.radius = config.base_radius;
        nodes_.push_back(root);
    }

    if (!vessel_grid || !vessel_grid->is_valid()) return;

    // 2. Monte Carlo rejection sampling of spores inside vessel volume on GPU
    const float spore_margin = (vessel_grid->get_min_sdf() < 0.0f) ? std::max(-0.15f, vessel_grid->get_min_sdf() * 0.45f) : -0.05f;
    rhizome::CudaRasterizer::sample_spores_gpu(*vessel_grid, config.spore_count, config.seed, spore_margin, spores_);
}

void SpaceColonization::run(std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
                            const ColonizationConfig& config,
                            ProfilingStats& stats) {
    auto start_total = std::chrono::high_resolution_clock::now();

    static const rhizome::CudaVolumeGrid empty_grid{};
    const rhizome::CudaVolumeGrid& grid_ref = (vessel_grid && vessel_grid->is_valid()) ? *vessel_grid : empty_grid;

    int iter = 0;
    for (; iter < config.max_iterations; ++iter) {
        if (spores_.empty() || nodes_.empty()) break;

        size_t cur_count = nodes_.size();
        size_t max_cap = cur_count * 2 + 16;
        nodes_.resize(max_cap);

        bool grew = rhizome::CudaRasterizer::step_classic_colonization_gpu(
            grid_ref,
            spores_,
            nodes_.data(),
            cur_count,
            max_cap,
            config.search_radius,
            config.kill_radius,
            config.growth_step,
            config.branching_rate,
            config.base_radius,
            config.tip_radius,
            config.max_iterations,
            config.seed + static_cast<uint32_t>(iter),
            false,
            false,
            config.variable_step,
            config.min_growth_step,
            config.max_growth_step,
            config.wall_clearance_scale,
            config.variable_taper,
            config.steering_cone
        );
        nodes_.resize(cur_count);
        if (!grew) break;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    stats.iterations_executed = iter;
    stats.simulation_total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    stats.avg_iteration_ms = (iter > 0) ? (stats.simulation_total_ms / iter) : 0.0;
}

void SpaceColonization::run_root_bound(std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
                                       const ColonizationConfig& config,
                                       ProfilingStats& stats) {
    auto start_total = std::chrono::high_resolution_clock::now();
    if (!vessel_grid || !vessel_grid->is_valid()) {
        stats.iterations_executed = 0;
        stats.simulation_total_ms = 0.0;
        stats.avg_iteration_ms = 0.0;
        return;
    }

    std::vector<glm::vec3> seeds;
    if (!nodes_.empty()) {
        rhizome::CudaRasterizer::extract_root_seeds_from_nodes_gpu(nodes_.data(), nodes_.size(), seeds);
    }
    if (seeds.empty()) {
        glm::vec3 min_w = glm::vec3(vessel_grid->min_coord) * vessel_grid->voxel_size;
        glm::vec3 max_w = glm::vec3(vessel_grid->max_coord) * vessel_grid->voxel_size;
        glm::vec3 default_seed = (min_w + max_w) * 0.5f;
        default_seed.z = min_w.z + 2.0f * vessel_grid->voxel_size;
        float seed_ms = 0.0f;
        rhizome::CudaRasterizer::find_default_seed_gpu(*vessel_grid, min_w, max_w, -config.base_radius, default_seed, seed_ms);
        seeds.push_back(default_seed);
    }

    initialize_volumetric_session(vessel_grid, seeds, config);
    const int total_steps = std::max(config.max_iterations, static_cast<int>(config.max_iterations * (1.0f + 0.85f * config.space_fill_drive)));
    step_volumetric_session(vessel_grid, config, vessel_grid->voxel_size, 1.0f, total_steps);

    // Sync nodes_ from vol_sections_ on GPU for 1D skeleton visualization
    nodes_.resize(vol_sections_.size());
    if (!vol_sections_.empty()) {
        rhizome::CudaRasterizer::convert_vol_sections_to_nodes_gpu(
            reinterpret_cast<const rhizome::CudaVolSection*>(vol_sections_.data()),
            vol_sections_.size(),
            nodes_.data()
        );
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    stats.iterations_executed = current_iter_;
    stats.simulation_total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    stats.avg_iteration_ms = (current_iter_ > 0) ? (stats.simulation_total_ms / current_iter_) : 0.0;
}

void SpaceColonization::initialize_volumetric_session(
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    const std::vector<glm::vec3>& seed_positions,
    const ColonizationConfig& config
) {
    nodes_.clear();
    spores_.clear();
    vol_sections_.clear();
    vol_active_tips_.clear();
    vol_next_root_id_ = 1;
    current_iter_ = 0;
    bonus_iterations_ = 0;
    min_vessel_sdf_ = 0.0f;

    if (!vessel_grid || !vessel_grid->is_valid()) {
        return;
    }

    const float vs = vessel_grid->voxel_size;
    min_vessel_sdf_ = vessel_grid->get_min_sdf();
    glm::vec3 min_w = glm::vec3(vessel_grid->min_coord) * vs;
    glm::vec3 max_w = glm::vec3(vessel_grid->max_coord) * vs;

    // 1. Sample 3D environment nutrient pockets inside vessel volume strictly on GPU
    // v24.12 Track F R1: mode 4 grows purely from the developmental program
    // (no spores) — skip the sampling entirely.
    if (config.algorithm_mode != 4) {
        const float spore_margin = (min_vessel_sdf_ < 0.0f) ? std::max(-0.15f, min_vessel_sdf_ * 0.45f) : -0.05f;
        const int target_spores = config.spore_count;
        spores_.reserve(target_spores);
        if (!rhizome::CudaRasterizer::sample_spores_gpu(*vessel_grid, target_spores, config.seed + 2027u, spore_margin, spores_)) {
            std::cerr << "[CUDA Error] GPU spore generation failed in sample_spores_gpu.\n";
        }
    }

    std::vector<glm::vec3> seeds = seed_positions;
    if (seeds.empty()) {
        seeds.push_back((min_w + max_w) * 0.5f);
    }

    // 2. Confine initial seeds to vessel interior and spawn primary root tips on GPU
    std::vector<rhizome::CudaVolSection> gpu_secs;
    std::vector<rhizome::CudaVolTip> gpu_tips;
    if (rhizome::CudaRasterizer::init_volumetric_seeds_gpu(
            *vessel_grid, seeds, config.base_radius, min_vessel_sdf_,
            config.seed + 2027u, gpu_secs, gpu_tips, vol_next_root_id_, config.wall_clearance_scale, config.primary_roots)) {
        if (!config.frozen_seeds.empty()) {
            bool spawned_dummy = false;
            rhizome::CudaRasterizer::sync_volumetric_seed_freeze_and_spawn_gpu(
                *vessel_grid, seeds, config.frozen_seeds, config.base_radius, min_vessel_sdf_,
                config.seed + 2027u, config.wall_clearance_scale,
                gpu_secs, gpu_tips, vol_next_root_id_, spawned_dummy, config.primary_roots
            );
        }
        vol_sections_.resize(gpu_secs.size());
        std::memcpy(vol_sections_.data(), gpu_secs.data(), gpu_secs.size() * sizeof(VolCrossSection));
        vol_active_tips_.resize(gpu_tips.size());
        std::memcpy(vol_active_tips_.data(), gpu_tips.data(), gpu_tips.size() * sizeof(VolRootTip));
        num_seminal_sections_ = vol_sections_.size();
        // v24.12 Track F R1: mode 4 has no decoration laterals (laterals are
        // real anatomy grown by the dev kernels) — skip the lateral pass.
        if (config.algorithm_mode != 4) {
            sync_lateral_roots_session(config);
        }
    }
}

bool SpaceColonization::step_volumetric_session(
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    const ColonizationConfig& config,
    float voxel_size,
    float taper_curve,
    int steps_to_advance
) {
    if (!vessel_grid || !vessel_grid->is_valid() || vol_sections_.empty()) {
        return false;
    }

    const bool dev_mode = (config.algorithm_mode == 4);

    // v24.12 Track F R1: section-budget guard (bake-size safety). Once the
    // cap is reached growth halts cleanly; the message lands once per call.
    if (dev_mode && static_cast<int>(vol_sections_.size()) >= config.dev_section_budget) {
        std::cerr << "[Dev] Section budget " << config.dev_section_budget
                  << " reached (" << vol_sections_.size() << " sections) - growth halted.\n";
        rhizome::RhizomeLog("REACTION", ("Developmental section budget " + std::to_string(config.dev_section_budget) + " reached - growth halted (" + std::to_string(vol_sections_.size()) + " sections)").c_str());
        return false;
    }

    // If lateral roots were previously appended to vol_sections_, strip them before advancing growth
    // v24.12 Track F R1: mode 4 NEVER strips — its laterals are real anatomy,
    // not decoration. num_seminal_sections_ stays a pure size snapshot.
    if (!dev_mode) {
        if (num_seminal_sections_ > 0 && num_seminal_sections_ < vol_sections_.size()) {
            vol_sections_.resize(num_seminal_sections_);
        } else {
            num_seminal_sections_ = vol_sections_.size();
        }
    } else {
        num_seminal_sections_ = vol_sections_.size();
    }

    const float vs = vessel_grid->voxel_size;
    // v24.12 Track F R1: mode 4 uses the exact iteration budget (no
    // space_fill scaling — the developmental program owns its timing).
    const int base_steps = dev_mode
        ? config.max_iterations
        : std::max(config.max_iterations, static_cast<int>(config.max_iterations * (1.0f + 0.85f * config.space_fill_drive)));
    const int total_steps = base_steps + bonus_iterations_;

    if (current_iter_ >= total_steps) {
        return false;
    }

    // Dynamically redistribute 3D section radii from Trunk Caliber (base) to Tip Caliber on GPU
    // v24.12 Track F R1: mode 4 skips the depth-based vascular taper both
    // before and after stepping — radii are per-order constants during
    // growth; structural caliber is rebuilt by the Murray census at bake.
    const float safe_taper = std::max(0.2f, taper_curve);
    auto& gpu_secs_ref = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
    auto& gpu_tips_ref = reinterpret_cast<std::vector<rhizome::CudaVolTip>&>(vol_active_tips_);

    if (!dev_mode) {
        rhizome::CudaRasterizer::taper_sections_gpu(gpu_secs_ref, config.base_radius, config.tip_radius, safe_taper, total_steps, config.variable_taper, config.seed);

        // Wake sleeping tips or respawn leaf tips on GPU if needed
        // (mode 4: retired roots stay retired — respawn would regrow them)
        bool any_awake_initial = false;
        rhizome::CudaRasterizer::wake_or_respawn_volumetric_tips_gpu(
            gpu_secs_ref.data(), gpu_secs_ref.size(), gpu_tips_ref, false, any_awake_initial
        );
    }

    (void)voxel_size;

    for (int s_step = 0; s_step < steps_to_advance; ++s_step) {
        bool any_awake_tip = false;
        rhizome::CudaRasterizer::check_any_awake_tips_gpu(gpu_tips_ref.data(), gpu_tips_ref.size(), any_awake_tip);
        if (!any_awake_tip || current_iter_ >= total_steps) {
            break;
        }

        std::mt19937 rng(config.seed + 2027u + static_cast<uint32_t>(current_iter_ * 131));

        // Execute the 3D Volumetric Root Colonization step directly on the NVIDIA CUDA GPU
        if (rhizome::CudaRasterizer::is_cuda_available()) {
            std::shuffle(vol_active_tips_.begin(), vol_active_tips_.end(), rng);

            rhizome::CudaVolGrowthParams gpu_params{};
            gpu_params.search_radius = config.search_radius;
            gpu_params.kill_radius = config.kill_radius;
            gpu_params.growth_step = config.growth_step;
            gpu_params.variable_step = config.variable_step;
            gpu_params.min_growth_step = config.min_growth_step;
            gpu_params.max_growth_step = config.max_growth_step;
            gpu_params.branching_rate = config.branching_rate;
            gpu_params.base_radius = config.base_radius;
            gpu_params.tip_radius = config.tip_radius;
            gpu_params.variable_taper = config.variable_taper;
            gpu_params.root_roughness = config.root_roughness;
            gpu_params.push_strength = config.push_strength;
            gpu_params.space_fill_drive = config.space_fill_drive;
            gpu_params.taper_curve = safe_taper;
            gpu_params.voxel_size = vs;
            gpu_params.min_vessel_sdf = min_vessel_sdf_;
            gpu_params.total_steps = total_steps;
            gpu_params.current_iter = current_iter_;
            gpu_params.seed = config.seed;
            gpu_params.tip_inertia_weight = config.tip_inertia_weight;
            gpu_params.spore_pull_weight = config.spore_pull_weight;
            gpu_params.roughness_amplitude = config.roughness_amplitude;
            gpu_params.feeler_nar_fwd = config.feeler_nar_fwd;
            gpu_params.feeler_nar_side = config.feeler_nar_side;
            gpu_params.feeler_med_fwd = config.feeler_med_fwd;
            gpu_params.feeler_med_side = config.feeler_med_side;
            gpu_params.feeler_wide_fwd = config.feeler_wide_fwd;
            gpu_params.feeler_wide_side = config.feeler_wide_side;
            gpu_params.feeler_wall_in = config.feeler_wall_in;
            gpu_params.feeler_wall_out = config.feeler_wall_out;
            gpu_params.clearance_score_weight = config.clearance_score_weight;
            gpu_params.align_bonus_weight = config.align_bonus_weight;
            gpu_params.lateral_fwd_weight = config.lateral_fwd_weight;
            gpu_params.lateral_side_weight = config.lateral_side_weight;
            gpu_params.lateral_radius_scale = config.lateral_radius_scale;
            gpu_params.max_branch_generations = config.max_branch_generations;
            gpu_params.skin_gap_scale = config.skin_gap_scale;
            gpu_params.push_base_gain = config.push_base_gain;
            gpu_params.push_strength_gain = config.push_strength_gain;
            gpu_params.max_step_disp_ratio = config.max_step_disp_ratio;
            gpu_params.tether_max_stretch = config.tether_max_stretch;
            gpu_params.tether_target_stretch = config.tether_target_stretch;
            gpu_params.wall_clearance_scale = config.wall_clearance_scale;
            gpu_params.steering_cone = config.steering_cone;
            gpu_params.backward_pull = config.backward_pull;
            gpu_params.tensile_straightening = config.tensile_straightening;
            gpu_params.algorithm_mode = config.algorithm_mode;
            gpu_params.murray_exponent = config.murray_exponent;
            gpu_params.murray_branch_angle = config.murray_branch_angle;
            gpu_params.surface_adherence = config.surface_adherence;
            // v24.12 Track F R1: developmental table onto the device params
            for (int o = 0; o < 4; ++o) {
                gpu_params.dev_gsa_deg[o] = config.dev_gsa_deg[o];
                gpu_params.dev_rate[o] = config.dev_rate[o];
                gpu_params.dev_spacing[o] = config.dev_spacing[o];
                gpu_params.dev_order_radius[o] = config.dev_order_radius[o];
                gpu_params.dev_tip_radius[o] = config.dev_tip_radius[o];
                gpu_params.dev_max_steps[o] = config.dev_max_steps[o];
            }
            gpu_params.dev_grav_gain = config.dev_grav_gain;
            gpu_params.dev_noise = config.dev_noise;
            gpu_params.dev_branch_angle = config.dev_branch_angle;
            gpu_params.dev_murray_gamma = config.dev_murray_gamma;
            gpu_params.dev_min_print_radius = config.dev_min_print_radius;
            gpu_params.dev_section_budget = config.dev_section_budget;

            if (rhizome::CudaRasterizer::step_volumetric_colonization_gpu(
                    *vessel_grid, gpu_params, spores_, gpu_secs_ref, gpu_tips_ref, vol_next_root_id_)) {
                current_iter_++;
                continue;
            } else {
                std::cerr << "[CUDA Error] GPU colonization step failed in step_volumetric_colonization_gpu.\n";
                break;
            }
        } else {
            std::cerr << "[CUDA Error] CUDA is not available for GPU colonization step.\n";
            break;
        }
    }

    // Final pass: ensure smooth Trunk -> Tip taper distribution across all current root sections on GPU
    // v24.12 Track F R1: mode 4 replaces the final depth-based taper with the
    // Murray census so live-view radii already carry structural caliber.
    if (dev_mode) {
        if (!apply_dev_murray_caliber(config)) {
            std::cerr << "[CUDA Error] Murray caliber pass failed in apply_dev_murray_caliber.\n";
        }
    } else if (!rhizome::CudaRasterizer::taper_sections_gpu(gpu_secs_ref, config.base_radius, config.tip_radius, safe_taper, total_steps, config.variable_taper, config.seed)) {
        std::cerr << "[CUDA Error] GPU tapering failed in taper_sections_gpu.\n";
    }

    bool any_awake_remaining = false;
    rhizome::CudaRasterizer::check_any_awake_tips_gpu(gpu_tips_ref.data(), gpu_tips_ref.size(), any_awake_remaining);

    // Snapshot seminal root count before generating lateral roots
    // v24.12 Track F R1: mode 4 skips the decoration pass (see init gate).
    num_seminal_sections_ = vol_sections_.size();
    if (!dev_mode) {
        sync_lateral_roots_session(config);
    }

    return (any_awake_remaining && current_iter_ < total_steps);
}

std::shared_ptr<rhizome::CudaVolumeGrid> SpaceColonization::rasterize_volumetric_session(
    float voxel_size,
    bool preserve_beaded,
    float branch_smooth_k
) const {
    auto volume = std::make_shared<rhizome::CudaVolumeGrid>();
    volume->voxel_size = voxel_size;
    if (vol_sections_.empty()) {
        return volume;
    }

    float cuda_ms = 0.0f;
    rhizome::CudaRasterizer::rasterize_volumetric_sections_gpu(
        reinterpret_cast<const rhizome::CudaVolSection*>(vol_sections_.data()),
        vol_sections_.size(),
        voxel_size,
        *volume,
        cuda_ms,
        preserve_beaded,
        branch_smooth_k
    );

    return volume;
}

bool SpaceColonization::step_classic_session(
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    const ColonizationConfig& config,
    int steps_to_advance
) {
    if (!vessel_grid || !vessel_grid->is_valid() || nodes_.empty() || spores_.empty()) {
        return false;
    }
    const int total_classic_steps = config.max_iterations + bonus_iterations_;
    if (current_iter_ >= total_classic_steps) {
        return false;
    }

    for (int s = 0; s < steps_to_advance; ++s) {
        if (spores_.empty() || current_iter_ >= total_classic_steps) {
            return false;
        }

        size_t cur_count = nodes_.size();
        size_t max_cap = cur_count * 2 + 16;
        nodes_.resize(max_cap);

        bool grew = rhizome::CudaRasterizer::step_classic_colonization_gpu(
            *vessel_grid,
            spores_,
            nodes_.data(),
            cur_count,
            max_cap,
            config.search_radius,
            config.kill_radius,
            config.growth_step,
            config.branching_rate,
            config.base_radius,
            config.tip_radius,
            total_classic_steps,
            config.seed + static_cast<uint32_t>(current_iter_),
            true,
            true,
            config.variable_step,
            config.min_growth_step,
            config.max_growth_step,
            config.wall_clearance_scale,
            config.variable_taper,
            config.steering_cone
        );
        nodes_.resize(cur_count);
        if (!grew) {
            return false;
        }
        current_iter_++;
    }

    return (!spores_.empty() && current_iter_ < total_classic_steps);
}

void SpaceColonization::sample_spores_only(
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    const ColonizationConfig& config
) {
    spores_.clear();
    if (!vessel_grid || !vessel_grid->is_valid()) return;

    const float min_v_sdf = std::min(0.0f, vessel_grid->get_min_sdf());
    const float spore_margin = (min_v_sdf < 0.0f) ? std::max(-0.15f, min_v_sdf * 0.45f) : -0.05f;
    const int target_spores = config.spore_count;
    spores_.reserve(target_spores);

    rhizome::CudaRasterizer::sample_spores_gpu(*vessel_grid, target_spores, config.seed + 2027u, spore_margin, spores_);
}

void SpaceColonization::prepare_resume_session(
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    const ColonizationConfig& config
) {
    if (!vessel_grid || !vessel_grid->is_valid()) return;

    const int base_steps = (config.algorithm_mode == 1 || config.algorithm_mode == 2)
        ? std::max(config.max_iterations, static_cast<int>(config.max_iterations * (1.0f + 0.85f * config.space_fill_drive)))
        : config.max_iterations;

    if (current_iter_ >= base_steps + bonus_iterations_) {
        bonus_iterations_ = (current_iter_ - base_steps) + std::max(10, base_steps);
    }

    // Replenish spores if depleted so continuing root tips have fresh nutrients to chase
    int min_needed_spores = std::max(250, config.spore_count / 10);
    if (static_cast<int>(spores_.size()) < min_needed_spores) {
        const float min_v_sdf = std::min(0.0f, vessel_grid->get_min_sdf());
        const float spore_margin = (min_v_sdf < 0.0f) ? std::max(-0.15f, min_v_sdf * 0.45f) : -0.05f;
        const int target_spores = std::max(min_needed_spores * 2, config.spore_count / 3);

        std::vector<glm::vec3> new_spores;
        if (rhizome::CudaRasterizer::sample_spores_gpu(*vessel_grid, target_spores, config.seed + 4099u + static_cast<uint32_t>(current_iter_ * 97), spore_margin, new_spores)) {
            spores_.insert(spores_.end(), new_spores.begin(), new_spores.end());
        } else {
            std::cerr << "[CUDA Error] GPU spore replenishment failed in sample_spores_gpu.\n";
        }
    }

    // Re-energize existing tips and awaken leaf tips on GPU if needed
    bool any_awake = false;
    auto& gpu_secs_ref = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
    auto& gpu_tips_ref = reinterpret_cast<std::vector<rhizome::CudaVolTip>&>(vol_active_tips_);
    rhizome::CudaRasterizer::wake_or_respawn_volumetric_tips_gpu(
        gpu_secs_ref.data(), gpu_secs_ref.size(), gpu_tips_ref, true, any_awake
    );
}

void SpaceColonization::sync_seeds_session(
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    const std::vector<glm::vec3>& seed_positions,
    const ColonizationConfig& config
) {
    if (!vessel_grid || !vessel_grid->is_valid() || seed_positions.empty()) return;

    // If growth has not advanced any steps yet, cleanly re-initialize all seeds and tips on GPU
    if (current_iter_ == 0) {
        if (config.algorithm_mode == 1 || config.algorithm_mode == 2) {
            initialize_volumetric_session(vessel_grid, seed_positions, config);
        } else {
            initialize(vessel_grid, seed_positions, config);
        }
        return;
    }

    bool spawned_new = false;
    if (config.algorithm_mode == 1 || config.algorithm_mode == 2) {
        if (vol_sections_.empty()) return;

        // Ensure vol_sections_ only contains seminal roots before synchronizing seeds
        if (num_seminal_sections_ > 0 && num_seminal_sections_ < vol_sections_.size()) {
            vol_sections_.resize(num_seminal_sections_);
        }

        auto& gpu_secs_ref = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
        auto& gpu_tips_ref = reinterpret_cast<std::vector<rhizome::CudaVolTip>&>(vol_active_tips_);
        rhizome::CudaRasterizer::sync_volumetric_seed_freeze_and_spawn_gpu(
            *vessel_grid, seed_positions, config.frozen_seeds, config.base_radius,
            vessel_grid->get_min_sdf(), config.seed + 2027u + static_cast<uint32_t>(current_iter_ * 31),
            config.wall_clearance_scale, gpu_secs_ref, gpu_tips_ref, vol_next_root_id_, spawned_new, config.primary_roots
        );

        if (spawned_new) {
            num_seminal_sections_ = vol_sections_.size();
        }
    } else {
        if (nodes_.empty()) return;
        size_t cur_cnt = nodes_.size();
        size_t max_cap = cur_cnt + seed_positions.size() + 16;
        nodes_.resize(max_cap);
        rhizome::CudaRasterizer::sync_classic_seed_freeze_and_spawn_gpu(
            seed_positions, config.frozen_seeds, config.base_radius,
            nodes_.data(), cur_cnt, max_cap, spawned_new
        );
        nodes_.resize(cur_cnt);
    }

    if (spawned_new) {
        const int base_steps = (config.algorithm_mode == 1 || config.algorithm_mode == 2)
            ? std::max(config.max_iterations, static_cast<int>(config.max_iterations * (1.0f + 0.85f * config.space_fill_drive)))
            : config.max_iterations;
        if (current_iter_ + base_steps > base_steps + bonus_iterations_) {
            bonus_iterations_ = current_iter_;
        }

        // Replenish nutrient spores on GPU so the newly spawned seed has a rich nutrient cloud to grow into
        const float min_v_sdf = std::min(0.0f, vessel_grid->get_min_sdf());
        const float spore_margin = (min_v_sdf < 0.0f) ? std::max(-0.15f, min_v_sdf * 0.45f) : -0.05f;
        const int target_spores = std::max(1000, config.spore_count - static_cast<int>(spores_.size()));
        if (target_spores > 0) {
            std::vector<glm::vec3> new_spores;
            if (rhizome::CudaRasterizer::sample_spores_gpu(
                    *vessel_grid, target_spores, config.seed + 6151u + static_cast<uint32_t>(current_iter_ * 131),
                    spore_margin, new_spores)) {
                spores_.insert(spores_.end(), new_spores.begin(), new_spores.end());
            }
        }
    }
}

void SpaceColonization::delete_seed(int seed_index) {
    if (seed_index < 0) return;

    if (!vol_sections_.empty()) {
        auto& gpu_secs_ref = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
        auto& gpu_tips_ref = reinterpret_cast<std::vector<rhizome::CudaVolTip>&>(vol_active_tips_);
        rhizome::CudaRasterizer::prune_volumetric_sections_by_seed_gpu(
            gpu_secs_ref, gpu_tips_ref, seed_index
        );
        num_seminal_sections_ = vol_sections_.size();

        nodes_.resize(vol_sections_.size());
        if (!vol_sections_.empty()) {
            rhizome::CudaRasterizer::convert_vol_sections_to_nodes_gpu(
                reinterpret_cast<const rhizome::CudaVolSection*>(vol_sections_.data()),
                vol_sections_.size(),
                nodes_.data()
            );
        } else {
            nodes_.clear();
        }
    } else if (!nodes_.empty()) {
        size_t cur_cnt = nodes_.size();
        if (rhizome::CudaRasterizer::prune_classic_nodes_by_seed_gpu(nodes_.data(), cur_cnt, seed_index)) {
            nodes_.resize(cur_cnt);
        }
    }
}

void SpaceColonization::update_caliber(
    float trunk_caliber,
    float tip_radius,
    float taper_curve,
    bool variable_taper,
    uint32_t rng_seed,
    const ColonizationConfig* dev_cfg
) {
    if (vol_sections_.empty() && nodes_.empty()) return;

    // v24.12 Track F R1: in developmental mode the Caliber sliders drive the
    // Murray census instead (trunk caliber acts as the radius cap; the
    // depth-based taper would fight the developmental scheme).
    if (dev_cfg && dev_cfg->algorithm_mode == 4) {
        if (!vol_sections_.empty()) {
            apply_dev_murray_caliber(*dev_cfg);
        }
        return;
    }

    if (!vol_sections_.empty()) {
        const float safe_taper = std::max(0.2f, taper_curve);
        const int base_steps = std::max(10, current_iter_);
        const int total_steps = base_steps + bonus_iterations_;
        auto& gpu_secs_ref = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
        rhizome::CudaRasterizer::taper_sections_gpu(
            gpu_secs_ref, trunk_caliber, tip_radius, safe_taper,
            total_steps, variable_taper, rng_seed
        );
    }
    if (!nodes_.empty()) {
        for (auto& n : nodes_) {
            if (n.parent_index == -1 || n.depth == 0) {
                n.radius = trunk_caliber;
            }
        }
    }
}

std::shared_ptr<rhizome::CudaVolumeGrid> SpaceColonization::simulate_volumetric_roots(
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    const std::vector<glm::vec3>& seed_positions,
    const ColonizationConfig& config,
    float voxel_size,
    float taper_curve,
    ProfilingStats& stats
) {
    auto start_total = std::chrono::high_resolution_clock::now();
    initialize_volumetric_session(vessel_grid, seed_positions, config);
    const int total_steps = std::max(config.max_iterations, static_cast<int>(config.max_iterations * (1.0f + 0.85f * config.space_fill_drive)));
    step_volumetric_session(vessel_grid, config, voxel_size, taper_curve, total_steps);
    auto volume = rasterize_volumetric_session(voxel_size);
    auto end_total = std::chrono::high_resolution_clock::now();
    stats.iterations_executed = current_iter_;
    stats.simulation_total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    stats.avg_iteration_ms = (current_iter_ > 0) ? (stats.simulation_total_ms / current_iter_) : 0.0;
    return volume;
}

void SpaceColonization::sync_lateral_roots_session(const ColonizationConfig& config) {
    if (vol_sections_.empty()) return;

    bool has_any_lateral = false;
    for (auto flag : config.seed_has_lateral) {
        if (flag != 0) {
            has_any_lateral = true;
            break;
        }
    }

    if (!has_any_lateral) {
        // Lateral roots are OFF: strip old lateral rootlets if present, otherwise keep all seminal roots
        if (num_seminal_sections_ > 0 && num_seminal_sections_ < vol_sections_.size()) {
            vol_sections_.resize(num_seminal_sections_);
        } else {
            num_seminal_sections_ = vol_sections_.size();
        }
        return;
    }

    // Reset to seminal roots only before appending fresh lateral roots
    if (num_seminal_sections_ > 0 && num_seminal_sections_ < vol_sections_.size()) {
        vol_sections_.resize(num_seminal_sections_);
    } else {
        num_seminal_sections_ = vol_sections_.size();
    }

    rhizome::CudaVolGrowthParams gpu_params{};
    gpu_params.lateral_density = config.lateral_density;
    gpu_params.lateral_trunk_caliber = config.lateral_trunk_caliber;
    gpu_params.lateral_tip_caliber = config.lateral_tip_caliber;
    gpu_params.lateral_max_length = config.lateral_max_length;
    gpu_params.lateral_wander = config.lateral_wander;
    gpu_params.lateral_branch_angle = config.lateral_branch_angle;
    gpu_params.growth_step = config.growth_step;
    gpu_params.seed = config.seed;

    // Generate lateral roots and append to seminal roots
    std::vector<rhizome::CudaVolSection> lateral_sections;
    float time_ms = 0.0f;
    const auto* seminal_data = reinterpret_cast<const rhizome::CudaVolSection*>(vol_sections_.data());
    const uint8_t* seed_lat_ptr = config.seed_has_lateral.empty() ? nullptr : config.seed_has_lateral.data();
    const uint8_t* seed_snap_ptr = config.seed_snapped_to_root.empty() ? nullptr : config.seed_snapped_to_root.data();
    int num_seeds = static_cast<int>(config.seed_has_lateral.size());

    if (rhizome::CudaRasterizer::generate_lateral_roots_gpu(
            seminal_data,
            static_cast<int>(num_seminal_sections_),
            seed_lat_ptr,
            seed_snap_ptr,
            num_seeds,
            gpu_params,
            lateral_sections,
            time_ms)) {
        // Reset to seminal roots only before appending fresh lateral roots
        vol_sections_.resize(num_seminal_sections_);
        
        if (!lateral_sections.empty()) {
            const int offset = static_cast<int>(num_seminal_sections_);
            for (auto& sec : lateral_sections) {
                if (sec.pinned == 64) {
                    sec.parent_sec += offset;
                    sec.pinned = 0;
                }
            }
            size_t prev_size = vol_sections_.size();
            vol_sections_.resize(prev_size + lateral_sections.size());
            std::memcpy(
                vol_sections_.data() + prev_size,
                lateral_sections.data(),
                lateral_sections.size() * sizeof(VolCrossSection)
            );
        }
    }
}

// =========================================================================
// v24.12 Track F R1 — DEVELOPMENTAL BAKE-TIME CALIBER + SLA PRINT AUDIT
// =========================================================================
bool SpaceColonization::apply_dev_murray_caliber(const ColonizationConfig& config) {
    if (vol_sections_.empty()) return false;

    auto& secs = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
    const int n = static_cast<int>(secs.size());
    const float gamma = std::max(1.5f, config.dev_murray_gamma);
    const float inv_gamma = 1.0f / gamma;
    const float floor_r = std::max(0.05f, config.dev_min_print_radius);
    const float cap_r = std::max(floor_r * 1.5f, config.base_radius * 1.25f);

    // Pass 1: child census (a section is a leaf iff child_count == 0).
    std::vector<int> child_count(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        int p = secs[i].parent_sec;
        if (p >= 0 && p < n) ++child_count[p];
    }

    // Pass 2: demand init. Frozen sections (pinned bit 0) keep their locked
    // subtree caliber (existing radius^gamma). Non-frozen leaves carry the
    // per-order terminal demand. Mid sections start at zero.
    // v24.13 Track F R2: the terminal demand is now the BIOLOGICAL tip radius
    // (not floored at the print radius) — the print floor is applied only at
    // the write below. With the floor inside the census every leaf demanded
    // the same 0.30 mm, so parents inflated toward uniform thickness and the
    // hierarchy visually collapsed (the field "shapeless" complaint); the
    // true subtree demand restores readable trunk-to-tip taper.
    std::vector<float> demand(static_cast<size_t>(n), 0.0f);
    for (int i = 0; i < n; ++i) {
        const bool frozen = (secs[i].pinned & 1) != 0;
        if (frozen) {
            demand[i] = std::pow(std::max(floor_r, secs[i].radius), gamma);
        } else if (child_count[i] == 0) {
            const int order = std::min(std::max(secs[i].pinned >> 1, 0), 3);
            const float term = std::max(config.dev_tip_radius[order], 0.05f);
            demand[i] = std::pow(term, gamma);
        }
    }

    // Pass 3: reverse accumulation. parent_sec < own index always holds
    // (append discipline — the same invariant dev_find_vol_section_seed_slot
    // relies on), so a single reverse sweep finalizes every child before its
    // parent reads it. Deterministic by construction (index order).
    for (int i = n - 1; i >= 0; --i) {
        int p = secs[i].parent_sec;
        if (p >= 0 && p < n) {
            demand[p] += demand[i];
        }
    }

    // Pass 4: write radii. Frozen sections untouched; seed bases (parent < 0,
    // pinned bit 1 = frozen seed) untouched — an unfrozen seed base naturally
    // carries the crown junction thickness (its demand aggregates all arms).
    int recomputed = 0;
    for (int i = 0; i < n; ++i) {
        const bool frozen = (secs[i].pinned & 1) != 0;
        const bool frozen_seed_base = (secs[i].parent_sec < 0) && (secs[i].pinned & 2) != 0;
        if (frozen || frozen_seed_base) continue;
        float r = std::pow(demand[i], inv_gamma);
        r = std::min(std::max(r, floor_r), cap_r);
        if (r != secs[i].radius) ++recomputed;
        secs[i].radius = r;
    }

    return true;
}

void SpaceColonization::emit_print_audit(const ColonizationConfig& config) {
    if (vol_sections_.empty()) return;

    auto& secs = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
    const int n = static_cast<int>(secs.size());
    const float floor_r = std::max(0.05f, config.dev_min_print_radius);

    // Per-order census, resin volume, min radius, connectivity.
    int per_order[4] = {0, 0, 0, 0};
    int seed_bases = 0;
    int parent_out_of_range = 0;
    float min_radius = 1e9f;
    double volume_mm3 = 0.0;
    for (int i = 0; i < n; ++i) {
        const int order = std::min(std::max(secs[i].pinned >> 1, 0), 3);
        ++per_order[order];
        if (secs[i].parent_sec < 0) {
            ++seed_bases;
        } else {
            if (secs[i].parent_sec >= n) ++parent_out_of_range;
            // Segment volume: capsule from parent to this section, average radius
            const auto& p = secs[secs[i].parent_sec].pos;
            const auto& q = secs[i].pos;
            double seg_len = std::sqrt(static_cast<double>(
                (q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y) + (q.z - p.z) * (q.z - p.z)));
            double r_avg = 0.5 * (static_cast<double>(secs[i].radius) + static_cast<double>(secs[secs[i].parent_sec].radius));
            volume_mm3 += 3.141592653589793 * r_avg * r_avg * seg_len;
        }
        if (secs[i].radius < min_radius) min_radius = secs[i].radius;
    }

    // Per-root slenderness (chain length / min radius of that root).
    // root_id groups are individual roots (commit assigns one per strand).
    struct RootStat { double len; float min_r; int secs; };
    std::unordered_map<int, RootStat> roots;
    roots.reserve(static_cast<size_t>(n) / 8 + 16);
    for (int i = 0; i < n; ++i) {
        if (secs[i].parent_sec < 0) continue;
        const auto& p = secs[secs[i].parent_sec].pos;
        const auto& q = secs[i].pos;
        double seg_len = std::sqrt(static_cast<double>(
            (q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y) + (q.z - p.z) * (q.z - p.z)));
        auto it = roots.find(secs[i].root_id);
        if (it == roots.end()) {
            roots.emplace(secs[i].root_id, RootStat{seg_len, secs[i].radius, 1});
        } else {
            it->second.len += seg_len;
            it->second.min_r = std::min(it->second.min_r, secs[i].radius);
            it->second.secs += 1;
        }
    }
    double worst_slen = 0.0; int worst_root = -1;
    int roots_over_60 = 0;
    for (const auto& kv : roots) {
        double slen = kv.second.len / std::max(0.05, static_cast<double>(kv.second.min_r));
        if (slen > 60.0) ++roots_over_60;
        if (slen > worst_slen) { worst_slen = slen; worst_root = kv.first; }
    }

    char line[512];
    std::snprintf(line, sizeof(line),
        "[Print Audit] sections=%d (order 0/1/2/3: %d/%d/%d/%d) | islands=%d seed base(s) %s | "
        "min radius=%.3f mm (floor %.3f) %s | est. resin volume=%.2f cm^3",
        n, per_order[0], per_order[1], per_order[2], per_order[3],
        seed_bases, (parent_out_of_range == 0 ? "connected" : "BROKEN"),
        min_radius, floor_r, (min_radius >= floor_r - 1e-4f ? "OK" : "BELOW"),
        volume_mm3 / 1000.0);
    rhizome::RhizomeLog("PRINT", line);
    std::snprintf(line, sizeof(line),
        "[Print Audit] slenderness: worst root #%d L/r=%.0f | roots with L/r > 60: %d "
        "(long thin spans - consider supports or a higher Min Print Radius)",
        worst_root, worst_slen, roots_over_60);
    rhizome::RhizomeLog("PRINT", line);
}

