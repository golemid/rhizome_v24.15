// =========================================================================
// RHIZOME APP STATE — core module (v23.29 Phase 2)
// =========================================================================
// The struct definitions live in core/app_state.hpp (Single Source of
// Truth for all persistable state). This translation unit owns the pure
// state logic that was previously inlined in UI panel code:
//
//   * GrowthState <-> GrowthPreset transfer, used by the Growth Presets
//     Library panel (CTRL053-CTRL059). The Apply/Update preset flows call
//     these helpers instead of hand-copying ~39 members at each call site.
//   * INI persistence (save/load_growth_presets_from_ini) intentionally
//     remains in core/algorithm_io.cpp — "Do Not Break" per CODE_MAP.md
//     Sec 3.2 — and is untouched by this migration.
// =========================================================================

#include "core/app_state.hpp"

namespace rhizome {

void apply_growth_preset_to_state(GrowthState& g, const GrowthPreset& p) {
    // Specimen detail
    g.detail_level = p.detail_level;

    // Seeds & Colonization
    g.spore_count = p.spore_count;
    g.primary_roots = p.primary_roots;
    g.search_radius = p.search_radius;
    g.kill_radius = p.kill_radius;

    // Branching & Growth
    g.growth_step = p.growth_step;
    g.variable_step = p.variable_step;
    g.min_growth_step = p.min_growth_step;
    g.max_growth_step = p.max_growth_step;
    g.branching_rate = p.branching_rate;
    g.steering_cone = p.steering_cone;
    g.max_iterations = p.max_iterations;

    // Vessel & Caliber
    g.trunk_caliber = p.trunk_caliber;
    g.tendril_caliber = p.tendril_caliber;
    g.variable_taper = p.variable_taper;
    g.taper_curve = p.taper_curve;
    g.branch_smooth_k = p.branch_smooth_k;
    g.wall_clearance_scale = p.wall_clearance_scale;
    g.wall_cut_smooth = p.wall_cut_smooth;

    // Lateral Roots Settings
    g.lateral_density = p.lateral_density;
    g.lateral_trunk_caliber = p.lateral_trunk_caliber;
    g.lateral_tip_caliber = p.lateral_tip_caliber;
    g.lateral_max_length = p.lateral_max_length;
    g.lateral_wander = p.lateral_wander;
    g.lateral_branch_angle = p.lateral_branch_angle;

    // Seeding
    g.seed_offset = p.seed_offset;

    // Growth Algorithm Mode & Physical Drivers
    g.algorithm_mode = p.algorithm_mode;
    g.root_roughness = p.root_roughness;
    g.push_strength = p.push_strength;
    g.space_fill_drive = p.space_fill_drive;
    g.backward_pull = p.backward_pull;
    g.tensile_straightening = p.tensile_straightening;
    g.murray_exponent = p.murray_exponent;
    g.murray_branch_angle = p.murray_branch_angle;
    g.surface_adherence = p.surface_adherence;

    // Live growth simulation controls
    g.live_growth = p.live_growth;
    g.live_steps_per_frame = p.live_steps_per_frame;
    g.live_fps = p.live_fps;
}

void capture_growth_preset_from_state(const GrowthState& g, GrowthPreset& p) {
    // Specimen detail
    p.detail_level = g.detail_level;

    // Seeds & Colonization
    p.spore_count = g.spore_count;
    p.primary_roots = g.primary_roots;
    p.search_radius = g.search_radius;
    p.kill_radius = g.kill_radius;

    // Branching & Growth
    p.growth_step = g.growth_step;
    p.variable_step = g.variable_step;
    p.min_growth_step = g.min_growth_step;
    p.max_growth_step = g.max_growth_step;
    p.branching_rate = g.branching_rate;
    p.steering_cone = g.steering_cone;
    p.max_iterations = g.max_iterations;

    // Vessel & Caliber
    p.trunk_caliber = g.trunk_caliber;
    p.tendril_caliber = g.tendril_caliber;
    p.variable_taper = g.variable_taper;
    p.taper_curve = g.taper_curve;
    p.branch_smooth_k = g.branch_smooth_k;
    p.wall_clearance_scale = g.wall_clearance_scale;
    p.wall_cut_smooth = g.wall_cut_smooth;

    // Lateral Roots Settings
    p.lateral_density = g.lateral_density;
    p.lateral_trunk_caliber = g.lateral_trunk_caliber;
    p.lateral_tip_caliber = g.lateral_tip_caliber;
    p.lateral_max_length = g.lateral_max_length;
    p.lateral_wander = g.lateral_wander;
    p.lateral_branch_angle = g.lateral_branch_angle;

    // Seeding
    p.seed_offset = g.seed_offset;

    // Growth Algorithm Mode & Physical Drivers
    p.algorithm_mode = g.algorithm_mode;
    p.root_roughness = g.root_roughness;
    p.push_strength = g.push_strength;
    p.space_fill_drive = g.space_fill_drive;
    p.backward_pull = g.backward_pull;
    p.tensile_straightening = g.tensile_straightening;
    p.murray_exponent = g.murray_exponent;
    p.murray_branch_angle = g.murray_branch_angle;
    p.surface_adherence = g.surface_adherence;

    // Live growth simulation controls
    p.live_growth = g.live_growth;
    p.live_steps_per_frame = g.live_steps_per_frame;
    p.live_fps = g.live_fps;
}

} // namespace rhizome
