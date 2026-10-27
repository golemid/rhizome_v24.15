#pragma once
// =========================================================================
// RHIZOME GROWTH CONTROLLER — core module (v23.29 Phase 2)
// =========================================================================
// High-level orchestration of the 100% GPU growth pipeline. Migrated from
// app/growth_controller.{hpp,cpp} with the v23.28 41-member mega-context
// replaced by the approved two-reference form (Phase 1 proposal Sec 5.2,
// fixing shadow-state finding F4):
//
//     struct GrowthControllerContext { AppState& app_state; EngineResources& res; };
//
// All state (growth flags, specimen info, seed points, status bar, UI
// visibility) is read/written directly through ctx.app_state — the Single
// Source of Truth. All live GPU/GL resources flow through ctx.res.
// main.cpp is rewired to construct AppState + EngineResources in Phase 4.
// =========================================================================

#include <string>
#include <vector>

#include "core/app_state.hpp"
#include "core/engine_resources.hpp"
#include "engine/colonization/colonization.hpp" // ColonizationConfig

namespace rhizome {

struct GrowthControllerContext {
    AppState& app_state;  // Single Source of Truth (persistable state)
    EngineResources& res; // Live GPU/OpenGL resources
};

void init_default_growth_algorithms(std::vector<GrowthAlgorithmProfile>& growth_algorithms, int& selected_algorithm_idx);
void apply_algorithm_profile_to_state(AppState& app_state, const std::vector<GrowthAlgorithmProfile>& growth_algorithms, int idx);
void capture_state_into_algorithm_profile(const AppState& app_state, GrowthAlgorithmProfile& a);
ColonizationConfig build_colonization_config_from_state(
    const AppState& app_state,
    const std::vector<GrowthAlgorithmProfile>& growth_algorithms,
    const std::vector<uint8_t>& seed_frozen
);

void run_all_tests_controller(GrowthControllerContext& ctx);
void update_bounding_box_controller(GrowthControllerContext& ctx);
void convert_and_update_model_controller(GrowthControllerContext& ctx, const TriangleMesh& mesh, const std::string& name);
void ensure_seed_points_controller(GrowthControllerContext& ctx);
void update_implicit_session_controller(GrowthControllerContext& ctx, bool finalize_sdf_volume);
void stop_live_growth_controller(GrowthControllerContext& ctx);
void run_growth_engine_controller(GrowthControllerContext& ctx);
void step_live_growth_frame_controller(GrowthControllerContext& ctx, double now_time);
void load_default_specimen_controller(GrowthControllerContext& ctx);

} // namespace rhizome
