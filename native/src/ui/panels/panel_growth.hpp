#pragma once
// =========================================================================
// RHIZOME GROWTH PANEL — ui/panels module (v23.29 Phase 4)
// =========================================================================
// The docked "Growth" window: command bar, algorithm manager, colonization
// seeds, branching & growth, vessel & caliber, lateral roots, seeding
// management and execution (CTRL001-CTRL052). Split from the monolithic
// src/app/ui_panels.cpp (1,732 lines -> four domain panels).
//
// Context: v23.28's 28-member GrowthPanelContext is replaced by the
// approved two-reference form (Phase 1 proposal Sec 5.2): AppState& (the
// Single Source of Truth) + EngineResources& (live GPU/GL resources) plus
// the command callbacks that remain main.cpp orchestration. Shadow locals
// (seed_points, is_growing_live, status_bar_message, ...) are gone — the
// panel reads/writes app_state.* directly.
// =========================================================================

#include <functional>
#include <string>

#include "core/app_state.hpp"
#include "core/engine_resources.hpp"
#include "engine/colonization/colonization.hpp"
#include "engine/io/mesh.hpp"

namespace rhizome {

struct GrowthPanelContext {
    AppState& app_state;  // Single Source of Truth
    EngineResources& res; // Live GPU/OpenGL resources

    // ---- commands owned by main.cpp orchestration ----
    std::function<void(const rhizome::TriangleMesh&, const std::string&)> convert_and_update_model;
    std::function<void()> run_growth_engine;
    std::function<void()> stop_live_growth;
    std::function<void(bool)> update_implicit_session;
    std::function<rhizome::ColonizationConfig()> build_colonization_config;
    std::function<void(int)> apply_algorithm_profile;
    std::function<void(GrowthAlgorithmProfile&)> capture_current_into_algorithm_profile;
};

void render_growth_panel(GrowthPanelContext& ctx);

} // namespace rhizome
