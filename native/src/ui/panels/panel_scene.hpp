#pragma once
// =========================================================================
// RHIZOME SCENE PANEL — ui/panels module (v23.29 Phase 4)
// =========================================================================
// The docked "Scene" window: verification suite triggers, implicit
// specimen, layers visibility, mesh generation & export options, bark
// texture & surface relief (CTRL107-CTRL108, CTRL112-CTRL148; v24.01
// Phase 0.2: the shading trio CTRL104-106 moved to the Styles panel's
// SPECIMEN APPEARANCE section). Split from
// the monolithic src/app/ui_scene_panel.cpp (802 lines -> two domain
// panels). Also renders the dockable Status Bar strip.
//
// Context: v23.28's 32-member ScenePanelContext is replaced by the
// approved two-reference form (Phase 1 proposal Sec 5.2): AppState& +
// EngineResources& + main.cpp command callbacks. The twelve specimen_*
// shadow locals now live in app_state.specimen.* (SpecimenInfoState).
// =========================================================================

#include <functional>
#include <string>

#include "core/app_state.hpp"
#include "core/engine_resources.hpp"
#include "engine/colonization/colonization.hpp"
#include "engine/io/mesh.hpp"

namespace rhizome {

struct ScenePanelContext {
    AppState& app_state;  // Single Source of Truth
    EngineResources& res; // Live GPU/OpenGL resources

    // ---- commands owned by main.cpp orchestration ----
    std::function<void()> run_all_tests;
    std::function<void(const rhizome::TriangleMesh&, const std::string&)> convert_and_update_model;
    std::function<rhizome::ColonizationConfig()> build_colonization_config;
};

void render_scene_panel(ScenePanelContext& ctx);

// Dockable bottom status strip (quiet styling, zero padding).
void render_status_bar_panel(const std::string& status_bar_message,
                             size_t section_count = 0, size_t spore_count = 0);

} // namespace rhizome
