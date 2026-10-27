#pragma once
// =========================================================================
// RHIZOME VERIFICATION PANEL — ui/panels module (v23.29 Phase 4)
// =========================================================================
// The independent dockable "Verification" window (CTRL109-CTRL111; ids
// shifted +3 by the v24.01 Phase 0 renumber): test
// harness run/clear buttons and the results list whose rows load test
// models into the viewport. Split from src/app/ui_scene_panel.cpp.
//
// (CTRL107-CTRL108, the "Verification Suite" section, physically render
// inside the Scene panel exactly as in v23.28 — see panel_scene.cpp.)
// =========================================================================

#include <functional>
#include <vector>

#include <imgui.h>

#include "core/app_state.hpp"
#include "core/engine_resources.hpp"
#include "engine/io/mesh.hpp"
#include "verification/harness.hpp"

namespace rhizome {

void render_verification_panel(
    AppState& app_state,
    EngineResources& res,
    const std::function<void()>& run_all_tests,
    const std::function<void(const rhizome::TriangleMesh&, const std::string&)>& convert_and_update_model,
    ImVec2 work_pos,
    ImVec2 work_size
);

} // namespace rhizome
