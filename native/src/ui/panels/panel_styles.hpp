#pragma once
// =========================================================================
// RHIZOME STYLES & PRESETS PANELS — ui/panels module (v23.29 Phase 4)
// =========================================================================
// Two docked windows, split from src/app/ui_panels.cpp:
//   * "Presets"  — Growth Presets Library + Record New Preset
//                  (CTRL053-CTRL059)
//   * "Styles"   — Color Themes & Presets, Display Font / Typography,
//                  Viewport Element Colors, 3D Lighting & Shading and the
//                  Dear ImGui style editor
//                  (CTRL060-CTRL103; v24.01 Phase 1: ten lighting
//                   sliders at CTRL093-102)
//
// The 39-field hand-copied preset transfers of v23.28 are replaced by the
// Phase 2 pure helpers apply_growth_preset_to_state() /
// capture_growth_preset_from_state() (core/app_state.cpp). The CTRL057
// "New Preset Name" buffer is homed in AppUIState::new_preset_name
// (Phase 2 finding F8: it was a function-local static).
// =========================================================================

#include <functional>
#include <string>
#include <vector>

#include <imgui.h>

#include "core/app_state.hpp"
#include "core/algorithm_io.hpp"
#include "engine/io/mesh.hpp"

namespace rhizome {

// Font table entry (owned by main.cpp; ImGui context lives there).
struct AppFontEntry {
    const char* name;
    const char* regular_filename;
    const char* bold_filename;
    float size;
    ImFont* regular_ptr;
    ImFont* bold_ptr;
};

// UI Styling and Theme Presets
void apply_dark_engineering_theme();

// v24.01 Phase 2 (proposal 2.1-2.3): the style editor renders through the
// registered catalog (CTRL149-236) — 30 widgets via render_control(), the
// ImGuiCol_COUNT color pickers panel-locally with commit-only telemetry.
void render_style_editor_with_modern_preset(AppState& app_state);

// v24.01 Phase 2 (proposal 2.4): ImGuiStyle restore — read side of the
// "Copy Layout INI" exporter's [Style] / [StyleColors] sections. Requires a
// live ImGui context (called from main.cpp after ImGui::CreateContext);
// absent sections keep the compiled defaults.
void load_imgui_style_from_ini(const std::string& filepath = "imgui.ini");

void render_styles_panel(
    AppState& app_state,
    std::vector<AppFontEntry>& app_fonts,
    const std::function<void()>& update_active_font,
    ImVec2 work_pos,
    ImVec2 work_size
);

void render_presets_panel(
    AppState& app_state,
    const std::function<void(const rhizome::TriangleMesh&, const std::string&)>& convert_and_update_model,
    const rhizome::TriangleMesh& original_mesh,
    const std::string& loaded_model_name,
    ImVec2 work_pos,
    ImVec2 work_size
);

} // namespace rhizome
