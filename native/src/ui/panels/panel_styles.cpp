// =========================================================================
// RHIZOME STYLES & PRESETS PANELS — implementation (v23.29 Phase 4)
// =========================================================================
// Migrated from src/app/ui_panels.cpp :: apply_dark_engineering_theme /
// render_style_editor_with_modern_preset / render_styles_panel /
// render_presets_panel (lines 89-1060). Rendering behavior preserved
// verbatim; what changed (per the approved Phase 1 proposal):
//   * value controls draw through the ControlDefinition registry;
//   * the 39-field hand-copied preset transfers are replaced by the
//     Phase 2 helpers apply_growth_preset_to_state() /
//     capture_growth_preset_from_state();
//   * the CTRL057 name buffer is AppUIState::new_preset_name (finding F8:
//     it was a function-local static in v23.28);
//   * growth_presets / status_bar_message now flow through AppState.
// =========================================================================

#include "ui/panels/panel_styles.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>
#include <filesystem>
#include <unordered_map>

#include "ui/widgets/widget_factory.hpp"
#include "ui/widgets/control_registry.hpp"
#include "core/unified_logger.hpp"
#include "core/algorithm_io.hpp"

namespace rhizome {

// UI Styling: Coherent Dark Palette (#07111C, #0D1723, #111E2B, #182738, #26384A, #19D3A2, #FF8A1F, #E9F0F4, #8998A7)
void apply_dark_engineering_theme() {
    ImGuiStyle& style = ImGui::GetStyle();

    // Refined moderate roundings for modern engineering look
    style.Alpha             = 1.0f;
    style.WindowRounding    = 6.0f;
    style.ChildRounding     = 6.0f;
    style.FrameRounding     = 4.0f;
    style.PopupRounding     = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding      = 6.0f;
    style.TabRounding       = 6.0f;

    // Compact, clean spacing for maximum workspace area
    style.WindowPadding     = ImVec2(10.0f, 6.0f);
    style.WindowMinSize     = ImVec2(32.0f, 32.0f);
    style.FramePadding      = ImVec2(8.0f, 3.0f);
    style.ItemSpacing       = ImVec2(8.0f, 4.0f);
    style.ItemInnerSpacing  = ImVec2(6.0f, 4.0f);
    style.IndentSpacing     = 16.0f;
    style.ScrollbarSize     = 10.0f;
    style.GrabMinSize       = 12.0f;

    // Soft, subtle 1px panel and popup borders
    style.WindowBorderSize  = 1.0f;
    style.ChildBorderSize   = 1.0f;
    style.FrameBorderSize   = 0.0f;
    style.PopupBorderSize   = 1.0f;
    style.TabBorderSize     = 0.0f;

    ImVec4* colors = style.Colors;

    // Palette:
    const ImVec4 bg_app          = ImVec4(0.027f, 0.067f, 0.110f, 1.0f);   // Very dark blue-black #07111C
    const ImVec4 bg_panel        = ImVec4(0.051f, 0.090f, 0.137f, 1.0f);   // Panel #0D1723
    const ImVec4 bg_card         = ImVec4(0.067f, 0.118f, 0.169f, 1.0f);   // Card #111E2B
    const ImVec4 bg_control      = ImVec4(0.094f, 0.153f, 0.220f, 1.0f);   // Elevated/control background #182738
    const ImVec4 bg_control_hover= ImVec4(0.125f, 0.196f, 0.275f, 1.0f);
    const ImVec4 bg_control_act  = ImVec4(0.157f, 0.235f, 0.322f, 1.0f);

    const ImVec4 border_subtle   = ImVec4(0.149f, 0.220f, 0.290f, 0.65f);  // Subtle blue-gray #26384A with soft transparency
    const ImVec4 text_main       = ImVec4(0.914f, 0.941f, 0.957f, 1.0f);   // Main text #E9F0F4
    const ImVec4 text_secondary  = ImVec4(0.537f, 0.596f, 0.655f, 1.0f);   // Secondary text #8998A7

    const ImVec4 primary_teal    = ImVec4(0.098f, 0.827f, 0.635f, 1.0f);   // Primary teal #19D3A2
    const ImVec4 teal_hover      = ImVec4(0.157f, 0.898f, 0.706f, 1.0f);   // Teal hover #28E5B4
    const ImVec4 teal_subtle     = ImVec4(0.098f, 0.827f, 0.635f, 0.25f);

    colors[ImGuiCol_Text]                  = text_main;
    colors[ImGuiCol_TextDisabled]          = text_secondary;
    colors[ImGuiCol_WindowBg]              = bg_app;
    colors[ImGuiCol_ChildBg]               = bg_panel;
    colors[ImGuiCol_PopupBg]               = bg_card;
    colors[ImGuiCol_Border]                = border_subtle;
    colors[ImGuiCol_BorderShadow]          = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    colors[ImGuiCol_FrameBg]               = bg_control;
    colors[ImGuiCol_FrameBgHovered]        = bg_control_hover;
    colors[ImGuiCol_FrameBgActive]         = bg_control_act;
    colors[ImGuiCol_TitleBg]               = bg_app;
    colors[ImGuiCol_TitleBgActive]         = bg_panel;
    colors[ImGuiCol_TitleBgCollapsed]      = bg_app;
    colors[ImGuiCol_MenuBarBg]             = bg_app;
    colors[ImGuiCol_ScrollbarBg]           = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    colors[ImGuiCol_ScrollbarGrab]         = bg_control;
    colors[ImGuiCol_ScrollbarGrabHovered]  = bg_control_hover;
    colors[ImGuiCol_ScrollbarGrabActive]   = bg_control_act;
    colors[ImGuiCol_CheckMark]             = primary_teal;
    colors[ImGuiCol_SliderGrab]            = primary_teal;
    colors[ImGuiCol_SliderGrabActive]      = teal_hover;
    colors[ImGuiCol_Button]                = bg_control;
    colors[ImGuiCol_ButtonHovered]         = bg_control_hover;
    colors[ImGuiCol_ButtonActive]          = bg_control_act;
    colors[ImGuiCol_Header]                = bg_card;
    colors[ImGuiCol_HeaderHovered]         = bg_control;
    colors[ImGuiCol_HeaderActive]          = bg_control_hover;
    colors[ImGuiCol_Separator]             = border_subtle;
    colors[ImGuiCol_SeparatorHovered]      = border_subtle;
    colors[ImGuiCol_SeparatorActive]       = border_subtle;
    colors[ImGuiCol_ResizeGrip]            = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    colors[ImGuiCol_ResizeGripHovered]     = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    colors[ImGuiCol_ResizeGripActive]      = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    colors[ImGuiCol_Tab]                   = bg_app;
    colors[ImGuiCol_TabHovered]            = bg_control;
    colors[ImGuiCol_TabActive]             = bg_panel;
    colors[ImGuiCol_TabUnfocused]          = bg_app;
    colors[ImGuiCol_TabUnfocusedActive]    = bg_panel;
    colors[ImGuiCol_DockingPreview]        = teal_subtle;
    colors[ImGuiCol_DockingEmptyBg]        = bg_app;
}

// ============================================================================
// Interface Style Editor - catalog boundary (documented v24.01, option (b))
// ============================================================================
// Rhizome-skinned re-imagining of ImGui::ShowStyleEditor(): three toolkit
// pills (SIZES, COLORS, UI RENDERING (IMGUI)) whose widgets bind DIRECTLY to
// ImGuiStyle. They are Dear ImGui *interface* state, not Rhizome domain
// state, and are therefore deliberately OUTSIDE the 138-control contract
// (CTRL001-CTRL138, param_controls_complete_list.csv): no registry rows, no
// CSV<->registry audit coverage, no ui_events.jsonl telemetry.
//
// v24.01 plan: register these widgets into the expanded catalog with full
// telemetry coverage (V2401_PROPOSAL.md, Phase 2). The three pills are
// grouped under the ADVANCED - INTERFACE STYLE EDITOR parent pill (collapsed
// by default) so the future registration has one stable section home; keep
// them grouped and nested when extending. The former RENDERING pill is
// renamed "UI RENDERING (IMGUI)" to stop it colliding with 3D viewport
// lighting vocabulary - it only affects Dear ImGui's own 2D interface
// drawing (UI transparency, line anti-aliasing, curve tessellation).
// ============================================================================
void render_style_editor_with_modern_preset(AppState& app_state) {
    ImGuiStyle& style = ImGui::GetStyle();
    using namespace ctrldefs;

    // v24.01 option (b): the three toolkit pills are nested under one
    // ADVANCED parent pill (collapsed by default) so the Styles panel reads
    // [domain sections] -> [advanced toolkit editor] top to bottom.
    if (render_floating_pill_header("ADVANCED - INTERFACE STYLE EDITOR##RhizomeStyleEditor", false)) {
    ImGui::Indent(12.0f);

    // v24.01 Phase 2 (proposal 2.1/2.2): every style-editor widget is a
    // registered control (CTRL149-241, catalog 241 / 25 sections). The 23
    // SIZES sliders and the 7 UI RENDERING widgets render through
    // render_control() exactly like the domain controls — same primitive,
    // draw label, CSV range and format as the v23.28 call sites, plus
    // automatic ui_events.jsonl telemetry. The 63 COLORS pickers keep the
    // verbatim v23.28 loop (filter + PushID(i) + GetStyleColorName label +
    // AlphaBar|AlphaPreviewHalf flags) and log commit-only through their
    // registry rows (see below) — per proposal 2.3, color drags fire
    // per-frame, so the event is emitted on IsItemDeactivatedAfterEdit.
    if (render_floating_pill_header("SIZES##StyleSizes", true)) {
        render_control(CTRL149, app_state);
        render_control(CTRL150, app_state);
        render_control(CTRL151, app_state);
        render_control(CTRL152, app_state);
        render_control(CTRL153, app_state);
        render_control(CTRL154, app_state);
        render_control(CTRL155, app_state);
        render_control(CTRL156, app_state);
        render_control(CTRL157, app_state);
        render_control(CTRL158, app_state);
        render_control(CTRL159, app_state);
        ImGui::Separator();
        render_control(CTRL160, app_state);
        render_control(CTRL161, app_state);
        render_control(CTRL162, app_state);
        render_control(CTRL163, app_state);
        render_control(CTRL164, app_state);
        ImGui::Separator();
        render_control(CTRL165, app_state);
        render_control(CTRL166, app_state);
        render_control(CTRL167, app_state);
        render_control(CTRL168, app_state);
        render_control(CTRL169, app_state);
        render_control(CTRL170, app_state);
        render_control(CTRL171, app_state);
    }

    if (render_floating_pill_header("COLORS##StyleColorsList", true)) {
        static ImGuiTextFilter color_filter;
        color_filter.Draw("Filter##StyleColorFilter", ImGui::GetFontSize() * 14.0f);
        // Drag-start snapshots for commit-only telemetry (proposal 2.3):
        // index-addressed by the ImGuiCol_ loop counter.
        static ImVec4 s_color_drag_start[ImGuiCol_COUNT];
        for (int i = 0; i < ImGuiCol_COUNT; ++i) {
            const char* name = ImGui::GetStyleColorName(i);
            if (!color_filter.PassFilter(name)) continue;
            ImGui::PushID(i);
            ImGui::ColorEdit4(name, (float*)&style.Colors[i], ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_AlphaPreviewHalf);
            if (ImGui::IsItemActivated()) {
                s_color_drag_start[i] = style.Colors[i];
            }
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                const ControlDefinition& def =
                    k_all_controls[k_style_colors_first + static_cast<size_t>(i)];
                if (def.logs) {
                    const ImVec4& o = s_color_drag_start[i];
                    const ImVec4& n = style.Colors[i];
                    char details[128];
                    std::snprintf(details, sizeof(details),
                                  "rgba(%.3f,%.3f,%.3f,%.3f) -> rgba(%.3f,%.3f,%.3f,%.3f)",
                                  o.x, o.y, o.z, o.w, n.x, n.y, n.z, n.w);
                    log_ui_interaction(def.id, 0.0f, 0.0f, details);
                }
            }
            ImGui::PopID();
        }
    }

    if (render_floating_pill_header("UI RENDERING (IMGUI)##StyleRendering", false)) {
        render_control(CTRL235, app_state);
        render_control(CTRL236, app_state);
        render_control(CTRL237, app_state);
        render_control(CTRL238, app_state);
        render_control(CTRL239, app_state);
        render_control(CTRL240, app_state);
        render_control(CTRL241, app_state);
    }

    ImGui::Unindent(12.0f);
    } // ADVANCED - INTERFACE STYLE EDITOR (v24.01 option (b))
}

// ---------------------------------------------------------------------------
// v24.01 Phase 2 (proposal 2.4): ImGuiStyle restore — the read side of the
// "Copy Layout INI" exporter's [Style] / [StyleColors] sections. Called once
// at startup (main.cpp, immediately after load_viewport_style_from_ini and
// AFTER ImGui::CreateContext, which is the one documented context-active
// use of the RHZ_BIND_IMGUI_STYLE_* thunk environment). Sections are
// optional: a fresh install (or an imgui.ini without them) keeps the
// compiled ImGuiStyle defaults, and a present section applies only the keys
// it carries — the load_viewport_style_from_ini convention.
//
// Key coverage = exactly what the exporter writes: 27 scalar/ImVec2 style
// members + the three AA booleans + all ImGuiCol_COUNT colors. The
// ActiveFont key is deliberately NOT applied here (font selection is owned
// by the font-system init in main.cpp; documented as a Phase 2 deviation).
// ---------------------------------------------------------------------------
void load_imgui_style_from_ini(const std::string& filepath) {
    std::error_code ec_ex;
    if (!std::filesystem::exists(filepath, ec_ex) || ec_ex) return;
    std::ifstream in(filepath);
    if (!in.is_open()) return;

    std::unordered_map<std::string, std::string> style;   // [Style]
    std::unordered_map<std::string, std::string> colors;  // [StyleColors]
    std::string line;
    std::unordered_map<std::string, std::string>* target = nullptr;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty()) continue;
        if (line.front() == '[') {
            if (line == "[Style]") target = &style;
            else if (line == "[StyleColors]") target = &colors;
            else target = nullptr;
            continue;
        }
        if (target != nullptr) {
            size_t eq = line.find('=');
            if (eq != std::string::npos) {
                (*target)[line.substr(0, eq)] = line.substr(eq + 1);
            }
        }
    }
    in.close();
    if (style.empty() && colors.empty()) return;

    ImGuiStyle& st = ImGui::GetStyle();

    // [Style] scalars (writer emission: std::to_string, 6 decimals)
    auto apply_flt = [&style](const char* key, float& out) {
        auto it = style.find(key);
        if (it == style.end()) return;
        out = static_cast<float>(std::atof(it->second.c_str()));
    };
    // [Style] ImVec2 members ("x,y")
    auto apply_vec2 = [&style](const char* key, ImVec2& out) {
        auto it = style.find(key);
        if (it == style.end()) return;
        float x, y;
        if (std::sscanf(it->second.c_str(), "%f,%f", &x, &y) == 2) {
            out = ImVec2(x, y);
        }
    };
    // [Style] booleans ("1"/"0")
    auto apply_bool = [&style](const char* key, bool& out) {
        auto it = style.find(key);
        if (it == style.end()) return;
        out = (std::atoi(it->second.c_str()) != 0);
    };

    apply_flt("Alpha", st.Alpha);
    apply_flt("DisabledAlpha", st.DisabledAlpha);
    apply_vec2("WindowPadding", st.WindowPadding);
    apply_flt("WindowRounding", st.WindowRounding);
    apply_flt("WindowBorderSize", st.WindowBorderSize);
    apply_vec2("WindowMinSize", st.WindowMinSize);
    apply_flt("ChildRounding", st.ChildRounding);
    apply_flt("ChildBorderSize", st.ChildBorderSize);
    apply_flt("PopupRounding", st.PopupRounding);
    apply_flt("PopupBorderSize", st.PopupBorderSize);
    apply_vec2("FramePadding", st.FramePadding);
    apply_flt("FrameRounding", st.FrameRounding);
    apply_flt("FrameBorderSize", st.FrameBorderSize);
    apply_vec2("ItemSpacing", st.ItemSpacing);
    apply_vec2("ItemInnerSpacing", st.ItemInnerSpacing);
    apply_flt("IndentSpacing", st.IndentSpacing);
    apply_flt("ScrollbarSize", st.ScrollbarSize);
    apply_flt("ScrollbarRounding", st.ScrollbarRounding);
    apply_flt("GrabMinSize", st.GrabMinSize);
    apply_flt("GrabRounding", st.GrabRounding);
    apply_flt("TabRounding", st.TabRounding);
    apply_flt("TabBorderSize", st.TabBorderSize);
    apply_bool("AntiAliasedLines", st.AntiAliasedLines);
    apply_bool("AntiAliasedLinesUseTex", st.AntiAliasedLinesUseTex);
    apply_bool("AntiAliasedFill", st.AntiAliasedFill);
    apply_flt("CurveTessellationTol", st.CurveTessellationTol);
    apply_flt("CircleTessellationMaxError", st.CircleTessellationMaxError);

    // [StyleColors] — "Name=r,g,b,a" rows (writer precision %.3f), matched
    // through GetStyleColorName so the section is version-stable.
    if (!colors.empty()) {
        for (int i = 0; i < ImGuiCol_COUNT; ++i) {
            auto it = colors.find(ImGui::GetStyleColorName(i));
            if (it == colors.end()) continue;
            float r, g, b, a;
            if (std::sscanf(it->second.c_str(), "%f,%f,%f,%f", &r, &g, &b, &a) == 4) {
                st.Colors[i] = ImVec4(r, g, b, a);
            }
        }
    }
}

void render_styles_panel(
    AppState& app_state,
    std::vector<AppFontEntry>& app_fonts,
    const std::function<void()>& update_active_font,
    ImVec2 work_pos,
    ImVec2 work_size
) {
    using namespace ctrldefs;

    (void)work_size;
    std::string& status_bar_message = app_state.ui.status_bar_message;

    ImGui::SetNextWindowSize(ImVec2(520.0f, 640.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(work_pos.x + 320.0f, work_pos.y + 60.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Styles", nullptr, ImGuiWindowFlags_None)) {
        // ------------------------------------------------------------------
        // COLOR THEMES & PRESETS (CTRL060-CTRL065)
        // ------------------------------------------------------------------
        if (render_floating_pill_header("COLOR THEMES & PRESETS", true)) {
            const char* theme_presets[] = { "Modern (Rhizome Dark & Teal)", "Dear ImGui Dark", "Dear ImGui Classic", "Dear ImGui Light" };
            if (render_pill_combo("##ThemePresetCombo", &app_state.ui.current_theme_preset, theme_presets, IM_ARRAYSIZE(theme_presets))) {
                if (app_state.ui.current_theme_preset == 0) apply_dark_engineering_theme();
                else if (app_state.ui.current_theme_preset == 1) ImGui::StyleColorsDark();
                else if (app_state.ui.current_theme_preset == 2) ImGui::StyleColorsClassic();
                else if (app_state.ui.current_theme_preset == 3) ImGui::StyleColorsLight();
            }
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 2.0f));
            float theme_btn_w = (ImGui::GetContentRegionAvail().x - 16.0f) * 0.20f;
            bool trigger_reset = false;
            bool trigger_revert = false;

            if (ImGui::Button("Reset##StyleResetMod", ImVec2(theme_btn_w, 22.0f))) {
                trigger_reset = true;
            }
            ImGui::SameLine(0.0f, 4.0f);
            struct SavedStyleRef {
                AppState app_state_snapshot;
                ImGuiStyle style;
                float font_scale = 1.0f;
                float item_spacing_y = 4.0f;
            };

            static SavedStyleRef s_saved_ref;
            static bool s_saved_ref_init = false;
            if (!s_saved_ref_init) {
                s_saved_ref.app_state_snapshot = app_state;
                s_saved_ref.style = ImGui::GetStyle();
                s_saved_ref.font_scale = ImGui::GetIO().FontGlobalScale;
                s_saved_ref.item_spacing_y = ImGui::GetStyle().ItemSpacing.y;
                s_saved_ref_init = true;
            }

            if (ImGui::Button("Save Ref##StyleSaveRef", ImVec2(theme_btn_w, 22.0f))) {
                s_saved_ref.app_state_snapshot = app_state;
                s_saved_ref.style = ImGui::GetStyle();
                s_saved_ref.font_scale = ImGui::GetIO().FontGlobalScale;
                s_saved_ref.item_spacing_y = ImGui::GetStyle().ItemSpacing.y;
                status_bar_message = "All style parameters saved to reference";
            }
            ImGui::SameLine(0.0f, 4.0f);
            if (ImGui::Button("Revert Ref##StyleRevertRef", ImVec2(theme_btn_w, 22.0f))) {
                trigger_revert = true;
            }
            ImGui::SameLine(0.0f, 4.0f);
            if (ImGui::Button("Copy C++##StyleCopyCPP", ImVec2(theme_btn_w, 22.0f))) {
                const ImGuiStyle& st = ImGui::GetStyle();
                std::string code = "// ==========================================================================\n";
                code += "// RHIZOME STYLES & VIEWPORT THEME C++ CONFIGURATION\n";
                code += "// ==========================================================================\n\n";

                code += "// 1. Theme Preset & Typography\n";
                code += "current_theme_preset = " + std::to_string(app_state.ui.current_theme_preset) + ";\n";
                code += "active_font_idx = " + std::to_string(app_state.ui.active_font_idx) + ";";
                if (app_state.ui.active_font_idx >= 0 && app_state.ui.active_font_idx < (int)app_fonts.size()) {
                    code += " // " + std::string(app_fonts[app_state.ui.active_font_idx].name);
                }
                code += "\n";
                code += "font_bold = " + std::string(app_state.ui.font_bold ? "true" : "false") + ";\n";
                code += "ImGui::GetIO().FontGlobalScale = " + std::to_string(ImGui::GetIO().FontGlobalScale) + "f;\n";
                code += "ImGui::GetStyle().ItemSpacing.y = " + std::to_string(st.ItemSpacing.y) + "f;\n\n";

                code += "// 2. Viewport Element Colors\n";
                auto append_vec4 = [](std::string& s, const char* var, const glm::vec4& v) {
                    char b[128];
                    snprintf(b, sizeof(b), "%s = glm::vec4(%.3ff, %.3ff, %.3ff, %.3ff);\n", var, v.x, v.y, v.z, v.w);
                    s += b;
                };
                append_vec4(code, "color_viewport_bg", app_state.display.color_viewport_bg);
                append_vec4(code, "color_floor_grid", app_state.display.color_floor_grid);
                append_vec4(code, "color_floor_axis_x", app_state.display.color_floor_axis_x);
                append_vec4(code, "color_floor_axis_y", app_state.display.color_floor_axis_y);
                append_vec4(code, "color_original_mesh", app_state.display.color_original_mesh);
                append_vec4(code, "color_converted_body", app_state.display.color_converted_body);
                append_vec4(code, "color_roots", app_state.display.color_roots);
                append_vec4(code, "color_branch_skeleton", app_state.display.color_branch_skeleton);
                append_vec4(code, "color_spores", app_state.display.color_spores);
                append_vec4(code, "color_seed_selected", app_state.display.color_seed_selected);
                append_vec4(code, "color_seed_unselected", app_state.display.color_seed_unselected);
                append_vec4(code, "color_bounding_box", app_state.display.color_bounding_box);
                append_vec4(code, "color_wireframe", app_state.display.color_wireframe);
                append_vec4(code, "color_test_pass", app_state.display.color_test_pass);
                append_vec4(code, "color_test_fail", app_state.display.color_test_fail);
                code += "\n";

                code += "// 3. 3D Lighting & Shading Parameters\n";
                auto append_flt = [](std::string& s, const char* var, float val) {
                    char b[128];
                    snprintf(b, sizeof(b), "%s = %.3ff;\n", var, val);
                    s += b;
                };
                append_flt(code, "param_glass_frosting", app_state.display.param_glass_frosting);
                append_flt(code, "param_shader_diffuse", app_state.display.param_shader_diffuse);
                append_flt(code, "param_shader_edge_darkening", app_state.display.param_shader_edge_darkening);
                append_flt(code, "param_shader_crevice_ao", app_state.display.param_shader_crevice_ao);
                append_flt(code, "param_shader_specular", app_state.display.param_shader_specular);
                append_flt(code, "param_shader_bark_strength", app_state.display.param_shader_bark_strength);
                append_flt(code, "param_shader_bark_frequency", app_state.display.param_shader_bark_frequency);
                // v24.01 Phase 1: promoted lighting constants
                append_flt(code, "param_light_azimuth", app_state.display.param_light_azimuth);
                append_flt(code, "param_light_elevation", app_state.display.param_light_elevation);
                append_flt(code, "param_fill_light", app_state.display.param_fill_light);
                append_flt(code, "param_ambient", app_state.display.param_ambient);
                append_flt(code, "param_glossiness", app_state.display.param_glossiness);
                append_flt(code, "param_highlight_tint", app_state.display.param_highlight_tint);
                append_flt(code, "param_fresnel_rim", app_state.display.param_fresnel_rim);
                append_flt(code, "param_volumetric_ao", app_state.display.param_volumetric_ao);
                append_flt(code, "param_sss_strength", app_state.display.param_sss_strength);
                append_flt(code, "param_sss_warmth", app_state.display.param_sss_warmth);
                code += "\n";

                code += "// 4. Dear ImGui Style Metrics\n";
                code += "ImGuiStyle& style = ImGui::GetStyle();\n";
                code += "style.Alpha = " + std::to_string(st.Alpha) + "f;\n";
                code += "style.DisabledAlpha = " + std::to_string(st.DisabledAlpha) + "f;\n";
                code += "style.WindowRounding = " + std::to_string(st.WindowRounding) + "f;\n";
                code += "style.ChildRounding = " + std::to_string(st.ChildRounding) + "f;\n";
                code += "style.FrameRounding = " + std::to_string(st.FrameRounding) + "f;\n";
                code += "style.PopupRounding = " + std::to_string(st.PopupRounding) + "f;\n";
                code += "style.ScrollbarRounding = " + std::to_string(st.ScrollbarRounding) + "f;\n";
                code += "style.GrabRounding = " + std::to_string(st.GrabRounding) + "f;\n";
                code += "style.TabRounding = " + std::to_string(st.TabRounding) + "f;\n";
                code += "style.WindowBorderSize = " + std::to_string(st.WindowBorderSize) + "f;\n";
                code += "style.ChildBorderSize = " + std::to_string(st.ChildBorderSize) + "f;\n";
                code += "style.FrameBorderSize = " + std::to_string(st.FrameBorderSize) + "f;\n";
                code += "style.PopupBorderSize = " + std::to_string(st.PopupBorderSize) + "f;\n";
                code += "style.TabBorderSize = " + std::to_string(st.TabBorderSize) + "f;\n\n";

                code += "// 5. Dear ImGui Style Colors\n";
                for (int i = 0; i < ImGuiCol_COUNT; ++i) {
                    const ImVec4& c = st.Colors[i];
                    const char* name = ImGui::GetStyleColorName(i);
                    char buf[128];
                    snprintf(buf, sizeof(buf), "style.Colors[%s] = ImVec4(%.3ff, %.3ff, %.3ff, %.3ff);\n", name, c.x, c.y, c.z, c.w);
                    code += buf;
                }
                ImGui::SetClipboardText(code.c_str());
                status_bar_message = "All style C++ code copied to clipboard";
            }
            ImGui::SameLine(0.0f, 4.0f);
            if (ImGui::Button("Copy Layout##StyleCopyLayout", ImVec2(theme_btn_w, 22.0f))) {
                ImGui::SaveIniSettingsToDisk("imgui.ini");

                size_t out_size = 0;
                const char* layout_ini = ImGui::SaveIniSettingsToMemory(&out_size);
                std::string combined_data;
                if (layout_ini && out_size > 0) {
                    combined_data.assign(layout_ini, out_size);
                    combined_data += "\n\n";
                }

                // Append App Window & Toggle States
                combined_data += "; ==========================================================================\n";
                combined_data += "; APPLICATION & WINDOW VISIBILITY STATES\n";
                combined_data += "; ==========================================================================\n";
                combined_data += "[AppState]\n";
                combined_data += "ShowStyleEditor=" + std::string(app_state.verification.show_style_editor ? "1" : "0") + "\n";
                combined_data += "ShowTestsWindow=" + std::string(app_state.verification.show_tests_window ? "1" : "0") + "\n";
                combined_data += "FontBold=" + std::string(app_state.ui.font_bold ? "1" : "0") + "\n";
                combined_data += "IsOrthographic=" + std::string(app_state.viewport.is_orthographic ? "1" : "0") + "\n";
                combined_data += "ThemePreset=" + std::to_string(app_state.ui.current_theme_preset) + "\n\n";

                // Append Style & Typography Information
                const ImGuiStyle& st = ImGui::GetStyle();
                combined_data += "; ==========================================================================\n";
                combined_data += "; DEAR IMGUI STYLE & THEME CONFIGURATION\n";
                combined_data += "; ==========================================================================\n";
                combined_data += "[Style]\n";
                if (app_state.ui.active_font_idx >= 0 && app_state.ui.active_font_idx < (int)app_fonts.size()) {
                    combined_data += "ActiveFont=" + std::string(app_fonts[app_state.ui.active_font_idx].name) + "\n";
                }
                combined_data += "Alpha=" + std::to_string(st.Alpha) + "\n";
                combined_data += "WindowPadding=" + std::to_string(st.WindowPadding.x) + "," + std::to_string(st.WindowPadding.y) + "\n";
                combined_data += "WindowRounding=" + std::to_string(st.WindowRounding) + "\n";
                combined_data += "WindowBorderSize=" + std::to_string(st.WindowBorderSize) + "\n";
                combined_data += "WindowMinSize=" + std::to_string(st.WindowMinSize.x) + "," + std::to_string(st.WindowMinSize.y) + "\n";
                combined_data += "ChildRounding=" + std::to_string(st.ChildRounding) + "\n";
                combined_data += "PopupRounding=" + std::to_string(st.PopupRounding) + "\n";
                combined_data += "FramePadding=" + std::to_string(st.FramePadding.x) + "," + std::to_string(st.FramePadding.y) + "\n";
                combined_data += "FrameRounding=" + std::to_string(st.FrameRounding) + "\n";
                combined_data += "FrameBorderSize=" + std::to_string(st.FrameBorderSize) + "\n";
                combined_data += "ItemSpacing=" + std::to_string(st.ItemSpacing.x) + "," + std::to_string(st.ItemSpacing.y) + "\n";
                combined_data += "ItemInnerSpacing=" + std::to_string(st.ItemInnerSpacing.x) + "," + std::to_string(st.ItemInnerSpacing.y) + "\n";
                combined_data += "IndentSpacing=" + std::to_string(st.IndentSpacing) + "\n";
                combined_data += "ScrollbarSize=" + std::to_string(st.ScrollbarSize) + "\n";
                combined_data += "ScrollbarRounding=" + std::to_string(st.ScrollbarRounding) + "\n";
                combined_data += "GrabMinSize=" + std::to_string(st.GrabMinSize) + "\n";
                combined_data += "GrabRounding=" + std::to_string(st.GrabRounding) + "\n";
                combined_data += "TabRounding=" + std::to_string(st.TabRounding) + "\n";
                // v24.01 Phase 2 (proposal 2.4): the remaining ImGuiStyle
                // members of the registered catalog — the three border sizes
                // the v23.28 exporter omitted plus the seven UI RENDERING
                // widgets. Read side: load_imgui_style_from_ini (below).
                combined_data += "ChildBorderSize=" + std::to_string(st.ChildBorderSize) + "\n";
                combined_data += "PopupBorderSize=" + std::to_string(st.PopupBorderSize) + "\n";
                combined_data += "TabBorderSize=" + std::to_string(st.TabBorderSize) + "\n";
                combined_data += "DisabledAlpha=" + std::to_string(st.DisabledAlpha) + "\n";
                combined_data += "AntiAliasedLines=" + std::string(st.AntiAliasedLines ? "1" : "0") + "\n";
                combined_data += "AntiAliasedLinesUseTex=" + std::string(st.AntiAliasedLinesUseTex ? "1" : "0") + "\n";
                combined_data += "AntiAliasedFill=" + std::string(st.AntiAliasedFill ? "1" : "0") + "\n";
                combined_data += "CurveTessellationTol=" + std::to_string(st.CurveTessellationTol) + "\n";
                combined_data += "CircleTessellationMaxError=" + std::to_string(st.CircleTessellationMaxError) + "\n";

                combined_data += "\n[StyleColors]\n";
                for (int i = 0; i < ImGuiCol_COUNT; ++i) {
                    const char* name = ImGui::GetStyleColorName(i);
                    const ImVec4& col = st.Colors[i];
                    char buf[128];
                    snprintf(buf, sizeof(buf), "%s=%.3f,%.3f,%.3f,%.3f\n", name, col.x, col.y, col.z, col.w);
                    combined_data += buf;
                }

                // Append Viewport Element Colors
                combined_data += "\n; ==========================================================================\n";
                combined_data += "; VIEWPORT ELEMENT COLORS\n";
                combined_data += "; ==========================================================================\n";
                combined_data += "[ViewportColors]\n";
                auto append_ini_col = [](std::string& s, const char* name, const glm::vec4& v) {
                    char b[128];
                    snprintf(b, sizeof(b), "%s=%.3f,%.3f,%.3f,%.3f\n", name, v.x, v.y, v.z, v.w);
                    s += b;
                };
                append_ini_col(combined_data, "ViewportBg", app_state.display.color_viewport_bg);
                append_ini_col(combined_data, "FloorGrid", app_state.display.color_floor_grid);
                append_ini_col(combined_data, "FloorAxisX", app_state.display.color_floor_axis_x);
                append_ini_col(combined_data, "FloorAxisY", app_state.display.color_floor_axis_y);
                append_ini_col(combined_data, "OriginalMesh", app_state.display.color_original_mesh);
                append_ini_col(combined_data, "ConvertedBody", app_state.display.color_converted_body);
                append_ini_col(combined_data, "Roots", app_state.display.color_roots);
                append_ini_col(combined_data, "BranchSkeleton", app_state.display.color_branch_skeleton);
                append_ini_col(combined_data, "Spores", app_state.display.color_spores);
                append_ini_col(combined_data, "SeedSelected", app_state.display.color_seed_selected);
                append_ini_col(combined_data, "SeedUnselected", app_state.display.color_seed_unselected);
                append_ini_col(combined_data, "BoundingBox", app_state.display.color_bounding_box);
                append_ini_col(combined_data, "Wireframe", app_state.display.color_wireframe);
                append_ini_col(combined_data, "TestPass", app_state.display.color_test_pass);
                append_ini_col(combined_data, "TestFail", app_state.display.color_test_fail);

                // Append 3D Lighting & Shading Parameters
                combined_data += "\n; ==========================================================================\n";
                combined_data += "; 3D LIGHTING & SHADING\n";
                combined_data += "; ==========================================================================\n";
                combined_data += "[LightingAndShading]\n";
                combined_data += "Diffuse=" + std::to_string(app_state.display.param_shader_diffuse) + "\n";
                combined_data += "EdgeDarkening=" + std::to_string(app_state.display.param_shader_edge_darkening) + "\n";
                combined_data += "CreviceAO=" + std::to_string(app_state.display.param_shader_crevice_ao) + "\n";
                combined_data += "Specular=" + std::to_string(app_state.display.param_shader_specular) + "\n";
                combined_data += "BarkStrength=" + std::to_string(app_state.display.param_shader_bark_strength) + "\n";
                combined_data += "BarkFrequency=" + std::to_string(app_state.display.param_shader_bark_frequency) + "\n";
                combined_data += "GlassFrosting=" + std::to_string(app_state.display.param_glass_frosting) + "\n";
                combined_data += "SpecimenOpacity=" + std::to_string(app_state.display.specimen_opacity) + "\n";
                // v24.01 Phase 1: promoted lighting constants (read side:
                // load_viewport_style_from_ini)
                combined_data += "LightAzimuth=" + std::to_string(app_state.display.param_light_azimuth) + "\n";
                combined_data += "LightElevation=" + std::to_string(app_state.display.param_light_elevation) + "\n";
                combined_data += "FillLight=" + std::to_string(app_state.display.param_fill_light) + "\n";
                combined_data += "Ambient=" + std::to_string(app_state.display.param_ambient) + "\n";
                combined_data += "Glossiness=" + std::to_string(app_state.display.param_glossiness) + "\n";
                combined_data += "HighlightTint=" + std::to_string(app_state.display.param_highlight_tint) + "\n";
                combined_data += "FresnelRim=" + std::to_string(app_state.display.param_fresnel_rim) + "\n";
                combined_data += "VolumetricAO=" + std::to_string(app_state.display.param_volumetric_ao) + "\n";
                combined_data += "SSSStrength=" + std::to_string(app_state.display.param_sss_strength) + "\n";
                combined_data += "SSSWarmth=" + std::to_string(app_state.display.param_sss_warmth) + "\n";

                // Append Growth Presets
                save_growth_presets_to_ini(app_state.growth.growth_presets, "imgui.ini");
                combined_data += "\n; ==========================================================================\n";
                combined_data += "; GROWTH SLIDER PRESETS\n";
                combined_data += "; ==========================================================================\n";
                combined_data += "[GrowthPresets]\n";
                combined_data += "Count=" + std::to_string(app_state.growth.growth_presets.size()) + "\n";
                for (size_t i = 0; i < app_state.growth.growth_presets.size(); ++i) {
                    const auto& p = app_state.growth.growth_presets[i];
                    std::string pfx = "Preset" + std::to_string(i) + "_";
                    combined_data += pfx + "Name=" + p.name + "\n";
                    combined_data += pfx + "DetailLevel=" + std::to_string(p.detail_level) + "\n";
                    combined_data += pfx + "Spores=" + std::to_string(p.spore_count) + "\n";
                    combined_data += pfx + "PrimaryRoots=" + std::to_string(p.primary_roots) + "\n";
                    combined_data += pfx + "SearchRadius=" + std::to_string(p.search_radius) + "\n";
                    combined_data += pfx + "KillRadius=" + std::to_string(p.kill_radius) + "\n";
                    combined_data += pfx + "GrowthStep=" + std::to_string(p.growth_step) + "\n";
                    combined_data += pfx + "VariableStep=" + std::string(p.variable_step ? "1" : "0") + "\n";
                    combined_data += pfx + "MinGrowthStep=" + std::to_string(p.min_growth_step) + "\n";
                    combined_data += pfx + "MaxGrowthStep=" + std::to_string(p.max_growth_step) + "\n";
                    combined_data += pfx + "BranchingRate=" + std::to_string(p.branching_rate) + "\n";
                    combined_data += pfx + "SteeringCone=" + std::to_string(p.steering_cone) + "\n";
                    combined_data += pfx + "MaxIterations=" + std::to_string(p.max_iterations) + "\n";
                    combined_data += pfx + "TrunkCaliber=" + std::to_string(p.trunk_caliber) + "\n";
                    combined_data += pfx + "TendrilCaliber=" + std::to_string(p.tendril_caliber) + "\n";
                    combined_data += pfx + "VariableTaper=" + std::string(p.variable_taper ? "1" : "0") + "\n";
                    combined_data += pfx + "TaperCurve=" + std::to_string(p.taper_curve) + "\n";
                    combined_data += pfx + "BranchSmoothK=" + std::to_string(p.branch_smooth_k) + "\n";
                    combined_data += pfx + "WallClearance=" + std::to_string(p.wall_clearance_scale) + "\n";
                    combined_data += pfx + "WallCutSmooth=" + std::to_string(p.wall_cut_smooth) + "\n";
                    combined_data += pfx + "LateralDensity=" + std::to_string(p.lateral_density) + "\n";
                    combined_data += pfx + "LateralTrunkCaliber=" + std::to_string(p.lateral_trunk_caliber) + "\n";
                    combined_data += pfx + "LateralTipCaliber=" + std::to_string(p.lateral_tip_caliber) + "\n";
                    combined_data += pfx + "LateralMaxLength=" + std::to_string(p.lateral_max_length) + "\n";
                    combined_data += pfx + "LateralWander=" + std::to_string(p.lateral_wander) + "\n";
                    combined_data += pfx + "LateralBranchAngle=" + std::to_string(p.lateral_branch_angle) + "\n";
                    combined_data += pfx + "SeedOffset=" + std::to_string(p.seed_offset) + "\n";
                    combined_data += pfx + "AlgorithmMode=" + std::to_string(p.algorithm_mode) + "\n";
                    combined_data += pfx + "RootRoughness=" + std::to_string(p.root_roughness) + "\n";
                    combined_data += pfx + "PushStrength=" + std::to_string(p.push_strength) + "\n";
                    combined_data += pfx + "SpaceFillDrive=" + std::to_string(p.space_fill_drive) + "\n";
                    combined_data += pfx + "BackwardPull=" + std::to_string(p.backward_pull) + "\n";
                    combined_data += pfx + "TensileStraightening=" + std::to_string(p.tensile_straightening) + "\n";
                    combined_data += pfx + "MurrayExponent=" + std::to_string(p.murray_exponent) + "\n";
                    combined_data += pfx + "MurrayBranchAngle=" + std::to_string(p.murray_branch_angle) + "\n";
                    combined_data += pfx + "SurfaceAdherence=" + std::to_string(p.surface_adherence) + "\n";
                    combined_data += pfx + "LiveGrowth=" + std::string(p.live_growth ? "1" : "0") + "\n";
                    combined_data += pfx + "LiveStepsPerFrame=" + std::to_string(p.live_steps_per_frame) + "\n";
                    combined_data += pfx + "LiveFps=" + std::to_string(p.live_fps) + "\n";
                }

                ImGui::SetClipboardText(combined_data.c_str());
                status_bar_message = "Layout, Style & Presets copied to clipboard";
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Copies current docking layout, active font, and style parameters to clipboard.");
            }
            ImGui::PopStyleVar();

            if (trigger_reset) {
                app_state.reset_all_styles_to_default();
                apply_dark_engineering_theme();
                if (update_active_font) {
                    update_active_font();
                }
                ImGui::GetIO().FontGlobalScale = 1.0f;
                ImGui::GetStyle().ItemSpacing.y = 4.0f;

                ImGuiStyle& st = ImGui::GetStyle();
                st.DisabledAlpha = 0.60f;
                st.AntiAliasedLines = true;
                st.AntiAliasedLinesUseTex = true;
                st.AntiAliasedFill = true;
                st.CurveTessellationTol = 1.25f;
                st.CircleTessellationMaxError = 0.30f;

                status_bar_message = "All style parameters reset to default";
            }
            if (trigger_revert) {
                app_state.display = s_saved_ref.app_state_snapshot.display;
                app_state.ui.current_theme_preset = s_saved_ref.app_state_snapshot.ui.current_theme_preset;
                app_state.ui.active_font_idx = s_saved_ref.app_state_snapshot.ui.active_font_idx;
                app_state.ui.font_bold = s_saved_ref.app_state_snapshot.ui.font_bold;
                ImGui::GetStyle() = s_saved_ref.style;
                if (update_active_font) {
                    update_active_font();
                }
                ImGui::GetIO().FontGlobalScale = s_saved_ref.font_scale;
                ImGui::GetStyle().ItemSpacing.y = s_saved_ref.item_spacing_y;
                status_bar_message = "All style parameters reverted from reference";
            }
        }

        // ------------------------------------------------------------------
        // DISPLAY FONT / TYPOGRAPHY (CTRL066-CTRL070)
        // ------------------------------------------------------------------
        if (render_floating_pill_header("DISPLAY FONT / TYPOGRAPHY", true)) {
            if (begin_pill_combo("##FontSelector", app_fonts[app_state.ui.active_font_idx].name)) {
                for (int i = 0; i < static_cast<int>(app_fonts.size()); ++i) {
                    bool is_selected = (app_state.ui.active_font_idx == i);
                    if (ImGui::Selectable(app_fonts[i].name, is_selected)) {
                        app_state.ui.active_font_idx = i;
                        update_active_font();
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                end_pill_combo();
            }

            if (render_control(CTRL067, app_state)) {
                update_active_font();
            }

            render_control(CTRL068, app_state);
            render_control(CTRL069, app_state);

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 2.0f));
            if (ImGui::Button("Reset Typography Settings", ImVec2(-1.0f, 22.0f))) {
                ImGui::GetIO().FontGlobalScale = 1.0f;
                ImGui::GetStyle().ItemSpacing.y = 4.0f;
            }
            ImGui::PopStyleVar();
        }

        // ------------------------------------------------------------------
        // VIEWPORT ELEMENT COLORS (CTRL071-CTRL086)
        // ------------------------------------------------------------------
        if (render_floating_pill_header("VIEWPORT ELEMENT COLORS", true)) {
            if (ImGui::BeginTable("ViewportColorsGrid", 2, ImGuiTableFlags_None)) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); render_control(CTRL071, app_state);
                ImGui::TableSetColumnIndex(1); render_control(CTRL072, app_state);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); render_control(CTRL073, app_state);
                ImGui::TableSetColumnIndex(1); render_control(CTRL074, app_state);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); render_control(CTRL075, app_state);
                ImGui::TableSetColumnIndex(1); render_control(CTRL076, app_state);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); render_control(CTRL077, app_state);
                ImGui::TableSetColumnIndex(1); render_control(CTRL078, app_state);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); render_control(CTRL079, app_state);
                ImGui::TableSetColumnIndex(1); render_control(CTRL080, app_state);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); render_control(CTRL081, app_state);
                ImGui::TableSetColumnIndex(1); render_control(CTRL082, app_state);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); render_control(CTRL083, app_state);
                ImGui::TableSetColumnIndex(1); render_control(CTRL084, app_state);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); render_control(CTRL085, app_state);
                ImGui::EndTable();
            }

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 2.0f));
            if (ImGui::Button("Reset Viewport Colors", ImVec2(-1.0f, 22.0f))) {
                app_state.display.reset_viewport_colors_to_default();
            }
            ImGui::PopStyleVar();
        }

        // ------------------------------------------------------------------
        // 3D LIGHTING & SHADING (CTRL087-CTRL103)
        // v24.01 Phase 1: 6 -> 16 sliders. CTRL093-102 promote the constants
        // that were hardcoded in the GL mesh shader and the CUDA raymarcher
        // (key/fill direction, ambient, fill weight, glossiness, highlight
        // tint, fresnel rim, volumetric AO, SSS strength/warmth); the
        // "(raymarch)" trio affects only the CUDA raymarch path — the GL
        // mesh path uses screen-space crevice occlusion instead.
        // ------------------------------------------------------------------
        if (render_floating_pill_header("3D LIGHTING & SHADING", true)) {
            render_control(CTRL087, app_state);
            render_control(CTRL088, app_state);
            render_control(CTRL089, app_state);
            render_control(CTRL090, app_state);
            render_control(CTRL091, app_state);
            render_control(CTRL092, app_state);
            render_control(CTRL093, app_state);
            render_control(CTRL094, app_state);
            render_control(CTRL095, app_state);
            render_control(CTRL096, app_state);
            render_control(CTRL097, app_state);
            render_control(CTRL098, app_state);
            render_control(CTRL099, app_state);
            render_control(CTRL100, app_state);
            render_control(CTRL101, app_state);
            render_control(CTRL102, app_state);

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 2.0f));
            if (ImGui::Button("Reset 3D Lighting", ImVec2(-1.0f, 22.0f))) {
                app_state.display.reset_lighting_to_default();
            }
            ImGui::PopStyleVar();
        }

        // ------------------------------------------------------------------
        // SPECIMEN APPEARANCE (CTRL104-CTRL106) — v24.01 Phase 0.2 re-home
        // (was Scene > Layers Visibility CTRL105-107; slid +10 by the Phase 1
        // lighting expansion). The shading trio belongs with the lighting
        // family: DisplayState declares all three in the "3D Shading &
        // Lighting" block, Reset 3D Lighting covers Glass Frosting, and the
        // [LightingAndShading] INI section serializes them. Nesting
        // conditions preserved from v23.29: the toggle shows only when the
        // converted body layer is visible, and the two sliders show only in
        // matte-glass mode.
        // ------------------------------------------------------------------
        if (render_floating_pill_header("SPECIMEN APPEARANCE", true)) {
            if (app_state.display.show_converted_body) {
                render_control(CTRL104, app_state);
                if (app_state.display.specimen_transparent) {
                    render_control(CTRL105, app_state);
                    render_control(CTRL106, app_state);
                }
            }
        }

        render_style_editor_with_modern_preset(app_state);
    }
    ImGui::End();
}

void render_presets_panel(
    AppState& app_state,
    const std::function<void(const rhizome::TriangleMesh&, const std::string&)>& convert_and_update_model,
    const rhizome::TriangleMesh& original_mesh,
    const std::string& loaded_model_name,
    ImVec2 work_pos,
    ImVec2 work_size
) {
    (void)work_size;
    using namespace ctrldefs;
    auto& growth_presets = app_state.growth.growth_presets;
    std::string& status_bar_message = app_state.ui.status_bar_message;

    ImGui::SetNextWindowSize(ImVec2(520.0f, 640.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(work_pos.x + 320.0f, work_pos.y + 60.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Presets", nullptr, ImGuiWindowFlags_None)) {
        // ------------------------------------------------------------------
        // GROWTH PRESETS LIBRARY (CTRL053-CTRL056)
        // ------------------------------------------------------------------
        if (render_floating_pill_header("GROWTH PRESETS LIBRARY", true)) {
            if (app_state.growth.selected_growth_preset_idx < 0) app_state.growth.selected_growth_preset_idx = 0;
            if (app_state.growth.selected_growth_preset_idx >= (int)growth_presets.size()) app_state.growth.selected_growth_preset_idx = (int)growth_presets.size() - 1;

            const char* cur_name = growth_presets.empty() ? "(None)" : growth_presets[app_state.growth.selected_growth_preset_idx].name.c_str();
            if (begin_pill_combo("##PresetCombo", cur_name)) {
                for (int i = 0; i < (int)growth_presets.size(); ++i) {
                    bool is_sel = (app_state.growth.selected_growth_preset_idx == i);
                    if (ImGui::Selectable(growth_presets[i].name.c_str(), is_sel)) {
                        app_state.growth.selected_growth_preset_idx = i;
                    }
                    if (is_sel) ImGui::SetItemDefaultFocus();
                }
                end_pill_combo();
            }

            // Actions: Apply Preset, Update & Delete
            if (!growth_presets.empty()) {
                const auto& p = growth_presets[app_state.growth.selected_growth_preset_idx];

                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 2.0f));
                int btn_cols = (growth_presets.size() > 1) ? 3 : 2;
                float p_btn_w = (ImGui::GetContentRegionAvail().x - (btn_cols - 1) * 4.0f) / static_cast<float>(btn_cols);

                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.098f, 0.827f, 0.635f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.157f, 0.898f, 0.706f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.078f, 0.700f, 0.530f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.027f, 0.067f, 0.110f, 1.0f));
                if (ImGui::Button("Apply##PresetApply", ImVec2(p_btn_w, 22.0f))) {
                    // 39-field transfer via the Phase 2 helper (core/app_state.cpp)
                    const float old_detail_level = app_state.growth.detail_level;
                    apply_growth_preset_to_state(app_state.growth, p);
                    if (std::abs(old_detail_level - p.detail_level) > 0.001f) {
                        convert_and_update_model(original_mesh, loaded_model_name);
                    }
                    status_bar_message = "Loaded preset '" + p.name + "' into Growth sliders";
                }
                ImGui::PopStyleColor(4);

                ImGui::SameLine(0.0f, 4.0f);
                if (ImGui::Button("Update##PresetUpdate", ImVec2(p_btn_w, 22.0f))) {
                    // 39-field capture via the Phase 2 helper (core/app_state.cpp)
                    auto& cur_p = growth_presets[app_state.growth.selected_growth_preset_idx];
                    capture_growth_preset_from_state(app_state.growth, cur_p);
                    save_growth_presets_to_ini(growth_presets, "imgui.ini");
                    status_bar_message = "Updated preset '" + cur_p.name + "' with all active growth parameters";
                }

                if (growth_presets.size() > 1) {
                    ImGui::SameLine(0.0f, 4.0f);
                    if (ImGui::Button("Delete##PresetDelete", ImVec2(p_btn_w, 22.0f))) {
                        std::string deleted_name = p.name;
                        growth_presets.erase(growth_presets.begin() + app_state.growth.selected_growth_preset_idx);
                        if (app_state.growth.selected_growth_preset_idx >= (int)growth_presets.size()) {
                            app_state.growth.selected_growth_preset_idx = (int)growth_presets.size() - 1;
                        }
                        save_growth_presets_to_ini(growth_presets, "imgui.ini");
                        status_bar_message = "Deleted preset '" + deleted_name + "'";
                    }
                }
                ImGui::PopStyleVar();

                ImGui::Separator();

                // Detailed Preset Parameters Summary Table
                ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 1.0f));
                if (ImGui::BeginTable("PresetDetailsTable", 4, ImGuiTableFlags_None)) {
                    // Row 1: Seeds & Spores
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Spores:");
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%d", p.spore_count);
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Primary Arms:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%d", p.primary_roots);

                    // Row 2: Search & Kill Radius
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Search R:");
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f mm", p.search_radius);
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Kill R:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.1f mm", p.kill_radius);

                    // Row 3: Step & Variable Step
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Step Size:");
                    if (p.variable_step) {
                        ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f-%.2f mm", p.min_growth_step, p.max_growth_step);
                    } else {
                        ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f mm", p.growth_step);
                    }
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Branch Rate:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.2f", p.branching_rate);

                    // Row 4: Steering & Max Steps
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Steering Cone:");
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%.0f deg", p.steering_cone);
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Max Steps:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%d", p.max_iterations);

                    // Row 5: Caliber & Tip
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Trunk R:");
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f mm", p.trunk_caliber);
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Tip R:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.2f mm", p.tendril_caliber);

                    // Row 6: Taper & Detail
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Taper:");
                    ImGui::TableSetColumnIndex(1); ImGui::Text(p.variable_taper ? "Variable" : "%.2f", p.taper_curve);
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Voxel Detail:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.2f mm", p.detail_level);

                    // Row 7: Branch Blend & Wall Clearance
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Branch Blend:");
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f mm", p.branch_smooth_k);
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Wall Clear:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.2f x", p.wall_clearance_scale);

                    // Row 8: Lateral Roots
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Lateral Spacing:");
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f mm", p.lateral_density);
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Lateral Reach:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.1f mm", p.lateral_max_length);

                    // Row 9: Lateral Caliber
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Lateral Base/Tip:");
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f / %.2f", p.lateral_trunk_caliber, p.lateral_tip_caliber);
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Lateral Wander:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.2f", p.lateral_wander);

                    // Row 10: Algorithm Physics
                    ImGui::TableNextRow();
                    const char* mode_names[] = { "Colonization", "Volumetric", "Vascular Murray", "Root Bound" };
                    const char* mode_str = (p.algorithm_mode >= 0 && p.algorithm_mode < 4) ? mode_names[p.algorithm_mode] : "Custom";
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Algorithm Mode:");
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%s", mode_str);
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Roughness:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.2f", p.root_roughness);

                    // Row 11: Forces
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Push Force:");
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f", p.push_strength);
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Space-Fill:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.2f", p.space_fill_drive);

                    // Row 12: Straightening & Adherence
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Straightening:");
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f", p.tensile_straightening);
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Adherence:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%.2f", p.surface_adherence);

                    // Row 13: Live Execution
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Live Growth:");
                    ImGui::TableSetColumnIndex(1); ImGui::Text(p.live_growth ? "Enabled" : "Disabled");
                    ImGui::TableSetColumnIndex(2); ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Rate & Cadence:");
                    ImGui::TableSetColumnIndex(3); ImGui::Text("%d fps (%d/fr)", p.live_fps, p.live_steps_per_frame);

                    ImGui::EndTable();
                }
                ImGui::PopStyleVar();
            }
        }

        // ------------------------------------------------------------------
        // RECORD NEW PRESET (CTRL057-CTRL059)
        // ------------------------------------------------------------------
        if (render_floating_pill_header("RECORD NEW PRESET", true)) {
            // CTRL057 buffer is homed in AppUIState::new_preset_name (F8 fix)
            ImGui::SetNextItemWidth(-1.0f);
            render_control(CTRL057, app_state);

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 2.0f));
            float rec_btn_w = (ImGui::GetContentRegionAvail().x - 4.0f) * 0.5f;
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 0.541f, 0.122f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.620f, 0.250f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.900f, 0.450f, 0.080f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
            if (ImGui::Button("Save New Preset", ImVec2(rec_btn_w, 22.0f))) {
                GrowthPreset p;
                p.name = (strlen(app_state.ui.new_preset_name) > 0)
                       ? app_state.ui.new_preset_name
                       : ("Preset " + std::to_string(growth_presets.size() + 1));
                // 39-field capture via the Phase 2 helper (core/app_state.cpp)
                capture_growth_preset_from_state(app_state.growth, p);

                growth_presets.push_back(p);
                app_state.growth.selected_growth_preset_idx = (int)growth_presets.size() - 1;
                save_growth_presets_to_ini(growth_presets, "imgui.ini");
                status_bar_message = "Recorded and saved new preset '" + p.name + "' with all Growth parameters";
            }
            ImGui::PopStyleColor(4);

            ImGui::SameLine(0.0f, 4.0f);
            if (ImGui::Button("Export to Clipboard", ImVec2(rec_btn_w, 22.0f))) {
                std::string out = "[GrowthPresets]\n";
                out += "Count=" + std::to_string(growth_presets.size()) + "\n\n";
                for (size_t i = 0; i < growth_presets.size(); ++i) {
                    const auto& p = growth_presets[i];
                    std::string pfx = "Preset" + std::to_string(i) + "_";
                    out += "; Preset " + std::to_string(i + 1) + ": " + p.name + "\n";
                    out += pfx + "Name=" + p.name + "\n";
                    out += pfx + "DetailLevel=" + std::to_string(p.detail_level) + "\n";
                    out += pfx + "Spores=" + std::to_string(p.spore_count) + "\n";
                    out += pfx + "PrimaryRoots=" + std::to_string(p.primary_roots) + "\n";
                    out += pfx + "SearchRadius=" + std::to_string(p.search_radius) + "\n";
                    out += pfx + "KillRadius=" + std::to_string(p.kill_radius) + "\n";
                    out += pfx + "GrowthStep=" + std::to_string(p.growth_step) + "\n";
                    out += pfx + "VariableStep=" + std::string(p.variable_step ? "1" : "0") + "\n";
                    out += pfx + "MinGrowthStep=" + std::to_string(p.min_growth_step) + "\n";
                    out += pfx + "MaxGrowthStep=" + std::to_string(p.max_growth_step) + "\n";
                    out += pfx + "BranchingRate=" + std::to_string(p.branching_rate) + "\n";
                    out += pfx + "SteeringCone=" + std::to_string(p.steering_cone) + "\n";
                    out += pfx + "MaxIterations=" + std::to_string(p.max_iterations) + "\n";
                    out += pfx + "TrunkCaliber=" + std::to_string(p.trunk_caliber) + "\n";
                    out += pfx + "TendrilCaliber=" + std::to_string(p.tendril_caliber) + "\n";
                    out += pfx + "VariableTaper=" + std::string(p.variable_taper ? "1" : "0") + "\n";
                    out += pfx + "TaperCurve=" + std::to_string(p.taper_curve) + "\n";
                    out += pfx + "BranchSmoothK=" + std::to_string(p.branch_smooth_k) + "\n";
                    out += pfx + "WallClearance=" + std::to_string(p.wall_clearance_scale) + "\n";
                    out += pfx + "WallCutSmooth=" + std::to_string(p.wall_cut_smooth) + "\n";
                    out += pfx + "LateralDensity=" + std::to_string(p.lateral_density) + "\n";
                    out += pfx + "LateralTrunkCaliber=" + std::to_string(p.lateral_trunk_caliber) + "\n";
                    out += pfx + "LateralTipCaliber=" + std::to_string(p.lateral_tip_caliber) + "\n";
                    out += pfx + "LateralMaxLength=" + std::to_string(p.lateral_max_length) + "\n";
                    out += pfx + "LateralWander=" + std::to_string(p.lateral_wander) + "\n";
                    out += pfx + "LateralBranchAngle=" + std::to_string(p.lateral_branch_angle) + "\n";
                    out += pfx + "SeedOffset=" + std::to_string(p.seed_offset) + "\n";
                    out += pfx + "AlgorithmMode=" + std::to_string(p.algorithm_mode) + "\n";
                    out += pfx + "RootRoughness=" + std::to_string(p.root_roughness) + "\n";
                    out += pfx + "PushStrength=" + std::to_string(p.push_strength) + "\n";
                    out += pfx + "SpaceFillDrive=" + std::to_string(p.space_fill_drive) + "\n";
                    out += pfx + "BackwardPull=" + std::to_string(p.backward_pull) + "\n";
                    out += pfx + "TensileStraightening=" + std::to_string(p.tensile_straightening) + "\n";
                    out += pfx + "MurrayExponent=" + std::to_string(p.murray_exponent) + "\n";
                    out += pfx + "MurrayBranchAngle=" + std::to_string(p.murray_branch_angle) + "\n";
                    out += pfx + "SurfaceAdherence=" + std::to_string(p.surface_adherence) + "\n";
                    out += pfx + "LiveGrowth=" + std::string(p.live_growth ? "1" : "0") + "\n";
                    out += pfx + "LiveStepsPerFrame=" + std::to_string(p.live_steps_per_frame) + "\n";
                    out += pfx + "LiveFps=" + std::to_string(p.live_fps) + "\n\n";
                }
                ImGui::SetClipboardText(out.c_str());
                status_bar_message = "All growth preset parameters copied to clipboard";
            }
            ImGui::PopStyleVar();
        }
    }
    ImGui::End();
}

} // namespace rhizome
