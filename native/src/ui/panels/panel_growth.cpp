// =========================================================================
// RHIZOME GROWTH PANEL — implementation (v23.29 Phase 4)
// =========================================================================
// Migrated from src/app/ui_panels.cpp :: render_growth_panel (lines
// 1062-1719) plus the Win32 file dialogs it owns. Rendering behavior is
// preserved verbatim: identical pill buttons, section order, tooltips,
// side effects and status messages. What changed (per the approved Phase 1
// proposal):
//   * value controls draw through the data-driven ControlDefinition
//     registry (render_control) — same primitive, same label, same CSV
//     range/format, plus uniform CSV-driven ACTION logging;
//   * the 28-member context is now AppState& + EngineResources& +
//     command callbacks; all shadow locals are gone.
// =========================================================================

#include "ui/panels/panel_growth.hpp"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <string>

#include "ui/widgets/widget_factory.hpp"
#include "ui/widgets/control_registry.hpp"
#include "core/unified_logger.hpp"
#include "core/algorithm_io.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/cuda_types.hpp"   // get_seed_color_vec4
#include "engine/io/parsers.hpp"        // ObjParser / StlParser (Import)

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

namespace rhizome {

#ifdef _WIN32
static std::string open_file_dialog() {
    char filename[MAX_PATH] = {0};
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "3D Models (*.obj;*.stl)\0*.obj;*.stl\0Wavefront OBJ (*.obj)\0*.obj\0Stereolithography STL (*.stl)\0*.stl\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn)) {
        return std::string(filename);
    }
    return "";
}

static std::string open_algorithm_file_dialog() {
    char filename[MAX_PATH] = {0};
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Growth Algorithm (*.json;*.cu)\0*.json;*.cu\0JSON Algorithm (*.json)\0*.json\0CUDA Kernel Source (*.cu)\0*.cu\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn)) {
        return std::string(filename);
    }
    return "";
}

static std::string save_algorithm_file_dialog(const std::string& default_stem = "custom_algorithm") {
    char filename[MAX_PATH] = {0};
    strncpy_s(filename, default_stem.c_str(), _TRUNCATE);
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "JSON Algorithm (*.json)\0*.json\0CUDA Kernel Source (*.cu)\0*.cu\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = "json";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;
    if (GetSaveFileNameA(&ofn)) {
        return std::string(filename);
    }
    return "";
}
#endif

void render_growth_panel(GrowthPanelContext& ctx) {
    using namespace ctrldefs;

    AppState& app_state = ctx.app_state;
    EngineResources& res = ctx.res;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 2.0f));
    ImGui::Begin("Growth", nullptr, ImGuiWindowFlags_None);

    const float cmd_avail_w = ImGui::GetContentRegionAvail().x;
    const float cmd_spacing = 4.0f;
    const float cmd_btn_w = std::max(44.0f, (cmd_avail_w - 3.0f * cmd_spacing) / 4.0f);
    const ImVec2 cmd_btn_size(cmd_btn_w, 22.0f);

    // Import Button (#19D3A2 Primary Teal Pill) — CTRL001
    if (render_pill_button("Import", cmd_btn_size, PillButtonAccent::TealPrimary)) {
        RhizomeLog("ACTION", "Import 3D Model button clicked");
#ifdef _WIN32
        std::string path = open_file_dialog();
        if (!path.empty()) {
            std::filesystem::path fp(path);
            std::string ext = fp.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            rhizome::TriangleMesh mesh;
            std::string err;
            bool ok = false;
            if (ext == ".obj") ok = rhizome::ObjParser::parse(path, mesh, err);
            else if (ext == ".stl") ok = rhizome::StlParser::parse(path, mesh, err);

            if (ok) {
                ctx.convert_and_update_model(mesh, fp.filename().string());
            } else {
                app_state.ui.status_bar_message = "Failed to load " + fp.filename().string();
            }
        }
#endif
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Import OBJ or STL 3D specimen mesh.");
    }

    ImGui::SameLine(0.0f, cmd_spacing);

    // Start Button (#FF8A1F Orange Growth Action Pill) — CTRL002
    if (render_pill_button("Start", cmd_btn_size, PillButtonAccent::OrangeAction)) {
        RhizomeLog("ACTION", "Start Growth button clicked");
        ctx.run_growth_engine();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Execute space-colonization root growth algorithm.");
    }

    ImGui::SameLine(0.0f, cmd_spacing);

    // Stop Button — CTRL003
    if (render_pill_button("Stop", cmd_btn_size)) {
        RhizomeLog("ACTION", "Stop Growth button clicked");
        ctx.stop_live_growth();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Halt or pause ongoing live root growth.");
    }

    ImGui::SameLine(0.0f, cmd_spacing);

    // Clear Button — CTRL004
    if (render_pill_button("Clear", cmd_btn_size)) {
        RhizomeLog("ACTION", "Clear Scene button clicked");
        app_state.growth.is_growing_live = false;
        app_state.growth.is_growth_paused = false;
        app_state.growth.has_live_session = false;
        res.live_colonizer.clear();
        res.current_roots_grid = nullptr;
        res.roots_mesh.vertices.clear();
        res.roots_mesh.indices.clear();
        res.roots_gl_buffers.cleanup();
        res.branch_gl_buffers.cleanup();
        res.spore_gl_buffers.cleanup();
        res.current_branch_nodes.clear();
        res.cached_roots_metrics = CachedMeshMetrics();
        app_state.mesh_export.has_generated_roots_mesh = false;
        app_state.display.show_roots = false;
        app_state.display.colorize_roots = false;
        app_state.display.colorize_strands = false;
        for (auto& obj : res.test_objects) {
            obj.buffers.cleanup();
        }
        res.test_objects.clear();
        res.last_test_results.clear();
        if (app_state.display.show_spores && res.current_sdf_grid) {
            res.live_colonizer.sample_spores_only(res.current_sdf_grid, ctx.build_colonization_config());
            res.spore_gl_buffers.upload(res.live_colonizer.get_spores());
        }
        app_state.ui.status_bar_message = "Scene cleared";
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Clear generated roots, reset growth simulation, and clear test objects.");
    }

    ImGui::Spacing();

    // ------------------------------------------------------------------
    // ALGORITHM (CTRL005-CTRL016)
    // ------------------------------------------------------------------
    if (render_floating_pill_header("ALGORITHM", true)) {
        auto& algos = app_state.growth.growth_algorithms;
        if (app_state.growth.selected_algorithm_idx < 0 || app_state.growth.selected_algorithm_idx >= static_cast<int>(algos.size())) {
            app_state.growth.selected_algorithm_idx = 0;
        }
        const char* cur_algo_label = algos.empty() ? "None" : algos[app_state.growth.selected_algorithm_idx].name.c_str();
        if (begin_pill_combo("##GrowthAlgoCombo", cur_algo_label)) {
            for (int i = 0; i < static_cast<int>(algos.size()); ++i) {
                bool is_sel = (app_state.growth.selected_algorithm_idx == i);
                if (ImGui::Selectable(algos[i].name.c_str(), is_sel)) {
                    // Save fine-tuned state of previous algorithm before switching
                    ctx.capture_current_into_algorithm_profile(algos[app_state.growth.selected_algorithm_idx]);
                    ctx.apply_algorithm_profile(i);
                    save_growth_algorithms_to_ini(algos, app_state.growth.selected_algorithm_idx, "imgui.ini");
                    app_state.ui.status_bar_message = "Switched to algorithm '" + algos[i].name + "'";
                }
                if (is_sel) ImGui::SetItemDefaultFocus();
            }
            end_pill_combo();
        }

        // Export, Import & Delete Algorithm Buttons (CTRL006-CTRL008)
        float algo_btn_w = (ImGui::GetContentRegionAvail().x - 8.0f) / 3.0f;
        if (ImGui::Button("Export##Algo", ImVec2(algo_btn_w, 22.0f))) {
#ifdef _WIN32
            if (!algos.empty()) {
                ctx.capture_current_into_algorithm_profile(algos[app_state.growth.selected_algorithm_idx]);
                save_growth_algorithms_to_ini(algos, app_state.growth.selected_algorithm_idx, "imgui.ini");
                std::string default_fn = algos[app_state.growth.selected_algorithm_idx].name;
                for (char& ch : default_fn) {
                    if (ch == ' ' || ch == '(' || ch == ')' || ch == '/' || ch == '\\') ch = '_';
                }
                std::string save_path = save_algorithm_file_dialog(default_fn);
                if (!save_path.empty()) {
                    if (export_algorithm_to_file(algos[app_state.growth.selected_algorithm_idx], save_path)) {
                        app_state.ui.status_bar_message = "Exported algorithm '" + algos[app_state.growth.selected_algorithm_idx].name + "' to " + save_path;
                    } else {
                        app_state.ui.status_bar_message = "Failed to export algorithm file";
                    }
                }
            }
#endif
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Export algorithm in your choice of JSON (.json) or CUDA Kernel Source (.cu) format");

        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::Button("Import##Algo", ImVec2(algo_btn_w, 22.0f))) {
#ifdef _WIN32
            std::string open_path = open_algorithm_file_dialog();
            if (!open_path.empty()) {
                GrowthAlgorithmProfile imported;
                if (import_algorithm_from_file(open_path, imported)) {
                    if (!algos.empty()) {
                        ctx.capture_current_into_algorithm_profile(algos[app_state.growth.selected_algorithm_idx]);
                    }
                    algos.push_back(imported);
                    ctx.apply_algorithm_profile(static_cast<int>(algos.size()) - 1);
                    save_growth_algorithms_to_ini(algos, app_state.growth.selected_algorithm_idx, "imgui.ini");
                    app_state.ui.status_bar_message = "Imported new algorithm '" + imported.name + "'";
                } else {
                    app_state.ui.status_bar_message = "Failed to import algorithm file";
                }
            }
#endif
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Import an algorithm file and add it as a new algorithm in the list");

        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::Button("Delete##Algo", ImVec2(algo_btn_w, 22.0f))) {
            if (algos.size() > 1 && app_state.growth.selected_algorithm_idx >= 0 && app_state.growth.selected_algorithm_idx < static_cast<int>(algos.size())) {
                std::string deleted_name = algos[app_state.growth.selected_algorithm_idx].name;
                algos.erase(algos.begin() + app_state.growth.selected_algorithm_idx);
                if (app_state.growth.selected_algorithm_idx >= static_cast<int>(algos.size())) {
                    app_state.growth.selected_algorithm_idx = static_cast<int>(algos.size()) - 1;
                }
                ctx.apply_algorithm_profile(app_state.growth.selected_algorithm_idx);
                save_growth_algorithms_to_ini(algos, app_state.growth.selected_algorithm_idx, "imgui.ini");
                app_state.ui.status_bar_message = "Deleted algorithm '" + deleted_name + "'";
            } else {
                app_state.ui.status_bar_message = "Cannot delete the last remaining algorithm";
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Delete the currently selected algorithm from the list");

        // Mode-conditional physical drivers (CTRL009-CTRL016)
        if (app_state.growth.algorithm_mode == 1) {
            render_control(CTRL009, app_state);
            render_control(CTRL010, app_state);
            render_control(CTRL011, app_state);
            render_control(CTRL012, app_state);
            render_control(CTRL013, app_state);
            render_control(CTRL014, app_state);
        } else if (app_state.growth.algorithm_mode == 2) {
            render_control(CTRL015, app_state);
            render_control(CTRL016, app_state);
            render_control(CTRL010, app_state);
            render_control(CTRL011, app_state);
            render_control(CTRL013, app_state);
            render_control(CTRL014, app_state);
        } else if (app_state.growth.algorithm_mode == 3) {
            render_control(CTRL014, app_state);
            render_control(CTRL009, app_state);
            render_control(CTRL010, app_state);
            render_control(CTRL011, app_state);
            render_control(CTRL013, app_state);
        } else if (app_state.growth.algorithm_mode == 4) {
            // v24.12 Track F R1: developmental user surface (CTRL242-251).
            // GSA = gravitropic setpoint angle from the downward vertical,
            // per root order (framework vs laterals); the Murray census at
            // bake derives structural caliber from the subtree. Min Print
            // Radius is the SLA floor — nothing thinner reaches the mesh.
            render_control(CTRL242, app_state);
            render_control(CTRL243, app_state);
            render_control(CTRL244, app_state);
            render_control(CTRL245, app_state);
            render_control(CTRL246, app_state);
            render_control(CTRL247, app_state);
            render_control(CTRL248, app_state);
            render_control(CTRL249, app_state);
            render_control(CTRL250, app_state);
            render_control(CTRL251, app_state);
        }

        if (!algos.empty()) {
            ctx.capture_current_into_algorithm_profile(algos[app_state.growth.selected_algorithm_idx]);
        }
    }

    // ------------------------------------------------------------------
    // COLONIZATION SEEDS (CTRL017-CTRL020)
    // ------------------------------------------------------------------
    if (render_floating_pill_header("COLONIZATION SEEDS", true)) {
        if (render_control(CTRL017, app_state)) {
            if (app_state.display.show_spores && !app_state.growth.is_growing_live && res.current_sdf_grid) {
                res.live_colonizer.sample_spores_only(res.current_sdf_grid, ctx.build_colonization_config());
                res.spore_gl_buffers.upload(res.live_colonizer.get_spores());
            }
        }
        render_control(CTRL018, app_state);
        render_control(CTRL019, app_state);
        render_control(CTRL020, app_state);
    }

    // ------------------------------------------------------------------
    // BRANCHING & GROWTH (CTRL021-CTRL027)
    // ------------------------------------------------------------------
    if (render_floating_pill_header("BRANCHING & GROWTH", true)) {
        float safe_step_floor = std::max(0.25f, app_state.growth.trunk_caliber * 1.15f);
        float step_slider_max = std::max(safe_step_floor + 2.50f, 6.00f);

        render_control(CTRL021, app_state);
        if (app_state.growth.variable_step) {
            if (app_state.growth.min_growth_step < safe_step_floor) {
                app_state.growth.min_growth_step = safe_step_floor;
            }
            float min_slider_max = std::max(safe_step_floor + 1.50f, 4.00f);
            if (render_control(CTRL022, app_state, &safe_step_floor, &min_slider_max)) {
                if (app_state.growth.min_growth_step > app_state.growth.max_growth_step) {
                    app_state.growth.max_growth_step = app_state.growth.min_growth_step;
                }
            }
            if (app_state.growth.max_growth_step < app_state.growth.min_growth_step) {
                app_state.growth.max_growth_step = app_state.growth.min_growth_step;
            }
            if (render_control(CTRL023, app_state, &app_state.growth.min_growth_step, &step_slider_max)) {
                if (app_state.growth.max_growth_step < app_state.growth.min_growth_step) {
                    app_state.growth.min_growth_step = app_state.growth.max_growth_step;
                }
            }
        } else {
            if (app_state.growth.growth_step < safe_step_floor) {
                app_state.growth.growth_step = safe_step_floor;
            }
            render_control(CTRL024, app_state, &safe_step_floor, &step_slider_max);
        }
        render_control(CTRL025, app_state);
        render_control(CTRL026, app_state);
        render_control(CTRL027, app_state);
    }

    // ------------------------------------------------------------------
    // VESSEL & CALIBER (CTRL028-CTRL034)
    // ------------------------------------------------------------------
    if (render_floating_pill_header("VESSEL & CALIBER", true)) {
        auto sync_caliber_live = [&]() {
            if (app_state.growth.has_live_session && res.current_sdf_grid) {
                ColonizationConfig c_cfg = ctx.build_colonization_config();
                res.live_colonizer.update_caliber(
                    app_state.growth.trunk_caliber,
                    app_state.growth.tendril_caliber,
                    app_state.growth.taper_curve,
                    app_state.growth.variable_taper,
                    c_cfg.seed,
                    &c_cfg
                );
                ctx.update_implicit_session(!app_state.growth.is_growing_live);
            }
        };

        if (render_control(CTRL028, app_state)) {
            float new_safe_floor = std::max(0.25f, app_state.growth.trunk_caliber * 1.15f);
            if (app_state.growth.min_growth_step < new_safe_floor) {
                app_state.growth.min_growth_step = new_safe_floor;
            }
            if (app_state.growth.max_growth_step < app_state.growth.min_growth_step) {
                app_state.growth.max_growth_step = std::max(app_state.growth.min_growth_step * 1.5f, app_state.growth.min_growth_step + 0.50f);
            }
            if (app_state.growth.growth_step < new_safe_floor) {
                app_state.growth.growth_step = new_safe_floor;
            }
            sync_caliber_live();
        }
        if (render_control(CTRL029, app_state)) {
            sync_caliber_live();
        }
        if (render_control(CTRL030, app_state)) {
            sync_caliber_live();
        }
        if (!app_state.growth.variable_taper) {
            if (render_control(CTRL031, app_state)) {
                sync_caliber_live();
            }
        }
        if (render_control(CTRL032, app_state)) {
            if (!app_state.growth.is_growing_live && app_state.growth.has_live_session && res.current_sdf_grid) {
                ctx.update_implicit_session(true);
            }
        }
        render_control(CTRL033, app_state);
        if (render_control(CTRL034, app_state)) {
            if (!app_state.growth.is_growing_live && app_state.growth.has_live_session && res.current_sdf_grid) {
                ctx.update_implicit_session(true);
            }
        }
    }

    // ------------------------------------------------------------------
    // LATERAL ROOTS (CTRL035-CTRL040)
    // ------------------------------------------------------------------
    if (render_floating_pill_header("LATERAL ROOTS", true)) {
        auto sync_lateral_live = [&]() {
            if (app_state.growth.has_live_session && res.current_sdf_grid) {
                auto c_cfg = ctx.build_colonization_config();
                res.live_colonizer.sync_lateral_roots_session(c_cfg);
                ctx.update_implicit_session(true);
            }
        };

        if (render_control(CTRL035, app_state)) {
            sync_lateral_live();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Spacing between lateral roots distributed equally along the seminal root branches");
        }

        if (render_control(CTRL036, app_state)) {
            sync_lateral_live();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Thin micro-diameter at the base of lateral roots");
        }

        if (render_control(CTRL037, app_state)) {
            sync_lateral_live();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Delicate, fragile diameter at lateral root tips");
        }

        if (render_control(CTRL038, app_state)) {
            sync_lateral_live();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Maximum length of lateral rootlets");
        }

        if (render_control(CTRL039, app_state)) {
            sync_lateral_live();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Organic curling and meandering along the surface");
        }

        if (render_control(CTRL040, app_state)) {
            sync_lateral_live();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Outward growth angle when Snap to Root is disabled");
        }
    }

    // ------------------------------------------------------------------
    // SEEDING (CTRL041-CTRL049)
    // ------------------------------------------------------------------
    if (render_floating_pill_header("SEEDING", true)) {
        if (render_control(CTRL041, app_state)) {
            if (app_state.seed.selected_seed_index >= 0 && app_state.seed.selected_seed_index < static_cast<int>(app_state.seed.points.size())) {
                if (app_state.seed.selected_seed_index < static_cast<int>(app_state.seed.snapped_to_root.size()) &&
                    app_state.seed.snapped_to_root[app_state.seed.selected_seed_index]) {
                    glm::vec3 cur_pos = app_state.seed.points[app_state.seed.selected_seed_index];
                    glm::vec3 snapped_pos(0.0f), root_normal(0.0f, 0.0f, 1.0f);
                    if (res.live_colonizer.snap_seed_to_root_surface(cur_pos, app_state.growth.seed_offset, snapped_pos, root_normal)) {
                        app_state.seed.points[app_state.seed.selected_seed_index] = snapped_pos;
                        if (app_state.growth.has_live_session && res.current_sdf_grid) {
                            auto c_cfg = ctx.build_colonization_config();
                            res.live_colonizer.sync_seeds_session(res.current_sdf_grid, app_state.seed.points, c_cfg);
                            ctx.update_implicit_session(false);
                        }
                    }
                }
            }
        }

        if (app_state.seed.frozen.size() < app_state.seed.points.size()) {
            app_state.seed.frozen.resize(app_state.seed.points.size(), 0);
        }
        if (app_state.seed.snapped_to_root.size() < app_state.seed.points.size()) {
            app_state.seed.snapped_to_root.resize(app_state.seed.points.size(), 0);
        }

        if (ImGui::Button("+##AddSeed", ImVec2(28.0f, 22.0f))) {
            glm::vec3 new_pos(0.0f);
            glm::vec3 min_p(0.0f), max_p(0.0f);
            float bbox_ms = 0.0f;
            if (res.original_mesh.vertex_count() > 0) {
                rhizome::CudaRasterizer::compute_bounding_box_gpu(res.original_mesh, min_p, max_p, bbox_ms);
            }
            glm::vec3 center = (min_p + max_p) * 0.5f;

            if (app_state.seed.points.empty()) {
                new_pos = glm::vec3(center.x, center.y, min_p.z + 2.0f * app_state.growth.detail_level);
            } else {
                float angle = static_cast<float>(app_state.seed.points.size()) * 2.39996323f; // Golden ratio spiral distribution
                float r_x = std::max(12.0f, (max_p.x - min_p.x) * 0.38f);
                float r_y = std::max(12.0f, (max_p.y - min_p.y) * 0.38f);
                float z_ratio = 0.20f + 0.55f * static_cast<float>(app_state.seed.points.size() % 5) * 0.20f;
                float z_pos = min_p.z + (max_p.z - min_p.z) * z_ratio;
                new_pos = glm::vec3(center.x + cosf(angle) * r_x, center.y + sinf(angle) * r_y, z_pos);

                if (res.current_sdf_grid && res.current_sdf_grid->is_valid()) {
                    float desired_margin = (app_state.growth.trunk_caliber + app_state.growth.seed_offset);
                    glm::vec3 r_orig = center;
                    glm::vec3 diff = new_pos - center;
                    float dl = glm::length(diff);
                    bool hit_ok = false;
                    if (dl > 1e-4f) {
                        glm::vec3 r_dir = diff / dl;
                        glm::vec3 hit_pt(0.0f), hit_norm(0.0f, 0.0f, 1.0f);
                        float hit_t = -1.0f, gpu_ms = 0.0f;
                        if (rhizome::CudaRasterizer::raycast_sdf_volume_gpu(*res.current_sdf_grid, r_orig, r_dir, hit_pt, hit_norm, hit_t, gpu_ms)) {
                            new_pos = hit_pt - hit_norm * desired_margin;
                            hit_ok = true;
                        }
                    }
                    if (!hit_ok || res.current_sdf_grid->sample_trilinear(new_pos) > -0.05f) {
                        glm::vec3 default_seed = new_pos;
                        float seed_ms = 0.0f;
                        if (rhizome::CudaRasterizer::find_default_seed_gpu(*res.current_sdf_grid, min_p, max_p, -desired_margin, default_seed, seed_ms)) {
                            new_pos = default_seed;
                        }
                    }
                }
            }
            app_state.seed.points.push_back(new_pos);
            app_state.seed.frozen.push_back(0);
            app_state.seed.snapped_to_root.push_back(0);
            app_state.seed.has_lateral.push_back(0);
            app_state.seed.radii.push_back(app_state.growth.trunk_caliber);
            app_state.seed.selected_seed_index = static_cast<int>(app_state.seed.points.size()) - 1;

            if (app_state.growth.has_live_session && res.current_sdf_grid) {
                auto c_cfg = ctx.build_colonization_config();
                res.live_colonizer.sync_seeds_session(res.current_sdf_grid, app_state.seed.points, c_cfg);
                res.live_colonizer.prepare_resume_session(res.current_sdf_grid, c_cfg);
                app_state.growth.is_growth_paused = true;
                app_state.growth.is_growing_live = false;
                ctx.update_implicit_session(false);
            }
            app_state.ui.status_bar_message = "Added Seed S" + std::to_string(app_state.seed.points.size()) + " (Paused - position or snap seed, then click Start to grow)";
        }
        auto delete_active_seed = [&]() {
            if (!app_state.seed.points.empty() && app_state.seed.selected_seed_index >= 0 && app_state.seed.selected_seed_index < static_cast<int>(app_state.seed.points.size())) {
                int removed_idx = app_state.seed.selected_seed_index;
                app_state.seed.points.erase(app_state.seed.points.begin() + removed_idx);
                if (removed_idx < static_cast<int>(app_state.seed.frozen.size())) {
                    app_state.seed.frozen.erase(app_state.seed.frozen.begin() + removed_idx);
                }
                if (removed_idx < static_cast<int>(app_state.seed.snapped_to_root.size())) {
                    app_state.seed.snapped_to_root.erase(app_state.seed.snapped_to_root.begin() + removed_idx);
                }
                if (removed_idx < static_cast<int>(app_state.seed.has_lateral.size())) {
                    app_state.seed.has_lateral.erase(app_state.seed.has_lateral.begin() + removed_idx);
                }
                if (removed_idx < static_cast<int>(app_state.seed.radii.size())) {
                    app_state.seed.radii.erase(app_state.seed.radii.begin() + removed_idx);
                }

                // 100% CUDA GPU Prune root branches grown from deleted seed
                if (app_state.growth.has_live_session) {
                    res.live_colonizer.delete_seed(removed_idx);
                    res.current_branch_nodes = res.live_colonizer.get_nodes();
                    res.branch_gl_buffers.upload(res.current_branch_nodes, app_state.display.colorize_roots || app_state.display.colorize_strands);
                }

                if (app_state.seed.points.empty()) {
                    glm::vec3 min_p(0.0f), max_p(0.0f);
                    float bbox_ms = 0.0f;
                    if (res.original_mesh.vertex_count() > 0) {
                        rhizome::CudaRasterizer::compute_bounding_box_gpu(res.original_mesh, min_p, max_p, bbox_ms);
                    }
                    glm::vec3 center = (min_p + max_p) * 0.5f;
                    glm::vec3 default_seed(center.x, center.y, min_p.z + 2.0f * app_state.growth.detail_level);
                    if (res.current_sdf_grid && res.current_sdf_grid->is_valid()) {
                        float desired_margin = -(app_state.growth.trunk_caliber + app_state.growth.seed_offset);
                        float seed_ms = 0.0f;
                        rhizome::CudaRasterizer::find_default_seed_gpu(*res.current_sdf_grid, min_p, max_p, desired_margin, default_seed, seed_ms);
                    }
                    app_state.seed.points.push_back(default_seed);
                    app_state.seed.frozen.push_back(0);
                    app_state.seed.snapped_to_root.push_back(0);
                    app_state.seed.has_lateral.push_back(0);
                    app_state.seed.radii.push_back(app_state.growth.trunk_caliber);
                    app_state.seed.selected_seed_index = 0;
                    if (app_state.growth.has_live_session) {
                        res.live_colonizer.clear();
                        app_state.growth.has_live_session = false;
                        ctx.update_implicit_session(false);
                    }
                } else {
                    if (app_state.seed.selected_seed_index >= static_cast<int>(app_state.seed.points.size())) {
                        app_state.seed.selected_seed_index = static_cast<int>(app_state.seed.points.size()) - 1;
                    }
                    if (app_state.growth.has_live_session && res.current_sdf_grid) {
                        auto c_cfg = ctx.build_colonization_config();
                        res.live_colonizer.sync_seeds_session(res.current_sdf_grid, app_state.seed.points, c_cfg);
                        ctx.update_implicit_session(!app_state.growth.is_growing_live);
                    }
                }
                app_state.ui.status_bar_message = "Removed selected seed and pruned its grown roots";
            }
        };

        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::Button("-##RemoveSeed", ImVec2(28.0f, 22.0f))) {
            delete_active_seed();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Delete selected seed and remove all roots grown from it (or press Delete)");
        }
        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && ImGui::IsKeyPressed(ImGuiKey_Delete)) {
            delete_active_seed();
        }

        ImGui::SameLine(0.0f, 8.0f);
        float all_btn_w = (ImGui::GetContentRegionAvail().x - 4.0f) * 0.5f;
        if (ImGui::Button("Freeze All##Seeds", ImVec2(all_btn_w, 22.0f))) {
            if (app_state.seed.radii.size() < app_state.seed.points.size()) {
                app_state.seed.radii.resize(app_state.seed.points.size(), app_state.growth.trunk_caliber);
            }
            for (size_t si = 0; si < app_state.seed.points.size(); ++si) {
                app_state.seed.frozen[si] = 1;
                app_state.seed.radii[si] = app_state.growth.trunk_caliber;
            }
            if (app_state.growth.has_live_session && res.current_sdf_grid) {
                res.live_colonizer.sync_seeds_session(res.current_sdf_grid, app_state.seed.points, ctx.build_colonization_config());
            }
            app_state.ui.status_bar_message = "All existing seeds frozen (locked diameter)";
        }
        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::Button("Unfreeze All##Seeds", ImVec2(all_btn_w, 22.0f))) {
            for (auto& f : app_state.seed.frozen) f = 0;
            if (app_state.growth.has_live_session && res.current_sdf_grid) {
                res.live_colonizer.sync_seeds_session(res.current_sdf_grid, app_state.seed.points, ctx.build_colonization_config());
            }
            app_state.ui.status_bar_message = "All seeds unfrozen";
        }

        if (!app_state.seed.points.empty()) {
            ImGui::Spacing();
            const bool colorize_by_seed = (app_state.display.colorize_roots || app_state.display.colorize_strands);
            for (size_t si = 0; si < app_state.seed.points.size(); ++si) {
                if (si > 0 && (si % 7 != 0)) ImGui::SameLine(0.0f, 4.0f);
                std::string label = "S" + std::to_string(si + 1);
                bool is_sel = (static_cast<int>(si) == app_state.seed.selected_seed_index);
                glm::vec4 c = get_seed_color_vec4(static_cast<int>(si));

                if (is_sel) {
                    ImVec4 btn_col = colorize_by_seed ? ImVec4(c.r, c.g, c.b, 1.0f) : ImVec4(0.098f, 0.827f, 0.635f, 1.0f);
                    ImGui::PushStyleColor(ImGuiCol_Button, btn_col);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(std::min(1.0f, btn_col.x + 0.15f), std::min(1.0f, btn_col.y + 0.15f), std::min(1.0f, btn_col.z + 0.15f), 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(std::max(0.0f, btn_col.x - 0.15f), std::max(0.0f, btn_col.y - 0.15f), std::max(0.0f, btn_col.z - 0.15f), 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.027f, 0.067f, 0.110f, 1.0f));
                } else if (colorize_by_seed) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.10f, 0.15f, 0.22f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.16f, 0.22f, 0.32f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.08f, 0.12f, 0.18f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(c.r, c.g, c.b, 1.0f));
                }
                if (ImGui::Button(label.c_str(), ImVec2(32.0f, 22.0f))) {
                    app_state.seed.selected_seed_index = static_cast<int>(si);
                }
                if (is_sel || colorize_by_seed) {
                    ImGui::PopStyleColor(4);
                }
            }

            if (app_state.seed.selected_seed_index >= 0 && app_state.seed.selected_seed_index < static_cast<int>(app_state.seed.points.size())) {
                int sel_seed_idx = app_state.seed.selected_seed_index;
                bool is_frozen = (sel_seed_idx < static_cast<int>(app_state.seed.frozen.size()) && app_state.seed.frozen[sel_seed_idx] != 0);
                if (render_toggle_switch(CTRL047.draw_label, &is_frozen)) {
                    log_control_interaction(CTRL047, is_frozen ? 0.0f : 1.0f, is_frozen ? 1.0f : 0.0f);
                    if (app_state.seed.frozen.size() <= static_cast<size_t>(sel_seed_idx)) {
                        app_state.seed.frozen.resize(app_state.seed.points.size(), 0);
                    }
                    app_state.seed.frozen[sel_seed_idx] = is_frozen ? 1 : 0;
                    if (app_state.seed.radii.size() < app_state.seed.points.size()) {
                        app_state.seed.radii.resize(app_state.seed.points.size(), app_state.growth.trunk_caliber);
                    }
                    if (is_frozen) {
                        app_state.seed.radii[sel_seed_idx] = app_state.growth.trunk_caliber;
                    }
                    if (app_state.growth.has_live_session && res.current_sdf_grid) {
                        res.live_colonizer.sync_seeds_session(res.current_sdf_grid, app_state.seed.points, ctx.build_colonization_config());
                    }
                    app_state.ui.status_bar_message = std::string(is_frozen ? "Frozen " : "Unfrozen ") + "Seed S" + std::to_string(sel_seed_idx + 1) + (is_frozen ? " (radius locked)" : "");
                }

                if (app_state.seed.snapped_to_root.size() < app_state.seed.points.size()) {
                    app_state.seed.snapped_to_root.resize(app_state.seed.points.size(), 0);
                }
                if (app_state.seed.has_lateral.size() < app_state.seed.points.size()) {
                    app_state.seed.has_lateral.resize(app_state.seed.points.size(), 0);
                }

                bool is_snapped = (app_state.seed.snapped_to_root[sel_seed_idx] != 0);
                if (render_toggle_switch(CTRL048.draw_label, &is_snapped)) {
                    log_control_interaction(CTRL048, is_snapped ? 0.0f : 1.0f, is_snapped ? 1.0f : 0.0f);
                    app_state.seed.snapped_to_root[sel_seed_idx] = is_snapped ? 1 : 0;
                    if (app_state.growth.has_live_session && res.current_sdf_grid) {
                        auto c_cfg = ctx.build_colonization_config();
                        res.live_colonizer.sync_lateral_roots_session(c_cfg);
                        ctx.update_implicit_session(true);
                    }
                    app_state.ui.status_bar_message = is_snapped
                        ? ("Seed S" + std::to_string(sel_seed_idx + 1) + ": Lateral roots commanded to follow & hug seminal surface")
                        : ("Seed S" + std::to_string(sel_seed_idx + 1) + ": Lateral roots commanded to sprout outward away from surface");
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("When enabled, lateral roots follow and cling flush to the seminal root surface. When disabled, they branch outward into space.");
                }

                bool has_lat = (app_state.seed.has_lateral[sel_seed_idx] != 0);
                if (render_toggle_switch(CTRL049.draw_label, &has_lat)) {
                    log_control_interaction(CTRL049, has_lat ? 0.0f : 1.0f, has_lat ? 1.0f : 0.0f);
                    app_state.seed.has_lateral[sel_seed_idx] = has_lat ? 1 : 0;
                    if (has_lat && sel_seed_idx < static_cast<int>(app_state.seed.frozen.size())) {
                        app_state.seed.frozen[sel_seed_idx] = 1;
                    }
                    if (app_state.growth.has_live_session && res.current_sdf_grid) {
                        auto c_cfg = ctx.build_colonization_config();
                        res.live_colonizer.sync_lateral_roots_session(c_cfg);
                        ctx.update_implicit_session(true);
                    }
                    app_state.ui.status_bar_message = has_lat
                        ? ("Enabled lateral roots for Seed S" + std::to_string(sel_seed_idx + 1) + " (seminal roots locked, lateral roots sprouted)")
                        : ("Disabled lateral roots for Seed S" + std::to_string(sel_seed_idx + 1));
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Sprout thin, fragile secondary lateral roots distributed equally along this seed's seminal branches.");
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // EXECUTION (CTRL050-CTRL052)
    // ------------------------------------------------------------------
    if (render_floating_pill_header("EXECUTION", true)) {
        render_control(CTRL050, app_state);
        if (app_state.growth.live_growth) {
            render_control(CTRL051, app_state);
            render_control(CTRL052, app_state);
        }
    }

    ImGui::End();
    ImGui::PopStyleVar(2);
}

} // namespace rhizome
