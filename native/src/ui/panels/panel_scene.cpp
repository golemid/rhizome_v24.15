// =========================================================================
// RHIZOME SCENE PANEL — implementation (v23.29 Phase 4)
// =========================================================================
// Migrated from src/app/ui_scene_panel.cpp :: render_scene_panel +
// render_status_bar_panel (lines 1-129, 205-800). Rendering behavior
// preserved verbatim; what changed (per the approved Phase 1 proposal):
//   * value controls draw through the ControlDefinition registry;
//   * the 32-member context is now AppState& + EngineResources& +
//     command callbacks; the twelve specimen_* shadow locals now live in
//     app_state.specimen.* (SpecimenInfoState).
// =========================================================================

#include "ui/panels/panel_scene.hpp"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <string>

#include "ui/widgets/widget_factory.hpp"
#include "ui/widgets/control_registry.hpp"
#include "core/unified_logger.hpp"
#include "core/engine_resources.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/root_field.hpp"
#include "engine/io/obj_exporter.hpp"
#include "verification/gpu_benchmark_runner.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

namespace rhizome {

#ifdef _WIN32
static std::string save_file_dialog(int format_choice = 0) {
    char filename[MAX_PATH] = "rhizome_model";
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    if (format_choice == 1) {
        std::strcat(filename, ".stl");
        ofn.lpstrFilter = "Binary STL (*.stl)\0*.stl\0All Files (*.*)\0*.*\0";
        ofn.lpstrDefExt = "stl";
    } else if (format_choice == 2) {
        std::strcat(filename, ".ply");
        ofn.lpstrFilter = "Binary PLY (*.ply)\0*.ply\0All Files (*.*)\0*.*\0";
        ofn.lpstrDefExt = "ply";
    } else {
        std::strcat(filename, ".obj");
        ofn.lpstrFilter = "Wavefront OBJ (*.obj)\0*.obj\0Binary STL (*.stl)\0*.stl\0Binary PLY (*.ply)\0*.ply\0All Files (*.*)\0*.*\0";
        ofn.lpstrDefExt = "obj";
    }
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;
    if (GetSaveFileNameA(&ofn)) {
        return std::string(filename);
    }
    return "";
}
#endif

void render_status_bar_panel(const std::string& status_bar_message, size_t section_count, size_t spore_count) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 2.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.027f, 0.067f, 0.110f, 1.0f));
    ImGui::Begin("Status Bar", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();

    ImVec2 win_pos = ImGui::GetWindowPos();
    ImVec2 content_min = ImGui::GetWindowContentRegionMin();
    ImVec2 content_max = ImGui::GetWindowContentRegionMax();
    float avail_y = content_max.y - content_min.y;
    ImDrawList* dl = ImGui::GetWindowDrawList();

    float line_h = ImGui::GetTextLineHeight();
    float text_y = win_pos.y + content_min.y + std::max(0.0f, (avail_y - line_h) * 0.5f);
    float dot_r = 3.0f;
    float dot_x = win_pos.x + content_min.x + 6.0f;
    float dot_y = text_y + line_h * 0.5f;

    dl->AddCircleFilled(ImVec2(dot_x, dot_y), dot_r, IM_COL32(59, 168, 142, 255));

    ImGui::SetCursorScreenPos(ImVec2(dot_x + dot_r + 6.0f, text_y));
    ImGui::TextColored(ImVec4(0.80f, 0.82f, 0.86f, 1.0f), "%s", status_bar_message.c_str());

    // Right-aligned stats: SpaceNavigator, VRAM, and buffer counts
    float right_cursor_x = win_pos.x + content_max.x - 6.0f;

    // 1. Live GPU VRAM & buffer metrics
    GpuMemoryStats vram = query_gpu_memory_stats();
    char mem_buf[160] = {0};
    if (vram.valid) {
        double used_gb = static_cast<double>(vram.used_bytes) / (1024.0 * 1024.0 * 1024.0);
        double total_gb = static_cast<double>(vram.total_bytes) / (1024.0 * 1024.0 * 1024.0);
        if (section_count > 0 || spore_count > 0) {
            if (spore_count >= 1000) {
                std::snprintf(mem_buf, sizeof(mem_buf), "VRAM: %.1f / %.1f GB (%.0f%%)  |  Secs: %zu  |  Spores: %.0fk",
                    used_gb, total_gb, vram.used_percent, section_count, static_cast<double>(spore_count) / 1000.0);
            } else {
                std::snprintf(mem_buf, sizeof(mem_buf), "VRAM: %.1f / %.1f GB (%.0f%%)  |  Secs: %zu  |  Spores: %zu",
                    used_gb, total_gb, vram.used_percent, section_count, spore_count);
            }
        } else {
            std::snprintf(mem_buf, sizeof(mem_buf), "VRAM: %.1f / %.1f GB (%.0f%%)",
                used_gb, total_gb, vram.used_percent);
        }
    } else if (section_count > 0 || spore_count > 0) {
        if (spore_count >= 1000) {
            std::snprintf(mem_buf, sizeof(mem_buf), "Secs: %zu  |  Spores: %.0fk",
                section_count, static_cast<double>(spore_count) / 1000.0);
        } else {
            std::snprintf(mem_buf, sizeof(mem_buf), "Secs: %zu  |  Spores: %zu",
                section_count, spore_count);
        }
    }

    if (mem_buf[0] != '\0') {
        float mem_w = ImGui::CalcTextSize(mem_buf).x;
        right_cursor_x -= mem_w;
        ImGui::SetCursorScreenPos(ImVec2(right_cursor_x, text_y));
        ImVec4 mem_col = (vram.valid && vram.used_percent > 85.0f)
            ? ImVec4(1.0f, 0.45f, 0.35f, 1.0f)
            : ImVec4(0.55f, 0.70f, 0.82f, 1.0f);
        ImGui::TextColored(mem_col, "%s", mem_buf);
        right_cursor_x -= 16.0f;
    }

#ifdef _WIN32
    if (rhizome::get_spacenav_state().connected) {
        std::string sn_label = "3Dconnexion SpaceNavigator [Active]";
        float sn_label_w = ImGui::CalcTextSize(sn_label.c_str()).x;
        right_cursor_x -= sn_label_w;
        ImGui::SetCursorScreenPos(ImVec2(right_cursor_x, text_y));
        ImGui::TextColored(ImVec4(0.35f, 0.75f, 0.95f, 1.0f), "%s", sn_label.c_str());
    }
#endif

    ImGui::End();
    ImGui::PopStyleVar(2);
}

void render_scene_panel(ScenePanelContext& ctx) {
    using namespace ctrldefs;

    AppState& app_state = ctx.app_state;
    EngineResources& res = ctx.res;
    SpecimenInfoState& spec = app_state.specimen;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 2.0f));
    ImGui::Begin("Scene", nullptr, ImGuiWindowFlags_None);

    // ------------------------------------------------------------------
    // VERIFICATION SUITE (CTRL107-CTRL108) + GPU benchmark trigger
    // ------------------------------------------------------------------
    if (render_floating_pill_header("VERIFICATION SUITE", false)) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 0.541f, 0.122f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.620f, 0.250f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.900f, 0.450f, 0.080f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));

        float ver_btn_w = (ImGui::GetContentRegionAvail().x - 4.0f) * 0.5f;
        if (ImGui::Button(app_state.verification.tests_running ? "Running..." : "Run All Tests", ImVec2(ver_btn_w, 22.0f))) {
            if (!app_state.verification.tests_running) {
                ctx.run_all_tests();
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Run complete 20-test verification suite");

        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::Button(app_state.verification.show_tests_window ? "Hide Log" : "View Log", ImVec2(ver_btn_w, 22.0f))) {
            app_state.verification.show_tests_window = !app_state.verification.show_tests_window;
        }

        static bool benchmark_running = false;
        static std::string benchmark_status = "";
        if (ImGui::Button(benchmark_running ? "Running GPU Benchmark..." : "Run GPU Benchmark (100% CUDA)", ImVec2(-1.0f, 22.0f))) {
            if (!benchmark_running) {
                benchmark_running = true;
                auto suite = GpuBenchmarkRunner::run_full_benchmark("logs/gpu_benchmark_report.json", "logs/gpu_benchmark_report.md");
                benchmark_status = suite.summary_text;
                benchmark_running = false;
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Stress-test parameter combinations from the 138-control catalog on CUDA GPU");

        if (!benchmark_status.empty()) {
            ImGui::Spacing();
            ImGui::TextWrapped("%s", benchmark_status.c_str());
        }

        ImGui::PopStyleColor(4);
    }

    // ------------------------------------------------------------------
    // IMPLICIT SPECIMEN (CTRL112-CTRL115)
    // ------------------------------------------------------------------
    if (render_floating_pill_header("IMPLICIT SPECIMEN", true)) {
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 1.0f));
        if (ImGui::BeginTable("SpecimenMetaTable", 2, ImGuiTableFlags_None)) {
            ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 108.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Source");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(ImVec4(0.914f, 0.941f, 0.957f, 1.0f), "%s", spec.specimen_source_name.c_str());
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.098f, 0.827f, 0.635f, 1.0f), "[%s]", spec.specimen_source_format.c_str());

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Dimensions");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(ImVec4(0.914f, 0.941f, 0.957f, 1.0f), "%.0f x %.0f x %.0f MM",
                               spec.specimen_dimensions.x, spec.specimen_dimensions.y, spec.specimen_dimensions.z);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Shell Count");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(ImVec4(0.914f, 0.941f, 0.957f, 1.0f), "%zu Tris", spec.specimen_triangle_count);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Implicit Status");
            ImGui::TableSetColumnIndex(1);
            if (spec.specimen_implicit_status == "Built") {
                ImGui::TextColored(ImVec4(0.098f, 0.827f, 0.635f, 1.0f), "Built (%.0f ms)", spec.specimen_build_time_ms);
            } else if (spec.specimen_implicit_status == "Failed") {
                ImGui::TextColored(ImVec4(0.90f, 0.25f, 0.25f, 1.0f), "Failed");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.541f, 0.122f, 1.0f), "%s", spec.specimen_implicit_status.c_str());
            }

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Interior Volume");
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(ImVec4(0.914f, 0.941f, 0.957f, 1.0f), "%.1f mL", spec.specimen_interior_volume_ml);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Shell Validity");
            ImGui::TableSetColumnIndex(1);
            if (spec.specimen_shell_valid) {
                ImGui::TextColored(ImVec4(0.098f, 0.827f, 0.635f, 1.0f), "%s", spec.specimen_shell_valid_msg.c_str());
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.541f, 0.122f, 1.0f), "%s", spec.specimen_shell_valid_msg.c_str());
            }

            ImGui::EndTable();
        }
        ImGui::PopStyleVar();

        bool detail_changed = render_control(CTRL112, app_state);
        if (ImGui::IsItemDeactivatedAfterEdit() || (detail_changed && !ImGui::IsItemActive())) {
            ctx.convert_and_update_model(res.original_mesh, res.loaded_model_name);
        }

        const float btn_w = (ImGui::GetContentRegionAvail().x - 8.0f) / 3.0f;
        if (ImGui::Button("Sphere##GpuShape", ImVec2(btn_w, 22.0f))) {
            rhizome::TriangleMesh m = rhizome::CudaRasterizer::generate_sphere_gpu(20.0f, 64, 64);
            ctx.convert_and_update_model(m, "GPU Sphere (R=20mm)");
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Generate 3D Sphere on GPU");

        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::Button("Torus##GpuShape", ImVec2(btn_w, 22.0f))) {
            rhizome::TriangleMesh m = rhizome::CudaRasterizer::generate_torus_gpu(25.0f, 10.0f, 64, 64);
            ctx.convert_and_update_model(m, "GPU Torus (R=25, r=10)");
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Generate 3D Torus on GPU");

        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::Button("Box##GpuShape", ImVec2(btn_w, 22.0f))) {
            rhizome::TriangleMesh m = rhizome::CudaRasterizer::generate_filleted_box_gpu(glm::vec3(20.0f), 4.0f, 48);
            ctx.convert_and_update_model(m, "GPU Filleted Box (40mm)");
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Generate 3D Filleted Box on GPU");
    }

    // ------------------------------------------------------------------
    // LAYERS (CTRL116-CTRL126; v24.01 Phase 0.2: the shading trio
    // CTRL104-106 moved to Styles > SPECIMEN APPEARANCE)
    // ------------------------------------------------------------------
    if (render_floating_pill_header("LAYERS", true)) {
        render_control(CTRL116, app_state);
        render_control(CTRL117, app_state);
        render_control(CTRL118, app_state);
        if (render_control(CTRL119, app_state)) {
            if (app_state.display.colorize_roots) {
                app_state.display.colorize_strands = false;
            }
            if (!res.current_branch_nodes.empty()) {
                res.branch_gl_buffers.upload(res.current_branch_nodes, app_state.display.colorize_roots || app_state.display.colorize_strands);
            }
        }
        if (render_control(CTRL120, app_state)) {
            if (app_state.display.colorize_strands) {
                app_state.display.colorize_roots = false;
            }
            if (!res.current_branch_nodes.empty()) {
                res.branch_gl_buffers.upload(res.current_branch_nodes, app_state.display.colorize_roots || app_state.display.colorize_strands);
            }
        }
        if (render_control(CTRL121, app_state)) {
            if (app_state.display.show_branch_skeleton && !res.current_branch_nodes.empty()) {
                res.branch_gl_buffers.upload(res.current_branch_nodes, app_state.display.colorize_roots || app_state.display.colorize_strands);
            }
        }
        render_control(CTRL122, app_state);
        if (render_control(CTRL123, app_state)) {
            if (app_state.display.show_spores) {
                if (res.live_colonizer.get_spores().empty() && res.current_sdf_grid) {
                    res.live_colonizer.sample_spores_only(res.current_sdf_grid, ctx.build_colonization_config());
                }
                res.spore_gl_buffers.upload(res.live_colonizer.get_spores());
            }
        }
        if (app_state.display.show_spores) {
            ImGui::TextColored(
                ImVec4(0.098f, 0.827f, 0.635f, 1.0f),
                "   Spore Population: %zu / %d",
                res.live_colonizer.get_spores().size(),
                app_state.growth.spore_count
            );
        }
        render_control(CTRL124, app_state);
        render_control(CTRL125, app_state);
        render_control(CTRL126, app_state);
    }

    auto post_process_generated_mesh_gpu = [&](rhizome::TriangleMesh& m) {
        if (m.vertex_count() == 0 || m.triangle_count() == 0) return;
        float gpu_ms = 0.0f;
        if (app_state.mesh_export.param_adaptive_mesh_voxel && app_state.mesh_export.param_max_mesh_voxel_size > app_state.mesh_export.param_min_mesh_voxel_size + 1e-4f) {
            rhizome::CudaRasterizer::decimate_mesh_adaptive_gpu(
                m, app_state.mesh_export.param_min_mesh_voxel_size, app_state.mesh_export.param_max_mesh_voxel_size, gpu_ms
            );
        }
        if (app_state.mesh_export.param_mesh_smooth_passes > 0) {
            rhizome::CudaRasterizer::smooth_mesh_laplacian_gpu(
                m, app_state.mesh_export.param_mesh_smooth_passes, app_state.mesh_export.param_mesh_smooth_strength, gpu_ms
            );
        }
        if (app_state.mesh_export.param_mesh_manifold_repair) {
            rhizome::CudaRasterizer::repair_mesh_manifold_gpu(m, gpu_ms);
        }
    };

    auto generate_target_mesh_gpu = [&]() {
        float target_vox = app_state.mesh_export.param_adaptive_mesh_voxel
            ? std::clamp(app_state.mesh_export.param_min_mesh_voxel_size, 0.05f, 0.60f)
            : std::clamp(app_state.mesh_export.param_mesh_voxel_size, 0.05f, 0.60f);

        auto build_roots_mesh_at_vox = [&]() {
            if (!res.current_sdf_grid) return;
            std::shared_ptr<rhizome::CudaVolumeGrid> r_grid = nullptr;
            if ((app_state.growth.algorithm_mode == 1 || app_state.growth.algorithm_mode == 2) && !res.live_colonizer.get_volumetric_sections().empty()) {
                auto raw_r = res.live_colonizer.rasterize_volumetric_session(target_vox, false, app_state.growth.branch_smooth_k);
                if (raw_r && raw_r->is_valid()) {
                    r_grid = RootImplicitField::intersect_with_vessel(raw_r, res.current_sdf_grid, app_state.growth.wall_cut_smooth);
                }
            } else if (!res.live_colonizer.get_nodes().empty()) {
                auto raw_r = RootImplicitField::rasterize_root_graph(
                    res.live_colonizer.get_nodes(),
                    target_vox,
                    app_state.growth.trunk_caliber,
                    app_state.growth.tendril_caliber,
                    app_state.growth.variable_taper ? 0.0f : app_state.growth.taper_curve,
                    4.0f,
                    false,
                    app_state.growth.branch_smooth_k
                );
                if (raw_r && raw_r->is_valid()) {
                    r_grid = RootImplicitField::intersect_with_vessel(raw_r, res.current_sdf_grid, app_state.growth.wall_cut_smooth);
                }
            } else if (res.current_roots_grid && res.current_roots_grid->is_valid()) {
                r_grid = res.current_roots_grid;
            }
            if (r_grid && r_grid->is_valid()) {
                res.current_roots_grid = r_grid;
                res.roots_mesh = rhizome::CudaRasterizer::extract_iso_surface(*r_grid, 0.0f);
                post_process_generated_mesh_gpu(res.roots_mesh);
                res.roots_gl_buffers.upload(res.roots_mesh, true);
                res.cached_roots_metrics = compute_mesh_metrics(res.roots_mesh);
                app_state.mesh_export.has_generated_roots_mesh = (res.roots_mesh.triangle_count() > 0);
            }
        };

        auto build_vessel_mesh_at_vox = [&]() {
            std::shared_ptr<rhizome::CudaVolumeGrid> v_grid = res.current_sdf_grid;
            if (res.original_mesh.vertex_count() > 0 &&
                (!v_grid || !v_grid->is_valid() || std::abs(v_grid->voxel_size - target_vox) > 1e-4f)) {
                float build_ms = 0.0f;
                auto rebuilt = rhizome::CudaRasterizer::build_from_mesh(res.original_mesh, target_vox, build_ms);
                if (rebuilt && rebuilt->is_valid()) {
                    v_grid = rebuilt;
                }
            }
            if (v_grid && v_grid->is_valid()) {
                res.converted_body_mesh = rhizome::CudaRasterizer::extract_iso_surface(*v_grid, 0.0f);
                post_process_generated_mesh_gpu(res.converted_body_mesh);
                res.body_gl_buffers.upload(res.converted_body_mesh, true);
                res.cached_body_metrics = compute_mesh_metrics(res.converted_body_mesh);
                app_state.mesh_export.has_generated_vessel_mesh = (res.converted_body_mesh.triangle_count() > 0);
            }
        };

        if (app_state.mesh_export.param_export_target == 0) {
            build_roots_mesh_at_vox();
            app_state.display.show_wireframe = app_state.mesh_export.has_generated_roots_mesh;
            app_state.ui.status_bar_message = "Generated Roots Mesh: " + std::to_string(res.roots_mesh.triangle_count()) + " triangles (GPU)";
            RhizomeLog("REACTION", app_state.ui.status_bar_message.c_str());
            RhizomeLogVramSnapshot("Roots Mesh Generated");
        } else if (app_state.mesh_export.param_export_target == 1) {
            build_vessel_mesh_at_vox();
            app_state.display.show_wireframe = app_state.mesh_export.has_generated_vessel_mesh;
            app_state.ui.status_bar_message = "Generated Vessel Mesh: " + std::to_string(res.converted_body_mesh.triangle_count()) + " triangles (GPU)";
            RhizomeLog("REACTION", app_state.ui.status_bar_message.c_str());
            RhizomeLogVramSnapshot("Vessel Mesh Generated");
        } else {
            build_roots_mesh_at_vox();
            build_vessel_mesh_at_vox();
            const rhizome::TriangleMesh& m1 = res.roots_mesh;
            const rhizome::TriangleMesh& m2 = (res.converted_body_mesh.vertex_count() > 0) ? res.converted_body_mesh : res.original_mesh;
            float merge_ms = 0.0f;
            rhizome::CudaRasterizer::merge_meshes_gpu(m1, m2, res.generated_merged_mesh, merge_ms);
            res.merged_gl_buffers.upload(res.generated_merged_mesh, true);
            res.cached_merged_metrics = compute_mesh_metrics(res.generated_merged_mesh);
            app_state.mesh_export.has_generated_merged_mesh = (res.generated_merged_mesh.triangle_count() > 0);
            app_state.display.show_wireframe = app_state.mesh_export.has_generated_merged_mesh;
            app_state.ui.status_bar_message = "Generated Merged Mesh: " + std::to_string(res.generated_merged_mesh.triangle_count()) + " triangles (GPU)";
            RhizomeLog("REACTION", app_state.ui.status_bar_message.c_str());
            RhizomeLogVramSnapshot("Merged Mesh Generated");
        }
    };

    // ------------------------------------------------------------------
    // EXPORT OPTIONS (CTRL127-CTRL138)
    // ------------------------------------------------------------------
    if (render_floating_pill_header("EXPORT OPTIONS", true)) {
        const char* target_items[] = { "Roots Only", "Converted Vessel", "Merged Solid (Union)" };
        render_pill_combo("##ExportTargetCombo", &app_state.mesh_export.param_export_target, target_items, IM_ARRAYSIZE(target_items));

        render_control(CTRL128, app_state);
        if (!app_state.mesh_export.param_adaptive_mesh_voxel) {
            render_control(CTRL129, app_state);
        } else {
            if (render_control(CTRL130, app_state)) {
                if (app_state.mesh_export.param_min_mesh_voxel_size > app_state.mesh_export.param_max_mesh_voxel_size) {
                    app_state.mesh_export.param_max_mesh_voxel_size = app_state.mesh_export.param_min_mesh_voxel_size;
                }
            }
            if (render_control(CTRL131, app_state)) {
                if (app_state.mesh_export.param_max_mesh_voxel_size < app_state.mesh_export.param_min_mesh_voxel_size) {
                    app_state.mesh_export.param_min_mesh_voxel_size = app_state.mesh_export.param_max_mesh_voxel_size;
                }
            }
        }

        render_control(CTRL132, app_state);
        if (app_state.mesh_export.param_mesh_smooth_passes > 0) {
            render_control(CTRL133, app_state);
        }

        render_control(CTRL134, app_state);

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.45f, 0.82f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.54f, 0.92f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.14f, 0.36f, 0.68f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.96f, 0.98f, 1.0f, 1.0f));
        if (ImGui::Button("Generate Mesh", ImVec2(-1.0f, 22.0f))) {
            generate_target_mesh_gpu();
        }
        ImGui::PopStyleColor(4);

        size_t total_tris = 0;
        size_t total_edges = 0;
        float avg_edge_mm = 0.0f;
        float min_edge_mm = 0.0f;
        float max_edge_mm = 0.0f;

        if (app_state.mesh_export.param_export_target == 0) {
            total_tris = res.cached_roots_metrics.total_tris;
            total_edges = res.cached_roots_metrics.total_edges;
            avg_edge_mm = res.cached_roots_metrics.avg_edge_mm;
            min_edge_mm = res.cached_roots_metrics.min_edge_mm;
            max_edge_mm = res.cached_roots_metrics.max_edge_mm;
        } else if (app_state.mesh_export.param_export_target == 1) {
            total_tris = res.cached_body_metrics.total_tris;
            total_edges = res.cached_body_metrics.total_edges;
            avg_edge_mm = res.cached_body_metrics.avg_edge_mm;
            min_edge_mm = res.cached_body_metrics.min_edge_mm;
            max_edge_mm = res.cached_body_metrics.max_edge_mm;
        } else if (app_state.mesh_export.param_export_target == 2) {
            if (app_state.mesh_export.has_generated_merged_mesh) {
                total_tris = res.cached_merged_metrics.total_tris;
                total_edges = res.cached_merged_metrics.total_edges;
                avg_edge_mm = res.cached_merged_metrics.avg_edge_mm;
                min_edge_mm = res.cached_merged_metrics.min_edge_mm;
                max_edge_mm = res.cached_merged_metrics.max_edge_mm;
            } else {
                total_tris = res.cached_roots_metrics.total_tris + res.cached_body_metrics.total_tris;
                total_edges = total_tris * 3 / 2;
                avg_edge_mm = (res.cached_roots_metrics.avg_edge_mm > 0.0f && res.cached_body_metrics.avg_edge_mm > 0.0f)
                    ? (res.cached_roots_metrics.avg_edge_mm + res.cached_body_metrics.avg_edge_mm) * 0.5f
                    : std::max(res.cached_roots_metrics.avg_edge_mm, res.cached_body_metrics.avg_edge_mm);
                min_edge_mm = std::min(res.cached_roots_metrics.min_edge_mm, res.cached_body_metrics.min_edge_mm);
                max_edge_mm = std::max(res.cached_roots_metrics.max_edge_mm, res.cached_body_metrics.max_edge_mm);
            }
        }

        ImGui::Columns(2, "ExportMetricsCols", false);
        ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Triangles:"); ImGui::NextColumn();
        ImGui::Text("%zu", total_tris); ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Edges:"); ImGui::NextColumn();
        ImGui::Text("%zu", total_edges); ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Avg Edge:"); ImGui::NextColumn();
        ImGui::Text("%.2f mm", avg_edge_mm); ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Min / Max Edge:"); ImGui::NextColumn();
        ImGui::Text("%.2f / %.2f mm", min_edge_mm, max_edge_mm); ImGui::NextColumn();
        ImGui::Columns(1);

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.098f, 0.827f, 0.635f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.157f, 0.898f, 0.706f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.078f, 0.700f, 0.530f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.027f, 0.067f, 0.110f, 1.0f));

        auto trigger_export_format = [&](int fmt_id, const char* fmt_name) {
#ifdef _WIN32
            std::string save_path = save_file_dialog(fmt_id);
            if (!save_path.empty()) {
                if ((app_state.mesh_export.param_export_target == 0 && !app_state.mesh_export.has_generated_roots_mesh) ||
                    (app_state.mesh_export.param_export_target == 1 && !app_state.mesh_export.has_generated_vessel_mesh) ||
                    (app_state.mesh_export.param_export_target == 2 && !app_state.mesh_export.has_generated_merged_mesh)) {
                    generate_target_mesh_gpu();
                }

                const rhizome::TriangleMesh* mesh_to_save = nullptr;
                if (app_state.mesh_export.param_export_target == 0) {
                    mesh_to_save = &res.roots_mesh;
                } else if (app_state.mesh_export.param_export_target == 1) {
                    mesh_to_save = (res.converted_body_mesh.vertex_count() > 0) ? &res.converted_body_mesh : &res.original_mesh;
                } else {
                    mesh_to_save = &res.generated_merged_mesh;
                }

                if (mesh_to_save && mesh_to_save->vertex_count() > 0 && rhizome::ObjExporter::export_mesh(save_path, *mesh_to_save)) {
                    app_state.ui.status_bar_message = "Saved " + std::to_string(mesh_to_save->triangle_count()) + " triangles to " + save_path;
                } else {
                    app_state.ui.status_bar_message = std::string("Failed to save ") + fmt_name;
                }
            }
#endif
        };

        if (ImGui::Button("Export Mesh (OBJ)", ImVec2(-1.0f, 22.0f))) {
            trigger_export_format(0, "OBJ");
        }

        ImGui::Spacing();
        float half_btn_w = (ImGui::GetContentRegionAvail().x - 4.0f) * 0.5f;
        if (ImGui::Button("Export Binary STL", ImVec2(half_btn_w, 22.0f))) {
            trigger_export_format(1, "STL");
        }
        ImGui::SameLine();
        if (ImGui::Button("Export Binary PLY", ImVec2(half_btn_w, 22.0f))) {
            trigger_export_format(2, "PLY");
        }
        ImGui::PopStyleColor(4);
    }

    // ------------------------------------------------------------------
    // BARK TEXTURE & SURFACE RELIEF (CTRL139-CTRL148)
    // ------------------------------------------------------------------
    if (render_floating_pill_header("BARK TEXTURE & SURFACE RELIEF", false)) {
        auto& bark = app_state.bark_texture;

        ImGui::TextColored(ImVec4(0.098f, 0.827f, 0.635f, 1.0f), "Step 1: 2D Grayscale Texture (GPU)");
        render_control(CTRL139, app_state);
        render_control(CTRL140, app_state);
        render_control(CTRL141, app_state);
        render_control(CTRL142, app_state);

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.45f, 0.82f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.54f, 0.92f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.14f, 0.36f, 0.68f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.96f, 0.98f, 1.0f, 1.0f));

        auto upload_bark_gl_preview = [](BarkTextureState& b) {
            if (b.rgba_preview.empty() || b.tex_width <= 0 || b.tex_height <= 0) return;
            if (b.gl_preview_tex_id == 0) {
                glGenTextures(1, &b.gl_preview_tex_id);
            }
            glBindTexture(GL_TEXTURE_2D, b.gl_preview_tex_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, b.tex_width, b.tex_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, b.rgba_preview.data());
            glBindTexture(GL_TEXTURE_2D, 0);
        };

        if (ImGui::Button("Generate 2D Bark Texture (GPU)", ImVec2(-1.0f, 22.0f))) {
            rhizome::CudaRasterizer::BarkTexture2DParams p;
            p.width = bark.tex_width;
            p.height = bark.tex_height;
            p.scale = bark.scale;
            p.stretch_y = bark.stretch_y;
            p.furrow_depth = bark.furrow_depth;
            p.octaves = bark.octaves;
            p.persistence = bark.persistence;
            p.seed = bark.seed;

            float gen_ms = 0.0f;
            if (rhizome::CudaRasterizer::generate_bark_texture_2d_gpu(p, bark.heightmap, bark.rgba_preview, gen_ms)) {
                bark.has_generated_texture = true;
                bark.last_gen_time_ms = gen_ms;
                upload_bark_gl_preview(bark);
                app_state.ui.status_bar_message = "Generated 2D Bark Texture " + std::to_string(p.width) + "x" + std::to_string(p.height) + " in " + std::to_string(static_cast<int>(gen_ms)) + " ms (GPU)";
                RhizomeLog("REACTION", app_state.ui.status_bar_message.c_str());
            }
        }
        ImGui::PopStyleColor(4);

        if (bark.has_generated_texture) {
            ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "Grayscale Map: %dx%d (%.1f ms)", bark.tex_width, bark.tex_height, bark.last_gen_time_ms);
            if (bark.gl_preview_tex_id == 0 && !bark.rgba_preview.empty()) {
                upload_bark_gl_preview(bark);
            }
            if (bark.gl_preview_tex_id != 0) {
                float avail_w = ImGui::GetContentRegionAvail().x;
                float preview_h = (avail_w > 220.0f) ? 220.0f : avail_w;
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.098f, 0.827f, 0.635f, 1.0f), "Live Texture Preview:");
                ImGui::Image(
                    (ImTextureID)(intptr_t)bark.gl_preview_tex_id,
                    ImVec2(avail_w, preview_h),
                    ImVec2(0.0f, 0.0f),
                    ImVec2(1.0f, 1.0f),
                    ImVec4(1.0f, 1.0f, 1.0f, 1.0f),
                    ImVec4(0.18f, 0.45f, 0.82f, 0.60f)
                );
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.098f, 0.827f, 0.635f, 1.0f), "Step 2: Surface Displacement (GPU)");
        render_control(CTRL144, app_state);
        render_control(CTRL145, app_state);
        render_control(CTRL146, app_state);
        render_control(CTRL147, app_state);

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.098f, 0.827f, 0.635f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.157f, 0.898f, 0.706f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.078f, 0.700f, 0.530f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.027f, 0.067f, 0.110f, 1.0f));

        if (ImGui::Button("Apply Bark Relief to Mesh (GPU)", ImVec2(-1.0f, 22.0f))) {
            if (!bark.has_generated_texture) {
                rhizome::CudaRasterizer::BarkTexture2DParams p;
                p.width = bark.tex_width;
                p.height = bark.tex_height;
                p.scale = bark.scale;
                p.stretch_y = bark.stretch_y;
                p.furrow_depth = bark.furrow_depth;
                p.octaves = bark.octaves;
                p.persistence = bark.persistence;
                p.seed = bark.seed;
                float gen_ms = 0.0f;
                rhizome::CudaRasterizer::generate_bark_texture_2d_gpu(p, bark.heightmap, bark.rgba_preview, gen_ms);
                bark.has_generated_texture = true;
                upload_bark_gl_preview(bark);
            }

            if (!app_state.mesh_export.has_generated_roots_mesh && res.roots_mesh.vertex_count() == 0) {
                generate_target_mesh_gpu();
            }

            rhizome::TriangleMesh* target_mesh = (res.roots_mesh.vertex_count() > 0) ? &res.roots_mesh : ((res.converted_body_mesh.vertex_count() > 0) ? &res.converted_body_mesh : const_cast<rhizome::TriangleMesh*>(&res.original_mesh));

            if (target_mesh && target_mesh->vertex_count() > 0) {
                rhizome::CudaRasterizer::MeshDisplacementParams dp;
                dp.displacement_depth_mm = bark.displacement_depth_mm;
                dp.tile_repeat_u = bark.tile_repeat_u;
                dp.tile_repeat_v = bark.tile_repeat_v;
                dp.recompute_normals = bark.recompute_normals;

                float disp_ms = 0.0f;
                if (rhizome::CudaRasterizer::displace_mesh_from_bark_texture_gpu(*target_mesh, bark.heightmap, bark.tex_width, bark.tex_height, dp, disp_ms)) {
                    bark.last_disp_time_ms = disp_ms;
                    if (target_mesh == &res.roots_mesh) {
                        res.roots_gl_buffers.upload(res.roots_mesh, true);
                        res.cached_roots_metrics = compute_mesh_metrics(res.roots_mesh);
                        app_state.mesh_export.has_generated_roots_mesh = true;
                        app_state.display.show_wireframe = true;
                    }
                    app_state.ui.status_bar_message = "Displaced " + std::to_string(target_mesh->vertex_count()) + " vertices with Bark Texture in " + std::to_string(static_cast<int>(disp_ms)) + " ms (GPU)";
                    RhizomeLog("REACTION", app_state.ui.status_bar_message.c_str());
                    RhizomeLogVramSnapshot("Bark Relief Applied");
                }
            }
        }
        ImGui::PopStyleColor(4);
    }

    ImGui::End();
    ImGui::PopStyleVar(2);
}

} // namespace rhizome
