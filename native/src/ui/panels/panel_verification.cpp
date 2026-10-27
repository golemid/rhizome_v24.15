// =========================================================================
// RHIZOME VERIFICATION PANEL — implementation (v23.29 Phase 4)
// =========================================================================
// Migrated from src/app/ui_scene_panel.cpp :: render_verification_panel
// (lines 131-203). Rendering behavior preserved verbatim; the loose
// boolean/vector reference parameters are replaced by AppState& +
// EngineResources& (show_tests_window -> app_state.verification.
// show_tests_window, tests_running -> app_state.verification.tests_running,
// test_objects / last_test_results -> res).
// =========================================================================

#include "ui/panels/panel_verification.hpp"

#include "ui/widgets/widget_factory.hpp"
#include "ui/widgets/control_registry.hpp"
#include "core/unified_logger.hpp"

namespace rhizome {

void render_verification_panel(
    AppState& app_state,
    EngineResources& res,
    const std::function<void()>& run_all_tests,
    const std::function<void(const rhizome::TriangleMesh&, const std::string&)>& convert_and_update_model,
    ImVec2 work_pos,
    ImVec2 work_size
) {
    using namespace ctrldefs;

    if (!app_state.verification.show_tests_window) return;

    ImGui::SetNextWindowSize(ImVec2(520.0f, 420.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(work_pos.x + work_size.x - 540.0f, work_pos.y + 60.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Verification", &app_state.verification.show_tests_window, ImGuiWindowFlags_None)) {
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 2.0f));
        float ver_win_btn_w = (ImGui::GetContentRegionAvail().x - 4.0f) * 0.5f;
        if (ImGui::Button(app_state.verification.tests_running ? "Running Tests..." : "Run All Tests", ImVec2(ver_win_btn_w, 22.0f))) {
            if (!app_state.verification.tests_running) {
                run_all_tests();
            }
        }
        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::Button("Clear 3D Tests", ImVec2(ver_win_btn_w, 22.0f))) {
            for (auto& obj : res.test_objects) {
                obj.buffers.cleanup();
            }
            res.test_objects.clear();
            res.last_test_results.clear();
        }
        ImGui::PopStyleVar();

        ImGui::Separator();

        if (res.last_test_results.empty()) {
            ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.64f, 1.0f), "Click 'Run All Tests' to generate 3D test models in the viewport.");
        } else {
            int p_cnt = 0, f_cnt = 0;
            for (const auto& r : res.last_test_results) {
                if (r.passed) p_cnt++; else f_cnt++;
            }
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.9f, 1.0f), "Suite Summary: %d Passed, %d Failed (%zu 3D models)", p_cnt, f_cnt, res.test_objects.size());

            ImGui::BeginChild("TestResultsSummaryList", ImVec2(0.0f, 0.0f), true);
            for (size_t i = 0; i < res.last_test_results.size(); ++i) {
                const auto& r = res.last_test_results[i];
                ImGui::PushID(static_cast<int>(i));
                if (r.passed) {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "[PASS]");
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "[FAIL]");
                }
                ImGui::SameLine();
                std::string label = "Test " + std::to_string(i + 1) + ": " + r.primitive_name;
                if (ImGui::Selectable(label.c_str())) {
                    if (r.result_mesh.vertex_count() > 0) {
                        convert_and_update_model(r.result_mesh, r.primitive_name);
                        app_state.display.show_converted_body = true;
                        app_state.ui.status_bar_message = "Loaded test model '" + r.primitive_name + "' as active Implicit Specimen.";
                    }
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Click to load '%s' into Implicit Specimen panel & viewport", r.primitive_name.c_str());
                }
                ImGui::PopID();
            }
            ImGui::EndChild();
        }
    }
    ImGui::End();
}

} // namespace rhizome
