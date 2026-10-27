// =========================================================================
// RHIZOME NATIVE — application entry point (v23.29 Phase 4)
// =========================================================================
// GLFW/ImGui bootstrap, main loop, RhizomeDockSpace layout and CLI flag
// parsing. Phase 4 rewrite per the approved Phase 1 proposal:
//   * Shadow state eradicated (finding F4): the ~30 main.cpp locals that
//     duplicated AppState members (seed_points, is_growing_live,
//     status_bar_message, specimen_*, show_tests_window, growth_presets,
//     ...) are gone. AppState is the Single Source of Truth; every live
//     GPU/GL resource lives in EngineResources.
//   * Context wiring: GrowthControllerContext / panel contexts are the
//     slim two-reference forms (AppState& + EngineResources&) plus
//     command callbacks.
//   * --verify CLI flag added (finding F3): headless full-suite run of
//     the verification harness, mirroring the --benchmark early-return
//     pattern; process exit code reflects pass/fail.
//   * Phase 5: --verify first runs the 138-control registry integrity
//     audit (AGENTS.md Sec 1.3) — the runtime twin of the build-time
//     CSV<->registry audit (native/tools/audit_control_registry.py).
// =========================================================================

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "core/app_state.hpp"
#include "core/engine_resources.hpp"
#include "core/growth_controller.hpp"
#include "core/algorithm_io.hpp"
#include "core/unified_logger.hpp"

#include "ui/panels/panel_growth.hpp"
#include "ui/panels/panel_styles.hpp"
#include "ui/panels/panel_scene.hpp"
#include "ui/panels/panel_verification.hpp"
#include "ui/viewport/viewport_renderer.hpp"
#include "ui/widgets/control_registry.hpp"

#include "verification/harness.hpp"
#include "verification/gpu_benchmark_runner.hpp"

using namespace rhizome;

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <filesystem>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

static void glfw_error_callback(int error, const char* description) {
    std::cerr << "[GLFW Error " << error << "]: " << description << std::endl;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    bool run_tests_on_startup = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test" || arg == "-t") {
            run_tests_on_startup = true;
        } else if (arg == "--verify") {
            // ------------------------------------------------------------
            // F3 (fixed): headless verification-harness run. Mirrors the
            // --benchmark early-return pattern: no window, no ImGui — the
            // full 20-test analytical & CUDA suite executes on the GPU and
            // the process exit code reflects pass/fail (CODE_MAP Sec 2/3.8).
            // Phase 5: the 138-control registry integrity audit (AGENTS.md
            // Sec 1.3) runs FIRST — structural drift between the CSV
            // contract and the ControlDefinition registry fails --verify
            // before any GPU work (build-time twin of this check:
            // native/tools/audit_control_registry.py via the
            // rhizome_control_audit CMake target).
            // ------------------------------------------------------------
            rhizome::init_ui_interaction_monitor("logs/ui_events.jsonl");
            std::cout << "\n=================================================================\n";
            std::cout << "  RHIZOME NATIVE: VERIFICATION HARNESS (--verify)\n";
            std::cout << "=================================================================\n";
            const bool registry_ok = rhizome::control_registry_integrity_check();
            // Counts derive from the registry arrays themselves (v24.01
            // Phase 0: 21 -> 22 CSV sections after the Specimen Appearance
            // re-home) so this preamble cannot drift from the catalog on
            // future renumbers or the Phase 2 style-editor registration.
            const size_t reg_controls = sizeof(rhizome::k_all_controls) / sizeof(rhizome::k_all_controls[0]);
            const size_t reg_sections = sizeof(rhizome::k_all_sections) / sizeof(rhizome::k_all_sections[0]);
            std::cout << "  [Control Registry] " << reg_controls << " controls / "
                      << reg_sections << " CSV sections: "
                      << (registry_ok ? "PASS" : "FAIL") << "\n";
            if (!registry_ok) {
                std::cout << "  Registry integrity check FAILED: the "
                             "ControlDefinition registry violates the\n"
                          "  241-control contract (AGENTS.md Sec 1.3).\n";
                std::cout << "=================================================================\n";
                rhizome::shutdown_ui_interaction_monitor();
                return 1;
            }
            auto results = rhizome::VerificationHarness::run_analytical_suite();
            int passed_count = 0;
            int failed_count = 0;
            for (size_t t = 0; t < results.size(); ++t) {
                const auto& r = results[t];
                std::cout << "  [" << (r.passed ? "PASS" : "FAIL") << "] Test " << (t + 1)
                          << ": " << r.primitive_name << "\n";
                if (r.passed) passed_count++; else failed_count++;
            }
            std::cout << "-----------------------------------------------------------------\n";
            std::cout << "  Suite Summary: " << passed_count << " Passed, "
                      << failed_count << " Failed (" << results.size() << " tests)\n";
            std::cout << "=================================================================\n";
            rhizome::shutdown_ui_interaction_monitor();
            return (failed_count == 0 && !results.empty()) ? 0 : 1;
        } else if (arg == "--benchmark" || arg == "-b") {
            rhizome::init_ui_interaction_monitor("logs/ui_events.jsonl");
            std::cout << "\n=================================================================\n";
            std::cout << "  RHIZOME NATIVE: AUTOMATED GPU BENCHMARK RUNNER (100% CUDA)\n";
            std::cout << "=================================================================\n";
            auto suite = rhizome::GpuBenchmarkRunner::run_full_benchmark("logs/gpu_benchmark_report.json", "logs/gpu_benchmark_report.md");
            rhizome::shutdown_ui_interaction_monitor();
            return suite.completed ? 0 : 1;
        }
    }

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1600, 920, "Rhizome Native", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

#ifdef _WIN32
    rhizome::init_spacenav(window);
#endif

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.IniFilename = "imgui.ini";

    // Set default docking layout if no imgui.ini exists on disk yet
    if (!std::filesystem::exists("imgui.ini")) {
        const char* default_layout_ini =
            "[Window][Command Bar]\n"
            "Pos=299,0\n"
            "Size=1320,82\n"
            "Collapsed=0\n"
            "DockId=0x00000005,0\n"
            "LastUsed=20261001\n"
            "\n"
            "[Window][Toolbar]\n"
            "Collapsed=0\n"
            "DockId=0x00000005\n"
            "\n"
            "[Window][Status Bar]\n"
            "Pos=299,0\n"
            "Size=1320,50\n"
            "Collapsed=0\n"
            "DockId=0x00000005,0\n"
            "LastUsed=20261011\n"
            "\n"
            "[Window][Status]\n"
            "Collapsed=0\n"
            "DockId=0x00000005\n"
            "\n"
            "[Window][Scene]\n"
            "Pos=1621,0\n"
            "Size=299,1009\n"
            "Collapsed=0\n"
            "DockId=0x00000004,0\n"
            "LastUsed=20261001\n"
            "\n"
            "[Window][Growth]\n"
            "Pos=0,0\n"
            "Size=297,1009\n"
            "Collapsed=0\n"
            "DockId=0x00000001,0\n"
            "LastUsed=20261001\n"
            "\n"
            "[Window][Controls]\n"
            "Collapsed=0\n"
            "DockId=0x00000001\n"
            "\n"
            "[Window][3D Viewport]\n"
            "Pos=299,52\n"
            "Size=1320,957\n"
            "Collapsed=0\n"
            "DockId=0x00000007,0\n"
            "LastUsed=20261001\n"
            "\n"
            "[Window][Viewport]\n"
            "Collapsed=0\n"
            "DockId=0x00000007\n"
            "\n"
            "[Window][WindowOverViewport_11111111]\n"
            "Pos=0,0\n"
            "Size=1920,1009\n"
            "Collapsed=0\n"
            "LastUsed=20261001\n"
            "\n"
            "[Window][Debug##Default]\n"
            "Pos=60,60\n"
            "Size=400,400\n"
            "Collapsed=0\n"
            "LastUsed=20261001\n"
            "\n"
            "[Window][Styles]\n"
            "Pos=299,52\n"
            "Size=1320,957\n"
            "Collapsed=0\n"
            "DockId=0x00000007,1\n"
            "LastUsed=20261001\n"
            "\n"
            "[Window][Style Editor]\n"
            "Pos=299,52\n"
            "Size=1320,957\n"
            "Collapsed=0\n"
            "DockId=0x00000007,1\n"
            "LastUsed=20261001\n"
            "\n"
            "[Window][Presets]\n"
            "Pos=299,52\n"
            "Size=1320,957\n"
            "Collapsed=0\n"
            "DockId=0x00000007,2\n"
            "LastUsed=20261001\n"
            "\n"
            "[Docking][Data]\n"
            "DockSpace         ID=0x03B69481 Window=0x1BBC0F80 Pos=0,0 Size=1920,1009 Split=X\n"
            "  DockNode        ID=0x00000001 Parent=0x03B69481 SizeRef=297,920 Selected=0xAFAA95F4\n"
            "  DockNode        ID=0x00000002 Parent=0x03B69481 SizeRef=1301,920 Split=X\n"
            "    DockNode      ID=0x00000003 Parent=0x00000002 SizeRef=1000,920 Split=Y\n"
            "      DockNode    ID=0x00000005 Parent=0x00000003 SizeRef=1000,50 Selected=0xDB008D27\n"
            "      DockNode    ID=0x00000007 Parent=0x00000003 SizeRef=1000,957 CentralNode=1 Selected=0xA7B63EB9\n"
            "    DockNode      ID=0x00000004 Parent=0x00000002 SizeRef=299,920 Selected=0xD3D7729F\n";
        ImGui::LoadIniSettingsFromMemory(default_layout_ini);
    }

    std::vector<AppFontEntry> app_fonts = {
        {"1. Consolas (Monospace)", "consola.ttf", "consolab.ttf", 13.0f, nullptr, nullptr},
        {"2. Segoe UI (Modern Clean)", "segoeui.ttf", "segoeuib.ttf", 14.0f, nullptr, nullptr},
        {"3. Segoe UI Semibold (Crisp Bold)", "seguisb.ttf", "segoeuib.ttf", 14.0f, nullptr, nullptr},
        {"4. Cascadia Code (Developer Mono)", "CascadiaCode.ttf", "CascadiaMono.ttf", 13.0f, nullptr, nullptr},
        {"5. Calibri (Clean Sans)", "calibri.ttf", "calibrib.ttf", 14.0f, nullptr, nullptr},
        {"6. Arial (Universal Sans)", "arial.ttf", "arialbd.ttf", 13.0f, nullptr, nullptr},
        {"7. Verdana (High Legibility)", "verdana.ttf", "verdanab.ttf", 13.0f, nullptr, nullptr},
        {"8. Tahoma (Compact Engineering)", "tahoma.ttf", "tahomabd.ttf", 13.0f, nullptr, nullptr},
        {"9. Trebuchet MS (Humanist Sans)", "trebuc.ttf", "trebucbd.ttf", 13.0f, nullptr, nullptr},
        {"10. Lucida Console (Classic Mono)", "lucon.ttf", "lucon.ttf", 12.0f, nullptr, nullptr}
    };

    // =========================================================================
    // SINGLE SOURCE OF TRUTH: AppState (all persistable state) + 
    //                        EngineResources (all live GPU/OpenGL resources)
    // No shadow locals — panels, controllers and the viewport resolve every
    // piece of state through these two objects (AGENTS.md Sec A.3).
    // =========================================================================
    AppState app_state;
    EngineResources res;

    #ifdef _WIN32
    char windir[MAX_PATH];
    std::string font_dir = "C:\\Windows\\Fonts\\";
    if (GetWindowsDirectoryA(windir, MAX_PATH)) {
        font_dir = std::string(windir) + "\\Fonts\\";
    }

    for (size_t i = 0; i < app_fonts.size(); ++i) {
        std::error_code ec_reg;
        std::string reg_path = font_dir + app_fonts[i].regular_filename;
        if (std::filesystem::exists(reg_path, ec_reg) && !ec_reg) {
            app_fonts[i].regular_ptr = io.Fonts->AddFontFromFileTTF(reg_path.c_str(), app_fonts[i].size);
        } else {
            app_fonts[i].regular_ptr = io.Fonts->AddFontDefault();
        }

        std::error_code ec_bold;
        std::string bold_path = font_dir + app_fonts[i].bold_filename;
        if (std::filesystem::exists(bold_path, ec_bold) && !ec_bold) {
            app_fonts[i].bold_ptr = io.Fonts->AddFontFromFileTTF(bold_path.c_str(), app_fonts[i].size);
        } else {
            app_fonts[i].bold_ptr = app_fonts[i].regular_ptr;
        }
    }
    #else
    for (size_t i = 0; i < app_fonts.size(); ++i) {
        app_fonts[i].regular_ptr = io.Fonts->AddFontDefault();
        app_fonts[i].bold_ptr = io.Fonts->AddFontDefault();
    }
    #endif

    auto update_active_font = [&]() {
        if (app_state.ui.active_font_idx >= 0 && app_state.ui.active_font_idx < static_cast<int>(app_fonts.size())) {
            ImFont* target = (app_state.ui.font_bold && app_fonts[app_state.ui.active_font_idx].bold_ptr)
                ? app_fonts[app_state.ui.active_font_idx].bold_ptr
                : app_fonts[app_state.ui.active_font_idx].regular_ptr;
            if (target) {
                io.FontDefault = target;
            }
        }
    };

    update_active_font();

    apply_dark_engineering_theme();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    ViewportRenderer vp_renderer;
    vp_renderer.init(1280, 720);

    // Unit sphere buffer for rendering seed markers (live GL resource)
    rhizome::TriangleMesh unit_sphere_mesh = rhizome::VerificationHarness::generate_sphere(1.0f, 20, 20);
    res.seed_sphere_buffer.upload(unit_sphere_mesh, true);

    // Growth algorithms & presets live in the AppState SSOT
    init_default_growth_algorithms(app_state.growth.growth_algorithms, app_state.growth.selected_algorithm_idx);

    auto apply_algorithm_profile = [&](int idx) {
        apply_algorithm_profile_to_state(app_state, app_state.growth.growth_algorithms, idx);
    };

    auto capture_current_into_algorithm_profile = [&](GrowthAlgorithmProfile& a) {
        capture_state_into_algorithm_profile(app_state, a);
    };

    apply_algorithm_profile(app_state.growth.selected_algorithm_idx);

    // Growth Presets State (stored in memory & imgui.ini)
    app_state.growth.growth_presets = []() {
        std::vector<GrowthPreset> list;
        {
            GrowthPreset p;
            p.name = "Balanced Default";
            // v24.04 Track D: 0.30 -> 0.10 mm — the Fast SDF path makes the
            // finest interactive resolution the balanced default.
            p.detail_level = 0.10f;
            p.spore_count = 25000;
            p.primary_roots = 3;
            p.search_radius = 16.0f;
            p.kill_radius = 2.6f;
            p.growth_step = 1.45f;
            p.variable_step = false;
            p.branching_rate = 0.58f;
            p.steering_cone = 35.0f;
            p.max_iterations = 45;
            p.trunk_caliber = 1.00f;
            p.tendril_caliber = 0.35f;
            p.taper_curve = 1.45f;
            p.algorithm_mode = 1;
            p.root_roughness = 0.28f;
            p.push_strength = 0.65f;
            p.space_fill_drive = 0.90f;
            p.backward_pull = 1.0f;
            list.push_back(p);
        }
        {
            GrowthPreset p;
            p.name = "Dense Mycelium";
            p.detail_level = 0.25f;
            p.spore_count = 45000;
            p.primary_roots = 4;
            p.search_radius = 16.0f;
            p.kill_radius = 1.2f;
            p.growth_step = 1.00f;
            p.variable_step = false;
            p.branching_rate = 0.95f;
            p.steering_cone = 45.0f;
            p.max_iterations = 50;
            p.trunk_caliber = 1.10f;
            p.tendril_caliber = 0.45f;
            p.taper_curve = 2.00f;
            p.algorithm_mode = 1;
            p.root_roughness = 0.35f;
            p.push_strength = 0.75f;
            p.space_fill_drive = 0.95f;
            p.lateral_density = 3.50f;
            p.lateral_max_length = 12.0f;
            list.push_back(p);
        }
        {
            GrowthPreset p;
            p.name = "Woody Trunk";
            p.detail_level = 0.40f;
            p.spore_count = 15000;
            p.primary_roots = 2;
            p.search_radius = 10.0f;
            p.kill_radius = 3.5f;
            p.growth_step = 2.20f;
            p.variable_step = false;
            p.branching_rate = 0.60f;
            p.steering_cone = 25.0f;
            p.max_iterations = 30;
            p.trunk_caliber = 2.20f;
            p.tendril_caliber = 0.90f;
            p.taper_curve = 1.20f;
            p.algorithm_mode = 1;
            p.root_roughness = 0.20f;
            p.push_strength = 0.50f;
            p.space_fill_drive = 0.80f;
            list.push_back(p);
        }
        {
            GrowthPreset p;
            p.name = "Delicate Tendrils";
            p.detail_level = 0.20f;
            p.spore_count = 35000;
            p.primary_roots = 3;
            p.search_radius = 20.0f;
            p.kill_radius = 1.5f;
            p.growth_step = 0.80f;
            p.variable_step = true;
            p.min_growth_step = 0.60f;
            p.max_growth_step = 1.80f;
            p.branching_rate = 0.90f;
            p.steering_cone = 40.0f;
            p.max_iterations = 45;
            p.trunk_caliber = 0.70f;
            p.tendril_caliber = 0.25f;
            p.taper_curve = 2.50f;
            p.algorithm_mode = 1;
            p.root_roughness = 0.40f;
            p.push_strength = 0.60f;
            p.space_fill_drive = 0.90f;
            p.lateral_density = 4.00f;
            p.lateral_max_length = 6.0f;
            list.push_back(p);
        }
        return list;
    }();
    load_growth_presets_from_ini(app_state.growth.growth_presets, "imgui.ini");
    // v24.01 Phase 0.4: viewport theme + lighting rig restore (optional
    // [ViewportColors] / [LightingAndShading] sections in imgui.ini, written
    // by the Styles panel's "Copy Layout INI" exporter; absent sections keep
    // the compiled defaults).
    load_viewport_style_from_ini(app_state.display, "imgui.ini");
    // v24.01 Phase 2.4: ImGuiStyle restore ([Style] / [StyleColors], written
    // by the Styles panel's "Copy Layout INI" exporter; absent sections keep
    // the compiled defaults). Runs after ImGui::CreateContext — the loader
    // writes into ImGui::GetStyle() directly.
    load_imgui_style_from_ini("imgui.ini");

    // Initialize Unified UI Interaction & Performance Monitoring System
    rhizome::init_ui_interaction_monitor("logs/ui_events.jsonl");

    // --test/-t: open the verification window & run the suite on startup
    app_state.verification.show_tests_window = run_tests_on_startup;

    // =========================================================================
    // GROWTH CONTROLLER (slim two-reference context, Phase 2 form)
    // =========================================================================
    GrowthControllerContext ctrl_ctx{
        app_state,
        res
    };

    auto run_all_tests = [&]() {
        run_all_tests_controller(ctrl_ctx);
    };

    if (run_tests_on_startup) {
        run_all_tests();
    }

    auto convert_and_update_model = [&](const rhizome::TriangleMesh& mesh, const std::string& name) {
        convert_and_update_model_controller(ctrl_ctx, mesh, name);
    };

    auto build_colonization_config = [&]() -> ColonizationConfig {
        return build_colonization_config_from_state(app_state, app_state.growth.growth_algorithms, app_state.seed.frozen);
    };

    auto update_implicit_session = [&](bool finalize_sdf_volume) {
        update_implicit_session_controller(ctrl_ctx, finalize_sdf_volume);
    };

    auto stop_live_growth = [&]() {
        stop_live_growth_controller(ctrl_ctx);
    };

    auto run_growth_engine = [&]() {
        run_growth_engine_controller(ctrl_ctx);
    };

    load_default_specimen_controller(ctrl_ctx);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        step_live_growth_frame_controller(ctrl_ctx, glfwGetTime());

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const ImVec2 work_pos = vp->WorkPos;
        const ImVec2 work_size = vp->WorkSize;

        // =========================================================================
        // 0. CENTRAL DOCKSPACE (The "Magnetic Floor" over Main Viewport)
        // =========================================================================
        const ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
        ImGuiID dockspace_id = ImGui::GetID("RhizomeDockSpace");
        dockspace_id = ImGui::DockSpaceOverViewport(dockspace_id, vp, dockspace_flags);

        static bool dockspace_initialized = false;
        ImGuiDockNode* central_node = ImGui::DockBuilderGetNode(dockspace_id);
        if (!dockspace_initialized || app_state.ui.force_reset_layout || central_node == nullptr || !central_node->IsSplitNode()) {
            dockspace_initialized = true;
            app_state.ui.force_reset_layout = false;

            ImGui::DockBuilderRemoveNode(dockspace_id);
            ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dockspace_id, work_size);

            ImGuiID dock_main_id = dockspace_id;
            ImGuiID dock_left = 0;
            ImGuiID dock_center_right = 0;

            // 1. Split Left Column (~298px / ratio ~0.155) for Specimen and Growth
            float left_ratio = 298.0f / (work_size.x > 0.0f ? work_size.x : 1920.0f);
            ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, left_ratio, &dock_left, &dock_center_right);

            // 2. Split Right Column (~299px / ratio ~0.185 of remaining)
            ImGuiID dock_center = 0;
            ImGuiID dock_right = 0;
            float right_ratio = 299.0f / (work_size.x > 298.0f ? (work_size.x - 298.0f) : 1620.0f);
            ImGui::DockBuilderSplitNode(dock_center_right, ImGuiDir_Right, right_ratio, &dock_right, &dock_center);

            // 3. Split Status Bar on top of center section (50px height)
            ImGuiID dock_up = 0;
            ImGuiID dock_viewport = 0;
            float top_ratio = 50.0f / (work_size.y > 0.0f ? work_size.y : 1009.0f);
            ImGui::DockBuilderSplitNode(dock_center, ImGuiDir_Up, top_ratio, &dock_up, &dock_viewport);

            // Dock windows into respective nodes matching the user layout
            ImGui::DockBuilderDockWindow("Status Bar", dock_up);
            ImGui::DockBuilderDockWindow("Toolbar", dock_up);
            ImGui::DockBuilderDockWindow("Status", dock_up);

            ImGui::DockBuilderDockWindow("Growth", dock_left);
            ImGui::DockBuilderDockWindow("Controls", dock_left);

            // Unified Scene panel in right dock column
            ImGui::DockBuilderDockWindow("Scene", dock_right);
            ImGui::DockBuilderDockWindow("Specimen", dock_right);
            ImGui::DockBuilderDockWindow("Display & Export", dock_right);

            ImGui::DockBuilderDockWindow("3D Viewport", dock_viewport);
            ImGui::DockBuilderDockWindow("Viewport", dock_viewport);
            ImGui::DockBuilderDockWindow("Styles", dock_viewport);
            ImGui::DockBuilderDockWindow("Style Editor", dock_viewport);
            ImGui::DockBuilderDockWindow("Presets", dock_viewport);

            ImGui::DockBuilderFinish(dockspace_id);
        }

        // =========================================================================
        // 1. GROWTH PANEL (Left Docked Panel with Command Bar Actions)
        // =========================================================================
        GrowthPanelContext growth_ctx{
            app_state,
            res,
            convert_and_update_model,
            run_growth_engine,
            stop_live_growth,
            update_implicit_session,
            build_colonization_config,
            apply_algorithm_profile,
            capture_current_into_algorithm_profile
        };
        render_growth_panel(growth_ctx);

        // =========================================================================
        // 3. SCENE PANEL (Unified Specimen, Visibility, Mesh Generation & Export)
        // =========================================================================
        ScenePanelContext scene_ctx{
            app_state,
            res,
            run_all_tests,
            convert_and_update_model,
            build_colonization_config
        };
        render_scene_panel(scene_ctx);

        // Safeguard: Ensure spore GPU buffer is uploaded whenever spores are active
        if (app_state.display.show_spores && res.spore_gl_buffers.vao == 0 && !res.live_colonizer.get_spores().empty()) {
            res.spore_gl_buffers.upload(res.live_colonizer.get_spores());
        }

        // =========================================================================
        // 5. VIEWPORT (Central 3D Viewport)
        // =========================================================================
        ViewportContext vp_ctx{
            vp_renderer,
            app_state,
            res,
            build_colonization_config,
            update_implicit_session
        };
        render_viewport_window(vp_ctx);

        // =========================================================================
        // 7. VERIFICATION (Independent, dockable Verification panel)
        // =========================================================================
        render_verification_panel(
            app_state,
            res,
            run_all_tests,
            convert_and_update_model,
            work_pos,
            work_size
        );

        // =========================================================================
        // 8. STYLES (Permanent interactive Style & Theme Customizer)
        // =========================================================================
        render_styles_panel(
            app_state,
            app_fonts,
            update_active_font,
            work_pos,
            work_size
        );

        // =========================================================================
        // 9. PRESETS (Permanent panel for Growth parameters & presets management)
        // =========================================================================
        render_presets_panel(
            app_state,
            convert_and_update_model,
            res.original_mesh,
            res.loaded_model_name,
            work_pos,
            work_size
        );

        // =========================================================================
        // 10. STATUS BAR (Dockable bottom status strip, quiet styling, zero padding)
        // =========================================================================
        size_t stat_secs = res.live_colonizer.get_volumetric_section_count();
        if (stat_secs == 0) stat_secs = res.live_colonizer.get_nodes().size();
        size_t stat_spores = res.live_colonizer.get_spores().size();
        render_status_bar_panel(app_state.ui.status_bar_message, stat_secs, stat_spores);

        // Render to Screen
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.094f, 0.098f, 0.106f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    res.orig_gl_buffers.cleanup();
    res.body_gl_buffers.cleanup();
    res.roots_gl_buffers.cleanup();
    res.merged_gl_buffers.cleanup();
    res.seed_sphere_buffer.cleanup();
    res.bbox_buffers.cleanup();
    res.floor_grid_buffers.cleanup();
    res.branch_gl_buffers.cleanup();
    res.spore_gl_buffers.cleanup();
    if (app_state.bark_texture.gl_preview_tex_id != 0) {
        glDeleteTextures(1, &app_state.bark_texture.gl_preview_tex_id);
        app_state.bark_texture.gl_preview_tex_id = 0;
    }
    for (auto& obj : res.test_objects) {
        obj.buffers.cleanup();
    }
    res.test_objects.clear();
    vp_renderer.cleanup();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

#ifdef _WIN32
    rhizome::shutdown_spacenav(window);
#endif

    // Cleanly flush and shutdown Unified UI Interaction Monitor
    rhizome::shutdown_ui_interaction_monitor();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
