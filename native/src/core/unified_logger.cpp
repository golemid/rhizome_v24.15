// =========================================================================
// RHIZOME UNIFIED DEBUG & MEMORY LOGGER — core module (v23.29 Phase 2)
// =========================================================================
// Migrated from the bottom section of src/app/ui_widgets.cpp (v23.28) into
// core/ per AGENTS.md Sec 2 and CODE_MAP.md Sec 3.9, fixing layering
// violations F1/F2: rhizome_core code (GPU benchmark runner, verification
// harness) may now log without depending on the ImGui application layer.
//
// Implementation contract (CODE_MAP.md Sec 3.9 "Do Not Break"):
//   * Zero-allocation ring buffer     -> fixed-size std::array storage with
//                                        snprintf-formatted fixed records.
//   * Millisecond timestamp precision -> captured at enqueue time; console
//                                        format "[HH:MM:SS.mmm] [CAT] msg"
//                                        and ISO-8601 JSON timestamps are
//                                        byte-identical to v23.28.
//   * Asynchronous file flushing      -> one background flusher thread owns
//                                        console + JSON-Lines emission;
//                                        flush_now()/close session act as
//                                        drain guards. Oversize messages
//                                        bypass synchronously to preserve
//                                        full fidelity.
//
// The 138-control catalog (param_controls_complete_list.csv) is compiled in
// as read-only data so ACTION events resolve CTRL IDs exactly as in v23.28.
// =========================================================================

#include "core/unified_logger.hpp"

#include "engine/sdf/cuda_rasterizer.hpp" // CudaRasterizer::query_gpu_memory

#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>

namespace rhizome {

// ---------------------------------------------------------------------
// 251-CONTROL CATALOG (authoritative source:
// param_controls_complete_list.csv — CTRL001..CTRL251; v24.01 Phase 1:
// ten lighting sliders at CTRL093-102, ex-093..138 slid +10 to CTRL103-148;
// v24.01 Phase 2: the 93 style-editor widgets appended at CTRL149-241
// (Sizes 23 / Colors ImGuiCol_COUNT / UI Rendering 7, panel_styles.cpp);
// v24.12 Track F R1: the ten developmental sliders appended at CTRL242-251
// (Developmental (Mode 4), panel_growth.cpp);
// Source_File column names the real panel home and is audited)
// ---------------------------------------------------------------------
struct ControlRegistryItem {
    const char* id;
    const char* file;
    const char* section;
    const char* label;
    const char* category;
    const char* unit;
};

static const ControlRegistryItem g_all_controls[251] = {
    {"CTRL001", "panel_growth.cpp", "Growth Command Bar", "Import", "COMMAND_01", ""},
    {"CTRL002", "panel_growth.cpp", "Growth Command Bar", "Start", "COMMAND_01", ""},
    {"CTRL003", "panel_growth.cpp", "Growth Command Bar", "Stop", "COMMAND_01", ""},
    {"CTRL004", "panel_growth.cpp", "Growth Command Bar", "Clear", "COMMAND_01", ""},
    {"CTRL005", "panel_growth.cpp", "Algorithm", "Algorithm Selector", "ALGORITHM_01", ""},
    {"CTRL006", "panel_growth.cpp", "Algorithm", "Export Algorithm", "ALGORITHM_01", ""},
    {"CTRL007", "panel_growth.cpp", "Algorithm", "Import Algorithm", "ALGORITHM_01", ""},
    {"CTRL008", "panel_growth.cpp", "Algorithm", "Delete Algorithm", "ALGORITHM_01", ""},
    {"CTRL009", "panel_growth.cpp", "Algorithm", "Root Roughness", "GROWTH_01", ""},
    {"CTRL010", "panel_growth.cpp", "Algorithm", "Push Strength", "GROWTH_01", ""},
    {"CTRL011", "panel_growth.cpp", "Algorithm", "Space Fill Drive", "GROWTH_01", ""},
    {"CTRL012", "panel_growth.cpp", "Algorithm", "Backward Pull", "GROWTH_01", ""},
    {"CTRL013", "panel_growth.cpp", "Algorithm", "Tensile Straightening", "GROWTH_01", ""},
    {"CTRL014", "panel_growth.cpp", "Algorithm", "Surface Adherence", "GROWTH_03", ""},
    {"CTRL015", "panel_growth.cpp", "Algorithm", "Murray Exponent", "GROWTH_02", "gamma"},
    {"CTRL016", "panel_growth.cpp", "Algorithm", "Murray Branch Angle", "GROWTH_02", "Degrees"},
    {"CTRL017", "panel_growth.cpp", "Colonization Seeds", "Spore Count", "ALGORITHM_01", "count"},
    {"CTRL018", "panel_growth.cpp", "Colonization Seeds", "Primary Roots", "ALGORITHM_01", "count"},
    {"CTRL019", "panel_growth.cpp", "Colonization Seeds", "Search Radius", "ALGORITHM_01", "mm"},
    {"CTRL020", "panel_growth.cpp", "Colonization Seeds", "Kill Radius", "ALGORITHM_01", "mm"},
    {"CTRL021", "panel_growth.cpp", "Branching & Growth", "Variable Step", "GROWTH_01", ""},
    {"CTRL022", "panel_growth.cpp", "Branching & Growth", "Min Growth Step", "GROWTH_01", "mm"},
    {"CTRL023", "panel_growth.cpp", "Branching & Growth", "Max Growth Step", "GROWTH_01", "mm"},
    {"CTRL024", "panel_growth.cpp", "Branching & Growth", "Growth Step Size", "GROWTH_01", "mm"},
    {"CTRL025", "panel_growth.cpp", "Branching & Growth", "Branching Rate", "GROWTH_01", ""},
    {"CTRL026", "panel_growth.cpp", "Branching & Growth", "Steering Cone", "GROWTH_01", "Degrees"},
    {"CTRL027", "panel_growth.cpp", "Branching & Growth", "Max Steps", "GROWTH_01", "count"},
    {"CTRL028", "panel_growth.cpp", "Vessel & Caliber", "Trunk Radius", "GROWTH_01", "mm"},
    {"CTRL029", "panel_growth.cpp", "Vessel & Caliber", "Tip Radius", "GROWTH_01", "mm"},
    {"CTRL030", "panel_growth.cpp", "Vessel & Caliber", "Variable Taper", "GROWTH_01", ""},
    {"CTRL031", "panel_growth.cpp", "Vessel & Caliber", "Taper Curve", "GROWTH_01", ""},
    {"CTRL032", "panel_growth.cpp", "Vessel & Caliber", "Branch Smoothness (K)", "GROWTH_04", "mm"},
    {"CTRL033", "panel_growth.cpp", "Vessel & Caliber", "Wall Clearance", "GROWTH_01", "x"},
    {"CTRL034", "panel_growth.cpp", "Vessel & Caliber", "Wall Cut Smooth", "GROWTH_01", "mm"},
    {"CTRL035", "panel_growth.cpp", "Lateral Roots", "Sprout Spacing", "LATERAL_01", "mm"},
    {"CTRL036", "panel_growth.cpp", "Lateral Roots", "Lateral Base", "LATERAL_01", "mm"},
    {"CTRL037", "panel_growth.cpp", "Lateral Roots", "Lateral Tip", "LATERAL_01", "mm"},
    {"CTRL038", "panel_growth.cpp", "Lateral Roots", "Max Reach", "LATERAL_01", "mm"},
    {"CTRL039", "panel_growth.cpp", "Lateral Roots", "Surface Wander", "LATERAL_01", ""},
    {"CTRL040", "panel_growth.cpp", "Lateral Roots", "Branch Angle", "LATERAL_01", "Degrees"},
    {"CTRL041", "panel_growth.cpp", "Seeding", "Seed Offset", "SEED_01", "mm"},
    {"CTRL042", "panel_growth.cpp", "Seeding", "Add Seed (+)", "SEED_01", ""},
    {"CTRL043", "panel_growth.cpp", "Seeding", "Remove Seed (-)", "SEED_01", ""},
    {"CTRL044", "panel_growth.cpp", "Seeding", "Freeze All Seeds", "SEED_01", ""},
    {"CTRL045", "panel_growth.cpp", "Seeding", "Unfreeze All Seeds", "SEED_01", ""},
    {"CTRL046", "panel_growth.cpp", "Seeding", "Select Seed (S1-Sn)", "SEED_01", "slot"},
    {"CTRL047", "panel_growth.cpp", "Seeding", "Freeze Seed", "SEED_01", ""},
    {"CTRL048", "panel_growth.cpp", "Seeding", "Snap to Root", "SEED_01", ""},
    {"CTRL049", "panel_growth.cpp", "Seeding", "Lateral Roots", "SEED_01", ""},
    {"CTRL050", "panel_growth.cpp", "Execution", "Live Animated Growth", "GROWTH_04", ""},
    {"CTRL051", "panel_growth.cpp", "Execution", "Step Rate (Live FPS)", "GROWTH_04", "fps"},
    {"CTRL052", "panel_growth.cpp", "Execution", "Steps Per Frame", "GROWTH_04", "count"},
    {"CTRL053", "panel_styles.cpp", "Growth Presets Library", "Growth Preset Selector", "PRESET_01", ""},
    {"CTRL054", "panel_styles.cpp", "Growth Presets Library", "Apply Preset", "PRESET_01", ""},
    {"CTRL055", "panel_styles.cpp", "Growth Presets Library", "Update Preset", "PRESET_01", ""},
    {"CTRL056", "panel_styles.cpp", "Growth Presets Library", "Delete Preset", "PRESET_01", ""},
    {"CTRL057", "panel_styles.cpp", "Record New Preset", "New Preset Name", "PRESET_01", "Text"},
    {"CTRL058", "panel_styles.cpp", "Record New Preset", "Save New Preset", "PRESET_01", ""},
    {"CTRL059", "panel_styles.cpp", "Record New Preset", "Export to Clipboard", "PRESET_01", ""},
    {"CTRL060", "panel_styles.cpp", "Color Themes & Presets", "Theme Preset Selector", "THEME_01", ""},
    {"CTRL061", "panel_styles.cpp", "Color Themes & Presets", "Reset Theme", "THEME_01", ""},
    {"CTRL062", "panel_styles.cpp", "Color Themes & Presets", "Save Style Reference", "THEME_01", ""},
    {"CTRL063", "panel_styles.cpp", "Color Themes & Presets", "Revert Style Reference", "THEME_01", ""},
    {"CTRL064", "panel_styles.cpp", "Color Themes & Presets", "Copy Style C++", "THEME_01", ""},
    {"CTRL065", "panel_styles.cpp", "Color Themes & Presets", "Copy Layout INI", "THEME_01", ""},
    {"CTRL066", "panel_styles.cpp", "Display Font / Typography", "Display Font Selector", "TYPO_01", ""},
    {"CTRL067", "panel_styles.cpp", "Display Font / Typography", "Bold Weight", "TYPO_01", ""},
    {"CTRL068", "panel_styles.cpp", "Display Font / Typography", "Font Scale", "TYPO_01", "x"},
    {"CTRL069", "panel_styles.cpp", "Display Font / Typography", "Item Spacing Y", "TYPO_01", "px"},
    {"CTRL070", "panel_styles.cpp", "Display Font / Typography", "Reset Typography Settings", "TYPO_01", ""},
    {"CTRL071", "panel_styles.cpp", "Viewport Element Colors", "Viewport Background Color", "COLOR_01", "RGBA"},
    {"CTRL072", "panel_styles.cpp", "Viewport Element Colors", "Floor Grid Color", "COLOR_01", "RGBA"},
    {"CTRL073", "panel_styles.cpp", "Viewport Element Colors", "Floor X-Axis Color", "COLOR_01", "RGBA"},
    {"CTRL074", "panel_styles.cpp", "Viewport Element Colors", "Floor Y-Axis Color", "COLOR_01", "RGBA"},
    {"CTRL075", "panel_styles.cpp", "Viewport Element Colors", "Original Mesh Color", "COLOR_01", "RGBA"},
    {"CTRL076", "panel_styles.cpp", "Viewport Element Colors", "Body / Glass Color", "COLOR_01", "RGBA"},
    {"CTRL077", "panel_styles.cpp", "Viewport Element Colors", "Root Network Color", "COLOR_01", "RGBA"},
    {"CTRL078", "panel_styles.cpp", "Viewport Element Colors", "Branch Skeleton Color", "COLOR_01", "RGBA"},
    {"CTRL079", "panel_styles.cpp", "Viewport Element Colors", "Spores Cloud Color", "COLOR_01", "RGBA"},
    {"CTRL080", "panel_styles.cpp", "Viewport Element Colors", "Active Seed Color", "COLOR_01", "RGBA"},
    {"CTRL081", "panel_styles.cpp", "Viewport Element Colors", "Inactive Seeds Color", "COLOR_01", "RGBA"},
    {"CTRL082", "panel_styles.cpp", "Viewport Element Colors", "Bounding Box Color", "COLOR_01", "RGBA"},
    {"CTRL083", "panel_styles.cpp", "Viewport Element Colors", "Wireframe Color", "COLOR_01", "RGBA"},
    {"CTRL084", "panel_styles.cpp", "Viewport Element Colors", "Test Pass Color", "COLOR_01", "RGBA"},
    {"CTRL085", "panel_styles.cpp", "Viewport Element Colors", "Test Fail Color", "COLOR_01", "RGBA"},
    {"CTRL086", "panel_styles.cpp", "Viewport Element Colors", "Reset Viewport Colors", "COLOR_01", ""},
    {"CTRL087", "panel_styles.cpp", "3D Lighting & Shading", "Diffuse Light", "SHADING_01", ""},
    {"CTRL088", "panel_styles.cpp", "3D Lighting & Shading", "Edge Darkening", "SHADING_01", ""},
    {"CTRL089", "panel_styles.cpp", "3D Lighting & Shading", "Crevice Occlusion", "SHADING_01", ""},
    {"CTRL090", "panel_styles.cpp", "3D Lighting & Shading", "Specular", "SHADING_01", ""},
    {"CTRL091", "panel_styles.cpp", "3D Lighting & Shading", "Bark Relief Shading", "SHADING_01", ""},
    {"CTRL092", "panel_styles.cpp", "3D Lighting & Shading", "Bark Relief Scale", "SHADING_01", ""},
    {"CTRL093", "panel_styles.cpp", "3D Lighting & Shading", "Light Azimuth", "SHADING_01", "Degrees"},
    {"CTRL094", "panel_styles.cpp", "3D Lighting & Shading", "Light Elevation", "SHADING_01", "Degrees"},
    {"CTRL095", "panel_styles.cpp", "3D Lighting & Shading", "Fill Light", "SHADING_01", ""},
    {"CTRL096", "panel_styles.cpp", "3D Lighting & Shading", "Ambient", "SHADING_01", ""},
    {"CTRL097", "panel_styles.cpp", "3D Lighting & Shading", "Glossiness", "SHADING_01", ""},
    {"CTRL098", "panel_styles.cpp", "3D Lighting & Shading", "Highlight Tint", "SHADING_01", ""},
    {"CTRL099", "panel_styles.cpp", "3D Lighting & Shading", "Fresnel Rim", "SHADING_01", ""},
    {"CTRL100", "panel_styles.cpp", "3D Lighting & Shading", "Volumetric AO (raymarch)", "SHADING_01", ""},
    {"CTRL101", "panel_styles.cpp", "3D Lighting & Shading", "SSS Strength (raymarch)", "SHADING_01", ""},
    {"CTRL102", "panel_styles.cpp", "3D Lighting & Shading", "SSS Warmth (raymarch)", "SHADING_01", ""},
    {"CTRL103", "panel_styles.cpp", "3D Lighting & Shading", "Reset 3D Lighting", "SHADING_01", ""},
    {"CTRL104", "panel_styles.cpp", "Specimen Appearance", "Matte Glass Specimen", "SHADING_02", ""},
    {"CTRL105", "panel_styles.cpp", "Specimen Appearance", "Glass Frosting", "SHADING_02", ""},
    {"CTRL106", "panel_styles.cpp", "Specimen Appearance", "Opacity", "SHADING_02", ""},
    {"CTRL107", "panel_scene.cpp", "Verification Suite", "Run All Tests", "TESTS_01", ""},
    {"CTRL108", "panel_scene.cpp", "Verification Suite", "Hide/View Log", "TESTS_02", ""},
    {"CTRL109", "panel_verification.cpp", "Verification Window", "Run Tests (Window)", "TESTS_01", ""},
    {"CTRL110", "panel_verification.cpp", "Verification Window", "Clear 3D Tests", "TESTS_01", ""},
    {"CTRL111", "panel_verification.cpp", "Verification Window", "Load Test Model to Viewport", "TESTS_01", "item"},
    {"CTRL112", "panel_scene.cpp", "Implicit Specimen", "Detail Level Override", "SPECIMEN_01", "mm"},
    {"CTRL113", "panel_scene.cpp", "Implicit Specimen", "Sphere (GPU Shape)", "SPECIMEN_01", ""},
    {"CTRL114", "panel_scene.cpp", "Implicit Specimen", "Torus (GPU Shape)", "SPECIMEN_01", ""},
    {"CTRL115", "panel_scene.cpp", "Implicit Specimen", "Box (GPU Shape)", "SPECIMEN_01", ""},
    {"CTRL116", "panel_scene.cpp", "Layers Visibility", "Original Mesh", "VISIBILITY_01", ""},
    {"CTRL117", "panel_scene.cpp", "Layers Visibility", "Converted Body", "VISIBILITY_01", ""},
    {"CTRL118", "panel_scene.cpp", "Layers Visibility", "Roots", "VISIBILITY_03", ""},
    {"CTRL119", "panel_scene.cpp", "Layers Visibility", "Colorize Roots", "VISIBILITY_03", ""},
    {"CTRL120", "panel_scene.cpp", "Layers Visibility", "Colorize Strands", "VISIBILITY_03", ""},
    {"CTRL121", "panel_scene.cpp", "Layers Visibility", "Branch Skeleton", "VISIBILITY_03", ""},
    {"CTRL122", "panel_scene.cpp", "Layers Visibility", "Seeds", "VISIBILITY_04", ""},
    {"CTRL123", "panel_scene.cpp", "Layers Visibility", "Spores", "VISIBILITY_04", ""},
    {"CTRL124", "panel_scene.cpp", "Layers Visibility", "Bounding Box", "VISIBILITY_05", ""},
    {"CTRL125", "panel_scene.cpp", "Layers Visibility", "Generated Mesh (Wireframe)", "VISIBILITY_05", ""},
    {"CTRL126", "panel_scene.cpp", "Layers Visibility", "Auto Orbit", "VISIBILITY_06", ""},
    {"CTRL127", "panel_scene.cpp", "Export Options", "Export Target", "EXPORT_01", ""},
    {"CTRL128", "panel_scene.cpp", "Export Options", "Adaptive Voxel Size", "MESH_01", ""},
    {"CTRL129", "panel_scene.cpp", "Export Options", "Mesh Voxel Size", "MESH_01", "mm"},
    {"CTRL130", "panel_scene.cpp", "Export Options", "Min Voxel Size", "MESH_01", "mm"},
    {"CTRL131", "panel_scene.cpp", "Export Options", "Max Voxel Size", "MESH_01", "mm"},
    {"CTRL132", "panel_scene.cpp", "Export Options", "Smoothing Passes", "MESH_02", "count"},
    {"CTRL133", "panel_scene.cpp", "Export Options", "Smooth Strength", "MESH_02", ""},
    {"CTRL134", "panel_scene.cpp", "Export Options", "Manifold Repair", "MESH_03", ""},
    {"CTRL135", "panel_scene.cpp", "Export Options", "Generate Mesh", "MESH_01", ""},
    {"CTRL136", "panel_scene.cpp", "Export Options", "Export Mesh (OBJ)", "EXPORT_02", ""},
    {"CTRL137", "panel_scene.cpp", "Export Options", "Export Binary STL", "EXPORT_02", ""},
    {"CTRL138", "panel_scene.cpp", "Export Options", "Export Binary PLY", "EXPORT_02", ""},
    {"CTRL139", "panel_scene.cpp", "Bark Texture", "Grain Scale", "SURFACE_01", ""},
    {"CTRL140", "panel_scene.cpp", "Bark Texture", "Vertical Stretch", "SURFACE_01", ""},
    {"CTRL141", "panel_scene.cpp", "Bark Texture", "Furrow Depth", "SURFACE_01", ""},
    {"CTRL142", "panel_scene.cpp", "Bark Texture", "Noise Octaves", "SURFACE_01", "count"},
    {"CTRL143", "panel_scene.cpp", "Bark Texture", "Generate 2D Bark Texture (GPU)", "SURFACE_01", ""},
    {"CTRL144", "panel_scene.cpp", "Bark Relief", "Displacement Depth", "SURFACE_02", "mm"},
    {"CTRL145", "panel_scene.cpp", "Bark Relief", "Tile Repeat U", "SURFACE_02", ""},
    {"CTRL146", "panel_scene.cpp", "Bark Relief", "Tile Repeat V", "SURFACE_02", ""},
    {"CTRL147", "panel_scene.cpp", "Bark Relief", "Recompute Normals", "SURFACE_02", ""},
    {"CTRL148", "panel_scene.cpp", "Bark Relief", "Apply Bark Relief to Mesh (GPU)", "SURFACE_02", ""},
    {"CTRL149", "panel_styles.cpp", "Style Editor: Sizes", "WindowPadding X", "STYLE_01", "px"},
    {"CTRL150", "panel_styles.cpp", "Style Editor: Sizes", "WindowPadding Y", "STYLE_01", "px"},
    {"CTRL151", "panel_styles.cpp", "Style Editor: Sizes", "FramePadding X", "STYLE_01", "px"},
    {"CTRL152", "panel_styles.cpp", "Style Editor: Sizes", "FramePadding Y", "STYLE_01", "px"},
    {"CTRL153", "panel_styles.cpp", "Style Editor: Sizes", "ItemSpacing X", "STYLE_01", "px"},
    {"CTRL154", "panel_styles.cpp", "Style Editor: Sizes", "ItemSpacing Y", "STYLE_01", "px"},
    {"CTRL155", "panel_styles.cpp", "Style Editor: Sizes", "InnerSpacing X", "STYLE_01", "px"},
    {"CTRL156", "panel_styles.cpp", "Style Editor: Sizes", "InnerSpacing Y", "STYLE_01", "px"},
    {"CTRL157", "panel_styles.cpp", "Style Editor: Sizes", "IndentSpacing", "STYLE_01", "px"},
    {"CTRL158", "panel_styles.cpp", "Style Editor: Sizes", "ScrollbarSize", "STYLE_01", "px"},
    {"CTRL159", "panel_styles.cpp", "Style Editor: Sizes", "GrabMinSize", "STYLE_01", "px"},
    {"CTRL160", "panel_styles.cpp", "Style Editor: Sizes", "WindowBorder", "STYLE_01", "px"},
    {"CTRL161", "panel_styles.cpp", "Style Editor: Sizes", "ChildBorder", "STYLE_01", "px"},
    {"CTRL162", "panel_styles.cpp", "Style Editor: Sizes", "PopupBorder", "STYLE_01", "px"},
    {"CTRL163", "panel_styles.cpp", "Style Editor: Sizes", "FrameBorder", "STYLE_01", "px"},
    {"CTRL164", "panel_styles.cpp", "Style Editor: Sizes", "TabBorder", "STYLE_01", "px"},
    {"CTRL165", "panel_styles.cpp", "Style Editor: Sizes", "WindowRounding", "STYLE_01", "px"},
    {"CTRL166", "panel_styles.cpp", "Style Editor: Sizes", "ChildRounding", "STYLE_01", "px"},
    {"CTRL167", "panel_styles.cpp", "Style Editor: Sizes", "FrameRounding", "STYLE_01", "px"},
    {"CTRL168", "panel_styles.cpp", "Style Editor: Sizes", "PopupRounding", "STYLE_01", "px"},
    {"CTRL169", "panel_styles.cpp", "Style Editor: Sizes", "ScrollbarRound", "STYLE_01", "px"},
    {"CTRL170", "panel_styles.cpp", "Style Editor: Sizes", "GrabRounding", "STYLE_01", "px"},
    {"CTRL171", "panel_styles.cpp", "Style Editor: Sizes", "TabRounding", "STYLE_01", "px"},
    {"CTRL172", "panel_styles.cpp", "Style Editor: Colors", "Text", "STYLE_02", "RGBA"},
    {"CTRL173", "panel_styles.cpp", "Style Editor: Colors", "TextDisabled", "STYLE_02", "RGBA"},
    {"CTRL174", "panel_styles.cpp", "Style Editor: Colors", "WindowBg", "STYLE_02", "RGBA"},
    {"CTRL175", "panel_styles.cpp", "Style Editor: Colors", "ChildBg", "STYLE_02", "RGBA"},
    {"CTRL176", "panel_styles.cpp", "Style Editor: Colors", "PopupBg", "STYLE_02", "RGBA"},
    {"CTRL177", "panel_styles.cpp", "Style Editor: Colors", "Border", "STYLE_02", "RGBA"},
    {"CTRL178", "panel_styles.cpp", "Style Editor: Colors", "BorderShadow", "STYLE_02", "RGBA"},
    {"CTRL179", "panel_styles.cpp", "Style Editor: Colors", "FrameBg", "STYLE_02", "RGBA"},
    {"CTRL180", "panel_styles.cpp", "Style Editor: Colors", "FrameBgHovered", "STYLE_02", "RGBA"},
    {"CTRL181", "panel_styles.cpp", "Style Editor: Colors", "FrameBgActive", "STYLE_02", "RGBA"},
    {"CTRL182", "panel_styles.cpp", "Style Editor: Colors", "TitleBg", "STYLE_02", "RGBA"},
    {"CTRL183", "panel_styles.cpp", "Style Editor: Colors", "TitleBgActive", "STYLE_02", "RGBA"},
    {"CTRL184", "panel_styles.cpp", "Style Editor: Colors", "TitleBgCollapsed", "STYLE_02", "RGBA"},
    {"CTRL185", "panel_styles.cpp", "Style Editor: Colors", "MenuBarBg", "STYLE_02", "RGBA"},
    {"CTRL186", "panel_styles.cpp", "Style Editor: Colors", "ScrollbarBg", "STYLE_02", "RGBA"},
    {"CTRL187", "panel_styles.cpp", "Style Editor: Colors", "ScrollbarGrab", "STYLE_02", "RGBA"},
    {"CTRL188", "panel_styles.cpp", "Style Editor: Colors", "ScrollbarGrabHovered", "STYLE_02", "RGBA"},
    {"CTRL189", "panel_styles.cpp", "Style Editor: Colors", "ScrollbarGrabActive", "STYLE_02", "RGBA"},
    {"CTRL190", "panel_styles.cpp", "Style Editor: Colors", "CheckMark", "STYLE_02", "RGBA"},
    {"CTRL191", "panel_styles.cpp", "Style Editor: Colors", "CheckboxSelectedBg", "STYLE_02", "RGBA"},
    {"CTRL192", "panel_styles.cpp", "Style Editor: Colors", "SliderGrab", "STYLE_02", "RGBA"},
    {"CTRL193", "panel_styles.cpp", "Style Editor: Colors", "SliderGrabActive", "STYLE_02", "RGBA"},
    {"CTRL194", "panel_styles.cpp", "Style Editor: Colors", "Button", "STYLE_02", "RGBA"},
    {"CTRL195", "panel_styles.cpp", "Style Editor: Colors", "ButtonHovered", "STYLE_02", "RGBA"},
    {"CTRL196", "panel_styles.cpp", "Style Editor: Colors", "ButtonActive", "STYLE_02", "RGBA"},
    {"CTRL197", "panel_styles.cpp", "Style Editor: Colors", "Header", "STYLE_02", "RGBA"},
    {"CTRL198", "panel_styles.cpp", "Style Editor: Colors", "HeaderHovered", "STYLE_02", "RGBA"},
    {"CTRL199", "panel_styles.cpp", "Style Editor: Colors", "HeaderActive", "STYLE_02", "RGBA"},
    {"CTRL200", "panel_styles.cpp", "Style Editor: Colors", "Separator", "STYLE_02", "RGBA"},
    {"CTRL201", "panel_styles.cpp", "Style Editor: Colors", "SeparatorHovered", "STYLE_02", "RGBA"},
    {"CTRL202", "panel_styles.cpp", "Style Editor: Colors", "SeparatorActive", "STYLE_02", "RGBA"},
    {"CTRL203", "panel_styles.cpp", "Style Editor: Colors", "ResizeGrip", "STYLE_02", "RGBA"},
    {"CTRL204", "panel_styles.cpp", "Style Editor: Colors", "ResizeGripHovered", "STYLE_02", "RGBA"},
    {"CTRL205", "panel_styles.cpp", "Style Editor: Colors", "ResizeGripActive", "STYLE_02", "RGBA"},
    {"CTRL206", "panel_styles.cpp", "Style Editor: Colors", "InputTextCursor", "STYLE_02", "RGBA"},
    {"CTRL207", "panel_styles.cpp", "Style Editor: Colors", "TabHovered", "STYLE_02", "RGBA"},
    {"CTRL208", "panel_styles.cpp", "Style Editor: Colors", "Tab", "STYLE_02", "RGBA"},
    {"CTRL209", "panel_styles.cpp", "Style Editor: Colors", "TabSelected", "STYLE_02", "RGBA"},
    {"CTRL210", "panel_styles.cpp", "Style Editor: Colors", "TabSelectedOverline", "STYLE_02", "RGBA"},
    {"CTRL211", "panel_styles.cpp", "Style Editor: Colors", "TabDimmed", "STYLE_02", "RGBA"},
    {"CTRL212", "panel_styles.cpp", "Style Editor: Colors", "TabDimmedSelected", "STYLE_02", "RGBA"},
    {"CTRL213", "panel_styles.cpp", "Style Editor: Colors", "TabDimmedSelectedOverline", "STYLE_02", "RGBA"},
    {"CTRL214", "panel_styles.cpp", "Style Editor: Colors", "DockingPreview", "STYLE_02", "RGBA"},
    {"CTRL215", "panel_styles.cpp", "Style Editor: Colors", "DockingEmptyBg", "STYLE_02", "RGBA"},
    {"CTRL216", "panel_styles.cpp", "Style Editor: Colors", "PlotLines", "STYLE_02", "RGBA"},
    {"CTRL217", "panel_styles.cpp", "Style Editor: Colors", "PlotLinesHovered", "STYLE_02", "RGBA"},
    {"CTRL218", "panel_styles.cpp", "Style Editor: Colors", "PlotHistogram", "STYLE_02", "RGBA"},
    {"CTRL219", "panel_styles.cpp", "Style Editor: Colors", "PlotHistogramHovered", "STYLE_02", "RGBA"},
    {"CTRL220", "panel_styles.cpp", "Style Editor: Colors", "TableHeaderBg", "STYLE_02", "RGBA"},
    {"CTRL221", "panel_styles.cpp", "Style Editor: Colors", "TableBorderStrong", "STYLE_02", "RGBA"},
    {"CTRL222", "panel_styles.cpp", "Style Editor: Colors", "TableBorderLight", "STYLE_02", "RGBA"},
    {"CTRL223", "panel_styles.cpp", "Style Editor: Colors", "TableRowBg", "STYLE_02", "RGBA"},
    {"CTRL224", "panel_styles.cpp", "Style Editor: Colors", "TableRowBgAlt", "STYLE_02", "RGBA"},
    {"CTRL225", "panel_styles.cpp", "Style Editor: Colors", "TextLink", "STYLE_02", "RGBA"},
    {"CTRL226", "panel_styles.cpp", "Style Editor: Colors", "TextSelectedBg", "STYLE_02", "RGBA"},
    {"CTRL227", "panel_styles.cpp", "Style Editor: Colors", "TreeLines", "STYLE_02", "RGBA"},
    {"CTRL228", "panel_styles.cpp", "Style Editor: Colors", "DragDropTarget", "STYLE_02", "RGBA"},
    {"CTRL229", "panel_styles.cpp", "Style Editor: Colors", "DragDropTargetBg", "STYLE_02", "RGBA"},
    {"CTRL230", "panel_styles.cpp", "Style Editor: Colors", "UnsavedMarker", "STYLE_02", "RGBA"},
    {"CTRL231", "panel_styles.cpp", "Style Editor: Colors", "NavCursor", "STYLE_02", "RGBA"},
    {"CTRL232", "panel_styles.cpp", "Style Editor: Colors", "NavWindowingHighlight", "STYLE_02", "RGBA"},
    {"CTRL233", "panel_styles.cpp", "Style Editor: Colors", "NavWindowingDimBg", "STYLE_02", "RGBA"},
    {"CTRL234", "panel_styles.cpp", "Style Editor: Colors", "ModalWindowDimBg", "STYLE_02", "RGBA"},
    {"CTRL235", "panel_styles.cpp", "Style Editor: UI Rendering", "Global Alpha", "STYLE_03", ""},
    {"CTRL236", "panel_styles.cpp", "Style Editor: UI Rendering", "Disabled Alpha", "STYLE_03", ""},
    {"CTRL237", "panel_styles.cpp", "Style Editor: UI Rendering", "Anti-aliased lines", "STYLE_03", ""},
    {"CTRL238", "panel_styles.cpp", "Style Editor: UI Rendering", "AA lines texture", "STYLE_03", ""},
    {"CTRL239", "panel_styles.cpp", "Style Editor: UI Rendering", "Anti-aliased fill", "STYLE_03", ""},
    {"CTRL240", "panel_styles.cpp", "Style Editor: UI Rendering", "Curve Tess Tol", "STYLE_03", ""},
    {"CTRL241", "panel_styles.cpp", "Style Editor: UI Rendering", "Circle Max Err", "STYLE_03", ""},
    // v24.12 Track F R1: developmental section (append-only terminal)
    {"CTRL242", "panel_growth.cpp", "Developmental (Mode 4)", "Framework GSA", "GROWTH_04", "deg"},
    {"CTRL243", "panel_growth.cpp", "Developmental (Mode 4)", "Lateral GSA", "GROWTH_04", "deg"},
    {"CTRL244", "panel_growth.cpp", "Developmental (Mode 4)", "Gravitropic Gain", "GROWTH_04", ""},
    {"CTRL245", "panel_growth.cpp", "Developmental (Mode 4)", "Wandering Noise", "GROWTH_04", ""},
    {"CTRL246", "panel_growth.cpp", "Developmental (Mode 4)", "Branch Angle", "GROWTH_04", "deg"},
    {"CTRL247", "panel_growth.cpp", "Developmental (Mode 4)", "Lateral Frequency", "GROWTH_04", "x"},
    {"CTRL248", "panel_growth.cpp", "Developmental (Mode 4)", "Vigor", "GROWTH_04", "x"},
    {"CTRL249", "panel_growth.cpp", "Developmental (Mode 4)", "Murray Gamma", "GROWTH_04", "gamma"},
    {"CTRL250", "panel_growth.cpp", "Developmental (Mode 4)", "Min Print Radius", "GROWTH_04", "mm"},
    {"CTRL251", "panel_growth.cpp", "Developmental (Mode 4)", "Section Budget", "GROWTH_04", "count"}
};

// =========================================================================
// Shared formatting helpers (verbatim from v23.28 ui_widgets.cpp)
// =========================================================================

static std::string escape_json_str(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (char c : input) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\b') out += "\\b";
        else if (c == '\f') out += "\\f";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (static_cast<unsigned char>(c) < 32) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
            out += buf;
        } else {
            out += c;
        }
    }
    return out;
}

static const ControlRegistryItem* find_control_by_label_or_id(const char* key) {
    if (!key || key[0] == '\0') return nullptr;
    std::string skey(key);

    // Direct ID match (e.g. "CTRL028")
    if (skey.rfind("CTRL", 0) == 0 && skey.size() == 7) {
        for (const auto& item : g_all_controls) {
            if (skey == item.id) return &item;
        }
    }

    // Exact label match
    for (const auto& item : g_all_controls) {
        if (skey == item.label) return &item;
    }

    // Substring or prefix matching for action log messages
    for (const auto& item : g_all_controls) {
        if (skey.find(item.label) != std::string::npos) {
            return &item;
        }
    }

    // Clean common prefixes ("Adjusted ", "Toggled ")
    std::string stripped = skey;
    if (stripped.rfind("Adjusted ", 0) == 0) stripped = stripped.substr(9);
    else if (stripped.rfind("Toggled ", 0) == 0) stripped = stripped.substr(8);

    for (const auto& item : g_all_controls) {
        if (stripped.find(item.label) != std::string::npos || item.label == stripped) {
            return &item;
        }
    }

    return nullptr;
}

static std::string get_iso_timestamp(const std::chrono::system_clock::time_point& tp) {
    auto time_t_now = std::chrono::system_clock::to_time_t(tp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()) % 1000;
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &time_t_now);
#else
    gmtime_r(&time_t_now, &tm_buf);
#endif
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
        tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
        tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec, static_cast<int>(ms.count()));
    return std::string(buf);
}

// =========================================================================
// UnifiedLogger — singleton ring-buffer core
// =========================================================================

UnifiedLogger& UnifiedLogger::instance() {
    static UnifiedLogger s_instance; // C++11 magic static: thread-safe init
    return s_instance;
}

UnifiedLogger::UnifiedLogger() {
    m_running.store(true, std::memory_order_release);
    m_flusher = std::thread(&UnifiedLogger::flusher_loop, this);
}

UnifiedLogger::~UnifiedLogger() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_running.store(false, std::memory_order_release);
    }
    m_cv.notify_all();
    if (m_flusher.joinable()) {
        m_flusher.join();
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    drain_locked();
    if (m_jsonl_file.is_open()) {
        m_jsonl_file.flush();
        m_jsonl_file.close();
    }
    m_jsonl_active.store(false, std::memory_order_release);
    if (m_dropped.load(std::memory_order_relaxed) > 0) {
        std::cout << "[LOGGER] " << m_dropped.load(std::memory_order_relaxed)
                  << " record(s) dropped due to ring buffer overflow\n";
    }
}

void UnifiedLogger::flusher_loop() {
    constexpr size_t kFlushBatch = 64;
    constexpr auto kFlushInterval = std::chrono::milliseconds(50);
    while (true) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait_for(lock, kFlushInterval, [&] {
            return !m_running.load(std::memory_order_acquire) || m_count >= kFlushBatch;
        });
        drain_locked();
        const bool stop = !m_running.load(std::memory_order_acquire) && m_count == 0;
        lock.unlock();
        if (stop) break;
    }
}

void UnifiedLogger::drain_locked() {
    // Requires m_mutex. Emits pending records in FIFO order.
    size_t tail = (m_head + kRingCapacity - m_count) % kRingCapacity;
    while (m_count > 0) {
        const Record& rec = m_ring[tail];

        if (rec.sink == Sink::ConsoleOnly || rec.sink == Sink::ConsoleAndJsonl) {
            // Console: "[HH:MM:SS.mmm] [CATEGORY] message" (local time)
            auto time_t_now = std::chrono::system_clock::to_time_t(rec.timestamp);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          rec.timestamp.time_since_epoch()) % 1000;
            std::tm tm_buf{};
#ifdef _WIN32
            localtime_s(&tm_buf, &time_t_now);
#else
            localtime_r(&time_t_now, &tm_buf);
#endif
            char time_str[32];
            std::snprintf(time_str, sizeof(time_str), "%02d:%02d:%02d.%03d",
                tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec, static_cast<int>(ms.count()));
            std::cout << "[" << time_str << "] [" << rec.category << "] "
                      << rec.console_line << "\n";
        }

        if (rec.sink == Sink::JsonlOnly || rec.sink == Sink::ConsoleAndJsonl) {
            if (m_jsonl_active.load(std::memory_order_relaxed) && m_jsonl_file.is_open()) {
                m_jsonl_file << rec.jsonl_line << "\n";
            }
        }

        tail = (tail + 1) % kRingCapacity;
        --m_count;
    }
    std::cout.flush();
    if (m_jsonl_file.is_open()) {
        m_jsonl_file.flush();
    }
}

void UnifiedLogger::enqueue(Record&& rec) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_count >= kRingCapacity) {
            // Overflow guard: drop the newest record, never block the caller.
            m_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        m_ring[m_head] = std::move(rec);
        m_head = (m_head + 1) % kRingCapacity;
        ++m_count;
    }
    m_cv.notify_one();
}

void UnifiedLogger::flush_now() {
    std::lock_guard<std::mutex> lock(m_mutex);
    drain_locked();
}

void UnifiedLogger::write_jsonl_direct(const char* json_line) {
    // Synchronous oversize bypass path (see file header).
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_jsonl_active.load(std::memory_order_relaxed) && m_jsonl_file.is_open()) {
        m_jsonl_file << json_line << "\n";
        m_jsonl_file.flush();
    }
}

void UnifiedLogger::open_jsonl_session(const char* file_path) {
    if (file_path && file_path[0] != '\0') {
        m_jsonl_path = file_path;
    }
    try {
        std::filesystem::path fp(m_jsonl_path);
        if (fp.has_parent_path()) {
            std::filesystem::create_directories(fp.parent_path());
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_jsonl_file.is_open()) {
            return; // idempotent
        }
        m_jsonl_file.open(m_jsonl_path, std::ios::out | std::ios::app);
        if (!m_jsonl_file.is_open()) {
            return;
        }
        m_jsonl_active.store(true, std::memory_order_release);

        auto now = std::chrono::system_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      now.time_since_epoch()).count();
        char line[256];
        std::snprintf(line, sizeof(line),
            "{\"timestamp_ns\":%lld,\"iso_time\":\"%s\",\"event_type\":\"SESSION_START\""
            ",\"registered_controls\":138"
            ",\"message\":\"Rhizome Unified UI Interaction Monitor started\"}",
            static_cast<long long>(ns), get_iso_timestamp(now).c_str());
        m_jsonl_file << line << "\n";
        m_jsonl_file.flush();
    } catch (...) {
        m_jsonl_active.store(false, std::memory_order_release);
    }
}

void UnifiedLogger::close_jsonl_session() {
    flush_now();
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_jsonl_file.is_open()) {
        auto now = std::chrono::system_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      now.time_since_epoch()).count();
        char line[256];
        std::snprintf(line, sizeof(line),
            "{\"timestamp_ns\":%lld,\"iso_time\":\"%s\",\"event_type\":\"SESSION_END\""
            ",\"message\":\"Rhizome Unified UI Interaction Monitor session ended\"}",
            static_cast<long long>(ns), get_iso_timestamp(now).c_str());
        m_jsonl_file << line << "\n";
        m_jsonl_file.flush();
        m_jsonl_file.close();
    }
    m_jsonl_active.store(false, std::memory_order_release);
}

// ---- CODE_MAP.md Sec 3.9 canonical API ----

void UnifiedLogger::log_info(const char* message) {
    RhizomeLog("INFO", message);
}

void UnifiedLogger::log_gpu_memory(const char* context) {
    RhizomeLogVramSnapshot(context);
}

void UnifiedLogger::log_step(const char* stage, double elapsed_ms) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s: %.3f ms",
                  stage ? stage : "STEP", elapsed_ms);
    RhizomeLog("STEP", buf);
}

void UnifiedLogger::log_interaction(const char* control_id_or_label, float old_val,
                                    float new_val, const char* details) {
    log_ui_interaction(control_id_or_label, old_val, new_val, details);
}

// =========================================================================
// FREE-FUNCTION COMPATIBILITY API (all v23.28 call sites unchanged)
// =========================================================================

GpuMemoryStats query_gpu_memory_stats() {
    GpuMemoryStats stats;
    size_t free_b = 0, total_b = 0;
    if (CudaRasterizer::query_gpu_memory(free_b, total_b) && total_b > 0) {
        stats.free_bytes = free_b;
        stats.total_bytes = total_b;
        stats.used_bytes = (total_b >= free_b) ? (total_b - free_b) : 0;
        stats.used_percent = (static_cast<float>(stats.used_bytes) / static_cast<float>(total_b)) * 100.0f;
        stats.valid = true;
    }
    return stats;
}

void RhizomeLog(const char* category, const char* message) {
    const char* cat = category ? category : "INFO";
    const char* msg = message ? message : "";
    UnifiedLogger& logger = UnifiedLogger::instance();

    // ---- Oversize bypass: emit synchronously, full text fidelity ----
    if (std::strlen(msg) >= UnifiedLogger::kConsoleChars) {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) % 1000;
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      now.time_since_epoch()).count();
        std::tm tm_buf{};
#ifdef _WIN32
        localtime_s(&tm_buf, &time_t_now);
#else
        localtime_r(&time_t_now, &tm_buf);
#endif
        char time_str[32];
        std::snprintf(time_str, sizeof(time_str), "%02d:%02d:%02d.%03d",
            tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec, static_cast<int>(ms.count()));
        std::cout << "[" << time_str << "] [" << cat << "] " << msg << std::endl;

        if (logger.jsonl_active()) {
            std::string json = "{\"timestamp_ns\":" + std::to_string(static_cast<long long>(ns)) +
                ",\"iso_time\":\"" + get_iso_timestamp(now) + "\"" +
                ",\"event_type\":\"" + std::string(cat) + "\"";
            const ControlRegistryItem* ctrl = nullptr;
            if (std::strcmp(cat, "ACTION") == 0) {
                ctrl = find_control_by_label_or_id(msg);
            }
            if (ctrl) {
                json += ",\"control_id\":\"" + std::string(ctrl->id) + "\""
                        ",\"label\":\"" + escape_json_str(ctrl->label) + "\"" +
                        ",\"section\":\"" + std::string(ctrl->section) + "\"" +
                        ",\"category\":\"" + std::string(ctrl->category) + "\"";
            }
            if (msg[0] != '\0') {
                json += ",\"message\":\"" + escape_json_str(msg) + "\"";
            }
            json += "}";
            logger.write_jsonl_direct(json.c_str());
        }
        return;
    }

    // ---- Regular path: fixed-size record into the zero-allocation ring ----
    UnifiedLogger::Record rec;
    rec.timestamp = std::chrono::system_clock::now();
    rec.sink = UnifiedLogger::Sink::ConsoleAndJsonl;
    std::snprintf(rec.category, sizeof(rec.category), "%s", cat);
    std::snprintf(rec.console_line, sizeof(rec.console_line), "%s", msg);

    if (logger.jsonl_active()) {
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      rec.timestamp.time_since_epoch()).count();
        const ControlRegistryItem* ctrl = nullptr;
        if (std::strcmp(cat, "ACTION") == 0) {
            ctrl = find_control_by_label_or_id(msg);
        }
        int n = std::snprintf(rec.jsonl_line, sizeof(rec.jsonl_line),
            "{\"timestamp_ns\":%lld,\"iso_time\":\"%s\",\"event_type\":\"%s\"",
            static_cast<long long>(ns), get_iso_timestamp(rec.timestamp).c_str(), cat);
        if (ctrl && n > 0 && n < static_cast<int>(sizeof(rec.jsonl_line))) {
            n += std::snprintf(rec.jsonl_line + n, sizeof(rec.jsonl_line) - n,
                ",\"control_id\":\"%s\",\"label\":\"%s\",\"section\":\"%s\",\"category\":\"%s\"",
                ctrl->id, escape_json_str(ctrl->label).c_str(), ctrl->section, ctrl->category);
        }
        if (msg[0] != '\0' && n > 0 && n < static_cast<int>(sizeof(rec.jsonl_line))) {
            n += std::snprintf(rec.jsonl_line + n, sizeof(rec.jsonl_line) - n,
                ",\"message\":\"%s\"}", escape_json_str(msg).c_str());
        } else if (n > 0 && n < static_cast<int>(sizeof(rec.jsonl_line))) {
            n += std::snprintf(rec.jsonl_line + n, sizeof(rec.jsonl_line) - n, "}");
        }
    } else {
        rec.sink = UnifiedLogger::Sink::ConsoleOnly;
    }

    logger.enqueue(std::move(rec));
}

void RhizomeLogMemory(const char* buffer_name, size_t count, size_t element_size_bytes) {
    size_t total_bytes = count * element_size_bytes;
    char size_buf[64];
    if (total_bytes >= 1024ULL * 1024ULL * 1024ULL) {
        std::snprintf(size_buf, sizeof(size_buf), "%.2f GB", static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0));
    } else if (total_bytes >= 1024ULL * 1024ULL) {
        std::snprintf(size_buf, sizeof(size_buf), "%.2f MB", static_cast<double>(total_bytes) / (1024.0 * 1024.0));
    } else if (total_bytes >= 1024ULL) {
        std::snprintf(size_buf, sizeof(size_buf), "%.1f KB", static_cast<double>(total_bytes) / 1024.0);
    } else {
        std::snprintf(size_buf, sizeof(size_buf), "%zu B", total_bytes);
    }

    std::string msg = std::string(buffer_name ? buffer_name : "Buffer") + ": " +
                      std::to_string(count) + " items (" + size_buf + ")";
    RhizomeLog("MEM:BUFFER", msg.c_str());
}

void RhizomeLogVramSnapshot(const char* context) {
    GpuMemoryStats stats = query_gpu_memory_stats();
    if (stats.valid) {
        double used_mb = static_cast<double>(stats.used_bytes) / (1024.0 * 1024.0);
        double total_mb = static_cast<double>(stats.total_bytes) / (1024.0 * 1024.0);
        double free_mb = static_cast<double>(stats.free_bytes) / (1024.0 * 1024.0);
        char buf[256];
        std::snprintf(buf, sizeof(buf), "Used: %.0f MB / %.0f MB (%.1f%%) | Free: %.0f MB%s%s",
            used_mb, total_mb, stats.used_percent, free_mb,
            context ? " [" : "", context ? context : "");
        if (context) std::strcat(buf, "]");
        RhizomeLog("MEM:VRAM", buf);
    }
}

void init_ui_interaction_monitor(const char* log_file_path) {
    UnifiedLogger::instance().open_jsonl_session(log_file_path);
}

void shutdown_ui_interaction_monitor() {
    UnifiedLogger::instance().close_jsonl_session();
}

void log_ui_interaction(const char* control_id_or_label, float old_val, float new_val,
                        const char* details) {
    UnifiedLogger& logger = UnifiedLogger::instance();
    if (!logger.jsonl_active()) return; // v23.28 parity: no monitor session -> no event

    const ControlRegistryItem* ctrl = find_control_by_label_or_id(control_id_or_label);
    auto now = std::chrono::system_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  now.time_since_epoch()).count();

    char line[UnifiedLogger::kJsonlChars];
    int n = std::snprintf(line, sizeof(line),
        "{\"timestamp_ns\":%lld,\"iso_time\":\"%s\",\"event_type\":\"ACTION\""
        ",\"control_id\":\"%s\",\"label\":\"%s\",\"section\":\"%s\",\"category\":\"%s\""
        ",\"old_value\":%.6g,\"new_value\":%.6g",
        static_cast<long long>(ns), get_iso_timestamp(now).c_str(),
        ctrl ? ctrl->id : "CTRL_UNKNOWN",
        escape_json_str(ctrl ? ctrl->label : (control_id_or_label ? control_id_or_label : "")).c_str(),
        ctrl ? ctrl->section : "",
        ctrl ? ctrl->category : "",
        old_val, new_val);
    if (n > 0 && n < static_cast<int>(sizeof(line))) {
        if (ctrl && ctrl->unit && ctrl->unit[0] != '\0') {
            n += std::snprintf(line + n, sizeof(line) - n, ",\"unit\":\"%s\"", ctrl->unit);
        }
        if (details && details[0] != '\0') {
            n += std::snprintf(line + n, sizeof(line) - n, ",\"details\":\"%s\"",
                               escape_json_str(details).c_str());
        }
        if (n > 0 && n < static_cast<int>(sizeof(line))) {
            std::snprintf(line + n, sizeof(line) - n, "}");
        }
    }

    UnifiedLogger::Record rec;
    rec.timestamp = now;
    rec.sink = UnifiedLogger::Sink::JsonlOnly;
    std::snprintf(rec.category, sizeof(rec.category), "ACTION");
    std::snprintf(rec.jsonl_line, sizeof(rec.jsonl_line), "%s", line);
    logger.enqueue(std::move(rec));
}

void log_ui_reaction(const char* control_id_or_label, int64_t elapsed_ms, float gpu_ms,
                     size_t tris_before, size_t tris_after, const char* message) {
    UnifiedLogger& logger = UnifiedLogger::instance();
    if (!logger.jsonl_active()) return;

    const ControlRegistryItem* ctrl = find_control_by_label_or_id(control_id_or_label);
    GpuMemoryStats vram = query_gpu_memory_stats();
    auto now = std::chrono::system_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  now.time_since_epoch()).count();

    char line[UnifiedLogger::kJsonlChars];
    int n = std::snprintf(line, sizeof(line),
        "{\"timestamp_ns\":%lld,\"iso_time\":\"%s\",\"event_type\":\"REACTION\""
        ",\"control_id\":\"%s\",\"elapsed_ms\":%lld,\"gpu_ms\":%.6g"
        ",\"triangles_before\":%zu,\"triangles_after\":%zu,\"geometry_delta\":%lld",
        static_cast<long long>(ns), get_iso_timestamp(now).c_str(),
        ctrl ? ctrl->id : (control_id_or_label ? control_id_or_label : ""),
        static_cast<long long>(elapsed_ms), gpu_ms,
        tris_before, tris_after,
        static_cast<long long>(static_cast<int64_t>(tris_after) - static_cast<int64_t>(tris_before)));
    if (vram.valid && n > 0 && n < static_cast<int>(sizeof(line))) {
        double used_gb = static_cast<double>(vram.used_bytes) / (1024.0 * 1024.0 * 1024.0);
        double total_gb = static_cast<double>(vram.total_bytes) / (1024.0 * 1024.0 * 1024.0);
        n += std::snprintf(line + n, sizeof(line) - n,
            ",\"vram_used_gb\":%.6g,\"vram_total_gb\":%.6g,\"vram_percent\":%.6g",
            used_gb, total_gb, vram.used_percent);
    }
    if (message && message[0] != '\0' && n > 0 && n < static_cast<int>(sizeof(line))) {
        n += std::snprintf(line + n, sizeof(line) - n, ",\"message\":\"%s\"",
                           escape_json_str(message).c_str());
    }
    if (n > 0 && n < static_cast<int>(sizeof(line))) {
        std::snprintf(line + n, sizeof(line) - n, "}");
    }

    UnifiedLogger::Record rec;
    rec.timestamp = now;
    rec.sink = UnifiedLogger::Sink::JsonlOnly;
    std::snprintf(rec.category, sizeof(rec.category), "REACTION");
    std::snprintf(rec.jsonl_line, sizeof(rec.jsonl_line), "%s", line);
    logger.enqueue(std::move(rec));
}

} // namespace rhizome
