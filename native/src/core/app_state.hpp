#pragma once

// =========================================================================
// RHIZOME APP STATE — core module (v23.29 Phase 2)
// =========================================================================
// Single Source of Truth for ALL persistable application state
// (AGENTS.md Sec A.3 / CODE_MAP.md Sec 3.2). Migrated from
// include/app/app_state.hpp; preset persistence logic lives in
// core/algorithm_io.cpp (save/load_growth_presets_from_ini) and the pure
// GrowthState <-> GrowthPreset transfer helpers live in core/app_state.cpp.
//
// 138-CONTROL CONTRACT: every value control in
// param_controls_complete_list.csv (CTRL001-CTRL138) resolves to a member
// of this struct (growth.* / seed.* / display.* / mesh_export.* /
// bark_texture.* / verification.* / ui.*). The full binding table is
// materialized as ControlDefinition data in Phase 4 (ui/widgets/
// control_registry). Action controls (buttons/dialogs) bind to handlers,
// not state.
// =========================================================================

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include "core/algorithm_io.hpp"

namespace rhizome {

struct GrowthState {
    // Voxel Resolution / Specimen detail level (mm)
    // v24.04 Track D: default 0.30 -> 0.10 mm. The Fast SDF path in
    // CudaRasterizer::compute_mesh_sdf (band splat + parity signs + eikonal
    // re-distancing) makes 0.10 mm interactive on mug-class specimens
    // (~66 s -> ~1 s on a 12 GB GPU); VRAM headroom is ample (16% used).
    float detail_level = 0.10f;

    // Spores & Colonization Search/Step
    int spore_count = 25000;
    float search_radius = 16.0f;
    float growth_step = 1.45f;
    bool variable_step = false;
    float min_growth_step = 0.65f;
    float max_growth_step = 2.25f;
    float branching_rate = 0.58f;
    float steering_cone = 35.0f;
    int max_iterations = 45;
    float kill_radius = 2.6f;

    // Caliber & Taper
    float trunk_caliber = 1.00f;
    float tendril_caliber = 0.35f;
    bool variable_taper = false;
    float taper_curve = 1.45f;
    float branch_smooth_k = 0.00f;
    float wall_clearance_scale = 0.85f;
    float wall_cut_smooth = 0.00f;
    float seed_offset = 0.50f;

    // Lateral Roots Settings
    float lateral_density = 5.00f;          // Spacing in mm between lateral rootlets
    float lateral_trunk_caliber = 0.60f;    // Base micro-diameter in mm
    float lateral_tip_caliber = 0.40f;      // Fragile tip diameter in mm
    float lateral_max_length = 8.0f;        // Max reach in mm
    float lateral_wander = 0.35f;           // Organic curl/meander along or away from surface
    float lateral_branch_angle = 45.0f;     // Outward angle in degrees when not surface-snapped

    // Growth Algorithm Mode & Physical Drivers
    int algorithm_mode = 1; // 0 = Corals (SCA), 1 = Roots (Root-Bound), 2 = Vascular Murray, 3 = Creeping Binder, 4 = Root Typ (Developmental FSPM, v24.12 Track F R1)
    int primary_roots = 3;  // Number of primary root arms spawned per seed [1..8]
    float root_roughness = 0.28f;
    float push_strength = 0.65f;
    float space_fill_drive = 0.90f;
    float backward_pull = 1.0f;
    float tensile_straightening = 0.0f;
    float murray_exponent = 3.0f;
    float murray_branch_angle = 37.5f;
    float surface_adherence = 0.0f; // [0..1] Attraction to and crawling along frozen root surfaces

    // v24.12 Track F R1: Developmental mode (algorithm_mode == 4) user
    // surface. The full per-order table is derived in build_colonization_
    // config_from_state (GSA[2]/[3] from dev_gsa_lateral; rate[] scaled by
    // dev_vigor; spacing[] divided by dev_lateral_freq). Registered as
    // CTRL242-CTRL251. Not carried by GrowthPreset round-trips in v1
    // (documented in TRACK_F_R1_NOTE.md Sec 6).
    // v24.13 Track F R2: defaults retuned through the structure-dump
    // analyzer loop (mock + eye reports, scripts/eye) — flaring GSA
    // hierarchy, stronger setpoint chase, sparser cascade. See
    // TRACK_F_R2_NOTE.md.
    float dev_gsa_framework = 55.0f;   // order-0 gravitropic setpoint (deg from downward vertical)
    float dev_gsa_lateral = 68.0f;     // order-1 setpoint; orders 2/3 derive +10/+18 deg (clamped 89)
    float dev_grav_gain = 0.26f;       // Firn-Digby gravitropic response gain [0.04..0.50]
    float dev_noise = 0.10f;           // directional wander amplitude (rad) [0..0.30] (order-0 damped 0.45x in-kernel)
    float dev_branch_angle = 55.0f;    // lateral departure angle from the parent axis (deg)
    float dev_lateral_freq = 1.0f;     // lateral spawn frequency multiplier (scales 1/spacing) [0.5..2.0]
    float dev_vigor = 1.0f;            // elongation rate scale across all orders [0.5..1.5]
    float dev_murray_gamma = 2.80f;    // Murray caliber exponent at bake [2.3..3.2]
    float dev_min_print_radius = 0.30f; // SLA printable floor radius (mm) [0.15..0.80]
    int dev_section_budget = 30000;    // hard section cap (bake-size guard) [10000..60000]

    // v24.13 Track F R2: internal (NOT a registered control — invisible to
    // the registry/catalog/CSV audit): largest vessel bbox extent in mm,
    // measured from the vessel SDF grid at Start. Drives the vessel-relative
    // per-order length caps (dev_len_frac x this) in mode 4 so the same
    // profile scales across specimen sizes. 0 = unknown (static fallback).
    float dev_vessel_ref_mm = 0.0f;

    // Live growth simulation controls
    // v24.09 Track E: live_growth defaults OFF (was true). Live mode steps
    // the session every frame at live_fps and re-syncs the implicit session
    // per tick (all accumulated sections -> branch nodes + GL re-upload,
    // live capsule render session held open) — the memory cost measured on
    // target hardware (VRAM 6,472 MB, 79-iteration live session). Default
    // Start path = one-shot batch bake; the Execution toggle re-enables.
    bool live_growth = false;
    int live_steps_per_frame = 1;
    int live_fps = 8;

    // Simple scalar selector indices
    int selected_algorithm_idx = 1;
    int selected_growth_preset_idx = 0;

    // Live growth session flags & timing
    double last_live_step_time = 0.0;
    bool is_growing_live = false;
    bool is_growth_paused = false;
    bool has_live_session = false;

    // Algorithm & Preset collections
    std::vector<GrowthAlgorithmProfile> growth_algorithms;
    std::vector<GrowthPreset> growth_presets;
};

struct SeedState {
    int selected_seed_index = 0;
    bool is_dragging_seed = false;
    int dragged_seed_index = -1;
    bool show_seeds = false;
    std::vector<glm::vec3> points;
    std::vector<uint8_t> frozen;
    std::vector<uint8_t> snapped_to_root;
    std::vector<uint8_t> has_lateral;
    std::vector<float> radii;
};

struct DisplayState {
    bool show_original_mesh = false;
    bool show_converted_body = true;
    bool specimen_transparent = false;
    float specimen_opacity = 0.40f;
    bool show_roots = false;
    bool colorize_roots = false;
    bool colorize_strands = false;
    bool show_branch_skeleton = false;
    bool show_spores = false;
    // (F9, resolved in Phase 4: a second, never-written show_seeds flag lived
    //  here in v23.28 and was OR-combined with SeedState::show_seeds in the
    //  viewport. CTRL112 drives SeedState::show_seeds — the single home.)
    bool show_bounding_box = false;
    bool show_wireframe = false;
    bool auto_orbit = false;

    // 3D Shading & Lighting parameters
    float param_glass_frosting = 0.60f;
    float param_shader_diffuse = 1.00f;
    float param_shader_edge_darkening = 0.55f;
    float param_shader_crevice_ao = 0.50f;
    float param_shader_specular = 0.12f;
    float param_shader_bark_strength = 0.35f;
    float param_shader_bark_frequency = 0.40f;

    // v24.01 Phase 1 (lighting expansion, 6 -> 16 sliders): every constant
    // below was hardcoded in the GL mesh shader (viewport_renderer.cpp)
    // and/or the CUDA raymarcher (cuda_viewport_gl.cu); the defaults
    // reproduce the v23.29 compiled-in look (key/fill direction derivation
    // in core/lighting_rig.hpp). "(raymarch)" fields affect only the CUDA
    // raymarch path — the GL mesh path uses screen-space crevice occlusion
    // instead (V2401_PROPOSAL.md Sec 2.1).
    float param_light_azimuth = -13.6f;   // deg, view-space key light; was (-0.22, 0.34, 0.91)
    float param_light_elevation = 20.0f;  // deg
    float param_fill_light = 0.16f;       // fill weight in baseIllum = ambient + 0.62*key + fill*fillDot
    float param_ambient = 0.32f;          // ambient floor in the same expression
    float param_glossiness = 24.0f;       // specular exponent pow(., gloss)
    float param_highlight_tint = 0.55f;   // specColor mix toward white
    float param_fresnel_rim = 0.30f;      // matte-glass fresnel rim gain
    float param_volumetric_ao = 0.85f;    // (raymarch) AO gain (roots site)
    float param_sss_strength = 0.45f;     // (raymarch) subsurface transmittance
    float param_sss_warmth = 0.35f;       // (raymarch) SSS warm tint mix

    // Customizable 3D Element Colors
    glm::vec4 color_viewport_bg = glm::vec4(0.027f, 0.067f, 0.110f, 1.0f);
    glm::vec4 color_floor_grid = glm::vec4(0.149f, 0.220f, 0.290f, 0.50f);
    glm::vec4 color_floor_axis_x = glm::vec4(0.70f, 0.35f, 0.35f, 0.85f);
    glm::vec4 color_floor_axis_y = glm::vec4(0.35f, 0.70f, 0.45f, 0.85f);
    glm::vec4 color_original_mesh = glm::vec4(0.25f, 0.75f, 0.90f, 0.90f);
    glm::vec4 color_converted_body = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    glm::vec4 color_roots = glm::vec4(0.098f, 0.827f, 0.635f, 1.00f);
    glm::vec4 color_branch_skeleton = glm::vec4(1.0f, 0.54f, 0.12f, 1.0f);
    glm::vec4 color_spores = glm::vec4(0.25f, 0.95f, 0.72f, 0.78f);
    glm::vec4 color_seed_selected = glm::vec4(1.0f, 0.72f, 0.188f, 1.00f);
    glm::vec4 color_seed_unselected = glm::vec4(0.0f, 0.898f, 1.0f, 1.00f);
    glm::vec4 color_bounding_box = glm::vec4(0.85f, 0.75f, 0.35f, 0.70f);
    glm::vec4 color_wireframe = glm::vec4(0.35f, 0.22f, 0.08f, 0.90f);
    glm::vec4 color_test_pass = glm::vec4(0.15f, 0.85f, 0.55f, 1.0f);
    glm::vec4 color_test_fail = glm::vec4(0.90f, 0.30f, 0.25f, 1.0f);

    void reset_viewport_colors_to_default() {
        const DisplayState def{};
        color_viewport_bg = def.color_viewport_bg;
        color_floor_grid = def.color_floor_grid;
        color_floor_axis_x = def.color_floor_axis_x;
        color_floor_axis_y = def.color_floor_axis_y;
        color_original_mesh = def.color_original_mesh;
        color_converted_body = def.color_converted_body;
        color_roots = def.color_roots;
        color_branch_skeleton = def.color_branch_skeleton;
        color_spores = def.color_spores;
        color_seed_selected = def.color_seed_selected;
        color_seed_unselected = def.color_seed_unselected;
        color_bounding_box = def.color_bounding_box;
        color_wireframe = def.color_wireframe;
        color_test_pass = def.color_test_pass;
        color_test_fail = def.color_test_fail;
    }

    void reset_lighting_to_default() {
        const DisplayState def{};
        param_glass_frosting = def.param_glass_frosting;
        param_shader_diffuse = def.param_shader_diffuse;
        param_shader_edge_darkening = def.param_shader_edge_darkening;
        param_shader_crevice_ao = def.param_shader_crevice_ao;
        param_shader_specular = def.param_shader_specular;
        param_shader_bark_strength = def.param_shader_bark_strength;
        param_shader_bark_frequency = def.param_shader_bark_frequency;
        // v24.01 Phase 1 additions (proposal item 1.5)
        param_light_azimuth = def.param_light_azimuth;
        param_light_elevation = def.param_light_elevation;
        param_fill_light = def.param_fill_light;
        param_ambient = def.param_ambient;
        param_glossiness = def.param_glossiness;
        param_highlight_tint = def.param_highlight_tint;
        param_fresnel_rim = def.param_fresnel_rim;
        param_volumetric_ao = def.param_volumetric_ao;
        param_sss_strength = def.param_sss_strength;
        param_sss_warmth = def.param_sss_warmth;
    }
};

struct ViewportInputState {
    bool is_orthographic = true;
    bool is_dragging_orbit = false;
    bool is_dragging_pan = false;
};

struct SpecimenInfoState {
    std::string loaded_model_name = "Default Shape";
    std::string specimen_source_name = "mug.obj";
    std::string specimen_source_format = "OBJ";
    glm::vec3 specimen_dimensions = glm::vec3(0.0f);
    size_t specimen_triangle_count = 0;
    std::string specimen_implicit_status = "Not Built";
    double specimen_build_time_ms = 0.0;
    float specimen_interior_volume_ml = 0.0f;
    bool specimen_shell_valid = true;
    std::string specimen_shell_valid_msg = "Pass (Manifold)";
    float specimen_extent = 100.0f;
};

struct MeshExportState {
    int param_export_target = 0; // 0: Roots Only, 1: Converted Vessel, 2: Merged
    float param_mesh_voxel_size = 0.25f;
    bool param_adaptive_mesh_voxel = false;
    float param_min_mesh_voxel_size = 0.05f;
    float param_max_mesh_voxel_size = 0.60f;
    int param_mesh_smooth_passes = 2;
    float param_mesh_smooth_strength = 0.35f;
    bool param_mesh_manifold_repair = true;

    bool has_generated_roots_mesh = false;
    bool has_generated_vessel_mesh = false;
    bool has_generated_merged_mesh = false;
};

struct BarkTextureState {
    int tex_width = 512;
    int tex_height = 512;
    float scale = 6.0f;
    float stretch_y = 3.5f;
    float furrow_depth = 0.65f;
    int octaves = 4;
    float persistence = 0.5f;
    uint32_t seed = 1337;

    float displacement_depth_mm = 1.50f;
    float tile_repeat_u = 4.0f;
    float tile_repeat_v = 8.0f;
    bool recompute_normals = true;

    bool has_generated_texture = false;
    std::vector<float> heightmap;
    std::vector<uint32_t> rgba_preview;
    unsigned int gl_preview_tex_id = 0;
    float last_gen_time_ms = 0.0f;
    float last_disp_time_ms = 0.0f;
};

struct VerificationUIState {
    bool show_tests_window = false;
    bool show_style_editor = true;
    bool tests_running = false;
};

struct UiThemeMetrics {
    float font_scale = 1.0f;
    float item_spacing_y = 4.0f;
    float window_rounding = 6.0f;
    float child_rounding = 6.0f;
    float frame_rounding = 4.0f;
    float popup_rounding = 6.0f;
    float scrollbar_rounding = 6.0f;
    float grab_rounding = 6.0f;
    float tab_rounding = 6.0f;
    float window_border_size = 1.0f;
    float child_border_size = 1.0f;
    float frame_border_size = 0.0f;
    float popup_border_size = 1.0f;
    float tab_border_size = 0.0f;
    glm::vec2 window_padding = glm::vec2(10.0f, 6.0f);
    glm::vec2 frame_padding = glm::vec2(8.0f, 3.0f);
    glm::vec2 item_spacing = glm::vec2(8.0f, 4.0f);
    glm::vec2 item_inner_spacing = glm::vec2(6.0f, 4.0f);
    float indent_spacing = 16.0f;
    float scrollbar_size = 10.0f;
    float grab_min_size = 12.0f;
    float alpha = 1.0f;
    float disabled_alpha = 0.60f;
};

struct AppUIState {
    std::string status_bar_message = "Ready. Click Import to load a 3D model.";
    bool force_reset_layout = false;
    int current_theme_preset = 0;
    int active_font_idx = 5;
    bool font_bold = true;
    UiThemeMetrics theme_metrics;

    // CTRL057 "New Preset Name" text input buffer (Growth Presets Library).
    // Promoted from a function-local static in ui_panels.cpp into AppState
    // in Phase 2 (shadow-state eradication, AGENTS.md Sec A.3). The presets
    // panel rewires to this member in Phase 4.
    char new_preset_name[64] = "Custom Growth";
};

struct AppState {
    GrowthState growth;
    SeedState seed;
    DisplayState display;
    ViewportInputState viewport;
    SpecimenInfoState specimen;
    MeshExportState mesh_export;
    BarkTextureState bark_texture;
    VerificationUIState verification;
    AppUIState ui;

    void reset_all_styles_to_default() {
        display.reset_viewport_colors_to_default();
        display.reset_lighting_to_default();
        const AppUIState def_ui{};
        ui.current_theme_preset = def_ui.current_theme_preset;
        ui.active_font_idx = def_ui.active_font_idx;
        ui.font_bold = def_ui.font_bold;
    }
};

// ---- Pure state-logic helpers (definitions in core/app_state.cpp) ----
// Used by the Growth Presets Library panel (CTRL053-CTRL059). INI
// round-trips remain in core/algorithm_io.cpp ("Do Not Break" per
// CODE_MAP.md Sec 3.2: load_growth_presets_from_ini).
void apply_growth_preset_to_state(GrowthState& g, const GrowthPreset& p);
void capture_growth_preset_from_state(const GrowthState& g, GrowthPreset& p);

} // namespace rhizome
