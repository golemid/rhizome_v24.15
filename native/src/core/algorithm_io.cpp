// v23.29 Phase 2: migrated verbatim from src/app/algorithm_io.cpp into core/.
// Owns the "Do Not Break" preset persistence logic (CODE_MAP.md Sec 3.2):
// save_growth_presets_to_ini / load_growth_presets_from_ini (imgui.ini).
// v24.01 Phase 0.4: adds load_viewport_style_from_ini — the read side of the
// [ViewportColors] / [LightingAndShading] theme sections written by the
// Styles panel's "Copy Layout INI" exporter.
#include "core/algorithm_io.hpp"
#include "core/app_state.hpp"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstdio>

void save_growth_presets_to_ini(const std::vector<GrowthPreset>& presets, const std::string& filepath) {
    std::string content;
    std::error_code ec_ex;
    if (std::filesystem::exists(filepath, ec_ex) && !ec_ex) {
        std::ifstream in(filepath, std::ios::in | std::ios::binary);
        if (in.is_open()) {
            std::ostringstream sstr;
            sstr << in.rdbuf();
            content = sstr.str();
            in.close();
        }
    }
    size_t pos = content.find("[GrowthPresets]");
    if (pos != std::string::npos) {
        size_t next_sec = content.find("\n[", pos + 15);
        if (next_sec != std::string::npos) {
            content.erase(pos, next_sec - pos + 1);
        } else {
            content.erase(pos);
        }
    }
    while (!content.empty() && (content.back() == '\n' || content.back() == '\r')) {
        content.pop_back();
    }
    content += "\n\n[GrowthPresets]\n";
    content += "Count=" + std::to_string(presets.size()) + "\n";
    for (size_t i = 0; i < presets.size(); ++i) {
        const auto& p = presets[i];
        std::string pfx = "Preset" + std::to_string(i) + "_";
        content += pfx + "Name=" + p.name + "\n";
        content += pfx + "DetailLevel=" + std::to_string(p.detail_level) + "\n";
        content += pfx + "Spores=" + std::to_string(p.spore_count) + "\n";
        content += pfx + "PrimaryRoots=" + std::to_string(p.primary_roots) + "\n";
        content += pfx + "SearchRadius=" + std::to_string(p.search_radius) + "\n";
        content += pfx + "KillRadius=" + std::to_string(p.kill_radius) + "\n";
        content += pfx + "GrowthStep=" + std::to_string(p.growth_step) + "\n";
        content += pfx + "VariableStep=" + std::string(p.variable_step ? "1" : "0") + "\n";
        content += pfx + "MinGrowthStep=" + std::to_string(p.min_growth_step) + "\n";
        content += pfx + "MaxGrowthStep=" + std::to_string(p.max_growth_step) + "\n";
        content += pfx + "BranchingRate=" + std::to_string(p.branching_rate) + "\n";
        content += pfx + "SteeringCone=" + std::to_string(p.steering_cone) + "\n";
        content += pfx + "MaxIterations=" + std::to_string(p.max_iterations) + "\n";
        content += pfx + "TrunkCaliber=" + std::to_string(p.trunk_caliber) + "\n";
        content += pfx + "TendrilCaliber=" + std::to_string(p.tendril_caliber) + "\n";
        content += pfx + "VariableTaper=" + std::string(p.variable_taper ? "1" : "0") + "\n";
        content += pfx + "TaperCurve=" + std::to_string(p.taper_curve) + "\n";
        content += pfx + "BranchSmoothK=" + std::to_string(p.branch_smooth_k) + "\n";
        content += pfx + "WallClearance=" + std::to_string(p.wall_clearance_scale) + "\n";
        content += pfx + "WallCutSmooth=" + std::to_string(p.wall_cut_smooth) + "\n";
        content += pfx + "LateralDensity=" + std::to_string(p.lateral_density) + "\n";
        content += pfx + "LateralTrunkCaliber=" + std::to_string(p.lateral_trunk_caliber) + "\n";
        content += pfx + "LateralTipCaliber=" + std::to_string(p.lateral_tip_caliber) + "\n";
        content += pfx + "LateralMaxLength=" + std::to_string(p.lateral_max_length) + "\n";
        content += pfx + "LateralWander=" + std::to_string(p.lateral_wander) + "\n";
        content += pfx + "LateralBranchAngle=" + std::to_string(p.lateral_branch_angle) + "\n";
        content += pfx + "SeedOffset=" + std::to_string(p.seed_offset) + "\n";
        content += pfx + "AlgorithmMode=" + std::to_string(p.algorithm_mode) + "\n";
        content += pfx + "RootRoughness=" + std::to_string(p.root_roughness) + "\n";
        content += pfx + "PushStrength=" + std::to_string(p.push_strength) + "\n";
        content += pfx + "SpaceFillDrive=" + std::to_string(p.space_fill_drive) + "\n";
        content += pfx + "BackwardPull=" + std::to_string(p.backward_pull) + "\n";
        content += pfx + "TensileStraightening=" + std::to_string(p.tensile_straightening) + "\n";
        content += pfx + "MurrayExponent=" + std::to_string(p.murray_exponent) + "\n";
        content += pfx + "MurrayBranchAngle=" + std::to_string(p.murray_branch_angle) + "\n";
        content += pfx + "SurfaceAdherence=" + std::to_string(p.surface_adherence) + "\n";
        content += pfx + "LiveGrowth=" + std::string(p.live_growth ? "1" : "0") + "\n";
        content += pfx + "LiveStepsPerFrame=" + std::to_string(p.live_steps_per_frame) + "\n";
        content += pfx + "LiveFps=" + std::to_string(p.live_fps) + "\n";
    }
    content += "\n";
    std::ofstream out(filepath, std::ios::out | std::ios::binary);
    if (out.is_open()) {
        out.write(content.data(), content.size());
        out.close();
    }
}

void load_growth_presets_from_ini(std::vector<GrowthPreset>& presets, const std::string& filepath) {
    std::error_code ec_ex;
    if (!std::filesystem::exists(filepath, ec_ex) || ec_ex) return;
    std::ifstream in(filepath);
    if (!in.is_open()) return;

    std::string line;
    bool in_section = false;
    std::unordered_map<std::string, std::string> kv;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty()) continue;
        if (line.front() == '[') {
            if (line == "[GrowthPresets]") {
                in_section = true;
                continue;
            } else if (in_section) {
                break;
            }
        }
        if (in_section) {
            size_t eq = line.find('=');
            if (eq != std::string::npos) {
                kv[line.substr(0, eq)] = line.substr(eq + 1);
            }
        }
    }
    in.close();

    if (kv.find("Count") != kv.end()) {
        int count = std::atoi(kv["Count"].c_str());
        if (count > 0) {
            presets.clear();
            for (int i = 0; i < count; ++i) {
                GrowthPreset p;
                std::string pfx = "Preset" + std::to_string(i) + "_";
                if (kv.count(pfx + "Name")) p.name = kv[pfx + "Name"];
                else p.name = "Preset " + std::to_string(i + 1);
                if (kv.count(pfx + "DetailLevel")) p.detail_level = static_cast<float>(std::atof(kv[pfx + "DetailLevel"].c_str()));
                if (kv.count(pfx + "Spores")) p.spore_count = std::atoi(kv[pfx + "Spores"].c_str());
                if (kv.count(pfx + "PrimaryRoots")) p.primary_roots = std::atoi(kv[pfx + "PrimaryRoots"].c_str());
                if (kv.count(pfx + "SearchRadius")) p.search_radius = static_cast<float>(std::atof(kv[pfx + "SearchRadius"].c_str()));
                if (kv.count(pfx + "KillRadius")) p.kill_radius = static_cast<float>(std::atof(kv[pfx + "KillRadius"].c_str()));
                if (kv.count(pfx + "GrowthStep")) p.growth_step = static_cast<float>(std::atof(kv[pfx + "GrowthStep"].c_str()));
                if (kv.count(pfx + "VariableStep")) p.variable_step = (kv[pfx + "VariableStep"] == "1" || kv[pfx + "VariableStep"] == "true");
                if (kv.count(pfx + "MinGrowthStep")) p.min_growth_step = static_cast<float>(std::atof(kv[pfx + "MinGrowthStep"].c_str()));
                if (kv.count(pfx + "MaxGrowthStep")) p.max_growth_step = static_cast<float>(std::atof(kv[pfx + "MaxGrowthStep"].c_str()));
                if (kv.count(pfx + "BranchingRate")) p.branching_rate = static_cast<float>(std::atof(kv[pfx + "BranchingRate"].c_str()));
                if (kv.count(pfx + "SteeringCone")) p.steering_cone = static_cast<float>(std::atof(kv[pfx + "SteeringCone"].c_str()));
                if (kv.count(pfx + "MaxIterations")) p.max_iterations = std::atoi(kv[pfx + "MaxIterations"].c_str());
                if (kv.count(pfx + "TrunkCaliber")) p.trunk_caliber = static_cast<float>(std::atof(kv[pfx + "TrunkCaliber"].c_str()));
                if (kv.count(pfx + "TendrilCaliber")) p.tendril_caliber = static_cast<float>(std::atof(kv[pfx + "TendrilCaliber"].c_str()));
                if (kv.count(pfx + "VariableTaper")) p.variable_taper = (kv[pfx + "VariableTaper"] == "1" || kv[pfx + "VariableTaper"] == "true");
                if (kv.count(pfx + "TaperCurve")) p.taper_curve = static_cast<float>(std::atof(kv[pfx + "TaperCurve"].c_str()));
                if (kv.count(pfx + "BranchSmoothK")) p.branch_smooth_k = static_cast<float>(std::atof(kv[pfx + "BranchSmoothK"].c_str()));
                if (kv.count(pfx + "WallClearance")) p.wall_clearance_scale = static_cast<float>(std::atof(kv[pfx + "WallClearance"].c_str()));
                if (kv.count(pfx + "WallCutSmooth")) p.wall_cut_smooth = static_cast<float>(std::atof(kv[pfx + "WallCutSmooth"].c_str()));
                if (kv.count(pfx + "LateralDensity")) p.lateral_density = static_cast<float>(std::atof(kv[pfx + "LateralDensity"].c_str()));
                if (kv.count(pfx + "LateralTrunkCaliber")) p.lateral_trunk_caliber = static_cast<float>(std::atof(kv[pfx + "LateralTrunkCaliber"].c_str()));
                if (kv.count(pfx + "LateralTipCaliber")) p.lateral_tip_caliber = static_cast<float>(std::atof(kv[pfx + "LateralTipCaliber"].c_str()));
                if (kv.count(pfx + "LateralMaxLength")) p.lateral_max_length = static_cast<float>(std::atof(kv[pfx + "LateralMaxLength"].c_str()));
                if (kv.count(pfx + "LateralWander")) p.lateral_wander = static_cast<float>(std::atof(kv[pfx + "LateralWander"].c_str()));
                if (kv.count(pfx + "LateralBranchAngle")) p.lateral_branch_angle = static_cast<float>(std::atof(kv[pfx + "LateralBranchAngle"].c_str()));
                if (kv.count(pfx + "SeedOffset")) p.seed_offset = static_cast<float>(std::atof(kv[pfx + "SeedOffset"].c_str()));
                if (kv.count(pfx + "AlgorithmMode")) p.algorithm_mode = std::atoi(kv[pfx + "AlgorithmMode"].c_str());
                if (kv.count(pfx + "RootRoughness")) p.root_roughness = static_cast<float>(std::atof(kv[pfx + "RootRoughness"].c_str()));
                if (kv.count(pfx + "PushStrength")) p.push_strength = static_cast<float>(std::atof(kv[pfx + "PushStrength"].c_str()));
                if (kv.count(pfx + "SpaceFillDrive")) p.space_fill_drive = static_cast<float>(std::atof(kv[pfx + "SpaceFillDrive"].c_str()));
                if (kv.count(pfx + "BackwardPull")) p.backward_pull = static_cast<float>(std::atof(kv[pfx + "BackwardPull"].c_str()));
                if (kv.count(pfx + "TensileStraightening")) p.tensile_straightening = static_cast<float>(std::atof(kv[pfx + "TensileStraightening"].c_str()));
                if (kv.count(pfx + "MurrayExponent")) p.murray_exponent = static_cast<float>(std::atof(kv[pfx + "MurrayExponent"].c_str()));
                if (kv.count(pfx + "MurrayBranchAngle")) p.murray_branch_angle = static_cast<float>(std::atof(kv[pfx + "MurrayBranchAngle"].c_str()));
                if (kv.count(pfx + "SurfaceAdherence")) p.surface_adherence = static_cast<float>(std::atof(kv[pfx + "SurfaceAdherence"].c_str()));
                if (kv.count(pfx + "LiveGrowth")) p.live_growth = (kv[pfx + "LiveGrowth"] == "1" || kv[pfx + "LiveGrowth"] == "true");
                if (kv.count(pfx + "LiveStepsPerFrame")) p.live_steps_per_frame = std::atoi(kv[pfx + "LiveStepsPerFrame"].c_str());
                if (kv.count(pfx + "LiveFps")) p.live_fps = std::atoi(kv[pfx + "LiveFps"].c_str());
                presets.push_back(p);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// v24.01 Phase 0.4: viewport theme loader ([ViewportColors] +
// [LightingAndShading], imgui.ini). Written by the Styles panel's "Copy
// Layout INI" exporter; this is the read side. Sections are optional: a
// fresh install (or an imgui.ini pasted without them) keeps the compiled
// DisplayState defaults, and a present section applies only the keys it
// actually carries — matching the load_growth_presets_from_ini convention.
// ---------------------------------------------------------------------------
void load_viewport_style_from_ini(rhizome::DisplayState& display, const std::string& filepath) {
    std::error_code ec_ex;
    if (!std::filesystem::exists(filepath, ec_ex) || ec_ex) return;
    std::ifstream in(filepath);
    if (!in.is_open()) return;

    std::unordered_map<std::string, std::string> cols;   // [ViewportColors]
    std::unordered_map<std::string, std::string> light;  // [LightingAndShading]
    std::string line;
    std::unordered_map<std::string, std::string>* target = nullptr;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty()) continue;
        if (line.front() == '[') {
            if (line == "[ViewportColors]") target = &cols;
            else if (line == "[LightingAndShading]") target = &light;
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

    // [ViewportColors] — "Name=r,g,b,a" rows (writer precision %.3f)
    auto apply_col = [&cols](const char* key, glm::vec4& out) {
        auto it = cols.find(key);
        if (it == cols.end()) return;
        float r, g, b, a;
        if (std::sscanf(it->second.c_str(), "%f,%f,%f,%f", &r, &g, &b, &a) == 4) {
            out = glm::vec4(r, g, b, a);
        }
    };
    apply_col("ViewportBg", display.color_viewport_bg);
    apply_col("FloorGrid", display.color_floor_grid);
    apply_col("FloorAxisX", display.color_floor_axis_x);
    apply_col("FloorAxisY", display.color_floor_axis_y);
    apply_col("OriginalMesh", display.color_original_mesh);
    apply_col("ConvertedBody", display.color_converted_body);
    apply_col("Roots", display.color_roots);
    apply_col("BranchSkeleton", display.color_branch_skeleton);
    apply_col("Spores", display.color_spores);
    apply_col("SeedSelected", display.color_seed_selected);
    apply_col("SeedUnselected", display.color_seed_unselected);
    apply_col("BoundingBox", display.color_bounding_box);
    apply_col("Wireframe", display.color_wireframe);
    apply_col("TestPass", display.color_test_pass);
    apply_col("TestFail", display.color_test_fail);

    // [LightingAndShading] — scalar rows (6 v23.28 keys + GlassFrosting and
    // SpecimenOpacity, both added by v24.01 Phase 0 alongside the trio's
    // re-home into the Specimen Appearance section; the ten v24.01 Phase 1
    // lighting-expansion keys — written by the Styles panel exporter in the
    // same order as the sliders)
    auto apply_flt = [&light](const char* key, float& out) {
        auto it = light.find(key);
        if (it == light.end()) return;
        out = static_cast<float>(std::atof(it->second.c_str()));
    };
    apply_flt("Diffuse", display.param_shader_diffuse);
    apply_flt("EdgeDarkening", display.param_shader_edge_darkening);
    apply_flt("CreviceAO", display.param_shader_crevice_ao);
    apply_flt("Specular", display.param_shader_specular);
    apply_flt("BarkStrength", display.param_shader_bark_strength);
    apply_flt("BarkFrequency", display.param_shader_bark_frequency);
    apply_flt("GlassFrosting", display.param_glass_frosting);
    apply_flt("SpecimenOpacity", display.specimen_opacity);
    apply_flt("LightAzimuth", display.param_light_azimuth);
    apply_flt("LightElevation", display.param_light_elevation);
    apply_flt("FillLight", display.param_fill_light);
    apply_flt("Ambient", display.param_ambient);
    apply_flt("Glossiness", display.param_glossiness);
    apply_flt("HighlightTint", display.param_highlight_tint);
    apply_flt("FresnelRim", display.param_fresnel_rim);
    apply_flt("VolumetricAO", display.param_volumetric_ao);
    apply_flt("SSSStrength", display.param_sss_strength);
    apply_flt("SSSWarmth", display.param_sss_warmth);
}

bool export_algorithm_to_json(const GrowthAlgorithmProfile& a, const std::string& filepath) {
    std::ofstream out(filepath, std::ios::out | std::ios::binary);
    if (!out.is_open()) return false;
    out << "{\n";
    out << "  \"format\": \"RhizomeColonizationAlgorithm\",\n";
    out << "  \"version\": \"2.0\",\n";
    out << "  \"name\": \"" << a.name << "\",\n";
    out << "  \"engine_type\": \"" << (a.engine_mode == 1 ? "volumetric_root_bound_cuda" : "classic_space_colonization_cuda") << "\",\n";
    out << "  \"EngineMode\": " << a.engine_mode << ",\n";
    out << "  \"PrimaryRoots\": " << a.primary_roots << ",\n";
    out << "  \"ui_parameters\": {\n";
    out << "    \"RootRoughness\": " << a.root_roughness << ",\n";
    out << "    \"PushStrength\": " << a.push_strength << ",\n";
    out << "    \"SpaceFillDrive\": " << a.space_fill_drive << ",\n";
    out << "    \"BackwardPull\": " << a.backward_pull << ",\n";
    out << "    \"TensileStraightening\": " << a.tensile_straightening << ",\n";
    out << "    \"MurrayExponent\": " << a.murray_exponent << ",\n";
    out << "    \"MurrayBranchAngle\": " << a.murray_branch_angle << ",\n";
    out << "    \"SurfaceAdherence\": " << a.surface_adherence << ",\n";
    out << "    \"Spores\": " << a.spore_count << ",\n";
    out << "    \"SearchRadius\": " << a.search_radius << ",\n";
    out << "    \"KillRadius\": " << a.kill_radius << ",\n";
    out << "    \"GrowthStep\": " << a.growth_step << ",\n";
    out << "    \"VariableStep\": " << (a.variable_step ? 1 : 0) << ",\n";
    out << "    \"MinGrowthStep\": " << a.min_growth_step << ",\n";
    out << "    \"MaxGrowthStep\": " << a.max_growth_step << ",\n";
    out << "    \"BranchingRate\": " << a.branching_rate << ",\n";
    out << "    \"SteeringCone\": " << a.steering_cone << ",\n";
    out << "    \"MaxIterations\": " << a.max_iterations << ",\n";
    out << "    \"TrunkCaliber\": " << a.trunk_caliber << ",\n";
    out << "    \"TendrilCaliber\": " << a.tendril_caliber << ",\n";
    out << "    \"VariableTaper\": " << (a.variable_taper ? 1 : 0) << ",\n";
    out << "    \"TaperCurve\": " << a.taper_curve << ",\n";
    out << "    \"WallClearanceScale\": " << a.wall_clearance_scale << ",\n";
    out << "    \"BranchSmoothK\": " << a.branch_smooth_k << ",\n";
    out << "    \"WallCutSmooth\": " << a.wall_cut_smooth << ",\n";
    out << "    \"SeedOffset\": " << a.seed_offset << "\n";
    out << "  },\n";
    out << "  \"cuda_kernel_rules\": {\n";
    out << "    \"TipInertiaWeight\": " << a.tip_inertia_weight << ",\n";
    out << "    \"SporePullWeight\": " << a.spore_pull_weight << ",\n";
    out << "    \"RoughnessAmplitude\": " << a.roughness_amplitude << ",\n";
    out << "    \"FeelerNarFwd\": " << a.feeler_nar_fwd << ",\n";
    out << "    \"FeelerNarSide\": " << a.feeler_nar_side << ",\n";
    out << "    \"FeelerMedFwd\": " << a.feeler_med_fwd << ",\n";
    out << "    \"FeelerMedSide\": " << a.feeler_med_side << ",\n";
    out << "    \"FeelerWideFwd\": " << a.feeler_wide_fwd << ",\n";
    out << "    \"FeelerWideSide\": " << a.feeler_wide_side << ",\n";
    out << "    \"FeelerWallIn\": " << a.feeler_wall_in << ",\n";
    out << "    \"FeelerWallOut\": " << a.feeler_wall_out << ",\n";
    out << "    \"ClearanceScoreWeight\": " << a.clearance_score_weight << ",\n";
    out << "    \"AlignBonusWeight\": " << a.align_bonus_weight << ",\n";
    out << "    \"LateralFwdWeight\": " << a.lateral_fwd_weight << ",\n";
    out << "    \"LateralSideWeight\": " << a.lateral_side_weight << ",\n";
    out << "    \"LateralRadiusScale\": " << a.lateral_radius_scale << ",\n";
    out << "    \"MaxBranchGenerations\": " << a.max_branch_generations << ",\n";
    out << "    \"SkinGapScale\": " << a.skin_gap_scale << ",\n";
    out << "    \"PushBaseGain\": " << a.push_base_gain << ",\n";
    out << "    \"PushStrengthGain\": " << a.push_strength_gain << ",\n";
    out << "    \"MaxStepDispRatio\": " << a.max_step_disp_ratio << ",\n";
    out << "    \"TetherMaxStretch\": " << a.tether_max_stretch << ",\n";
    out << "    \"TetherTargetStretch\": " << a.tether_target_stretch << ",\n";
    out << "    \"WallClearanceScale\": " << a.wall_clearance_scale << ",\n";
    out << "    \"BranchSmoothK\": " << a.branch_smooth_k << ",\n";
    out << "    \"WallCutSmooth\": " << a.wall_cut_smooth << "\n";
    out << "  }\n";
    out << "}\n";
    out.close();
    return true;
}

bool export_algorithm_to_cu(const GrowthAlgorithmProfile& a, const std::string& filepath) {
    std::ofstream out(filepath, std::ios::out | std::ios::binary);
    if (!out.is_open()) return false;

    out << "// ============================================================================\n";
    out << "// RHIZOME NATIVE CUDA GPU COLONIZATION ALGORITHM BACKUP (.cu)\n";
    out << "// Self-contained GPU algorithm definition + complete CUDA kernel source.\n";
    out << "// Can be imported directly into Rhizome (Import New / Replace) or compiled.\n";
    out << "// ============================================================================\n";
    out << "/* [RHIZOME_ALGORITHM_SPEC]\n";
    out << "Name=" << a.name << "\n";
    out << "EngineMode=" << a.engine_mode << "\n";
    out << "RootRoughness=" << a.root_roughness << "\n";
    out << "PushStrength=" << a.push_strength << "\n";
    out << "SpaceFillDrive=" << a.space_fill_drive << "\n";
    out << "BackwardPull=" << a.backward_pull << "\n";
    out << "TensileStraightening=" << a.tensile_straightening << "\n";
    out << "SurfaceAdherence=" << a.surface_adherence << "\n";
    out << "Spores=" << a.spore_count << "\n";
    out << "SearchRadius=" << a.search_radius << "\n";
    out << "KillRadius=" << a.kill_radius << "\n";
    out << "GrowthStep=" << a.growth_step << "\n";
    out << "VariableStep=" << (a.variable_step ? 1 : 0) << "\n";
    out << "MinGrowthStep=" << a.min_growth_step << "\n";
    out << "MaxGrowthStep=" << a.max_growth_step << "\n";
    out << "BranchingRate=" << a.branching_rate << "\n";
    out << "SteeringCone=" << a.steering_cone << "\n";
    out << "MaxIterations=" << a.max_iterations << "\n";
    out << "TrunkCaliber=" << a.trunk_caliber << "\n";
    out << "TendrilCaliber=" << a.tendril_caliber << "\n";
    out << "VariableTaper=" << (a.variable_taper ? 1 : 0) << "\n";
    out << "TaperCurve=" << a.taper_curve << "\n";
    out << "SeedOffset=" << a.seed_offset << "\n";
    out << "TipInertiaWeight=" << a.tip_inertia_weight << "\n";
    out << "SporePullWeight=" << a.spore_pull_weight << "\n";
    out << "RoughnessAmplitude=" << a.roughness_amplitude << "\n";
    out << "FeelerNarFwd=" << a.feeler_nar_fwd << "\n";
    out << "FeelerNarSide=" << a.feeler_nar_side << "\n";
    out << "FeelerMedFwd=" << a.feeler_med_fwd << "\n";
    out << "FeelerMedSide=" << a.feeler_med_side << "\n";
    out << "FeelerWideFwd=" << a.feeler_wide_fwd << "\n";
    out << "FeelerWideSide=" << a.feeler_wide_side << "\n";
    out << "FeelerWallIn=" << a.feeler_wall_in << "\n";
    out << "FeelerWallOut=" << a.feeler_wall_out << "\n";
    out << "ClearanceScoreWeight=" << a.clearance_score_weight << "\n";
    out << "AlignBonusWeight=" << a.align_bonus_weight << "\n";
    out << "LateralFwdWeight=" << a.lateral_fwd_weight << "\n";
    out << "LateralSideWeight=" << a.lateral_side_weight << "\n";
    out << "LateralRadiusScale=" << a.lateral_radius_scale << "\n";
    out << "MaxBranchGenerations=" << a.max_branch_generations << "\n";
    out << "SkinGapScale=" << a.skin_gap_scale << "\n";
    out << "PushBaseGain=" << a.push_base_gain << "\n";
    out << "PushStrengthGain=" << a.push_strength_gain << "\n";
    out << "MaxStepDispRatio=" << a.max_step_disp_ratio << "\n";
    out << "TetherMaxStretch=" << a.tether_max_stretch << "\n";
    out << "TetherTargetStretch=" << a.tether_target_stretch << "\n";
    out << "WallClearanceScale=" << a.wall_clearance_scale << "\n";
    out << "BranchSmoothK=" << a.branch_smooth_k << "\n";
    out << "WallCutSmooth=" << a.wall_cut_smooth << "\n";
    out << "[/RHIZOME_ALGORITHM_SPEC] */\n\n";

    out << "#include <cuda_runtime.h>\n";
    out << "#include <glm/glm.hpp>\n\n";
    out << "// Baked CUDA Algorithm Constants for: " << a.name << "\n";
    out << "namespace rhizome_baked_algo {\n";
    out << "    constexpr int   kEngineMode           = " << a.engine_mode << ";\n";
    out << "    constexpr float kRootRoughness        = " << a.root_roughness << "f;\n";
    out << "    constexpr float kPushStrength         = " << a.push_strength << "f;\n";
    out << "    constexpr float kSpaceFillDrive       = " << a.space_fill_drive << "f;\n";
    out << "    constexpr float kBackwardPull         = " << a.backward_pull << "f;\n";
    out << "    constexpr float kTensileStraightening = " << a.tensile_straightening << "f;\n";
    out << "    constexpr float kSurfaceAdherence     = " << a.surface_adherence << "f;\n";
    out << "    constexpr int   kSporeCount           = " << a.spore_count << ";\n";
    out << "    constexpr float kSearchRadius         = " << a.search_radius << "f;\n";
    out << "    constexpr float kKillRadius           = " << a.kill_radius << "f;\n";
    out << "    constexpr float kGrowthStep           = " << a.growth_step << "f;\n";
    out << "    constexpr int   kVariableStep         = " << (a.variable_step ? 1 : 0) << ";\n";
    out << "    constexpr float kMinGrowthStep        = " << a.min_growth_step << "f;\n";
    out << "    constexpr float kMaxGrowthStep        = " << a.max_growth_step << "f;\n";
    out << "    constexpr float kBranchingRate        = " << a.branching_rate << "f;\n";
    out << "    constexpr float kSteeringCone         = " << a.steering_cone << "f;\n";
    out << "    constexpr int   kMaxIterations        = " << a.max_iterations << ";\n";
    out << "    constexpr float kTrunkCaliber         = " << a.trunk_caliber << "f;\n";
    out << "    constexpr float kTendrilCaliber       = " << a.tendril_caliber << "f;\n";
    out << "    constexpr int   kVariableTaper        = " << (a.variable_taper ? 1 : 0) << ";\n";
    out << "    constexpr float kTaperCurve           = " << a.taper_curve << "f;\n";
    out << "    constexpr float kSeedOffset           = " << a.seed_offset << "f;\n";
    out << "    constexpr float kTipInertiaWeight     = " << a.tip_inertia_weight << "f;\n";
    out << "    constexpr float kSporePullWeight      = " << a.spore_pull_weight << "f;\n";
    out << "    constexpr float kRoughnessAmplitude   = " << a.roughness_amplitude << "f;\n";
    out << "    constexpr float kFeelerNarFwd         = " << a.feeler_nar_fwd << "f;\n";
    out << "    constexpr float kFeelerNarSide        = " << a.feeler_nar_side << "f;\n";
    out << "    constexpr float kFeelerMedFwd         = " << a.feeler_med_fwd << "f;\n";
    out << "    constexpr float kFeelerMedSide        = " << a.feeler_med_side << "f;\n";
    out << "    constexpr float kFeelerWideFwd        = " << a.feeler_wide_fwd << "f;\n";
    out << "    constexpr float kFeelerWideSide       = " << a.feeler_wide_side << "f;\n";
    out << "    constexpr float kFeelerWallIn         = " << a.feeler_wall_in << "f;\n";
    out << "    constexpr float kFeelerWallOut        = " << a.feeler_wall_out << "f;\n";
    out << "    constexpr float kClearanceScoreWeight = " << a.clearance_score_weight << "f;\n";
    out << "    constexpr float kAlignBonusWeight     = " << a.align_bonus_weight << "f;\n";
    out << "    constexpr float kLateralFwdWeight     = " << a.lateral_fwd_weight << "f;\n";
    out << "    constexpr float kLateralSideWeight    = " << a.lateral_side_weight << "f;\n";
    out << "    constexpr float kLateralRadiusScale   = " << a.lateral_radius_scale << "f;\n";
    out << "    constexpr int   kMaxBranchGenerations = " << a.max_branch_generations << ";\n";
    out << "    constexpr float kSkinGapScale         = " << a.skin_gap_scale << "f;\n";
    out << "    constexpr float kPushBaseGain         = " << a.push_base_gain << "f;\n";
    out << "    constexpr float kPushStrengthGain     = " << a.push_strength_gain << "f;\n";
    out << "    constexpr float kMaxStepDispRatio     = " << a.max_step_disp_ratio << "f;\n";
    out << "    constexpr float kTetherMaxStretch     = " << a.tether_max_stretch << "f;\n";
    out << "    constexpr float kTetherTargetStretch  = " << a.tether_target_stretch << "f;\n";
    out << "    constexpr float kWallClearanceScale   = " << a.wall_clearance_scale << "f;\n";
    out << "    constexpr float kBranchSmoothK        = " << a.branch_smooth_k << "f;\n";
    out << "    constexpr float kWallCutSmooth        = " << a.wall_cut_smooth << "f;\n";
    out << "}\n\n";

    // Also embed the live CUDA colonization kernel implementation if accessible
    const std::vector<std::string> cu_paths = {
        "native/src/sdf/cuda_colonization_volumetric.cu",
        "src/sdf/cuda_colonization_volumetric.cu",
        "../src/sdf/cuda_colonization_volumetric.cu",
        "../../native/src/sdf/cuda_colonization_volumetric.cu",
        "native/src/sdf/cuda_rasterizer.cu",
        "src/sdf/cuda_rasterizer.cu",
        "../src/sdf/cuda_rasterizer.cu",
        "../../native/src/sdf/cuda_rasterizer.cu"
    };
    bool embedded_full_kernels = false;
    for (const auto& cp : cu_paths) {
        std::error_code ec_cp;
        if (std::filesystem::exists(cp, ec_cp) && !ec_cp) {
            std::ifstream cu_in(cp, std::ios::in | std::ios::binary);
            if (cu_in.is_open()) {
                std::ostringstream ss;
                ss << cu_in.rdbuf();
                std::string full_cu = ss.str();
                size_t start_pos = full_cu.find("// 5. CUDA GPU 3D Volumetric Root Colonization Engine");
                size_t end_pos = full_cu.find("bool CudaRasterizer::intersect_volumes_gpu");
                if (end_pos == std::string::npos) {
                    end_pos = full_cu.rfind("} // namespace rhizome");
                }
                if (start_pos != std::string::npos && end_pos != std::string::npos && end_pos > start_pos) {
                    out << "// --- COMPLETE CUDA GPU COLONIZATION KERNEL PIPELINE ---\n";
                    out << full_cu.substr(start_pos, end_pos - start_pos) << "\n";
                    embedded_full_kernels = true;
                }
                break;
            }
        }
    }

    if (!embedded_full_kernels) {
        out << R"CUDA(
// 11-Feeler Cone Direction Synthesis & Scoring Equations (CUDA Device Kernel)
__device__ void compute_11_feeler_directions(
    const glm::vec3& base_dir,
    const glm::vec3& tang_fwd,
    const glm::vec3& side,
    const glm::vec3& wall_n,
    glm::vec3 out_dirs[11]
) {
    using namespace rhizome_baked_algo;
    out_dirs[0]  = base_dir;
    out_dirs[1]  = glm::normalize(tang_fwd * kFeelerNarFwd  + side * kFeelerNarSide);
    out_dirs[2]  = glm::normalize(tang_fwd * kFeelerNarFwd  - side * kFeelerNarSide);
    out_dirs[3]  = glm::normalize(tang_fwd * kFeelerMedFwd  + side * kFeelerMedSide);
    out_dirs[4]  = glm::normalize(tang_fwd * kFeelerMedFwd  - side * kFeelerMedSide);
    out_dirs[5]  = glm::normalize(tang_fwd * kFeelerWideFwd + side * kFeelerWideSide);
    out_dirs[6]  = glm::normalize(tang_fwd * kFeelerWideFwd - side * kFeelerWideSide);
    out_dirs[7]  = glm::normalize(tang_fwd * 0.80f - wall_n * kFeelerWallIn  + side * 0.22f);
    out_dirs[8]  = glm::normalize(tang_fwd * 0.80f - wall_n * kFeelerWallIn  - side * 0.22f);
    out_dirs[9]  = glm::normalize(tang_fwd * 0.80f + wall_n * kFeelerWallOut + side * 0.22f);
    out_dirs[10] = glm::normalize(tang_fwd * 0.80f + wall_n * kFeelerWallOut - side * 0.22f);
}
)CUDA";
    }

    out.close();
    return true;
}

bool export_algorithm_to_file(const GrowthAlgorithmProfile& a, const std::string& filepath) {
    std::filesystem::path fp(filepath);
    std::string ext = fp.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".cu") {
        return export_algorithm_to_cu(a, filepath);
    }
    return export_algorithm_to_json(a, filepath);
}

static std::string trim_token(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t' || s.back() == ',' || s.back() == ';')) {
        s.pop_back();
    }
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t')) {
        start++;
    }
    s = s.substr(start);
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        s = s.substr(1, s.size() - 2);
    }
    if (!s.empty() && s.back() == 'f' && (std::isdigit(static_cast<unsigned char>(s[0])) || s[0] == '-')) {
        s.pop_back();
    }
    return s;
}

void apply_kv_to_algorithm(const std::unordered_map<std::string, std::string>& kv, const std::string& fallback_name, GrowthAlgorithmProfile& out_algo) {
    if (kv.count("Name")) out_algo.name = kv.at("Name");
    else if (kv.count("name")) out_algo.name = kv.at("name");
    else out_algo.name = fallback_name;

    auto get_f = [&](const char* k1, const char* k2, float& target) {
        if (kv.count(k1)) target = static_cast<float>(std::atof(kv.at(k1).c_str()));
        else if (k2 && kv.count(k2)) target = static_cast<float>(std::atof(kv.at(k2).c_str()));
    };
    auto get_i = [&](const char* k1, const char* k2, int& target) {
        if (kv.count(k1)) target = std::atoi(kv.at(k1).c_str());
        else if (k2 && kv.count(k2)) target = std::atoi(kv.at(k2).c_str());
    };

    get_i("EngineMode", "kEngineMode", out_algo.engine_mode);
    out_algo.engine_mode = std::clamp(out_algo.engine_mode, 0, 3);
    get_i("PrimaryRoots", "kPrimaryRoots", out_algo.primary_roots);
    out_algo.primary_roots = std::clamp(out_algo.primary_roots, 1, 8);
    get_f("RootRoughness", "kRootRoughness", out_algo.root_roughness);
    get_f("PushStrength", "kPushStrength", out_algo.push_strength);
    get_f("SpaceFillDrive", "kSpaceFillDrive", out_algo.space_fill_drive);
    get_f("BackwardPull", "kBackwardPull", out_algo.backward_pull);
    get_f("TensileStraightening", "kTensileStraightening", out_algo.tensile_straightening);
    get_f("MurrayExponent", "kMurrayExponent", out_algo.murray_exponent);
    get_f("MurrayBranchAngle", "kMurrayBranchAngle", out_algo.murray_branch_angle);
    get_f("SurfaceAdherence", "kSurfaceAdherence", out_algo.surface_adherence);
    get_i("Spores", "kSporeCount", out_algo.spore_count);
    get_f("SearchRadius", "kSearchRadius", out_algo.search_radius);
    get_f("KillRadius", "kKillRadius", out_algo.kill_radius);
    get_f("GrowthStep", "kGrowthStep", out_algo.growth_step);
    int var_step_int = out_algo.variable_step ? 1 : 0;
    get_i("VariableStep", "kVariableStep", var_step_int);
    out_algo.variable_step = (var_step_int != 0);
    get_f("MinGrowthStep", "kMinGrowthStep", out_algo.min_growth_step);
    get_f("MaxGrowthStep", "kMaxGrowthStep", out_algo.max_growth_step);
    get_f("BranchingRate", "kBranchingRate", out_algo.branching_rate);
    get_f("SteeringCone", "kSteeringCone", out_algo.steering_cone);
    get_i("MaxIterations", "kMaxIterations", out_algo.max_iterations);
    get_f("TrunkCaliber", "kTrunkCaliber", out_algo.trunk_caliber);
    get_f("TendrilCaliber", "kTendrilCaliber", out_algo.tendril_caliber);
    int var_taper_int = out_algo.variable_taper ? 1 : 0;
    get_i("VariableTaper", "kVariableTaper", var_taper_int);
    out_algo.variable_taper = (var_taper_int != 0);
    get_f("TaperCurve", "kTaperCurve", out_algo.taper_curve);
    get_f("SeedOffset", "kSeedOffset", out_algo.seed_offset);

    // Internal GPU colonization algorithm rules
    get_f("TipInertiaWeight", "kTipInertiaWeight", out_algo.tip_inertia_weight);
    get_f("SporePullWeight", "kSporePullWeight", out_algo.spore_pull_weight);
    get_f("RoughnessAmplitude", "kRoughnessAmplitude", out_algo.roughness_amplitude);
    get_f("FeelerNarFwd", "kFeelerNarFwd", out_algo.feeler_nar_fwd);
    get_f("FeelerNarSide", "kFeelerNarSide", out_algo.feeler_nar_side);
    get_f("FeelerMedFwd", "kFeelerMedFwd", out_algo.feeler_med_fwd);
    get_f("FeelerMedSide", "kFeelerMedSide", out_algo.feeler_med_side);
    get_f("FeelerWideFwd", "kFeelerWideFwd", out_algo.feeler_wide_fwd);
    get_f("FeelerWideSide", "kFeelerWideSide", out_algo.feeler_wide_side);
    get_f("FeelerWallIn", "kFeelerWallIn", out_algo.feeler_wall_in);
    get_f("FeelerWallOut", "kFeelerWallOut", out_algo.feeler_wall_out);
    get_f("ClearanceScoreWeight", "kClearanceScoreWeight", out_algo.clearance_score_weight);
    get_f("AlignBonusWeight", "kAlignBonusWeight", out_algo.align_bonus_weight);
    get_f("LateralFwdWeight", "kLateralFwdWeight", out_algo.lateral_fwd_weight);
    get_f("LateralSideWeight", "kLateralSideWeight", out_algo.lateral_side_weight);
    get_f("LateralRadiusScale", "kLateralRadiusScale", out_algo.lateral_radius_scale);
    get_i("MaxBranchGenerations", "kMaxBranchGenerations", out_algo.max_branch_generations);
    get_f("SkinGapScale", "kSkinGapScale", out_algo.skin_gap_scale);
    get_f("PushBaseGain", "kPushBaseGain", out_algo.push_base_gain);
    get_f("PushStrengthGain", "kPushStrengthGain", out_algo.push_strength_gain);
    get_f("MaxStepDispRatio", "kMaxStepDispRatio", out_algo.max_step_disp_ratio);
    get_f("TetherMaxStretch", "kTetherMaxStretch", out_algo.tether_max_stretch);
    get_f("TetherTargetStretch", "kTetherTargetStretch", out_algo.tether_target_stretch);
    get_f("WallClearanceScale", "kWallClearanceScale", out_algo.wall_clearance_scale);
    get_f("BranchSmoothK", "kBranchSmoothK", out_algo.branch_smooth_k);
    get_f("WallCutSmooth", "kWallCutSmooth", out_algo.wall_cut_smooth);
}

bool import_algorithm_from_file(const std::string& filepath, GrowthAlgorithmProfile& out_algo) {
    if (!std::filesystem::exists(filepath)) return false;
    std::ifstream in(filepath);
    if (!in.is_open()) return false;

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        std::string t = trim_token(line);
        if (t.empty() || t.front() == ';' || t.front() == '#' || t.front() == '[' || t.front() == '{' || t.front() == '}') continue;

        // Parse constexpr lines from .cu files if edited directly in C++ syntax
        if (t.find("constexpr ") == 0) {
            size_t eq = t.find('=');
            if (eq != std::string::npos) {
                std::string lhs = trim_token(t.substr(0, eq));
                size_t last_sp = lhs.find_last_of(" \t");
                if (last_sp != std::string::npos) {
                    std::string key = trim_token(lhs.substr(last_sp + 1));
                    std::string val = trim_token(t.substr(eq + 1));
                    kv[key] = val;
                }
            }
            continue;
        }

        // Parse key=value (.cu header / .ini) or "key": value (.json)
        size_t sep = line.find('=');
        if (sep == std::string::npos) {
            sep = line.find(':');
        }
        if (sep != std::string::npos) {
            std::string key = trim_token(line.substr(0, sep));
            std::string val = trim_token(line.substr(sep + 1));
            if (!key.empty() && !val.empty() && val != "{") {
                kv[key] = val;
            }
        }
    }
    in.close();
    if (kv.empty()) return false;

    std::filesystem::path fp(filepath);
    apply_kv_to_algorithm(kv, fp.stem().string(), out_algo);
    return true;
}

void save_growth_algorithms_to_ini(const std::vector<GrowthAlgorithmProfile>& algos, int selected_idx, const std::string& filepath) {
    std::string content;
    if (std::filesystem::exists(filepath)) {
        std::ifstream in(filepath, std::ios::in | std::ios::binary);
        if (in.is_open()) {
            std::ostringstream sstr;
            sstr << in.rdbuf();
            content = sstr.str();
            in.close();
        }
    }
    size_t pos = content.find("[GrowthAlgorithms]");
    if (pos != std::string::npos) {
        size_t next_sec = content.find("\n[", pos + 18);
        if (next_sec != std::string::npos) {
            content.erase(pos, next_sec - pos + 1);
        } else {
            content.erase(pos);
        }
    }
    while (!content.empty() && (content.back() == '\n' || content.back() == '\r')) {
        content.pop_back();
    }
    content += "\n\n[GrowthAlgorithms]\n";
    content += "Selected=" + std::to_string(selected_idx) + "\n";
    content += "Count=" + std::to_string(algos.size()) + "\n";
    for (size_t i = 0; i < algos.size(); ++i) {
        const auto& a = algos[i];
        std::string pfx = "Algo" + std::to_string(i) + "_";
        content += pfx + "Name=" + a.name + "\n";
        content += pfx + "EngineMode=" + std::to_string(a.engine_mode) + "\n";
        content += pfx + "PrimaryRoots=" + std::to_string(a.primary_roots) + "\n";
        content += pfx + "RootRoughness=" + std::to_string(a.root_roughness) + "\n";
        content += pfx + "PushStrength=" + std::to_string(a.push_strength) + "\n";
        content += pfx + "SpaceFillDrive=" + std::to_string(a.space_fill_drive) + "\n";
        content += pfx + "BackwardPull=" + std::to_string(a.backward_pull) + "\n";
        content += pfx + "TensileStraightening=" + std::to_string(a.tensile_straightening) + "\n";
        content += pfx + "MurrayExponent=" + std::to_string(a.murray_exponent) + "\n";
        content += pfx + "MurrayBranchAngle=" + std::to_string(a.murray_branch_angle) + "\n";
        content += pfx + "SurfaceAdherence=" + std::to_string(a.surface_adherence) + "\n";
        content += pfx + "Spores=" + std::to_string(a.spore_count) + "\n";
        content += pfx + "SearchRadius=" + std::to_string(a.search_radius) + "\n";
        content += pfx + "KillRadius=" + std::to_string(a.kill_radius) + "\n";
        content += pfx + "GrowthStep=" + std::to_string(a.growth_step) + "\n";
        content += pfx + "VariableStep=" + std::to_string(a.variable_step ? 1 : 0) + "\n";
        content += pfx + "MinGrowthStep=" + std::to_string(a.min_growth_step) + "\n";
        content += pfx + "MaxGrowthStep=" + std::to_string(a.max_growth_step) + "\n";
        content += pfx + "BranchingRate=" + std::to_string(a.branching_rate) + "\n";
        content += pfx + "SteeringCone=" + std::to_string(a.steering_cone) + "\n";
        content += pfx + "MaxIterations=" + std::to_string(a.max_iterations) + "\n";
        content += pfx + "TrunkCaliber=" + std::to_string(a.trunk_caliber) + "\n";
        content += pfx + "TendrilCaliber=" + std::to_string(a.tendril_caliber) + "\n";
        content += pfx + "VariableTaper=" + std::to_string(a.variable_taper ? 1 : 0) + "\n";
        content += pfx + "TaperCurve=" + std::to_string(a.taper_curve) + "\n";
        content += pfx + "SeedOffset=" + std::to_string(a.seed_offset) + "\n";
        content += pfx + "TipInertiaWeight=" + std::to_string(a.tip_inertia_weight) + "\n";
        content += pfx + "SporePullWeight=" + std::to_string(a.spore_pull_weight) + "\n";
        content += pfx + "RoughnessAmplitude=" + std::to_string(a.roughness_amplitude) + "\n";
        content += pfx + "FeelerNarFwd=" + std::to_string(a.feeler_nar_fwd) + "\n";
        content += pfx + "FeelerNarSide=" + std::to_string(a.feeler_nar_side) + "\n";
        content += pfx + "FeelerMedFwd=" + std::to_string(a.feeler_med_fwd) + "\n";
        content += pfx + "FeelerMedSide=" + std::to_string(a.feeler_med_side) + "\n";
        content += pfx + "FeelerWideFwd=" + std::to_string(a.feeler_wide_fwd) + "\n";
        content += pfx + "FeelerWideSide=" + std::to_string(a.feeler_wide_side) + "\n";
        content += pfx + "FeelerWallIn=" + std::to_string(a.feeler_wall_in) + "\n";
        content += pfx + "FeelerWallOut=" + std::to_string(a.feeler_wall_out) + "\n";
        content += pfx + "ClearanceScoreWeight=" + std::to_string(a.clearance_score_weight) + "\n";
        content += pfx + "AlignBonusWeight=" + std::to_string(a.align_bonus_weight) + "\n";
        content += pfx + "LateralFwdWeight=" + std::to_string(a.lateral_fwd_weight) + "\n";
        content += pfx + "LateralSideWeight=" + std::to_string(a.lateral_side_weight) + "\n";
        content += pfx + "LateralRadiusScale=" + std::to_string(a.lateral_radius_scale) + "\n";
        content += pfx + "MaxBranchGenerations=" + std::to_string(a.max_branch_generations) + "\n";
        content += pfx + "SkinGapScale=" + std::to_string(a.skin_gap_scale) + "\n";
        content += pfx + "PushBaseGain=" + std::to_string(a.push_base_gain) + "\n";
        content += pfx + "PushStrengthGain=" + std::to_string(a.push_strength_gain) + "\n";
        content += pfx + "MaxStepDispRatio=" + std::to_string(a.max_step_disp_ratio) + "\n";
        content += pfx + "TetherMaxStretch=" + std::to_string(a.tether_max_stretch) + "\n";
        content += pfx + "TetherTargetStretch=" + std::to_string(a.tether_target_stretch) + "\n";
        content += pfx + "WallClearanceScale=" + std::to_string(a.wall_clearance_scale) + "\n";
        content += pfx + "BranchSmoothK=" + std::to_string(a.branch_smooth_k) + "\n";
        content += pfx + "WallCutSmooth=" + std::to_string(a.wall_cut_smooth) + "\n";
    }
    content += "\n";
    std::ofstream out(filepath, std::ios::out | std::ios::binary);
    if (out.is_open()) {
        out.write(content.data(), content.size());
        out.close();
    }
}

void load_growth_algorithms_from_ini(std::vector<GrowthAlgorithmProfile>& algos, int& selected_idx, const std::string& filepath) {
    if (!std::filesystem::exists(filepath)) return;
    std::ifstream in(filepath);
    if (!in.is_open()) return;

    std::string line;
    bool in_section = false;
    std::unordered_map<std::string, std::string> kv;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty()) continue;
        if (line.front() == '[') {
            if (line == "[GrowthAlgorithms]") {
                in_section = true;
                continue;
            } else if (in_section) {
                break;
            }
        }
        if (in_section) {
            size_t eq = line.find('=');
            if (eq != std::string::npos) {
                kv[line.substr(0, eq)] = line.substr(eq + 1);
            }
        }
    }
    in.close();

    if (kv.count("Count")) {
        int count = std::atoi(kv["Count"].c_str());
        if (count > 0) {
            algos.clear();
            for (int i = 0; i < count; ++i) {
                GrowthAlgorithmProfile a;
                std::string pfx = "Algo" + std::to_string(i) + "_";
                std::unordered_map<std::string, std::string> sub_kv;
                for (const auto& pair : kv) {
                    if (pair.first.find(pfx) == 0) {
                        sub_kv[pair.first.substr(pfx.size())] = pair.second;
                    }
                }
                apply_kv_to_algorithm(sub_kv, "Algorithm " + std::to_string(i + 1), a);
                algos.push_back(a);
            }
            if (kv.count("Selected")) {
                selected_idx = std::clamp(std::atoi(kv["Selected"].c_str()), 0, static_cast<int>(algos.size()) - 1);
            }
        }
    }
}
