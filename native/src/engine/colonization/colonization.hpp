#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>
#include "engine/sdf/cuda_rasterizer.hpp"

struct ColonizationNode {
    glm::vec3 position{0.0f};
    int parent_index{-1};
    float radius{0.5f};
    int depth{0};
    float energy{1.0f};
    int strand_id{0};
};

struct ColonizationConfig {
    int spore_count = 10000;
    float search_radius = 8.0f;
    float kill_radius = 2.0f;
    float growth_step = 1.5f;
    bool variable_step = false;
    float min_growth_step = 0.65f;
    float max_growth_step = 2.25f;
    float branching_rate = 0.6f;
    float steering_cone = 35.0f;
    float base_radius = 2.5f;
    float tip_radius = 0.5f;
    bool variable_taper = false;
    int max_iterations = 80;
    uint32_t seed = 42;
    int algorithm_mode = 0;       // 0 = Classic Space Colonization, 1 = Natural Root-Bound (Obstacle & Pushing), 2 = Vascular Murray Bifurcation
    int primary_roots = 3;        // Number of primary root arms spawned per seed [1..8]
    float root_roughness = 0.35f; // Organic gnarliness / micro-obstacle deflection [0..1]
    float push_strength = 0.45f;  // Root-to-root physical pushing force [0..1]
    float space_fill_drive = 0.85f; // Relentless continuation into remaining space [0..1]
    float backward_pull = 1.0f;   // Backward distance reeling [0..1]
    float tensile_straightening = 0.0f; // Forward tensile alignment / streamline flow [0..1]
    float murray_exponent = 3.0f; // Murray's law exponent (r0^gamma = r1^gamma + r2^gamma, typical 2.5 - 3.0)
    float murray_branch_angle = 37.5f; // Optimal bifurcation angle based on Murray vascular mechanics [deg]
    float surface_adherence = 0.0f; // [0..1] Attraction to and crawling along frozen root surfaces

    // Internal CUDA GPU colonization algorithm rules & constants
    float tip_inertia_weight = 1.25f;
    float spore_pull_weight = 0.75f;
    float roughness_amplitude = 0.65f;
    float feeler_nar_fwd = 0.92f;
    float feeler_nar_side = 0.38f;
    float feeler_med_fwd = 0.72f;
    float feeler_med_side = 0.69f;
    float feeler_wide_fwd = 0.42f;
    float feeler_wide_side = 0.90f;
    float feeler_wall_in = 0.55f;
    float feeler_wall_out = 0.45f;
    float clearance_score_weight = 2.50f;
    float align_bonus_weight = 0.65f;
    float lateral_fwd_weight = 0.52f;
    float lateral_side_weight = 0.85f;
    float lateral_radius_scale = 0.86f;
    int max_branch_generations = 4;
    float skin_gap_scale = 0.75f;
    float push_base_gain = 0.25f;
    float push_strength_gain = 0.55f;
    float max_step_disp_ratio = 0.28f;
    float tether_max_stretch = 1.25f;
    float tether_target_stretch = 1.15f;
    float wall_clearance_scale = 0.85f;
    float branch_smooth_k = 0.00f;
    float wall_cut_smooth = 0.0f;
    std::vector<uint8_t> frozen_seeds;

    // Lateral Roots Configuration & Per-Seed Control
    float lateral_density = 5.00f;
    float lateral_trunk_caliber = 0.60f;
    float lateral_tip_caliber = 0.40f;
    float lateral_max_length = 8.0f;
    float lateral_wander = 0.35f;
    float lateral_branch_angle = 45.0f;
    std::vector<uint8_t> seed_has_lateral;
    std::vector<uint8_t> seed_snapped_to_root;

    // v24.12 Track F R1: Developmental mode (algorithm_mode == 4) — FSPM-class
    // per-order growth (gravitropic setpoints, acropetal laterals, Murray
    // caliber, SLA print floor). Full table mirrors CudaVolGrowthParams
    // (cuda_types.hpp); the user surface (GSA[0]/[1], gain, noise, branch
    // angle, gamma, floor, budget, vigor, lateral frequency) is derived in
    // growth_controller build_colonization_config_from_state. Orders:
    // 0 = seminal framework, 1/2/3 = lateral orders. See TRACK_F_R1_NOTE.md
    // and TRACK_F_R2_NOTE.md (v24.13 eye-driven retune).
    // v24.13 Track F R2: table retuned through the structure-dump analyzer
    // (scripts/eye) — flaring GSA hierarchy 55/68/78/86, sparse cascade
    // (spacing 16/20/28/36), biological Murray demand {0.55,0.38,0.26,0.20},
    // and vessel-relative per-root length caps (dev_len_frac x dev_vessel_ref_mm,
    // set by the controller at Start; the static max_steps below are the
    // fallback when no vessel reference is known, sized for a ~90 mm vessel).
    float dev_gsa_deg[4] = {55.0f, 68.0f, 78.0f, 86.0f};      // gravitropic setpoint from the downward vertical (deg)
    float dev_rate[4] = {1.00f, 0.72f, 0.50f, 0.34f};         // per-order elongation multiplier on growth_step
    int   dev_spacing[4] = {16, 20, 28, 36};                   // min steps between lateral spawns of this order
    float dev_order_radius[4] = {1.10f, 0.62f, 0.42f, 0.32f}; // growth-time radius per order (mm)
    float dev_tip_radius[4] = {0.55f, 0.38f, 0.26f, 0.20f};   // Murray terminal demand per order (mm, biological)
    int   dev_max_steps[4] = {105, 61, 35, 26};               // per-root length cap (steps; fallback when dev_vessel_ref_mm unset)
    float dev_len_frac[4] = {1.55f, 0.65f, 0.26f, 0.13f};     // per-order target length as a fraction of the vessel reference (mm)
    float dev_vessel_ref_mm = 0.0f;                           // vessel reference size (largest bbox extent, mm); 0 = unknown -> static max_steps
    float dev_grav_gain = 0.26f;      // Firn-Digby gravitropic response gain
    float dev_noise = 0.10f;          // directional wander amplitude (rad; order-0 damped 0.45x in-kernel)
    float dev_branch_angle = 55.0f;   // lateral departure angle from the parent axis (deg)
    float dev_murray_gamma = 2.80f;   // Murray exponent for bake-time caliber
    float dev_min_print_radius = 0.30f; // SLA printable floor (mm)
    int   dev_section_budget = 30000; // hard cap on total sections
};

struct ProfilingStats {
    double spore_sampling_ms = 0.0;
    double simulation_total_ms = 0.0;
    double avg_iteration_ms = 0.0;
    int iterations_executed = 0;
    double vdb_raster_ms = 0.0;
    double mesh_extraction_ms = 0.0;
};

class SpaceColonization {
public:
    SpaceColonization() = default;

    // Seeds initial roots and rejection-samples spores inside vessel 3D distance volume
    void initialize(std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
                    const std::vector<glm::vec3>& seed_positions,
                    const ColonizationConfig& config);

    // Runs simulation until completion or max iterations
    void run(std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
             const ColonizationConfig& config,
             ProfilingStats& stats);

    // Runs Natural Root-Bound simulation (obstacle deflection, root-to-root pushing, relentless space filling)
    void run_root_bound(std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
                        const ColonizationConfig& config,
                        ProfilingStats& stats);

    // Direct 3D Volumetric Root Simulation (no 1D skeleton inflation; simulates 3D root bodies with real tip thickness,
    // 3D capsule-to-capsule surface sensing, obstacle deflection, and 3D root-to-root pushing)
    std::shared_ptr<rhizome::CudaVolumeGrid> simulate_volumetric_roots(
        std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
        const std::vector<glm::vec3>& seed_positions,
        const ColonizationConfig& config,
        float voxel_size,
        float taper_curve,
        ProfilingStats& stats
    );

    // Stateful Live Step-by-Step 3D Volumetric Root Simulation (supports real-time slider adjustments while growing)
    void initialize_volumetric_session(
        std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
        const std::vector<glm::vec3>& seed_positions,
        const ColonizationConfig& config
    );

    bool step_volumetric_session(
        std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
        const ColonizationConfig& config,
        float voxel_size,
        float taper_curve,
        int steps_to_advance
    );

    std::shared_ptr<rhizome::CudaVolumeGrid> rasterize_volumetric_session(
        float voxel_size,
        bool preserve_beaded = false,
        float branch_smooth_k = 0.35f
    ) const;

    // Stateful Live Step-by-Step Classic Space Colonization
    bool step_classic_session(
        std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
        const ColonizationConfig& config,
        int steps_to_advance
    );

    // Extends step budget, wakes up root tips, and replenishes spores if depleted so growth can continue past Max Steps
    void prepare_resume_session(
        std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
        const ColonizationConfig& config
    );

    // Synchronizes newly added seeds and per-seed freeze states on GPU during an existing growth session
    void sync_seeds_session(
        std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
        const std::vector<glm::vec3>& seed_positions,
        const ColonizationConfig& config
    );

    // Prunes all root branches, sections, and tips that originated from a deleted seed index on GPU
    void delete_seed(int seed_index);

    // Dynamically redistributes trunk/tip calibers and retapers existing root sections 100% on GPU
    // v24.12 Track F R1: in developmental mode (dev_cfg with algorithm_mode==4)
    // this routes to the Murray census instead — trunk caliber acts as the cap.
    void update_caliber(
        float trunk_caliber,
        float tip_radius,
        float taper_curve,
        bool variable_taper,
        uint32_t rng_seed,
        const ColonizationConfig* dev_cfg = nullptr
    );

    // Snaps a seed position directly onto the outer surface of existing root sections on GPU
    bool snap_seed_to_root_surface(glm::vec3 query_pos, float target_offset, glm::vec3& out_snapped_pos, glm::vec3& out_normal) const {
        float time_ms = 0.0f;
        if (!vol_sections_.empty()) {
            return rhizome::CudaRasterizer::snap_seed_to_root_surface_gpu(
                reinterpret_cast<const rhizome::CudaVolSection*>(vol_sections_.data()),
                vol_sections_.size(),
                query_pos,
                target_offset,
                out_snapped_pos,
                out_normal,
                time_ms
            );
        } else if (!nodes_.empty()) {
            return rhizome::CudaRasterizer::snap_seed_to_nodes_gpu(
                nodes_.data(),
                nodes_.size(),
                query_pos,
                target_offset,
                out_snapped_pos,
                out_normal,
                time_ms
            );
        }
        return false;
    }

    // Samples spore cloud inside vessel volume for immediate 3D visualization
    void sample_spores_only(
        std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
        const ColonizationConfig& config
    );

    // Clears all simulation state, nodes, volumetric sections, tips, and spores
    void clear() {
        nodes_.clear();
        spores_.clear();
        vol_sections_.clear();
        vol_active_tips_.clear();
        vol_next_root_id_ = 1;
        current_iter_ = 0;
        bonus_iterations_ = 0;
        min_vessel_sdf_ = 0.0f;
        num_seminal_sections_ = 0;
    }

    // Generates or updates fine lateral roots distributed equally across seminal roots 100% on GPU
    void sync_lateral_roots_session(const ColonizationConfig& config);

    // v24.12 Track F R1: bake-time developmental caliber — bottom-up Murray
    // census (r^gamma demand accumulated child->parent; parent index < own
    // index always holds by the append discipline), clamped to the SLA print
    // floor and the trunk-caliber cap. Host-side O(N); deterministic.
    bool apply_dev_murray_caliber(const ColonizationConfig& config);

    // v24.12 Track F R1: SLA print audit — per-order section census, minimum
    // radius vs floor, worst slenderness ratios, connectivity check, and
    // estimated resin volume from the section capsules.
    void emit_print_audit(const ColonizationConfig& config);

    int get_current_iteration() const { return current_iter_; }
    size_t get_volumetric_section_count() const { return vol_sections_.size(); }
    size_t get_active_tip_count() const { return vol_active_tips_.size(); }

    const std::vector<ColonizationNode>& get_nodes() const { return nodes_; }
    const std::vector<glm::vec3>& get_spores() const { return spores_; }

    struct VolCrossSection {
        glm::vec3 pos{0.0f};
        float radius{0.5f};
        int parent_sec{-1};
        int root_id{0};
        int depth{0};
        int pinned{0};
    };

    const std::vector<VolCrossSection>& get_volumetric_sections() const { return vol_sections_; }

    struct VolRootTip {
        int sec_idx{0};
        int root_id{0};
        glm::vec3 dir{0.0f, 0.0f, 1.0f};
        float radius{0.5f};
        int steps_since_branch{0};
        int stalled_steps{0};
        int generation{0};
        // v24.12 Track F R1: per-root step counter (Developmental mode 4).
        // Layout-mirrors CudaVolTip::steps_grown (cuda_types.hpp).
        int steps_grown{0};
    };

private:
    std::vector<ColonizationNode> nodes_;
    std::vector<glm::vec3> spores_;
    std::vector<VolCrossSection> vol_sections_;
    std::vector<VolRootTip> vol_active_tips_;
    size_t num_seminal_sections_{0};
    int vol_next_root_id_{1};
    int current_iter_{0};
    int bonus_iterations_{0};
    float min_vessel_sdf_{0.0f};
};

namespace rhizome {
    using SpaceColonization = ::SpaceColonization;
    using ColonizationConfig = ::ColonizationConfig;
    using ColonizationNode = ::ColonizationNode;
    using ProfilingStats = ::ProfilingStats;
}
