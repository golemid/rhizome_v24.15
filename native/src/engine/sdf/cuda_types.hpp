#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include <glm/glm.hpp>

namespace rhizome {

struct CudaCapsule {
    float ax, ay, az;
    float bx, by, bz;
    float ra, rb;
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    int strand_id{0};
};

struct CudaTriangle {
    glm::vec3 v0, v1, v2;
    glm::vec3 normal;
};

inline glm::vec4 get_seed_color_vec4(int seed_idx) {
    const int idx = (seed_idx >= 0 ? seed_idx : 0) % 10;
    switch (idx) {
        case 0: return glm::vec4(0.098f, 0.827f, 0.635f, 1.0f); // Primary Teal (#19D3A2)
        case 1: return glm::vec4(1.000f, 0.541f, 0.122f, 1.0f); // Warm Orange (#FF8A1F)
        case 2: return glm::vec4(0.220f, 0.741f, 0.973f, 1.0f); // Sky Cyan (#38BDF8)
        case 3: return glm::vec4(0.957f, 0.247f, 0.369f, 1.0f); // Coral Rose (#F43F5E)
        case 4: return glm::vec4(0.659f, 0.333f, 0.969f, 1.0f); // Vivid Violet (#A855F7)
        case 5: return glm::vec4(0.918f, 0.702f, 0.031f, 1.0f); // Amber Gold (#EAB308)
        case 6: return glm::vec4(0.518f, 0.800f, 0.086f, 1.0f); // Lime Green (#84CC16)
        case 7: return glm::vec4(0.925f, 0.282f, 0.600f, 1.0f); // Hot Magenta (#EC4899)
        case 8: return glm::vec4(0.024f, 0.714f, 0.831f, 1.0f); // Deep Aqua (#06B6D4)
        default: return glm::vec4(0.976f, 0.451f, 0.086f, 1.0f); // Tangerine (#F97316)
    }
}

struct CudaSporeData {
    glm::vec3 pos;
    bool alive;
    int closest_node_idx;
    float min_dist;
};

struct CudaBranchNode {
    glm::vec3 pos;
    float radius;
};

struct CudaVolSection {
    glm::vec3 pos;
    float radius;
    int parent_sec;
    int root_id;
    int depth;
    int pinned;
};

struct CudaVolTip {
    int sec_idx;
    int root_id;
    glm::vec3 dir;
    float radius;
    int steps_since_branch;
    int stalled_steps;
    int generation;
    // v24.12 Track F R1 (Developmental mode 4): per-root step counter
    // (drives the per-order length cap). Foreign creation sites (shared
    // init/respawn/commit kernels) initialize it to 0 so the field is
    // always defined; modes 0-3 never read it.
    int steps_grown;
};

struct CudaVolGrowthParams {
    float search_radius;
    float kill_radius;
    float growth_step;
    bool variable_step = false;
    float min_growth_step = 0.65f;
    float max_growth_step = 2.25f;
    float branching_rate;
    float base_radius;
    float tip_radius;
    bool variable_taper = false;
    float root_roughness;
    float push_strength;
    float space_fill_drive;
    float taper_curve;
    float voxel_size;
    float min_vessel_sdf;
    int total_steps;
    int current_iter;
    uint32_t seed;

    // Internal GPU colonization algorithm equations & behavior constants
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
    float steering_cone = 35.0f;
    float backward_pull = 1.0f;
    float tensile_straightening = 0.0f;
    int algorithm_mode = 0; // 0 = classic, 1 = root-bound, 2 = murray vascular, 3 = surface creeping
    float murray_exponent = 3.0f;
    float murray_branch_angle = 37.5f;
    float surface_adherence = 0.0f; // [0..1] Attraction to and crawling along frozen root surfaces

    // Lateral roots parameters
    float lateral_density = 5.00f;
    float lateral_trunk_caliber = 0.60f;
    float lateral_tip_caliber = 0.40f;
    float lateral_max_length = 8.0f;
    float lateral_wander = 0.35f;
    float lateral_branch_angle = 45.0f;

    // v24.12 Track F R1: Developmental mode (algorithm_mode == 4) —
    // FSPM-class per-order table. Mirrors ColonizationConfig (built per
    // step from GrowthState in colonization.cpp step_volumetric_session).
    // Orders: 0 = seminal framework, 1/2/3 = lateral orders.
    float dev_gsa_deg[4] = {62.0f, 78.0f, 82.0f, 86.0f};      // gravitropic setpoint from the downward vertical (deg)
    float dev_rate[4] = {1.00f, 0.72f, 0.50f, 0.34f};         // per-order elongation multiplier on growth_step
    int   dev_spacing[4] = {6, 5, 4, 4};                       // min steps between lateral spawns of this order
    float dev_order_radius[4] = {1.10f, 0.62f, 0.42f, 0.32f}; // growth-time radius per order (mm)
    float dev_tip_radius[4] = {0.55f, 0.40f, 0.34f, 0.30f};   // Murray terminal demand per order (mm)
    int   dev_max_steps[4] = {70, 46, 30, 20};                // per-root length cap (steps)
    float dev_grav_gain = 0.22f;      // Firn-Digby gravitropic response gain (rad/step at sin=1)
    float dev_noise = 0.10f;          // directional wander amplitude (rad)
    float dev_branch_angle = 62.0f;   // lateral departure angle from the parent axis (deg)
    float dev_murray_gamma = 2.80f;   // Murray exponent for bake-time caliber
    float dev_min_print_radius = 0.30f; // SLA printable floor (mm) — nothing thinner survives the bake
    int   dev_section_budget = 30000; // hard cap on total sections (bake-size guard)
};

class CudaVolumeGrid {
public:
    std::vector<float> data;
    glm::ivec3 min_coord = glm::ivec3(0);
    glm::ivec3 max_coord = glm::ivec3(0);
    glm::ivec3 dims = glm::ivec3(0);
    float voxel_size = 0.5f;
    uint64_t version = 1;
    mutable float cached_min_sdf_ = 0.0f;
    mutable size_t cached_min_sdf_count_ = 0;

    bool is_valid() const {
        return !data.empty() && dims.x > 0 && dims.y > 0 && dims.z > 0;
    }

    float get_min_sdf() const;

    float sample_nearest(const glm::vec3& world_pos) const;

    float sample_trilinear(const glm::vec3& world_pos) const;
};

} // namespace rhizome
