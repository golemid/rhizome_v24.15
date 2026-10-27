#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/cuda_common.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace rhizome {

// ============================================================================
// 5. CUDA GPU 3D Volumetric Root Colonization Engine
//    (11-Feeler Cone Sensing, Spore Pruning, 3D Capsule Pushing & Rope-Reeling)
// ============================================================================

struct CudaTipProposal {
    int action;        // 0 = sleeping, -1 = stalled, 1 = advance
    glm::vec3 best_p1;
    glm::vec3 best_dir;
    float r1;
    int next_depth;
    int has_lateral;   // 1 = sprout lateral branch
    glm::vec3 lat_dir;
    float lat_r;
};

#ifdef __CUDACC__

__device__ void dev_closest_points_segment_segment(
    const glm::vec3& p1, const glm::vec3& q1,
    const glm::vec3& p2, const glm::vec3& q2,
    float& s, float& t,
    glm::vec3& c1, glm::vec3& c2
) {
    glm::vec3 d1 = q1 - p1;
    glm::vec3 d2 = q2 - p2;
    glm::vec3 r = p1 - p2;
    float a = glm::dot(d1, d1);
    float e = glm::dot(d2, d2);
    float f = glm::dot(d2, r);

    if (a <= 1e-8f && e <= 1e-8f) {
        s = t = 0.0f;
        c1 = p1;
        c2 = p2;
        return;
    }
    if (a <= 1e-8f) {
        s = 0.0f;
        t = dev_clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = glm::dot(d1, r);
        if (e <= 1e-8f) {
            t = 0.0f;
            s = dev_clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = glm::dot(d1, d2);
            float denom = a * e - b * b;
            if (denom != 0.0f) {
                s = dev_clamp((b * f - c * e) / denom, 0.0f, 1.0f);
            } else {
                s = 0.0f;
            }
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = dev_clamp(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = dev_clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }
    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
}

__device__ bool dev_shares_recent_fork(
    const CudaVolSection* __restrict__ sections,
    int sec_a,
    int sec_b,
    int max_hops
) {
    if (sec_a < 0 || sec_b < 0) return false;
    int anc_a[10];
    int count_a = 0;
    int cur = sec_a;
    while (cur >= 0 && count_a <= max_hops && count_a < 10) {
        anc_a[count_a++] = cur;
        cur = sections[cur].parent_sec;
    }
    cur = sec_b;
    int hops_b = 0;
    while (cur >= 0 && hops_b <= max_hops) {
        for (int k = 0; k < count_a; ++k) {
            if (anc_a[k] == cur) return true;
        }
        cur = sections[cur].parent_sec;
        hops_b++;
    }
    return false;
}

__device__ float dev_sense_3d_root_environment(
    const CudaVolSection* __restrict__ sections,
    const int* __restrict__ body_head,
    const int* __restrict__ body_next,
    const glm::ivec3* __restrict__ body_cells,
    uint32_t hash_mask,
    float inv_body_cell,
    float step_len,
    float min_skin_gap,
    int collar_hops,
    const glm::vec3& p0,
    const glm::vec3& p1,
    float r0,
    float r1,
    int parent_sec_idx,
    glm::vec3* out_surface_repel,
    float surface_adherence = 0.0f,
    float* out_adherence_reward = nullptr
) {
    glm::vec3 mid = (p0 + p1) * 0.5f;
    int bcx = static_cast<int>(floorf(mid.x * inv_body_cell));
    int bcy = static_cast<int>(floorf(mid.y * inv_body_cell));
    int bcz = static_cast<int>(floorf(mid.z * inv_body_cell));

    glm::vec3 cand_dir = p1 - p0;
    float cand_len = sqrtf(glm::dot(cand_dir, cand_dir));
    if (cand_len > 1e-5f) cand_dir /= cand_len;

    float min_effective_margin = 1e9f;
    glm::vec3 repel_sum(0.0f);

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                int qx = bcx + dx, qy = bcy + dy, qz = bcz + dz;
                uint32_t slot = dev_hash_coords(qx, qy, qz, hash_mask);
                for (int sec_j = body_head[slot]; sec_j >= 0; sec_j = body_next[sec_j]) {
                    glm::ivec3 sc = body_cells[sec_j];
                    if (sc.x != qx || sc.y != qy || sc.z != qz) continue;

                    CudaVolSection sj = sections[sec_j];
                    if (sj.parent_sec < 0) continue;

                    bool is_collar_sibling = dev_shares_recent_fork(sections, parent_sec_idx, sec_j, collar_hops);
                    CudaVolSection spj = sections[sj.parent_sec];

                    float s = 0.0f, t = 0.0f;
                    glm::vec3 c1(0.0f), c2(0.0f);
                    dev_closest_points_segment_segment(p0, p1, spj.pos, sj.pos, s, t, c1, c2);

                    if (is_collar_sibling) {
                        glm::vec3 diff_sib = p1 - sj.pos;
                        float d_sib = sqrtf(glm::dot(diff_sib, diff_sib));
                        if (d_sib > 1e-5f && d_sib < (r1 + sj.radius) * 1.5f) {
                            repel_sum += (diff_sib / d_sib) * 0.35f;
                        }
                        continue;
                    }

                    float rc1 = r0 + (r1 - r0) * s;
                    float rc2 = spj.radius + (sj.radius - spj.radius) * t;

                    glm::vec3 diff = c1 - c2;
                    float center_dist = sqrtf(glm::dot(diff, diff));
                    float surface_gap = center_dist - (rc1 + rc2);

                    glm::vec3 other_dir = sj.pos - spj.pos;
                    float ol = sqrtf(glm::dot(other_dir, other_dir));
                    float parallel_ness = (ol > 1e-5f && cand_len > 1e-5f)
                        ? fabsf(glm::dot(cand_dir, other_dir / ol))
                        : 1.0f;

                    bool is_frozen_scaffold = ((sj.pinned & 1) != 0);
                    if (is_frozen_scaffold && surface_adherence > 0.01f) {
                        // Creeping root adherence to frozen root scaffold surface:
                        // Allow tight skin contact down to slight overlap for watertight fusion
                        float max_allowed_penetration = -r1 * 0.25f;
                        float local_margin = surface_gap - max_allowed_penetration;
                        if (local_margin < min_effective_margin) {
                            min_effective_margin = local_margin;
                        }

                        // Clinging proximity reward for candidate feelers
                        float cling_dist = fabsf(surface_gap);
                        if (cling_dist < step_len * 1.6f) {
                            float cling_w = 1.0f - (cling_dist / (step_len * 1.6f));
                            if (out_adherence_reward) {
                                *out_adherence_reward += cling_w * cling_w * surface_adherence;
                            }
                        }

                        // If penetrating too deeply into the frozen root interior, gently steer outward
                        if (surface_gap < max_allowed_penetration && center_dist > 1e-5f) {
                            float w = (max_allowed_penetration - surface_gap) / (rc1 + rc2);
                            repel_sum += (diff / center_dist) * (w * w);
                        }
                        continue;
                    }

                    float local_skin_gap = fmaxf(min_skin_gap, (rc1 + rc2) * 0.34f);
                    float required_gap = local_skin_gap * (0.50f + 0.65f * parallel_ness);
                    float margin = surface_gap - required_gap;

                    if (margin < min_effective_margin) {
                        min_effective_margin = margin;
                    }

                    float sense_band = required_gap + step_len * 1.10f;
                    if (surface_gap < sense_band && center_dist > 1e-5f) {
                        float w = (sense_band - surface_gap) / sense_band;
                        repel_sum += (diff / center_dist) * (w * w);
                    }
                }
            }
        }
    }

    if (out_surface_repel) *out_surface_repel = repel_sum;
    return min_effective_margin;
}

// Kernel 1: Insert all 3D root sections into GPU spatial hash table
__global__ void gpu_insert_sections_hash_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    float inv_cell_size,
    uint32_t hash_mask,
    int* __restrict__ body_head,
    int* __restrict__ body_next,
    glm::ivec3* __restrict__ body_cells
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;

    int p = sections[i].parent_sec;
    glm::vec3 mid = (p >= 0) ? ((sections[p].pos + sections[i].pos) * 0.5f) : sections[i].pos;
    int cx = static_cast<int>(floorf(mid.x * inv_cell_size));
    int cy = static_cast<int>(floorf(mid.y * inv_cell_size));
    int cz = static_cast<int>(floorf(mid.z * inv_cell_size));
    body_cells[i] = glm::ivec3(cx, cy, cz);

    uint32_t slot = dev_hash_coords(cx, cy, cz, hash_mask);
    int prev = atomicExch(&body_head[slot], i);
    body_next[i] = prev;
}

// Kernel 2: Prune spores occupied by 3D root bodies and insert surviving spores into GPU spatial hash table
__global__ void gpu_prune_and_hash_spores_kernel(
    const glm::vec3* __restrict__ spores,
    int num_spores,
    const CudaVolSection* __restrict__ sections,
    const int* __restrict__ body_head,
    const int* __restrict__ body_next,
    const glm::ivec3* __restrict__ body_cells,
    uint32_t body_hash_mask,
    float inv_body_cell,
    float kill_radius,
    float inv_spore_cell,
    uint32_t spore_hash_mask,
    uint8_t* __restrict__ spore_alive,
    int* __restrict__ spore_head,
    int* __restrict__ spore_next,
    glm::ivec3* __restrict__ spore_cells
) {
    int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (s_idx >= num_spores) return;

    glm::vec3 sp = spores[s_idx];
    int bcx = static_cast<int>(floorf(sp.x * inv_body_cell));
    int bcy = static_cast<int>(floorf(sp.y * inv_body_cell));
    int bcz = static_cast<int>(floorf(sp.z * inv_body_cell));

    bool occupied = false;
    for (int dx = -1; dx <= 1 && !occupied; ++dx) {
        for (int dy = -1; dy <= 1 && !occupied; ++dy) {
            for (int dz = -1; dz <= 1 && !occupied; ++dz) {
                int qx = bcx + dx, qy = bcy + dy, qz = bcz + dz;
                uint32_t slot = dev_hash_coords(qx, qy, qz, body_hash_mask);
                for (int sec_i = body_head[slot]; sec_i >= 0; sec_i = body_next[sec_i]) {
                    glm::ivec3 sc = body_cells[sec_i];
                    if (sc.x != qx || sc.y != qy || sc.z != qz) continue;

                    CudaVolSection sec = sections[sec_i];
                    // Frozen root sections (pinned bit 0 set) do not kill spores or starve new growing roots
                    if ((sec.pinned & 1) != 0) continue;

                    float d_surf = 0.0f;
                    if (sec.parent_sec >= 0) {
                        CudaVolSection psec = sections[sec.parent_sec];
                        glm::vec3 ba = sec.pos - psec.pos;
                        float l2 = glm::dot(ba, ba);
                        float t = (l2 > 1e-8f) ? dev_clamp(glm::dot(sp - psec.pos, ba) / l2, 0.0f, 1.0f) : 0.0f;
                        glm::vec3 proj = psec.pos + t * ba;
                        float r_at_t = psec.radius + (sec.radius - psec.radius) * t;
                        glm::vec3 dvec = sp - proj;
                        d_surf = sqrtf(glm::dot(dvec, dvec)) - r_at_t;
                    } else {
                        glm::vec3 dvec = sp - sec.pos;
                        d_surf = sqrtf(glm::dot(dvec, dvec)) - sec.radius;
                    }
                    if (d_surf <= kill_radius) {
                        occupied = true;
                        break;
                    }
                }
            }
        }
    }

    if (occupied) {
        spore_alive[s_idx] = 0;
        spore_next[s_idx] = -1;
    } else {
        spore_alive[s_idx] = 1;
        int scx = static_cast<int>(floorf(sp.x * inv_spore_cell));
        int scy = static_cast<int>(floorf(sp.y * inv_spore_cell));
        int scz = static_cast<int>(floorf(sp.z * inv_spore_cell));
        spore_cells[s_idx] = glm::ivec3(scx, scy, scz);
        uint32_t slot = dev_hash_coords(scx, scy, scz, spore_hash_mask);
        int prev = atomicExch(&spore_head[slot], s_idx);
        spore_next[s_idx] = prev;
    }
}

// Kernel 3: Parallel GPU 11-Feeler Cone Environment Sensing & Lateral Branch Probing (1 GPU thread per active root tip)
__global__ void gpu_sense_and_advance_tips_kernel(
    const CudaVolTip* __restrict__ tips,
    int num_tips,
    const CudaVolSection* __restrict__ sections,
    const int* __restrict__ body_head,
    const int* __restrict__ body_next,
    const glm::ivec3* __restrict__ body_cells,
    uint32_t body_hash_mask,
    float inv_body_cell,
    const glm::vec3* __restrict__ spores,
    const int* __restrict__ spore_head,
    const int* __restrict__ spore_next,
    const glm::ivec3* __restrict__ spore_cells,
    uint32_t spore_hash_mask,
    float inv_spore_cell,
    const float* __restrict__ vessel_sdf,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    CudaVolGrowthParams params,
    int collar_hops,
    int branch_interval,
    float min_skin_gap,
    CudaTipProposal* __restrict__ out_proposals
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= num_tips) return;

    CudaVolTip tip = tips[ti];
    CudaTipProposal prop;
    prop.action = 0;
    prop.best_p1 = glm::vec3(0.0f);
    prop.best_dir = tip.dir;
    prop.r1 = tip.radius;
    prop.next_depth = 0;
    prop.has_lateral = 0;
    prop.lat_dir = glm::vec3(0.0f);
    prop.lat_r = params.tip_radius;

    if (tip.stalled_steps > 4) {
        out_proposals[ti] = prop;
        return;
    }

    const glm::vec3 p0 = sections[tip.sec_idx].pos;
    const float r0 = sections[tip.sec_idx].radius;
    const int next_depth = sections[tip.sec_idx].depth + 1;
    const float safe_taper = fmaxf(0.2f, params.taper_curve);

    float cur_taper_span = static_cast<float>(max(max(params.total_steps, next_depth), 5));
    float progress = dev_clamp(static_cast<float>(next_depth) / cur_taper_span, 0.0f, 1.0f);
    float min_r = fminf(params.base_radius, params.tip_radius);
    float max_r = fmaxf(params.base_radius, params.tip_radius);
    float gen_decay = powf(fminf(params.lateral_radius_scale, 0.78f), static_cast<float>(tip.generation));
    float branch_hash = dev_hash_01(params.seed + static_cast<uint32_t>(tip.root_id * 7919 + tip.generation * 1013 + 53));
    float branch_scale = (tip.generation == 0) ? (0.72f + 0.28f * branch_hash) : (0.48f + 0.52f * branch_hash);
    float branch_max_r = min_r + (max_r - min_r) * dev_clamp(gen_decay * branch_scale, 0.08f, 1.0f);
    float r1 = params.tip_radius + (branch_max_r - params.tip_radius) * powf(1.0f - progress, safe_taper);
    if (params.variable_taper) {
        float wave = 0.5f + 0.5f * sinf(static_cast<float>(next_depth) * 0.65f + static_cast<float>(tip.root_id) * 2.17f);
        float jitter = dev_hash_01(params.seed + static_cast<uint32_t>(next_depth * 733 + tip.root_id * 1999 + tip.sec_idx * 37));
        float t_var = dev_clamp(0.45f * branch_hash + 0.35f * wave + 0.20f * jitter, 0.0f, 1.0f) * dev_clamp(gen_decay, 0.15f, 1.0f);
        r1 = min_r + (max_r - min_r) * t_var;
    }

    float local_step_floor = fmaxf(0.25f, fmaxf(r0 * 1.12f, r1 * 1.12f));
    float min_s = fmaxf(local_step_floor, fminf(params.min_growth_step, params.max_growth_step));
    float max_s = fmaxf(min_s, fmaxf(params.min_growth_step, params.max_growth_step));
    float step_len = fmaxf(local_step_floor, params.growth_step);
    if (params.variable_step) {
        float cur_sdf = dev_sample_sdf_trilinear(vessel_sdf, vessel_min_coord, vessel_dims, params.voxel_size, p0);
        float wall_clear = dev_clamp((-cur_sdf - r1) / fmaxf(max_s * 1.5f, 0.5f), 0.0f, 1.0f);
        float gen_decay = powf(0.75f, static_cast<float>(tip.generation));
        float branch_hash = dev_hash_01(params.seed + static_cast<uint32_t>(tip.root_id * 7919 + tip.generation * 1013 + 37));
        float branch_speed = (tip.generation == 0) ? (0.80f + 0.20f * branch_hash) : (0.50f + 0.50f * branch_hash);
        float depth_decay = 1.0f - 0.25f * progress;
        float t_step = dev_clamp(gen_decay * branch_speed * depth_decay * (0.40f + 0.60f * wall_clear), 0.0f, 1.0f);
        step_len = fmaxf(local_step_floor, min_s + (max_s - min_s) * t_step);
    }
    const float search_r = fmaxf(step_len * 2.5f, params.search_radius);
    const float search_r2 = search_r * search_r;

    // 1. Sample nearby surviving nutrient spores from GPU spatial hash
    glm::vec3 env_pull(0.0f);
    int pull_count = 0;
    int scx = static_cast<int>(floorf(p0.x * inv_spore_cell));
    int scy = static_cast<int>(floorf(p0.y * inv_spore_cell));
    int scz = static_cast<int>(floorf(p0.z * inv_spore_cell));

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                int qx = scx + dx, qy = scy + dy, qz = scz + dz;
                uint32_t slot = dev_hash_coords(qx, qy, qz, spore_hash_mask);
                for (int s_idx = spore_head[slot]; s_idx >= 0; s_idx = spore_next[s_idx]) {
                    glm::ivec3 sc = spore_cells[s_idx];
                    if (sc.x != qx || sc.y != qy || sc.z != qz) continue;
                    glm::vec3 diff = spores[s_idx] - p0;
                    float d2 = glm::dot(diff, diff);
                    if (d2 < search_r2 && d2 > 1e-5f) {
                        float d = sqrtf(d2);
                        glm::vec3 sdir = diff / d;
                        float fwd = fmaxf(0.10f, 0.65f + 0.35f * glm::dot(sdir, tip.dir));
                        env_pull += sdir * (fwd / (1.0f + d * 0.2f));
                        pull_count++;
                    }
                }
            }
        }
    }

    glm::vec3 base_dir = tip.dir * params.tip_inertia_weight;
    if (pull_count > 0 && glm::dot(env_pull, env_pull) > 1e-10f) {
        base_dir += dev_safe_norm(env_pull, tip.dir) * params.spore_pull_weight;
    }
    if (params.root_roughness > 0.0f) {
        float sc = 1.0f / fmaxf(step_len * 2.1f, 0.6f);
        uint32_t salt = static_cast<uint32_t>(tip.sec_idx * 37 + params.current_iter);
        float px = p0.x * sc + static_cast<float>(salt) * 1.37f;
        float py = p0.y * sc + static_cast<float>(salt) * 2.53f;
        float pz = p0.z * sc + static_cast<float>(salt) * 3.89f;
        float rx = sinf(px * 1.6f + cosf(py * 2.2f)) + 0.5f * cosf(pz * 2.9f + px);
        float ry = sinf(py * 1.8f + cosf(pz * 2.0f)) + 0.5f * cosf(px * 2.5f + py);
        float rz = sinf(pz * 2.0f + cosf(px * 1.7f)) + 0.5f * cosf(py * 2.8f + pz);
        glm::vec3 rough = dev_safe_norm(glm::vec3(rx, ry, rz), glm::vec3(1.0f, 0.0f, 0.0f));
        rough -= glm::dot(rough, tip.dir) * tip.dir;
        if (glm::dot(rough, rough) > 1e-10f) {
            base_dir += dev_safe_norm(rough, tip.dir) * (params.root_roughness * params.roughness_amplitude);
        }
    }
    base_dir = dev_clamp_to_cone(dev_safe_norm(base_dir, tip.dir), tip.dir, params.steering_cone);

    glm::vec3 wall_n = dev_sample_sdf_grad(vessel_sdf, vessel_min_coord, vessel_dims, params.voxel_size, p0);
    glm::vec3 side = dev_safe_norm(glm::cross(wall_n, base_dir), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::vec3 tang_fwd = dev_safe_norm(glm::cross(side, wall_n), base_dir);

    glm::vec3 candidate_dirs[11];
    candidate_dirs[0]  = base_dir;
    candidate_dirs[1]  = dev_clamp_to_cone(dev_safe_norm(tang_fwd * params.feeler_nar_fwd + side * params.feeler_nar_side, base_dir), tip.dir, params.steering_cone);
    candidate_dirs[2]  = dev_clamp_to_cone(dev_safe_norm(tang_fwd * params.feeler_nar_fwd - side * params.feeler_nar_side, base_dir), tip.dir, params.steering_cone);
    candidate_dirs[3]  = dev_clamp_to_cone(dev_safe_norm(tang_fwd * params.feeler_med_fwd + side * params.feeler_med_side, base_dir), tip.dir, params.steering_cone);
    candidate_dirs[4]  = dev_clamp_to_cone(dev_safe_norm(tang_fwd * params.feeler_med_fwd - side * params.feeler_med_side, base_dir), tip.dir, params.steering_cone);
    candidate_dirs[5]  = dev_clamp_to_cone(dev_safe_norm(tang_fwd * params.feeler_wide_fwd + side * params.feeler_wide_side, base_dir), tip.dir, params.steering_cone);
    candidate_dirs[6]  = dev_clamp_to_cone(dev_safe_norm(tang_fwd * params.feeler_wide_fwd - side * params.feeler_wide_side, base_dir), tip.dir, params.steering_cone);
    candidate_dirs[7]  = dev_clamp_to_cone(dev_safe_norm(tang_fwd * 0.80f - wall_n * params.feeler_wall_in + side * 0.22f, base_dir), tip.dir, params.steering_cone);
    candidate_dirs[8]  = dev_clamp_to_cone(dev_safe_norm(tang_fwd * 0.80f - wall_n * params.feeler_wall_in - side * 0.22f, base_dir), tip.dir, params.steering_cone);
    candidate_dirs[9]  = dev_clamp_to_cone(dev_safe_norm(tang_fwd * 0.80f + wall_n * params.feeler_wall_out + side * 0.22f, base_dir), tip.dir, params.steering_cone);
    candidate_dirs[10] = dev_clamp_to_cone(dev_safe_norm(tang_fwd * 0.80f + wall_n * params.feeler_wall_out - side * 0.22f, base_dir), tip.dir, params.steering_cone);

    float best_score = -1e9f;
    float best_margin = -1e9f;
    glm::vec3 best_p1 = p0 + base_dir * step_len;
    glm::vec3 best_dir = base_dir;
    glm::vec3 repel_v(0.0f);

    int num_feelers = (params.space_fill_drive > 0.1f) ? 11 : 5;
    for (int f = 0; f < num_feelers; ++f) {
        glm::vec3 fdir = candidate_dirs[f];
        glm::vec3 fp1 = p0 + fdir * step_len;
        dev_confine_sphere_to_vessel(vessel_sdf, vessel_min_coord, vessel_dims, params.voxel_size, params.min_vessel_sdf, fp1, r1, &fdir, params.wall_clearance_scale);

        glm::vec3 actual_vec = fp1 - p0;
        float actual_len = sqrtf(glm::dot(actual_vec, actual_vec));
        if (actual_len < step_len * 0.60f && glm::dot(fdir, fdir) > 1e-10f) {
            fp1 = p0 + dev_safe_norm(fdir, base_dir) * step_len;
            dev_confine_sphere_to_vessel(vessel_sdf, vessel_min_coord, vessel_dims, params.voxel_size, params.min_vessel_sdf, fp1, r1, nullptr, params.wall_clearance_scale);
            actual_vec = fp1 - p0;
            actual_len = sqrtf(glm::dot(actual_vec, actual_vec));
        }
        if (actual_len < step_len * 0.20f) continue;

        glm::vec3 feeler_repel(0.0f);
        float feeler_adherence = 0.0f;
        float margin = dev_sense_3d_root_environment(
            sections, body_head, body_next, body_cells, body_hash_mask, inv_body_cell,
            step_len, min_skin_gap, collar_hops, p0, fp1, r0, r1, tip.sec_idx, &feeler_repel,
            params.surface_adherence, &feeler_adherence
        );

        if (f == 0) {
            repel_v = feeler_repel;
        }

        if (f == 0 && (margin < 0.0f || glm::dot(feeler_repel, feeler_repel) > 1e-10f)) {
            glm::vec3 def_dir = dev_clamp_to_cone(dev_safe_norm(fdir + dev_safe_norm(feeler_repel, fdir) * 0.80f, fdir), tip.dir, params.steering_cone);
            glm::vec3 def_p1 = p0 + def_dir * step_len;
            dev_confine_sphere_to_vessel(vessel_sdf, vessel_min_coord, vessel_dims, params.voxel_size, params.min_vessel_sdf, def_p1, r1, &def_dir, params.wall_clearance_scale);
            float def_adherence = 0.0f;
            float def_margin = dev_sense_3d_root_environment(
                sections, body_head, body_next, body_cells, body_hash_mask, inv_body_cell,
                step_len, min_skin_gap, collar_hops, p0, def_p1, r0, r1, tip.sec_idx, nullptr,
                params.surface_adherence, &def_adherence
            );
            glm::vec3 d_step = def_p1 - p0;
            if (def_margin >= margin && sqrtf(glm::dot(d_step, d_step)) >= step_len * 0.25f) {
                fp1 = def_p1;
                fdir = def_dir;
                margin = def_margin;
                feeler_adherence = def_adherence;
            }
        }

        glm::vec3 step_unit = dev_clamp_to_cone(dev_safe_norm(fp1 - p0, base_dir), tip.dir, params.steering_cone);
        float align_bonus = glm::dot(step_unit, base_dir) * params.align_bonus_weight;
        float clearance_score = fminf(margin, min_skin_gap * 1.5f) * params.clearance_score_weight;
        float adherence_score = feeler_adherence * (params.surface_adherence * 4.5f);
        float score = clearance_score + align_bonus + adherence_score;

        if (score > best_score) {
            best_score = score;
            best_margin = margin;
            best_p1 = fp1;
            best_dir = step_unit;
            repel_v = feeler_repel;
        }
    }

    const float min_allowed_margin = -min_skin_gap * (0.35f + 0.45f * params.push_strength);
    if (best_margin < min_allowed_margin && next_depth <= 4) {
        // Young roots from newly added seeds actively steer away from neighboring root bodies
        glm::vec3 repel_bias(0.0f);
        if (glm::dot(repel_v, repel_v) > 1e-8f) {
            repel_bias = dev_safe_norm(repel_v, tang_fwd);
        } else {
            repel_bias = tang_fwd;
        }

        glm::vec3 inward = -dev_safe_norm(wall_n, glm::vec3(0.0f, 0.0f, -1.0f));
        for (int def_try = 0; def_try < 8; ++def_try) {
            float angle = static_cast<float>(def_try) * 0.785398f + 0.20f;
            glm::vec3 esc_dir = dev_safe_norm(cosf(angle) * repel_bias + sinf(angle) * side + 0.35f * inward, base_dir);
            glm::vec3 esc_p1 = p0 + esc_dir * step_len;
            dev_confine_sphere_to_vessel(vessel_sdf, vessel_min_coord, vessel_dims, params.voxel_size, params.min_vessel_sdf, esc_p1, r1, &esc_dir, params.wall_clearance_scale);
            glm::vec3 sub_repel(0.0f);
            float esc_margin = dev_sense_3d_root_environment(
                sections, body_head, body_next, body_cells, body_hash_mask, inv_body_cell,
                step_len, min_skin_gap, collar_hops, p0, esc_p1, r0, r1, tip.sec_idx, &sub_repel,
                params.surface_adherence, nullptr
            );
            if (esc_margin > best_margin) {
                best_margin = esc_margin;
                best_p1 = esc_p1;
                best_dir = esc_dir;
            }
        }
    }

    const float effective_min_margin = (next_depth <= 4) ? (min_allowed_margin * 2.50f) : min_allowed_margin;
    if (best_margin < effective_min_margin) {
        if (next_depth <= 4) {
            // Young roots must not stall on step 0: take the best escape step outward
            prop.action = 1;
            prop.best_p1 = best_p1;
            prop.best_dir = best_dir;
            prop.r1 = r1;
            prop.next_depth = next_depth;
            out_proposals[ti] = prop;
            return;
        }
        prop.action = -1; // Stalled on this step
        out_proposals[ti] = prop;
        return;
    }

    prop.action = 1;
    prop.best_p1 = best_p1;
    prop.best_dir = best_dir;
    prop.r1 = r1;
    prop.next_depth = next_depth;

    // 2. Evaluate lateral branch candidate on GPU
    uint32_t rng_key = params.seed + static_cast<uint32_t>(params.current_iter * 1999 + ti * 7919 + tip.sec_idx * 31);
    float rand01 = dev_hash_01(rng_key);
    if ((tip.steps_since_branch + 1) >= branch_interval &&
        tip.generation < params.max_branch_generations &&
        rand01 <= params.branching_rate) {
        float pref_sign = ((tip.root_id + tip.generation + params.current_iter) % 2 == 0) ? 1.0f : -1.0f;
        int child_gen = tip.generation + 1;
        float child_gen_decay = powf(fminf(params.lateral_radius_scale, 0.78f), static_cast<float>(child_gen));
        float lat_j = dev_hash_01(rng_key + 7919u);
        float lat_scale = 0.48f + 0.52f * lat_j;
        float lat_max_r = min_r + (max_r - min_r) * dev_clamp(child_gen_decay * lat_scale, 0.08f, 1.0f);
        float murray_gamma = (params.murray_exponent > 1.0f) ? params.murray_exponent : 3.0f;
        // Murray's law: r_parent^gamma = r_continuation^gamma + r_lateral^gamma
        // For balanced/asymmetric vascular bifurcation: r_lat = r_parent * (lat_ratio)^(1/gamma)
        float lat_r = fmaxf(min_r, fminf(r1 * params.lateral_radius_scale, params.tip_radius + (lat_max_r - params.tip_radius) * powf(1.0f - progress, safe_taper)));
        if (params.algorithm_mode == 2) {
            float split_fraction = 0.35f + 0.30f * lat_j; // branch flow fraction
            lat_r = fmaxf(min_r, r1 * powf(split_fraction, 1.0f / murray_gamma));
            // Recalculate continuation radius to preserve Murray's vascular volume
            float cont_fraction = fmaxf(0.01f, 1.0f - split_fraction);
            prop.r1 = fmaxf(min_r, r1 * powf(cont_fraction, 1.0f / murray_gamma));
        } else if (params.variable_taper) {
            lat_r = min_r + (max_r - min_r) * (lat_j * dev_clamp(child_gen_decay, 0.15f, 1.0f));
        }
        float best_lat_margin = -1e9f;
        glm::vec3 best_lat_p1(0.0f);

        // Branching angle determination: Murray optimal angle cos(theta) = (r0^4 + r1^4 - r2^4) / (2 * r0^2 * r1^2)
        float fwd_w = params.lateral_fwd_weight;
        float side_w = params.lateral_side_weight;
        if (params.algorithm_mode == 2 && params.murray_branch_angle > 0.0f) {
            float angle_rad = params.murray_branch_angle * (3.14159265f / 180.0f);
            fwd_w = cosf(angle_rad);
            side_w = sinf(angle_rad);
        }

        for (int side_try = 0; side_try < 2; ++side_try) {
            float sign = (side_try == 0) ? pref_sign : -pref_sign;
            glm::vec3 lat_dir = dev_safe_norm(best_dir * fwd_w + sign * side * side_w, best_dir);
            float lat_step = params.variable_step ? fmaxf(min_s, step_len * 0.75f) : step_len;
            glm::vec3 lat_p1 = best_p1 + lat_dir * lat_step;
            dev_confine_sphere_to_vessel(vessel_sdf, vessel_min_coord, vessel_dims, params.voxel_size, params.min_vessel_sdf, lat_p1, lat_r, &lat_dir, params.wall_clearance_scale);

            glm::vec3 d_lat = lat_p1 - best_p1;
            if (sqrtf(glm::dot(d_lat, d_lat)) < lat_step * 0.25f) continue;

            float lat_margin = dev_sense_3d_root_environment(
                sections, body_head, body_next, body_cells, body_hash_mask, inv_body_cell,
                lat_step, min_skin_gap, collar_hops, best_p1, lat_p1, r1, lat_r, tip.sec_idx, nullptr,
                params.surface_adherence, nullptr
            );
            if (lat_margin > best_lat_margin) {
                best_lat_margin = lat_margin;
                best_lat_p1 = lat_p1;
            }
        }

        if (best_lat_margin >= 0.0f) {
            prop.has_lateral = 1;
            prop.lat_dir = dev_safe_norm(best_lat_p1 - best_p1, best_dir);
            prop.lat_r = lat_r;
        }
    }

    out_proposals[ti] = prop;
}

// Kernel 4: Parallel GPU 3D Capsule-to-Capsule Pushing Physics (1 GPU thread per root section)
// v24.02 P1.2 DRAFT (2026-10-22) — UNVERIFIED UNTIL TRACK B:
// the 12 float atomicAdd accumulation sites below are replaced by fixed-point
// uint64 atomicAdd (Q = 2^24, quantum ~6e-8). CUDA exposes no signed 64-bit
// atomicAdd overload, so the accumulator is unsigned long long: two's-complement
// addition mod 2^64 is associative, so the per-section displacement total is
// bit-identical regardless of pair-visit order (deterministic). The float buffer
// is refilled by a dequantize pass before the apply/relax kernels.
// Track B fix 1 (2026-10-22): original draft used long long, which does not
// compile (no such atomicAdd overload) — flavor changed to uint64, semantics equal.
__global__ void gpu_push_sections_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    const int* __restrict__ body_head,
    const int* __restrict__ body_next,
    const glm::ivec3* __restrict__ body_cells,
    uint32_t body_hash_mask,
    float inv_body_cell,
    float min_skin_gap,
    float push_gain,
    unsigned long long* __restrict__ sec_disp_q // v24.02 P1.2 DRAFT: quantized accumulator (3 uint64 per section)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;

    CudaVolSection si = sections[i];
    if (si.parent_sec < 0) return;
    CudaVolSection spi = sections[si.parent_sec];

    glm::vec3 mid = (spi.pos + si.pos) * 0.5f;
    int bcx = static_cast<int>(floorf(mid.x * inv_body_cell));
    int bcy = static_cast<int>(floorf(mid.y * inv_body_cell));
    int bcz = static_cast<int>(floorf(mid.z * inv_body_cell));

    glm::vec3 dir_i = si.pos - spi.pos;
    float len_i = sqrtf(glm::dot(dir_i, dir_i));
    if (len_i > 1e-5f) dir_i /= len_i;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                int qx = bcx + dx, qy = bcy + dy, qz = bcz + dz;
                uint32_t slot = dev_hash_coords(qx, qy, qz, body_hash_mask);
                for (int j = body_head[slot]; j >= 0; j = body_next[j]) {
                    if (j <= i) continue;
                    glm::ivec3 sc = body_cells[j];
                    if (sc.x != qx || sc.y != qy || sc.z != qz) continue;

                    CudaVolSection sj = sections[j];
                    if (sj.parent_sec < 0) continue;
                    if (si.parent_sec == j || sj.parent_sec == i || si.parent_sec == sj.parent_sec) continue;

                    CudaVolSection spj = sections[sj.parent_sec];
                    float s = 0.0f, t = 0.0f;
                    glm::vec3 c1(0.0f), c2(0.0f);
                    dev_closest_points_segment_segment(spi.pos, si.pos, spj.pos, sj.pos, s, t, c1, c2);

                    float rc1 = spi.radius + (si.radius - spi.radius) * s;
                    float rc2 = spj.radius + (sj.radius - spj.radius) * t;
                    glm::vec3 diff = c1 - c2;
                    float dist = sqrtf(glm::dot(diff, diff));

                    glm::vec3 dir_j = sj.pos - spj.pos;
                    float len_j = sqrtf(glm::dot(dir_j, dir_j));
                    float parallel_ness = (len_i > 1e-5f && len_j > 1e-5f)
                        ? fabsf(glm::dot(dir_i, dir_j / len_j))
                        : 1.0f;

                    float target_sep = (rc1 + rc2) + min_skin_gap * (0.45f + 0.70f * parallel_ness);
                    if (dist < target_sep && dist > 1e-5f) {
                        float pen = target_sep - dist;
                        glm::vec3 n = diff / dist;
                        glm::vec3 impulse = n * (pen * push_gain * 0.5f);

                        if (!(spi.pinned & 1)) {
                            float w = 1.0f - s;
                            atomicAdd(&sec_disp_q[si.parent_sec * 3 + 0], (unsigned long long)llroundf(impulse.x * w * 16777216.0f));
                            atomicAdd(&sec_disp_q[si.parent_sec * 3 + 1], (unsigned long long)llroundf(impulse.y * w * 16777216.0f));
                            atomicAdd(&sec_disp_q[si.parent_sec * 3 + 2], (unsigned long long)llroundf(impulse.z * w * 16777216.0f));
                        }
                        if (!(si.pinned & 1)) {
                            atomicAdd(&sec_disp_q[i * 3 + 0], (unsigned long long)llroundf(impulse.x * s * 16777216.0f));
                            atomicAdd(&sec_disp_q[i * 3 + 1], (unsigned long long)llroundf(impulse.y * s * 16777216.0f));
                            atomicAdd(&sec_disp_q[i * 3 + 2], (unsigned long long)llroundf(impulse.z * s * 16777216.0f));
                        }
                        if (!(spj.pinned & 1)) {
                            float w = 1.0f - t;
                            atomicAdd(&sec_disp_q[sj.parent_sec * 3 + 0], (unsigned long long)llroundf(-impulse.x * w * 16777216.0f));
                            atomicAdd(&sec_disp_q[sj.parent_sec * 3 + 1], (unsigned long long)llroundf(-impulse.y * w * 16777216.0f));
                            atomicAdd(&sec_disp_q[sj.parent_sec * 3 + 2], (unsigned long long)llroundf(-impulse.z * w * 16777216.0f));
                        }
                        if (!(sj.pinned & 1)) {
                            atomicAdd(&sec_disp_q[j * 3 + 0], (unsigned long long)llroundf(-impulse.x * t * 16777216.0f));
                            atomicAdd(&sec_disp_q[j * 3 + 1], (unsigned long long)llroundf(-impulse.y * t * 16777216.0f));
                            atomicAdd(&sec_disp_q[j * 3 + 2], (unsigned long long)llroundf(-impulse.z * t * 16777216.0f));
                        }
                    }
                }
            }
        }
    }
}

// v24.02 P1.2 DRAFT (2026-10-22) — UNVERIFIED UNTIL TRACK B:
// dequantize pass for the fixed-point uint64 accumulator (Q = 2^24). One thread
// per element; per-element conversion is deterministic by construction. The
// unsigned->signed conversion is well-defined modulo 2^64 under C++20
// (CMAKE_CUDA_STANDARD 20), so the two's-complement sum maps to the signed total.
// v24.02 fix 2 (2026-10-22): renamed with the _vol suffix — cuda_colonization.cu carries
// its own file-local dequantize kernel under the original name; identical names in the
// two translation units were an ODR violation at link time (LNK2005). Symbol rename only.
__global__ void gpu_dequantize_ll_to_float_vol_kernel(
    const unsigned long long* __restrict__ q,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<float>(static_cast<long long>(q[i])) * (1.0f / 16777216.0f);
}

__global__ void gpu_compact_surviving_spores_vol_kernel(
    const glm::vec3* __restrict__ in_spores,
    const uint8_t* __restrict__ spore_alive,
    int num_spores,
    glm::vec3* __restrict__ out_spores,
    int* __restrict__ out_surviving_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_spores) return;
    if (spore_alive[i]) {
        int slot = atomicAdd(out_surviving_count, 1);
        out_spores[slot] = in_spores[i];
    }
}

__global__ void gpu_commit_tip_proposals_kernel(
    const CudaVolTip* __restrict__ in_tips,
    const CudaTipProposal* __restrict__ proposals,
    int num_tips,
    float min_skin_gap,
    CudaVolSection* __restrict__ io_sections,
    int* __restrict__ d_num_sections,
    int max_sections,
    CudaVolTip* __restrict__ out_next_tips,
    int* __restrict__ d_num_next_tips,
    int max_next_tips,
    int* __restrict__ d_next_root_id
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= num_tips) return;

    CudaVolTip tip = in_tips[ti];
    CudaTipProposal prop = proposals[ti];

    if (prop.action == 0) {
        int slot = atomicAdd(d_num_next_tips, 1);
        if (slot < max_next_tips) out_next_tips[slot] = tip;
        return;
    }
    if (prop.action < 0) {
        tip.stalled_steps++;
        int slot = atomicAdd(d_num_next_tips, 1);
        if (slot < max_next_tips) out_next_tips[slot] = tip;
        return;
    }

    // Parallel conflict check against lower-index advancing tips in the same step
    bool conflict = false;
    for (int oj = 0; oj < ti; ++oj) {
        if (proposals[oj].action <= 0) continue;
        if (in_tips[oj].sec_idx == tip.sec_idx) continue;
        glm::vec3 d = prop.best_p1 - proposals[oj].best_p1;
        float min_d = (prop.r1 + proposals[oj].r1) + min_skin_gap * 0.55f;
        if (glm::dot(d, d) < min_d * min_d) {
            conflict = true;
            break;
        }
    }

    if (conflict) {
        tip.stalled_steps++;
        int slot = atomicAdd(d_num_next_tips, 1);
        if (slot < max_next_tips) out_next_tips[slot] = tip;
        return;
    }

    int new_sec_idx = atomicAdd(d_num_sections, 1);
    if (new_sec_idx >= max_sections) return;

    CudaVolSection new_sec;
    new_sec.pos = prop.best_p1;
    new_sec.radius = prop.r1;
    new_sec.parent_sec = tip.sec_idx;
    new_sec.root_id = tip.root_id;
    new_sec.depth = prop.next_depth;
    new_sec.pinned = (tip.generation << 1);
    io_sections[new_sec_idx] = new_sec;

    tip.sec_idx = new_sec_idx;
    tip.dir = prop.best_dir;
    tip.radius = prop.r1;
    tip.steps_since_branch = prop.has_lateral ? 0 : (tip.steps_since_branch + 1);
    tip.stalled_steps = 0;

    int slot_main = atomicAdd(d_num_next_tips, 1);
    if (slot_main < max_next_tips) {
        out_next_tips[slot_main] = tip;
    }

    if (prop.has_lateral) {
        CudaVolTip child_tip;
        child_tip.sec_idx = new_sec_idx;
        child_tip.root_id = atomicAdd(d_next_root_id, 1);
        child_tip.dir = prop.lat_dir;
        child_tip.radius = prop.lat_r;
        child_tip.steps_since_branch = 0;
        child_tip.stalled_steps = 0;
        child_tip.generation = tip.generation + 1;
        // v24.12 Track F R1: always-defined per-root counter (read only by
        // the developmental advance kernel; harmless to modes 0-3).
        child_tip.steps_grown = 0;

        int slot_lat = atomicAdd(d_num_next_tips, 1);
        if (slot_lat < max_next_tips) {
            out_next_tips[slot_lat] = child_tip;
        }
    }
}

__global__ void gpu_apply_section_displacements_kernel(
    CudaVolSection* __restrict__ sections,
    int num_sections,
    const float* __restrict__ sec_disp,
    float max_step_disp,
    glm::vec3* __restrict__ out_old_pos,
    uint8_t* __restrict__ sec_moved
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;

    sec_moved[i] = 0;
    out_old_pos[i] = sections[i].pos;
    if (sections[i].pinned & 1) return;

    glm::vec3 disp(sec_disp[i * 3 + 0], sec_disp[i * 3 + 1], sec_disp[i * 3 + 2]);
    float dl2 = glm::dot(disp, disp);
    if (dl2 > 1e-10f) {
        float dl = sqrtf(dl2);
        if (dl > max_step_disp) disp = (disp / dl) * max_step_disp;
        sections[i].pos += disp;
    }
}

__global__ void gpu_rope_reel_tethers_kernel(
    CudaVolSection* __restrict__ sections,
    int num_sections,
    const glm::vec3* __restrict__ old_pos,
    float step_len,
    float tether_max_stretch,
    float tether_target_stretch,
    float backward_pull,
    float tensile_straightening,
    const float* __restrict__ vessel_sdf,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    float voxel_size,
    float min_vessel_sdf,
    float wall_clearance_scale,
    uint8_t* __restrict__ sec_moved
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;
    if (sections[i].pinned & 1) return;

    int p = sections[i].parent_sec;
    float safe_pull = dev_clamp(backward_pull, 0.0f, 1.0f);
    float safe_straight = dev_clamp(tensile_straightening, 0.0f, 1.0f);

    if (p >= 0 && p < num_sections) {
        glm::vec3 link = sections[i].pos - sections[p].pos;
        float ll2 = glm::dot(link, link);
        float ll = sqrtf(ll2);

        // 1. Distance stretch reeling: controlled strictly by "Backward Pull"
        if (safe_pull > 0.001f) {
            float eff_scale = 1.0f / fmaxf(1.0f, safe_pull);
            float max_l = step_len * (1.0f + (tether_max_stretch - 1.0f) * eff_scale);
            float target_l = step_len * (1.0f + (tether_target_stretch - 1.0f) * eff_scale);
            if (ll > max_l && ll > 1e-5f) {
                glm::vec3 reeled = sections[p].pos + (link / ll) * target_l;
                sections[i].pos = glm::mix(sections[i].pos, reeled, safe_pull);
                link = sections[i].pos - sections[p].pos;
                ll = sqrtf(glm::dot(link, link));
            }
        }

        // 2. Tensile straightening: controlled strictly by "Straightening"
        int gp = sections[p].parent_sec;
        if (safe_straight > 0.001f && gp >= 0 && gp < num_sections && ll > 1e-5f) {
            glm::vec3 parent_seg = sections[p].pos - sections[gp].pos;
            float pl2 = glm::dot(parent_seg, parent_seg);
            if (pl2 > 1e-5f) {
                glm::vec3 fwd_dir = parent_seg * rsqrtf(pl2);
                glm::vec3 straightened = sections[p].pos + fwd_dir * ll;
                float align_fraction = safe_straight * 0.30f;
                glm::vec3 cand = glm::mix(sections[i].pos, straightened, align_fraction);
                if (isfinite(cand.x) && isfinite(cand.y) && isfinite(cand.z)) {
                    sections[i].pos = cand;
                }
            }
        }
    }

    dev_confine_sphere_to_vessel(
        vessel_sdf, vessel_min_coord, vessel_dims,
        voxel_size, min_vessel_sdf,
        sections[i].pos, sections[i].radius, nullptr,
        wall_clearance_scale
    );

    glm::vec3 delta = sections[i].pos - old_pos[i];
    if (glm::dot(delta, delta) > (step_len * 0.02f) * (step_len * 0.02f)) {
        sec_moved[i] = 1;
    }
}

__global__ void gpu_wake_tips_kernel(
    CudaVolTip* __restrict__ tips,
    int num_tips,
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    const uint8_t* __restrict__ sec_moved
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= num_tips) return;

    if (tips[ti].stalled_steps > 4 && tips[ti].stalled_steps < 1000 && tips[ti].sec_idx >= 0 && tips[ti].sec_idx < num_sections) {
        int sec_idx = tips[ti].sec_idx;
        if ((sections[sec_idx].pinned & 1) != 0) return; // Do not wake tips attached to frozen sections
        int p_idx = sections[sec_idx].parent_sec;
        if (sec_moved[sec_idx] || (p_idx >= 0 && sec_moved[p_idx])) {
            tips[ti].stalled_steps = 0;
            if (p_idx >= 0) {
                glm::vec3 link = sections[sec_idx].pos - sections[p_idx].pos;
                float ll2 = glm::dot(link, link);
                if (ll2 > 1e-10f) {
                    float ll = sqrtf(ll2);
                    tips[ti].dir = dev_safe_norm((link / ll) * 0.75f + tips[ti].dir * 0.25f, tips[ti].dir);
                }
            }
        }
    }
}

__global__ void gpu_reduce_max_depth_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    int* __restrict__ out_max_depth
) {
    __shared__ int s_max[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int local_max = 0;
    while (i < num_sections) {
        if (sections[i].depth > local_max) {
            local_max = sections[i].depth;
        }
        i += gridDim.x * blockDim.x;
    }
    s_max[tid] = local_max;
    __syncthreads();

    // Parallel tree reduction within thread block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid + s] > s_max[tid]) {
                s_max[tid] = s_max[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(out_max_depth, s_max[0]);
    }
}

__global__ void gpu_vascular_taper_kernel(
    CudaVolSection* __restrict__ sections,
    int num_sections,
    const int* __restrict__ in_max_depth,
    float base_radius,
    float tip_radius,
    float safe_taper,
    int total_steps,
    bool variable_taper,
    uint32_t rng_seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;

    float min_r = fminf(base_radius, tip_radius);
    float max_r = fmaxf(base_radius, tip_radius);
    if (sections[i].parent_sec < 0) {
        // Unfrozen seed base section dynamically updates to current base_radius
        if (!(sections[i].pinned & 2)) {
            sections[i].radius = base_radius;
        }
        return;
    }
    if (sections[i].pinned & 1) {
        // Frozen root sections retain their locked caliber completely intact
        return;
    }

    int gen = max(0, sections[i].pinned >> 1);
    int root_id = sections[i].root_id;
    float gen_decay = powf(0.74f, static_cast<float>(gen));
    float branch_hash = dev_hash_01(rng_seed + static_cast<uint32_t>(root_id * 7919 + gen * 1013 + 53));

    if (variable_taper) {
        float wave = 0.5f + 0.5f * sinf(static_cast<float>(sections[i].depth) * 0.65f + static_cast<float>(root_id) * 2.17f);
        float jitter = dev_hash_01(rng_seed + static_cast<uint32_t>(i * 1999 + sections[i].depth * 733 + root_id * 37));
        float t_var = dev_clamp(0.45f * branch_hash + 0.35f * wave + 0.20f * jitter, 0.0f, 1.0f) * dev_clamp(gen_decay, 0.15f, 1.0f);
        sections[i].radius = min_r + (max_r - min_r) * t_var;
        return;
    }

    int max_depth = *in_max_depth;
    float taper_span = static_cast<float>(max(max(total_steps, max_depth + 1), 5));
    float progress = dev_clamp(static_cast<float>(sections[i].depth) / taper_span, 0.0f, 1.0f);
    float branch_scale = (gen == 0) ? (0.72f + 0.28f * branch_hash) : (0.48f + 0.52f * branch_hash);
    float branch_max_r = min_r + (max_r - min_r) * dev_clamp(gen_decay * branch_scale, 0.08f, 1.0f);
    sections[i].radius = tip_radius + (branch_max_r - tip_radius) * powf(1.0f - progress, safe_taper);
}

__global__ void gpu_taper_radii_kernel(
    CudaVolSection* __restrict__ sections,
    int num_sections,
    float base_radius,
    float tip_radius,
    float safe_taper,
    float taper_span
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;
    float progress = dev_clamp(static_cast<float>(sections[i].depth) / taper_span, 0.0f, 1.0f);
    sections[i].radius = tip_radius + (base_radius - tip_radius) * powf(1.0f - progress, safe_taper);
}

__global__ void gpu_init_volumetric_seeds_kernel(
    const glm::vec3* __restrict__ seeds,
    int num_seeds,
    const float* __restrict__ vessel_sdf,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    float voxel_size,
    float min_vessel_sdf,
    float base_radius,
    uint32_t rng_seed,
    float wall_clearance_scale,
    CudaVolSection* __restrict__ out_sections,
    CudaVolTip* __restrict__ out_tips,
    int primary_roots
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_seeds) return;

    CudaVolSection seed_sec;
    seed_sec.pos = seeds[s];
    seed_sec.radius = base_radius;
    seed_sec.parent_sec = -1;
    seed_sec.root_id = s;
    seed_sec.depth = 0;
    seed_sec.pinned = 1;

    dev_confine_sphere_to_vessel(
        vessel_sdf, vessel_min_coord, vessel_dims,
        voxel_size, min_vessel_sdf,
        seed_sec.pos, seed_sec.radius, nullptr,
        wall_clearance_scale
    );
    out_sections[s] = seed_sec;

    glm::vec3 wall_n = dev_sample_sdf_grad(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, seed_sec.pos);
    glm::vec3 up = (fabsf(wall_n.z) < 0.85f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 t1 = dev_safe_norm(glm::cross(wall_n, up), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::vec3 t2 = dev_safe_norm(glm::cross(wall_n, t1), glm::vec3(0.0f, 1.0f, 0.0f));

    const int primary_count = (primary_roots > 0) ? primary_roots : 3;
    for (int k = 0; k < primary_count; ++k) {
        float u = dev_hash_01(rng_seed + static_cast<uint32_t>(s * 17 + k * 31));
        float ang = (static_cast<float>(k) + 0.25f * u) * (2.0f * 3.14159265f / static_cast<float>(primary_count));
        glm::vec3 tangent_dir = cosf(ang) * t1 + sinf(ang) * t2;
        glm::vec3 inward_dir = -wall_n * 0.45f + tangent_dir * 0.85f;
        glm::vec3 d = dev_safe_norm(inward_dir, tangent_dir);
        CudaVolTip tip;
        tip.sec_idx = s;
        tip.root_id = num_seeds + s * primary_count + k;
        tip.dir = d;
        tip.radius = base_radius;
        tip.steps_since_branch = k;
        tip.stalled_steps = 0;
        tip.generation = 0;
        tip.steps_grown = 0;
        out_tips[s * primary_count + k] = tip;
    }
}

__global__ void gpu_convert_vol_sections_to_nodes_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    GpuColonizationNodeLayout* __restrict__ out_nodes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;
    GpuColonizationNodeLayout cn;
    cn.position = sections[i].pos;
    cn.parent_index = sections[i].parent_sec;
    cn.radius = sections[i].radius;
    cn.depth = sections[i].depth;
    cn.energy = 1.0f;
    cn.strand_id = sections[i].root_id;
    out_nodes[i] = cn;
}

__global__ void gpu_check_awake_tips_kernel(
    const CudaVolTip* __restrict__ tips,
    int num_tips,
    int* __restrict__ out_awake_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tips) return;
    if (tips[i].stalled_steps <= 4) {
        atomicAdd(out_awake_count, 1);
    }
}

__global__ void gpu_reset_stalled_tips_kernel(
    CudaVolTip* __restrict__ tips,
    int num_tips
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tips) return;
    if (tips[i].stalled_steps < 1000) {
        tips[i].stalled_steps = 0;
    }
}

__global__ void gpu_mark_section_children_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    int* __restrict__ out_has_child
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;
    int p = sections[i].parent_sec;
    if (p >= 0 && p < num_sections) {
        out_has_child[p] = 1;
    }
}

__global__ void gpu_mark_active_tip_sections_kernel(
    const CudaVolTip* __restrict__ tips,
    int num_tips,
    int num_sections,
    int* __restrict__ out_has_child
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tips) return;
    int sec = tips[i].sec_idx;
    if (sec >= 0 && sec < num_sections) {
        out_has_child[sec] = 1;
    }
}

__global__ void gpu_respawn_leaf_tips_kernel(
    const CudaVolSection* __restrict__ sections,
    const int* __restrict__ has_child,
    int num_sections,
    CudaVolTip* __restrict__ io_tips,
    int* __restrict__ d_tip_count,
    int max_tips
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sections) return;
    int i = (num_sections - 1) - idx;
    if (has_child[i] == 0 && sections[i].parent_sec >= 0 && (sections[i].pinned & 1) == 0) {
        glm::vec3 d = sections[i].pos - sections[sections[i].parent_sec].pos;
        float dl = sqrtf(glm::dot(d, d));
        if (dl > 1e-5f) {
            int slot = atomicAdd(d_tip_count, 1);
            if (slot < max_tips) {
                CudaVolTip tip;
                tip.sec_idx = i;
                tip.root_id = sections[i].root_id;
                tip.dir = d / dl;
                tip.radius = sections[i].radius;
                tip.steps_since_branch = 2;
                tip.stalled_steps = 0;
                tip.generation = max(1, sections[i].pinned >> 1);
                tip.steps_grown = 0;
                io_tips[slot] = tip;
            }
        }
    }
}

__device__ inline int dev_find_vol_section_seed_slot(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    int idx
) {
    int cur = idx;
    for (int hop = 0; hop < 4096; ++hop) {
        int p = sections[cur].parent_sec;
        if (p < 0 || p >= num_sections || p == cur) break;
        cur = p;
    }
    return sections[cur].root_id;
}

__global__ void gpu_sync_volumetric_freeze_sections_kernel(
    CudaVolSection* __restrict__ sections,
    int num_sections,
    const glm::vec3* __restrict__ seeds,
    const uint8_t* __restrict__ frozen_seeds,
    int num_seeds,
    int* __restrict__ out_seed_exists
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;

    int seed_slot = dev_find_vol_section_seed_slot(sections, num_sections, i);
    if (sections[i].parent_sec < 0 && seed_slot >= 0 && seed_slot < num_seeds) {
        out_seed_exists[seed_slot] = 1;
        // Keep seed root section position in sync with moved seed position
        if (frozen_seeds == nullptr || frozen_seeds[seed_slot] == 0) {
            sections[i].pos = seeds[seed_slot];
        }
    }
    bool is_frozen = (seed_slot >= 0 && seed_slot < num_seeds && frozen_seeds != nullptr && frozen_seeds[seed_slot] != 0);
    if (sections[i].parent_sec < 0) {
        sections[i].pinned = is_frozen ? 3 : 1;
    } else if (is_frozen) {
        sections[i].pinned |= 1;
    } else {
        sections[i].pinned &= ~1;
    }
}

__global__ void gpu_sync_volumetric_freeze_tips_kernel(
    CudaVolTip* __restrict__ tips,
    int num_tips,
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    const uint8_t* __restrict__ frozen_seeds,
    int num_seeds
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tips) return;
    int sec = tips[i].sec_idx;
    if (sec < 0 || sec >= num_sections) return;
    int seed_slot = dev_find_vol_section_seed_slot(sections, num_sections, sec);
    bool is_frozen = (seed_slot >= 0 && seed_slot < num_seeds && frozen_seeds != nullptr && frozen_seeds[seed_slot] != 0);
    if (is_frozen) {
        tips[i].stalled_steps = 10000;
    } else if (tips[i].stalled_steps >= 1000) {
        tips[i].stalled_steps = 0;
    }
}

__global__ void gpu_spawn_missing_volumetric_seeds_kernel(
    const glm::vec3* __restrict__ seeds,
    const uint8_t* __restrict__ frozen_seeds,
    const int* __restrict__ seed_exists,
    int num_seeds,
    const float* __restrict__ vessel_sdf,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    float voxel_size,
    float min_vessel_sdf,
    float base_radius,
    uint32_t rng_seed,
    float wall_clearance_scale,
    CudaVolSection* __restrict__ io_sections,
    int* __restrict__ d_num_sections,
    int max_sections,
    CudaVolTip* __restrict__ io_tips,
    int* __restrict__ d_num_tips,
    int max_tips,
    int* __restrict__ d_next_root_id,
    int* __restrict__ d_spawned_count,
    int primary_roots
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_seeds) return;
    if (seed_exists[s] != 0) return;

    int sec_idx = atomicAdd(d_num_sections, 1);
    if (sec_idx >= max_sections) return;

    CudaVolSection seed_sec;
    seed_sec.pos = seeds[s];
    seed_sec.radius = base_radius;
    seed_sec.parent_sec = -1;
    seed_sec.root_id = s;
    seed_sec.depth = 0;
    seed_sec.pinned = 1;

    dev_confine_sphere_to_vessel(
        vessel_sdf, vessel_min_coord, vessel_dims,
        voxel_size, min_vessel_sdf,
        seed_sec.pos, seed_sec.radius, nullptr,
        wall_clearance_scale
    );
    io_sections[sec_idx] = seed_sec;
    atomicAdd(d_spawned_count, 1);

    glm::vec3 wall_n = dev_sample_sdf_grad(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, seed_sec.pos);
    glm::vec3 up = (fabsf(wall_n.z) < 0.85f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 t1 = dev_safe_norm(glm::cross(wall_n, up), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::vec3 t2 = dev_safe_norm(glm::cross(wall_n, t1), glm::vec3(0.0f, 1.0f, 0.0f));

    bool is_frozen = (frozen_seeds != nullptr && frozen_seeds[s] != 0);
    const int primary_count = (primary_roots > 0) ? primary_roots : 3;
    for (int k = 0; k < primary_count; ++k) {
        int tip_slot = atomicAdd(d_num_tips, 1);
        if (tip_slot >= max_tips) break;
        float u = dev_hash_01(rng_seed + static_cast<uint32_t>(s * 17 + k * 31));
        float ang = (static_cast<float>(k) + 0.25f * u) * (2.0f * 3.14159265f / static_cast<float>(primary_count));
        glm::vec3 tangent_dir = cosf(ang) * t1 + sinf(ang) * t2;
        glm::vec3 inward_dir = -wall_n * 0.45f + tangent_dir * 0.85f;
        glm::vec3 d = dev_safe_norm(inward_dir, tangent_dir);
        CudaVolTip tip;
        tip.sec_idx = sec_idx;
        tip.root_id = atomicAdd(d_next_root_id, 1);
        tip.dir = d;
        tip.radius = base_radius;
        tip.steps_since_branch = k;
        tip.stalled_steps = is_frozen ? 10000 : 0;
        tip.generation = 0;
        tip.steps_grown = 0;
        io_tips[tip_slot] = tip;
    }
}

// =========================================================================
// v24.12 Track F R1 — DEVELOPMENTAL MODE (algorithm_mode == 4)
// FSPM-class apex growth: the root is the trace the apex leaves. Per-order
// gravitropic setpoints chased by the Firn-Digby sine law, per-order
// elongation/spacing/length caps, acropetal lateral emergence with hashed
// azimuth, vessel-wall confinement through the shared helper. No spores,
// no feelers, no body sensing, no push/reel corpse physics — sections are
// append-only. Radii during growth are per-order constants (floored at the
// SLA print radius); the bake-time Murray census (host, colonization.cpp
// apply_dev_murray_caliber) rebuilds structural caliber from the subtree.
// =========================================================================
__global__ void gpu_dev_advance_tips_kernel(
    const CudaVolTip* __restrict__ tips,
    int num_tips,
    const CudaVolSection* __restrict__ sections,
    const float* __restrict__ vessel_sdf,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    CudaVolGrowthParams params,
    CudaTipProposal* __restrict__ out_proposals
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= num_tips) return;

    CudaVolTip tip = tips[ti];
    CudaTipProposal prop;
    prop.action = 0;
    prop.best_p1 = glm::vec3(0.0f);
    prop.best_dir = tip.dir;
    prop.r1 = tip.radius;
    prop.next_depth = 0;
    prop.has_lateral = 0;
    prop.lat_dir = glm::vec3(0.0f);
    prop.lat_r = fmaxf(params.dev_min_print_radius, 0.10f);

    // Existing sleep gate (retired tips stay parked; stall counts accumulate
    // through the commit kernel so check_any_awake eventually ends the batch)
    if (tip.stalled_steps > 4) {
        out_proposals[ti] = prop;
        return;
    }

    const int order = min(max(tip.generation, 0), 3);
    const glm::vec3 p0 = sections[tip.sec_idx].pos;
    const int next_depth = sections[tip.sec_idx].depth + 1;

    // Per-root length cap: past the cap the apex parks itself via the stall
    // channel (action -1); five quiet iterations later it sleeps for good.
    int steps_grown = tip.steps_grown;
    if (steps_grown < 0) steps_grown = 0; // foreign-creation garbage guard
    if (steps_grown >= params.dev_max_steps[order]) {
        prop.action = -1;
        out_proposals[ti] = prop;
        return;
    }

    // Growth-time radius: per-order constant, never below the SLA floor.
    const float r1 = fmaxf(params.dev_order_radius[order], params.dev_min_print_radius);

    // Elongation: shared step size scaled by the per-order rate, with the
    // same local step floor discipline as the classic path.
    float step_len = fmaxf(0.25f, params.growth_step * params.dev_rate[order]);
    step_len = fmaxf(step_len, r1 * 1.12f);

    glm::vec3 dir = tip.dir;

    // 1. Directional wander first, so the gravitropic correction visibly
    //    re-straightens it (damped return = smooth organic arcs).
    //    v24.13 Track F R2: order-0 wander damped to 0.45x — framework arms
    //    keep smooth catenary arcs (the field blob complaint was aggravated
    //    by zig-zag primaries); fine orders keep the full organic wiggle.
    if (params.dev_noise > 0.001f) {
        const float noise_eff = params.dev_noise * (order == 0 ? 0.45f : 1.0f);
        uint32_t salt = params.seed + static_cast<uint32_t>(tip.sec_idx * 37 + params.current_iter * 101);
        float nx = dev_hash_01(salt) * 2.0f - 1.0f;
        float ny = dev_hash_01(salt + 733u) * 2.0f - 1.0f;
        float nz = dev_hash_01(salt + 1531u) * 2.0f - 1.0f;
        glm::vec3 jitter(nx, ny, nz);
        jitter -= glm::dot(jitter, dir) * dir;
        if (glm::dot(jitter, jitter) > 1e-10f) {
            dir = dev_safe_norm(dir + dev_safe_norm(jitter, dir) * (noise_eff * 3.0f), dir);
        }
    }

    // 2. Firn-Digby gravitropic setpoint chase (sine law): turn rate
    //    proportional to sin(angle between heading and the GSA setpoint).
    {
        const float gsa_deg = fminf(fmaxf(params.dev_gsa_deg[order], 0.0f), 89.0f);
        const float gsa_rad = gsa_deg * (3.14159265f / 180.0f);
        const glm::vec3 down(0.0f, 0.0f, -1.0f);
        glm::vec3 horiz = dir - down * glm::dot(dir, down);
        glm::vec3 plane1;
        if (glm::dot(horiz, horiz) > 1e-8f) {
            plane1 = glm::normalize(horiz); // outward horizontal heading
        } else {
            // Plumb degenerate: hashed per-root azimuth so near-vertical roots
            // do not all snap into one common vertical plane.
            float az = dev_hash_01(params.seed + static_cast<uint32_t>(tip.root_id * 40503u)) * 6.2831853f;
            plane1 = glm::vec3(cosf(az), sinf(az), 0.0f);
        }
        glm::vec3 set_dir = down * cosf(gsa_rad) + plane1 * sinf(gsa_rad);
        float cosang = fminf(fmaxf(glm::dot(dir, set_dir), -1.0f), 1.0f);
        float ang = acosf(cosang);
        if (ang > 1e-4f) {
            float turn = params.dev_grav_gain * sinf(ang);
            if (turn > ang) turn = ang;
            glm::vec3 axis = glm::cross(dir, set_dir);
            float ax2 = glm::dot(axis, axis);
            if (ax2 > 1e-12f) {
                axis *= rsqrtf(ax2);
                glm::vec3 rotated = dir * cosf(turn)
                    + glm::cross(axis, dir) * sinf(turn)
                    + axis * (glm::dot(axis, dir) * (1.0f - cosf(turn)));
                dir = dev_safe_norm(rotated, dir);
            } else {
                dir = set_dir; // anti-parallel edge
            }
        }
    }

    // 3. Proposed step, confined to the vessel interior by the shared wall
    //    helper (deflects along the wall; keeps print-radius clearance).
    glm::vec3 p1 = p0 + dir * step_len;
    glm::vec3 intended = dir; // v24.13: pre-confinement heading (for the wall-slide test)
    glm::vec3 odir = dir;
    dev_confine_sphere_to_vessel(
        vessel_sdf, vessel_min_coord, vessel_dims,
        params.voxel_size, params.min_vessel_sdf,
        p1, r1, &odir, params.wall_clearance_scale
    );
    glm::vec3 moved = p1 - p0;
    float moved_len = sqrtf(glm::dot(moved, moved));
    if (moved_len < step_len * 0.25f) {
        // Wall-blocked (tip buried against the vessel): stall channel; the
        // existing sleep bookkeeping parks it after five attempts.
        prop.action = -1;
        out_proposals[ti] = prop;
        return;
    }
    dir = dev_safe_norm(moved * (1.0f / moved_len), dir);

    // v24.13 Track F R2: wall-slide park. If the confinement deflected the
    // step far from the intended heading (>65 deg), the tip is grinding
    // along a wall or the floor — park it through the stall channel instead
    // of letting it crawl (v24.12 floor-crawling piled the whole cascade
    // into the vessel bottom = the reported "shapeless blob").
    {
        float cos_defl = fminf(fmaxf(glm::dot(dir, intended), -1.0f), 1.0f);
        if (acosf(cos_defl) > 1.1345f) { // 65 degrees in radians
            prop.action = -1;
            out_proposals[ti] = prop;
            return;
        }
    }

    prop.action = 1;
    prop.best_p1 = p1;
    prop.best_dir = dir;
    prop.r1 = r1;
    prop.next_depth = next_depth;

    // 4. Acropetal lateral emergence: fixed per-order spacing, hashed azimuth
    //    around the parent axis (non-planar), departure at dev_branch_angle.
    //    The child's own GSA takes over from its first step onward.
    if (tip.generation < 3 &&
        (tip.steps_since_branch + 1) >= params.dev_spacing[order] &&
        (steps_grown + 1) < params.dev_max_steps[order]) {
        const int child_order = order + 1;
        glm::vec3 ref = (fabsf(dir.z) < 0.9f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 perp1 = dev_safe_norm(glm::cross(dir, ref), glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec3 perp2 = glm::cross(dir, perp1); // unit: dir x perp1, both unit, orthogonal
        uint32_t az_hash = params.seed + static_cast<uint32_t>(tip.root_id * 40503u + params.current_iter * 613u + tip.sec_idx * 83u);
        float az = dev_hash_01(az_hash) * 6.2831853f;
        float ba = fminf(fmaxf(params.dev_branch_angle, 15.0f), 89.0f) * (3.14159265f / 180.0f);
        glm::vec3 radial_dir = perp1 * cosf(az) + perp2 * sinf(az);
        glm::vec3 lat_dir = dev_safe_norm(dir * cosf(ba) + radial_dir * sinf(ba), dir);
        const float lat_r = fmaxf(params.dev_order_radius[child_order], params.dev_min_print_radius);
        glm::vec3 lat_p1 = p1 + lat_dir * fmaxf(0.25f, step_len * 0.6f);
        glm::vec3 ldir = lat_dir;
        dev_confine_sphere_to_vessel(
            vessel_sdf, vessel_min_coord, vessel_dims,
            params.voxel_size, params.min_vessel_sdf,
            lat_p1, lat_r, &ldir, params.wall_clearance_scale
        );
        glm::vec3 lm = lat_p1 - p1;
        if (sqrtf(glm::dot(lm, lm)) >= step_len * 0.20f) {
            prop.has_lateral = 1;
            prop.lat_dir = dev_safe_norm(lm, lat_dir);
            prop.lat_r = lat_r;
        }
    }

    out_proposals[ti] = prop;
}

// v24.12 Track F R1: developmental commit — the shared append discipline
// (conflict check against lower-index tips, atomic section append, lateral
// child creation) plus steps_grown bookkeeping the shared kernel does not
// carry. Stand-off for the same-step conflict check is a fixed 10% + 0.1 mm
// (no min_skin_gap coupling: developmental roots may touch and graft — the
// SDF bake unions them into one printable solid).
__global__ void gpu_dev_commit_proposals_kernel(
    const CudaVolTip* __restrict__ in_tips,
    const CudaTipProposal* __restrict__ proposals,
    int num_tips,
    CudaVolSection* __restrict__ io_sections,
    int* __restrict__ d_num_sections,
    int max_sections,
    CudaVolTip* __restrict__ out_next_tips,
    int* __restrict__ d_num_next_tips,
    int max_next_tips,
    int* __restrict__ d_next_root_id
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= num_tips) return;

    CudaVolTip tip = in_tips[ti];
    CudaTipProposal prop = proposals[ti];

    if (prop.action == 0) {
        int slot = atomicAdd(d_num_next_tips, 1);
        if (slot < max_next_tips) out_next_tips[slot] = tip;
        return;
    }
    if (prop.action < 0) {
        tip.stalled_steps++;
        int slot = atomicAdd(d_num_next_tips, 1);
        if (slot < max_next_tips) out_next_tips[slot] = tip;
        return;
    }

    // Parallel conflict check against lower-index advancing tips in the same step
    bool conflict = false;
    for (int oj = 0; oj < ti; ++oj) {
        if (proposals[oj].action <= 0) continue;
        if (in_tips[oj].sec_idx == tip.sec_idx) continue;
        glm::vec3 d = prop.best_p1 - proposals[oj].best_p1;
        float min_d = (prop.r1 + proposals[oj].r1) * 1.10f + 0.10f;
        if (glm::dot(d, d) < min_d * min_d) {
            conflict = true;
            break;
        }
    }
    if (conflict) {
        tip.stalled_steps++;
        int slot = atomicAdd(d_num_next_tips, 1);
        if (slot < max_next_tips) out_next_tips[slot] = tip;
        return;
    }

    int new_sec_idx = atomicAdd(d_num_sections, 1);
    if (new_sec_idx >= max_sections) return;

    CudaVolSection new_sec;
    new_sec.pos = prop.best_p1;
    new_sec.radius = prop.r1;
    new_sec.parent_sec = tip.sec_idx;
    new_sec.root_id = tip.root_id;
    new_sec.depth = prop.next_depth;
    new_sec.pinned = (tip.generation << 1); // order storage for Murray/taper (existing convention)
    io_sections[new_sec_idx] = new_sec;

    tip.sec_idx = new_sec_idx;
    tip.dir = prop.best_dir;
    tip.radius = prop.r1;
    tip.steps_since_branch = prop.has_lateral ? 0 : (tip.steps_since_branch + 1);
    tip.stalled_steps = 0;
    tip.steps_grown = (tip.steps_grown < 0 ? 0 : tip.steps_grown) + 1;

    int slot_main = atomicAdd(d_num_next_tips, 1);
    if (slot_main < max_next_tips) {
        out_next_tips[slot_main] = tip;
    }

    if (prop.has_lateral) {
        CudaVolTip child_tip;
        child_tip.sec_idx = new_sec_idx;
        child_tip.root_id = atomicAdd(d_next_root_id, 1);
        child_tip.dir = prop.lat_dir;
        child_tip.radius = prop.lat_r;
        child_tip.steps_since_branch = 0;
        child_tip.stalled_steps = 0;
        child_tip.generation = tip.generation + 1;
        child_tip.steps_grown = 0;
        int slot_lat = atomicAdd(d_num_next_tips, 1);
        if (slot_lat < max_next_tips) {
            out_next_tips[slot_lat] = child_tip;
        }
    }
}

#endif // __CUDACC__

bool CudaRasterizer::step_volumetric_colonization_gpu(
    const CudaVolumeGrid& vessel_grid,
    const CudaVolGrowthParams& params,
    std::vector<glm::vec3>& io_spores,
    std::vector<CudaVolSection>& io_sections,
    std::vector<CudaVolTip>& io_tips,
    int& io_next_root_id
) {
#ifdef __CUDACC__
    if (!vessel_grid.is_valid() || io_sections.empty() || io_tips.empty()) {
        return false;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return false;
    }

    // 1. Cache the 3D Vessel SDF grid in GPU VRAM across growth steps so it is only uploaded once per specimen
    static float* s_d_vessel_sdf = nullptr;
    static const float* s_cached_host_ptr = nullptr;
    static size_t s_cached_sdf_count = 0;
    static uint64_t s_cached_version = 0;

    if (s_cached_host_ptr != vessel_grid.data.data() || s_cached_sdf_count != vessel_grid.data.size() || s_cached_version != vessel_grid.version) {
        if (s_d_vessel_sdf) {
            cudaFree(s_d_vessel_sdf);
            s_d_vessel_sdf = nullptr;
        }
        size_t bytes = vessel_grid.data.size() * sizeof(float);
        if (cudaMalloc(&s_d_vessel_sdf, bytes) != cudaSuccess) {
            s_d_vessel_sdf = nullptr;
            return false;
        }
        cudaMemcpy(s_d_vessel_sdf, vessel_grid.data.data(), bytes, cudaMemcpyHostToDevice);
        s_cached_host_ptr = vessel_grid.data.data();
        s_cached_sdf_count = vessel_grid.data.size();
        s_cached_version = vessel_grid.version;
    }

    const float ref_step = params.variable_step
        ? std::max(params.min_growth_step, params.max_growth_step)
        : params.growth_step;
    const float step_len = std::max(0.25f, ref_step);
    const float search_r = std::max(step_len * 2.5f, params.search_radius);
    const float min_skin_gap = std::max(params.voxel_size * 1.85f, params.tip_radius * params.skin_gap_scale);
    const float seg_cell_size = std::max(step_len * 2.4f, (params.base_radius * 2.2f + min_skin_gap) * 1.6f);
    const float inv_body_cell = 1.0f / seg_cell_size;
    const float inv_spore_cell = 1.0f / search_r;

    const int collar_hops = std::clamp(
        static_cast<int>(std::ceil((params.base_radius * 2.5f + min_skin_gap) / std::max(0.4f, step_len))) + 1,
        3, 7
    );
    const int branch_interval = std::max(4, static_cast<int>(std::round(12.0f - 7.0f * std::clamp(params.branching_rate, 0.1f, 1.0f))));

    const uint32_t body_hash_size = 65536u;
    const uint32_t body_hash_mask = body_hash_size - 1u;
    const uint32_t spore_hash_size = 65536u;
    const uint32_t spore_hash_mask = spore_hash_size - 1u;

    // Persistent reusable GPU buffers to avoid per-step cudaMalloc overhead
    static CudaVolSection* s_d_sections = nullptr;
    static size_t s_cap_sections = 0;
    static int* s_d_body_head = nullptr;
    static int* s_d_body_next = nullptr;
    static glm::ivec3* s_d_body_cells = nullptr;
    static float* s_d_sec_disp = nullptr;
    static unsigned long long* s_d_sec_disp_q = nullptr; // v24.02 P1.2 DRAFT: quantized shadow accumulator (Q = 2^24, uint64)
    static uint8_t* s_d_sec_moved = nullptr;

    static glm::vec3* s_d_spores = nullptr;
    static glm::vec3* s_d_spores_compact = nullptr;
    static size_t s_cap_spores = 0;
    static uint8_t* s_d_spore_alive = nullptr;
    static int* s_d_spore_head = nullptr;
    static int* s_d_spore_next = nullptr;
    static glm::ivec3* s_d_spore_cells = nullptr;

    static CudaVolTip* s_d_tips = nullptr;
    static CudaVolTip* s_d_next_tips = nullptr;
    static CudaTipProposal* s_d_proposals = nullptr;
    static size_t s_cap_tips = 0;
    static glm::vec3* s_d_sec_old_pos = nullptr;
    static int* s_d_step_counters = nullptr; // [0]=surviving_spores, [1]=num_sections, [2]=num_next_tips, [3]=next_root_id

    if (!s_d_body_head) {
        cudaMalloc(&s_d_body_head, body_hash_size * sizeof(int));
        cudaMalloc(&s_d_spore_head, spore_hash_size * sizeof(int));
        cudaMalloc(&s_d_step_counters, 4 * sizeof(int));
    }

    size_t need_sections = std::max<size_t>(io_sections.size() + io_tips.size() * 2 + 256, 16384);
    if (need_sections > s_cap_sections) {
        if (s_d_sections) cudaFree(s_d_sections);
        if (s_d_body_next) cudaFree(s_d_body_next);
        if (s_d_body_cells) cudaFree(s_d_body_cells);
        if (s_d_sec_disp) cudaFree(s_d_sec_disp);
        if (s_d_sec_disp_q) cudaFree(s_d_sec_disp_q);
        if (s_d_sec_moved) cudaFree(s_d_sec_moved);
        if (s_d_sec_old_pos) cudaFree(s_d_sec_old_pos);
        s_cap_sections = need_sections * 2;
        cudaMalloc(&s_d_sections, s_cap_sections * sizeof(CudaVolSection));
        cudaMalloc(&s_d_body_next, s_cap_sections * sizeof(int));
        cudaMalloc(&s_d_body_cells, s_cap_sections * sizeof(glm::ivec3));
        cudaMalloc(&s_d_sec_disp, s_cap_sections * 3 * sizeof(float));
        cudaMalloc(&s_d_sec_disp_q, s_cap_sections * 3 * sizeof(unsigned long long));
        cudaMalloc(&s_d_sec_moved, s_cap_sections * sizeof(uint8_t));
        cudaMalloc(&s_d_sec_old_pos, s_cap_sections * sizeof(glm::vec3));
    }

    size_t need_spores = std::max<size_t>(io_spores.size() + 256, 16384);
    if (need_spores > s_cap_spores) {
        if (s_d_spores) cudaFree(s_d_spores);
        if (s_d_spores_compact) cudaFree(s_d_spores_compact);
        if (s_d_spore_alive) cudaFree(s_d_spore_alive);
        if (s_d_spore_next) cudaFree(s_d_spore_next);
        if (s_d_spore_cells) cudaFree(s_d_spore_cells);
        s_cap_spores = need_spores * 2;
        cudaMalloc(&s_d_spores, s_cap_spores * sizeof(glm::vec3));
        cudaMalloc(&s_d_spores_compact, s_cap_spores * sizeof(glm::vec3));
        cudaMalloc(&s_d_spore_alive, s_cap_spores * sizeof(uint8_t));
        cudaMalloc(&s_d_spore_next, s_cap_spores * sizeof(int));
        cudaMalloc(&s_d_spore_cells, s_cap_spores * sizeof(glm::ivec3));
    }

    size_t need_tips = std::max<size_t>(io_tips.size() * 2 + 256, 4096);
    if (need_tips > s_cap_tips) {
        if (s_d_tips) cudaFree(s_d_tips);
        if (s_d_next_tips) cudaFree(s_d_next_tips);
        if (s_d_proposals) cudaFree(s_d_proposals);
        s_cap_tips = need_tips * 2;
        cudaMalloc(&s_d_tips, s_cap_tips * sizeof(CudaVolTip));
        cudaMalloc(&s_d_next_tips, s_cap_tips * sizeof(CudaVolTip));
        cudaMalloc(&s_d_proposals, s_cap_tips * sizeof(CudaTipProposal));
    }

    const int num_sec_initial = static_cast<int>(io_sections.size());
    cudaMemset(s_d_body_head, 0xFF, body_hash_size * sizeof(int));
    cudaMemcpy(s_d_sections, io_sections.data(), num_sec_initial * sizeof(CudaVolSection), cudaMemcpyHostToDevice);

    // Launch Kernel 1: Spatially hash all existing 3D root sections on GPU
    // v24.12 Track F R1: mode 4 skips the body hash (no corpse physics, no
    // body sensing — the buffers stay memset-empty and every consumer of the
    // hash is inside the mode!=4 branches below).
    if (params.algorithm_mode != 4) {
        int threads = 256;
        int blocks = (num_sec_initial + threads - 1) / threads;
        gpu_insert_sections_hash_kernel<<<blocks, threads>>>(
            s_d_sections, num_sec_initial, inv_body_cell, body_hash_mask,
            s_d_body_head, s_d_body_next, s_d_body_cells
        );
    }

    // Launch Kernel 2: Prune occupied nutrient spores and compact surviving spores on GPU
    cudaMemset(s_d_spore_head, 0xFF, spore_hash_size * sizeof(int));
    const int num_spores = static_cast<int>(io_spores.size());
    if (num_spores > 0) {
        cudaMemcpy(s_d_spores, io_spores.data(), num_spores * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (num_spores + threads - 1) / threads;
        gpu_prune_and_hash_spores_kernel<<<blocks, threads>>>(
            s_d_spores, num_spores,
            s_d_sections, s_d_body_head, s_d_body_next, s_d_body_cells,
            body_hash_mask, inv_body_cell, params.kill_radius,
            inv_spore_cell, spore_hash_mask,
            s_d_spore_alive, s_d_spore_head, s_d_spore_next, s_d_spore_cells
        );

        cudaMemset(&s_d_step_counters[0], 0, sizeof(int));
        gpu_compact_surviving_spores_vol_kernel<<<blocks, threads>>>(
            s_d_spores, s_d_spore_alive, num_spores,
            s_d_spores_compact, &s_d_step_counters[0]
        );
        int h_surviving = 0;
        cudaMemcpy(&h_surviving, &s_d_step_counters[0], sizeof(int), cudaMemcpyDeviceToHost);
        h_surviving = std::min(h_surviving, num_spores);
        io_spores.resize(h_surviving);
        if (h_surviving > 0) {
            cudaMemcpy(io_spores.data(), s_d_spores_compact, h_surviving * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        }
    }

    // Launch Kernel 3: Parallel 11-Feeler Cone Environment Sensing & Lateral Branch Probing on GPU
    // v24.12 Track F R1: mode 4 substitutes the developmental advance kernel
    // (gravitropic setpoint chase; no spores, no feelers, no body sensing).
    const int num_tips = static_cast<int>(io_tips.size());
    cudaMemcpy(s_d_tips, io_tips.data(), num_tips * sizeof(CudaVolTip), cudaMemcpyHostToDevice);
    {
        int threads = 128;
        int blocks = (num_tips + threads - 1) / threads;
        if (params.algorithm_mode == 4) {
            gpu_dev_advance_tips_kernel<<<blocks, threads>>>(
                s_d_tips, num_tips,
                s_d_sections,
                s_d_vessel_sdf, vessel_grid.min_coord, vessel_grid.dims,
                params, s_d_proposals
            );
        } else {
            gpu_sense_and_advance_tips_kernel<<<blocks, threads>>>(
                s_d_tips, num_tips,
                s_d_sections, s_d_body_head, s_d_body_next, s_d_body_cells,
                body_hash_mask, inv_body_cell,
                s_d_spores, s_d_spore_head, s_d_spore_next, s_d_spore_cells,
                spore_hash_mask, inv_spore_cell,
                s_d_vessel_sdf, vessel_grid.min_coord, vessel_grid.dims,
                params, collar_hops, branch_interval, min_skin_gap,
                s_d_proposals
            );
        }

        // Commit non-overlapping new root sections and lateral tips in parallel on GPU
        // v24.12 Track F R1: mode 4 uses the developmental commit kernel
        // (steps_grown bookkeeping; graft-tolerant stand-off).
        int h_init_counters[3] = { num_sec_initial, 0, io_next_root_id };
        cudaMemcpy(&s_d_step_counters[1], h_init_counters, 3 * sizeof(int), cudaMemcpyHostToDevice);

        if (params.algorithm_mode == 4) {
            gpu_dev_commit_proposals_kernel<<<blocks, threads>>>(
                s_d_tips, s_d_proposals, num_tips,
                s_d_sections, &s_d_step_counters[1], static_cast<int>(s_cap_sections),
                s_d_next_tips, &s_d_step_counters[2], static_cast<int>(s_cap_tips),
                &s_d_step_counters[3]
            );
        } else {
            gpu_commit_tip_proposals_kernel<<<blocks, threads>>>(
                s_d_tips, s_d_proposals, num_tips, min_skin_gap,
                s_d_sections, &s_d_step_counters[1], static_cast<int>(s_cap_sections),
                s_d_next_tips, &s_d_step_counters[2], static_cast<int>(s_cap_tips),
                &s_d_step_counters[3]
            );
        }
    }

    int h_out_counters[3] = { num_sec_initial, 0, io_next_root_id };
    cudaMemcpy(h_out_counters, &s_d_step_counters[1], 3 * sizeof(int), cudaMemcpyDeviceToHost);
    const int num_sec_after = std::min(h_out_counters[0], static_cast<int>(s_cap_sections));
    const int num_tips_after = std::min(h_out_counters[1], static_cast<int>(s_cap_tips));
    io_next_root_id = h_out_counters[2];

    std::swap(s_d_tips, s_d_next_tips);

    // Launch Kernel 4 & 5: Parallel GPU 3D Capsule-to-Capsule Pushing Physics & Parallel Tether Relaxation
    // v24.12 Track F R1: mode 4 skips the corpse-physics suite entirely —
    // sections are append-only history (real roots never move after they are
    // laid down); the wake kernel inside this block is skipped with it.
    if (params.algorithm_mode != 4 && num_sec_after > 8) {
        cudaMemset(s_d_body_head, 0xFF, body_hash_size * sizeof(int));
        cudaMemset(s_d_sec_disp, 0, num_sec_after * 3 * sizeof(float));
        cudaMemset(s_d_sec_disp_q, 0, num_sec_after * 3 * sizeof(unsigned long long));

        int threads = 256;
        int blocks = (num_sec_after + threads - 1) / threads;
        gpu_insert_sections_hash_kernel<<<blocks, threads>>>(
            s_d_sections, num_sec_after, inv_body_cell, body_hash_mask,
            s_d_body_head, s_d_body_next, s_d_body_cells
        );

        const float push_gain = params.push_base_gain + params.push_strength_gain * params.push_strength;
        gpu_push_sections_kernel<<<blocks, threads>>>(
            s_d_sections, num_sec_after,
            s_d_body_head, s_d_body_next, s_d_body_cells,
            body_hash_mask, inv_body_cell,
            min_skin_gap, push_gain,
            s_d_sec_disp_q
        );
        // v24.02 P1.2 DRAFT (2026-10-22) — UNVERIFIED UNTIL TRACK B: dequantize pass
        // (must land before gpu_apply_section_displacements_kernel consumes s_d_sec_disp)
        int dq_blocks = (num_sec_after * 3 + threads - 1) / threads;
        gpu_dequantize_ll_to_float_vol_kernel<<<dq_blocks, threads>>>(
            s_d_sec_disp_q, s_d_sec_disp, num_sec_after * 3
        );

        // Parallel GPU section displacement & 2-pass tether relaxation
        const float max_step_disp = step_len * params.max_step_disp_ratio;
        gpu_apply_section_displacements_kernel<<<blocks, threads>>>(
            s_d_sections, num_sec_after, s_d_sec_disp, max_step_disp,
            s_d_sec_old_pos, s_d_sec_moved
        );
        for (int pass = 0; pass < 2; ++pass) {
            gpu_rope_reel_tethers_kernel<<<blocks, threads>>>(
                s_d_sections, num_sec_after, s_d_sec_old_pos, step_len,
                params.tether_max_stretch, params.tether_target_stretch,
                params.backward_pull, params.tensile_straightening,
                s_d_vessel_sdf, vessel_grid.min_coord, vessel_grid.dims, params.voxel_size, params.min_vessel_sdf,
                params.wall_clearance_scale,
                s_d_sec_moved
            );
        }

        if (num_tips_after > 0) {
            int wake_threads = 128;
            int wake_blocks = (num_tips_after + wake_threads - 1) / wake_threads;
            gpu_wake_tips_kernel<<<wake_blocks, wake_threads>>>(
                s_d_tips, num_tips_after, s_d_sections, num_sec_after, s_d_sec_moved
            );
        }
    }

    io_sections.resize(num_sec_after);
    if (num_sec_after > 0) {
        cudaMemcpy(io_sections.data(), s_d_sections, num_sec_after * sizeof(CudaVolSection), cudaMemcpyDeviceToHost);
    }
    io_tips.resize(num_tips_after);
    if (num_tips_after > 0) {
        cudaMemcpy(io_tips.data(), s_d_tips, num_tips_after * sizeof(CudaVolTip), cudaMemcpyDeviceToHost);
    }

    return true;
#else
    (void)vessel_grid;
    (void)params;
    (void)io_spores;
    (void)io_sections;
    (void)io_tips;
    (void)io_next_root_id;
    return false;
#endif
}

bool CudaRasterizer::taper_sections_gpu(
    std::vector<CudaVolSection>& io_sections,
    float base_radius,
    float tip_radius,
    float taper_curve,
    int total_steps,
    bool variable_taper,
    uint32_t rng_seed
) {
    if (io_sections.empty()) return true;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        const float safe_taper = std::max(0.2f, taper_curve);
        const int num_secs = static_cast<int>(io_sections.size());

        CudaVolSection* d_sec = nullptr;
        int* d_max_depth = nullptr;
        size_t bytes = io_sections.size() * sizeof(CudaVolSection);
        if (cudaMalloc(&d_sec, bytes) == cudaSuccess &&
            cudaMalloc(&d_max_depth, sizeof(int)) == cudaSuccess) {

            int initial_depth = 1;
            cudaMemcpy(d_max_depth, &initial_depth, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_sec, io_sections.data(), bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (num_secs + threads - 1) / threads;

            // 1. Parallel reduction to determine maximum vascular depth on GPU
            gpu_reduce_max_depth_kernel<<<blocks, threads>>>(d_sec, num_secs, d_max_depth);

            // 2. Parallel vascular caliber tapering on GPU
            gpu_vascular_taper_kernel<<<blocks, threads>>>(
                d_sec, num_secs, d_max_depth,
                base_radius, tip_radius, safe_taper, total_steps,
                variable_taper, rng_seed
            );

            cudaMemcpy(io_sections.data(), d_sec, bytes, cudaMemcpyDeviceToHost);
            cudaFree(d_sec);
            cudaFree(d_max_depth);
            return true;
        }
        if (d_sec) cudaFree(d_sec);
        if (d_max_depth) cudaFree(d_max_depth);
    }
#endif
    return false;
}

bool CudaRasterizer::init_volumetric_seeds_gpu(
    const CudaVolumeGrid& vessel_grid,
    const std::vector<glm::vec3>& seed_positions,
    float base_radius,
    float min_vessel_sdf,
    uint32_t rng_seed,
    std::vector<CudaVolSection>& out_sections,
    std::vector<CudaVolTip>& out_tips,
    int& out_next_root_id,
    float wall_clearance_scale,
    int primary_roots
) {
    out_sections.clear();
    out_tips.clear();
    if (!vessel_grid.is_valid() || seed_positions.empty()) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int pr = std::clamp(primary_roots, 1, 16);
        int num_seeds = static_cast<int>(seed_positions.size());
        int num_tips = num_seeds * pr;

        glm::vec3* d_seeds = nullptr;
        float* d_vessel_sdf = nullptr;
        CudaVolSection* d_secs = nullptr;
        CudaVolTip* d_tips = nullptr;

        if (cudaMalloc(&d_seeds, num_seeds * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_vessel_sdf, vessel_grid.data.size() * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_secs, num_seeds * sizeof(CudaVolSection)) == cudaSuccess &&
            cudaMalloc(&d_tips, num_tips * sizeof(CudaVolTip)) == cudaSuccess) {

            cudaMemcpy(d_seeds, seed_positions.data(), num_seeds * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vessel_sdf, vessel_grid.data.data(), vessel_grid.data.size() * sizeof(float), cudaMemcpyHostToDevice);

            int threads = 64;
            int blocks = (num_seeds + threads - 1) / threads;
            gpu_init_volumetric_seeds_kernel<<<blocks, threads>>>(
                d_seeds, num_seeds, d_vessel_sdf,
                vessel_grid.min_coord, vessel_grid.dims, vessel_grid.voxel_size,
                min_vessel_sdf, base_radius, rng_seed,
                wall_clearance_scale,
                d_secs, d_tips,
                pr
            );

            out_sections.resize(num_seeds);
            out_tips.resize(num_tips);
            cudaMemcpy(out_sections.data(), d_secs, num_seeds * sizeof(CudaVolSection), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_tips.data(), d_tips, num_tips * sizeof(CudaVolTip), cudaMemcpyDeviceToHost);
            out_next_root_id = num_seeds + num_tips;

            cudaFree(d_seeds);
            cudaFree(d_vessel_sdf);
            cudaFree(d_secs);
            cudaFree(d_tips);
            return true;
        }
        if (d_seeds) cudaFree(d_seeds);
        if (d_vessel_sdf) cudaFree(d_vessel_sdf);
        if (d_secs) cudaFree(d_secs);
        if (d_tips) cudaFree(d_tips);
    }
#endif
    return false;
}

bool CudaRasterizer::convert_vol_sections_to_nodes_gpu(
    const CudaVolSection* sections_data,
    size_t num_sections,
    void* out_nodes_data
) {
    if (!sections_data || !out_nodes_data || num_sections == 0) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n = static_cast<int>(num_sections);
        CudaVolSection* d_secs = nullptr;
        GpuColonizationNodeLayout* d_nodes = nullptr;
        if (cudaMalloc(&d_secs, n * sizeof(CudaVolSection)) == cudaSuccess &&
            cudaMalloc(&d_nodes, n * sizeof(GpuColonizationNodeLayout)) == cudaSuccess) {

            cudaMemcpy(d_secs, sections_data, n * sizeof(CudaVolSection), cudaMemcpyHostToDevice);
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            gpu_convert_vol_sections_to_nodes_kernel<<<blocks, threads>>>(d_secs, n, d_nodes);
            cudaMemcpy(out_nodes_data, d_nodes, n * sizeof(GpuColonizationNodeLayout), cudaMemcpyDeviceToHost);

            cudaFree(d_secs);
            cudaFree(d_nodes);
            return true;
        }
        if (d_secs) cudaFree(d_secs);
        if (d_nodes) cudaFree(d_nodes);
    }
#endif
    return false;
}

bool CudaRasterizer::check_any_awake_tips_gpu(
    const CudaVolTip* tips_data,
    size_t num_tips,
    bool& out_any_awake
) {
    out_any_awake = false;
    if (!tips_data || num_tips == 0) return true;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n = static_cast<int>(num_tips);
        CudaVolTip* d_tips = nullptr;
        int* d_awake = nullptr;
        if (cudaMalloc(&d_tips, n * sizeof(CudaVolTip)) == cudaSuccess &&
            cudaMalloc(&d_awake, sizeof(int)) == cudaSuccess) {
            cudaMemcpy(d_tips, tips_data, n * sizeof(CudaVolTip), cudaMemcpyHostToDevice);
            cudaMemset(d_awake, 0, sizeof(int));
            int threads = 128;
            int blocks = (n + threads - 1) / threads;
            gpu_check_awake_tips_kernel<<<blocks, threads>>>(d_tips, n, d_awake);
            int h_awake = 0;
            cudaMemcpy(&h_awake, d_awake, sizeof(int), cudaMemcpyDeviceToHost);
            out_any_awake = (h_awake > 0);
            cudaFree(d_tips);
            cudaFree(d_awake);
            return true;
        }
        if (d_tips) cudaFree(d_tips);
        if (d_awake) cudaFree(d_awake);
    }
#endif
    return false;
}

bool CudaRasterizer::wake_or_respawn_volumetric_tips_gpu(
    const CudaVolSection* sections_data,
    size_t num_sections,
    std::vector<CudaVolTip>& io_tips,
    bool force_reset_stalled,
    bool& out_any_awake
) {
    out_any_awake = false;
    if (!sections_data || num_sections == 0) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n_sec = static_cast<int>(num_sections);
        int n_tips = static_cast<int>(io_tips.size());
        int max_tips = std::max(n_tips + 24, 32);

        CudaVolSection* d_secs = nullptr;
        CudaVolTip* d_tips = nullptr;
        int* d_has_child = nullptr;
        int* d_cnt = nullptr; // [0]=awake_count, [1]=tip_count

        if (cudaMalloc(&d_secs, n_sec * sizeof(CudaVolSection)) == cudaSuccess &&
            cudaMalloc(&d_tips, max_tips * sizeof(CudaVolTip)) == cudaSuccess &&
            cudaMalloc(&d_has_child, n_sec * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_cnt, 2 * sizeof(int)) == cudaSuccess) {

            cudaMemcpy(d_secs, sections_data, n_sec * sizeof(CudaVolSection), cudaMemcpyHostToDevice);
            if (n_tips > 0) {
                cudaMemcpy(d_tips, io_tips.data(), n_tips * sizeof(CudaVolTip), cudaMemcpyHostToDevice);
            }
            int init_cnt[2] = { 0, n_tips };
            cudaMemcpy(d_cnt, init_cnt, 2 * sizeof(int), cudaMemcpyHostToDevice);

            int threads = 256;
            if (n_tips > 0) {
                int t_blocks = (n_tips + threads - 1) / threads;
                if (force_reset_stalled) {
                    gpu_reset_stalled_tips_kernel<<<t_blocks, threads>>>(d_tips, n_tips);
                } else {
                    gpu_check_awake_tips_kernel<<<t_blocks, threads>>>(d_tips, n_tips, &d_cnt[0]);
                    int h_awake = 0;
                    cudaMemcpy(&h_awake, &d_cnt[0], sizeof(int), cudaMemcpyDeviceToHost);
                    if (h_awake == 0) {
                        gpu_reset_stalled_tips_kernel<<<t_blocks, threads>>>(d_tips, n_tips);
                    }
                }
            }

            bool need_respawn = force_reset_stalled ? (n_tips < 6 && n_sec > 1) : (n_tips == 0 && n_sec > 1);
            if (need_respawn) {
                cudaMemset(d_has_child, 0, n_sec * sizeof(int));
                int s_blocks = (n_sec + threads - 1) / threads;
                gpu_mark_section_children_kernel<<<s_blocks, threads>>>(d_secs, n_sec, d_has_child);
                if (n_tips > 0) {
                    int t_blocks = (n_tips + threads - 1) / threads;
                    gpu_mark_active_tip_sections_kernel<<<t_blocks, threads>>>(d_tips, n_tips, n_sec, d_has_child);
                }
                gpu_respawn_leaf_tips_kernel<<<s_blocks, threads>>>(
                    d_secs, d_has_child, n_sec, d_tips, &d_cnt[1], 24
                );
            }

            int h_final_tips = n_tips;
            cudaMemcpy(&h_final_tips, &d_cnt[1], sizeof(int), cudaMemcpyDeviceToHost);
            if (need_respawn) {
                h_final_tips = std::min(h_final_tips, 24);
            } else {
                h_final_tips = n_tips;
            }

            io_tips.resize(h_final_tips);
            if (h_final_tips > 0) {
                cudaMemcpy(io_tips.data(), d_tips, h_final_tips * sizeof(CudaVolTip), cudaMemcpyDeviceToHost);
                cudaMemset(&d_cnt[0], 0, sizeof(int));
                int t_blocks = (h_final_tips + threads - 1) / threads;
                gpu_check_awake_tips_kernel<<<t_blocks, threads>>>(d_tips, h_final_tips, &d_cnt[0]);
                int h_awake = 0;
                cudaMemcpy(&h_awake, &d_cnt[0], sizeof(int), cudaMemcpyDeviceToHost);
                out_any_awake = (h_awake > 0);
            }

            cudaFree(d_secs);
            cudaFree(d_tips);
            cudaFree(d_has_child);
            cudaFree(d_cnt);
            return true;
        }
        if (d_secs) cudaFree(d_secs);
        if (d_tips) cudaFree(d_tips);
        if (d_has_child) cudaFree(d_has_child);
        if (d_cnt) cudaFree(d_cnt);
    }
#endif
    return false;
}

bool CudaRasterizer::sync_volumetric_seed_freeze_and_spawn_gpu(
    const CudaVolumeGrid& vessel_grid,
    const std::vector<glm::vec3>& seed_positions,
    const std::vector<uint8_t>& frozen_seeds,
    float base_radius,
    float min_vessel_sdf,
    uint32_t rng_seed,
    float wall_clearance_scale,
    std::vector<CudaVolSection>& io_sections,
    std::vector<CudaVolTip>& io_tips,
    int& io_next_root_id,
    bool& out_spawned_new,
    int primary_roots
) {
    out_spawned_new = false;
    if (!vessel_grid.is_valid() || seed_positions.empty()) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int pr = std::clamp(primary_roots, 1, 16);
        int num_seeds = static_cast<int>(seed_positions.size());
        int num_secs = static_cast<int>(io_sections.size());
        int num_tips = static_cast<int>(io_tips.size());
        int max_secs = num_secs + num_seeds + 16;
        int max_tips = num_tips + num_seeds * pr + 32;

        std::vector<uint8_t> h_frozen(num_seeds, 0);
        for (int s = 0; s < num_seeds && s < static_cast<int>(frozen_seeds.size()); ++s) {
            h_frozen[s] = frozen_seeds[s];
        }

        glm::vec3* d_seeds = nullptr;
        uint8_t* d_frozen = nullptr;
        int* d_seed_exists = nullptr;
        float* d_vessel_sdf = nullptr;
        CudaVolSection* d_secs = nullptr;
        CudaVolTip* d_tips = nullptr;
        int* d_cnt = nullptr; // [0]=num_secs, [1]=num_tips, [2]=next_root_id, [3]=spawned_count

        if (cudaMalloc(&d_seeds, num_seeds * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_frozen, num_seeds * sizeof(uint8_t)) == cudaSuccess &&
            cudaMalloc(&d_seed_exists, num_seeds * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_vessel_sdf, vessel_grid.data.size() * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_secs, max_secs * sizeof(CudaVolSection)) == cudaSuccess &&
            cudaMalloc(&d_tips, max_tips * sizeof(CudaVolTip)) == cudaSuccess &&
            cudaMalloc(&d_cnt, 4 * sizeof(int)) == cudaSuccess) {

            cudaMemcpy(d_seeds, seed_positions.data(), num_seeds * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            cudaMemcpy(d_frozen, h_frozen.data(), num_seeds * sizeof(uint8_t), cudaMemcpyHostToDevice);
            cudaMemset(d_seed_exists, 0, num_seeds * sizeof(int));
            cudaMemcpy(d_vessel_sdf, vessel_grid.data.data(), vessel_grid.data.size() * sizeof(float), cudaMemcpyHostToDevice);

            if (num_secs > 0) {
                cudaMemcpy(d_secs, io_sections.data(), num_secs * sizeof(CudaVolSection), cudaMemcpyHostToDevice);
            }
            if (num_tips > 0) {
                cudaMemcpy(d_tips, io_tips.data(), num_tips * sizeof(CudaVolTip), cudaMemcpyHostToDevice);
            }

            int init_cnt[4] = { num_secs, num_tips, std::max(io_next_root_id, num_seeds * 16 + 1), 0 };
            cudaMemcpy(d_cnt, init_cnt, 4 * sizeof(int), cudaMemcpyHostToDevice);

            int threads = 256;
            if (num_secs > 0) {
                int s_blocks = (num_secs + threads - 1) / threads;
                gpu_sync_volumetric_freeze_sections_kernel<<<s_blocks, threads>>>(
                    d_secs, num_secs, d_seeds, d_frozen, num_seeds, d_seed_exists
                );
            }
            if (num_tips > 0) {
                int t_blocks = (num_tips + threads - 1) / threads;
                gpu_sync_volumetric_freeze_tips_kernel<<<t_blocks, threads>>>(
                    d_tips, num_tips, d_secs, num_secs, d_frozen, num_seeds
                );
            }

            int seed_blocks = (num_seeds + 63) / 64;
            gpu_spawn_missing_volumetric_seeds_kernel<<<seed_blocks, 64>>>(
                d_seeds, d_frozen, d_seed_exists, num_seeds,
                d_vessel_sdf, vessel_grid.min_coord, vessel_grid.dims, vessel_grid.voxel_size,
                min_vessel_sdf, base_radius, rng_seed, wall_clearance_scale,
                d_secs, &d_cnt[0], max_secs,
                d_tips, &d_cnt[1], max_tips,
                &d_cnt[2], &d_cnt[3],
                pr
            );

            int out_cnt[4] = { num_secs, num_tips, io_next_root_id, 0 };
            cudaMemcpy(out_cnt, d_cnt, 4 * sizeof(int), cudaMemcpyDeviceToHost);

            int final_secs = std::min(out_cnt[0], max_secs);
            int final_tips = std::min(out_cnt[1], max_tips);
            io_next_root_id = out_cnt[2];
            out_spawned_new = (out_cnt[3] > 0);

            io_sections.resize(final_secs);
            if (final_secs > 0) {
                cudaMemcpy(io_sections.data(), d_secs, final_secs * sizeof(CudaVolSection), cudaMemcpyDeviceToHost);
            }
            io_tips.resize(final_tips);
            if (final_tips > 0) {
                cudaMemcpy(io_tips.data(), d_tips, final_tips * sizeof(CudaVolTip), cudaMemcpyDeviceToHost);
            }

            cudaFree(d_seeds);
            cudaFree(d_frozen);
            cudaFree(d_seed_exists);
            cudaFree(d_vessel_sdf);
            cudaFree(d_secs);
            cudaFree(d_tips);
            cudaFree(d_cnt);
            return true;
        }
        if (d_seeds) cudaFree(d_seeds);
        if (d_frozen) cudaFree(d_frozen);
        if (d_seed_exists) cudaFree(d_seed_exists);
        if (d_vessel_sdf) cudaFree(d_vessel_sdf);
        if (d_secs) cudaFree(d_secs);
        if (d_tips) cudaFree(d_tips);
        if (d_cnt) cudaFree(d_cnt);
    }
#endif
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_snap_seed_to_root_surface_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    glm::vec3 query_pos,
    float target_offset,
    float* __restrict__ block_best_dist,
    glm::vec3* __restrict__ block_best_pos,
    glm::vec3* __restrict__ block_best_norm
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float my_dist = 1e9f;
    glm::vec3 my_pos(0.0f);
    glm::vec3 my_norm(0.0f, 0.0f, 1.0f);

    if (i < num_sections) {
        CudaVolSection sec = sections[i];
        glm::vec3 seg_pt(0.0f);
        float seg_r = sec.radius;

        if (sec.parent_sec >= 0 && sec.parent_sec < num_sections) {
            CudaVolSection psec = sections[sec.parent_sec];
            glm::vec3 ba = sec.pos - psec.pos;
            float l2 = glm::dot(ba, ba);
            float t = (l2 > 1e-8f) ? dev_clamp(glm::dot(query_pos - psec.pos, ba) / l2, 0.0f, 1.0f) : 0.0f;
            seg_pt = psec.pos + t * ba;
            seg_r = psec.radius + (sec.radius - psec.radius) * t;
        } else {
            seg_pt = sec.pos;
        }

        glm::vec3 diff = query_pos - seg_pt;
        float d = sqrtf(glm::dot(diff, diff));
        if (d > 1e-5f) {
            my_norm = diff / d;
        } else {
            my_norm = glm::vec3(0.0f, 0.0f, 1.0f);
        }
        my_dist = fabsf(d - seg_r);
        my_pos = seg_pt + my_norm * (seg_r + target_offset);
    }

    __shared__ float s_dist[256];
    __shared__ glm::vec3 s_pos[256];
    __shared__ glm::vec3 s_norm[256];

    int tid = threadIdx.x;
    s_dist[tid] = my_dist;
    s_pos[tid] = my_pos;
    s_norm[tid] = my_norm;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_dist[tid + stride] < s_dist[tid]) {
                s_dist[tid] = s_dist[tid + stride];
                s_pos[tid] = s_pos[tid + stride];
                s_norm[tid] = s_norm[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_best_dist[blockIdx.x] = s_dist[0];
        block_best_pos[blockIdx.x] = s_pos[0];
        block_best_norm[blockIdx.x] = s_norm[0];
    }
}

__global__ void gpu_reduce_snap_seed_blocks_kernel(
    const float* __restrict__ block_best_dist,
    const glm::vec3* __restrict__ block_best_pos,
    const glm::vec3* __restrict__ block_best_norm,
    int num_blocks,
    glm::vec3* __restrict__ out_snapped_pos,
    glm::vec3* __restrict__ out_normal
) {
    if (threadIdx.x == 0) {
        float min_d = 1e9f;
        int best_idx = 0;
        for (int b = 0; b < num_blocks; ++b) {
            if (block_best_dist[b] < min_d) {
                min_d = block_best_dist[b];
                best_idx = b;
            }
        }
        *out_snapped_pos = block_best_pos[best_idx];
        *out_normal = block_best_norm[best_idx];
    }
}
#endif

bool CudaRasterizer::snap_seed_to_root_surface_gpu(
    const CudaVolSection* sections_data,
    size_t num_sections,
    glm::vec3 query_pos,
    float target_offset,
    glm::vec3& out_snapped_pos,
    glm::vec3& out_surface_normal,
    float& out_time_ms
) {
    out_time_ms = 0.0f;
    out_snapped_pos = query_pos;
    out_surface_normal = glm::vec3(0.0f, 0.0f, 1.0f);
    if (!sections_data || num_sections == 0) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int n = static_cast<int>(num_sections);
        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        CudaVolSection* d_secs = nullptr;
        float* d_b_dist = nullptr;
        glm::vec3* d_b_pos = nullptr;
        glm::vec3* d_b_norm = nullptr;
        glm::vec3* d_res_pos = nullptr;
        glm::vec3* d_res_norm = nullptr;

        if (cudaMalloc(&d_secs, n * sizeof(CudaVolSection)) == cudaSuccess &&
            cudaMalloc(&d_b_dist, blocks * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_b_pos, blocks * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_b_norm, blocks * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_res_pos, sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_res_norm, sizeof(glm::vec3)) == cudaSuccess) {

            cudaMemcpy(d_secs, sections_data, n * sizeof(CudaVolSection), cudaMemcpyHostToDevice);

            gpu_snap_seed_to_root_surface_kernel<<<blocks, threads>>>(
                d_secs, n, query_pos, target_offset, d_b_dist, d_b_pos, d_b_norm
            );

            gpu_reduce_snap_seed_blocks_kernel<<<1, 1>>>(
                d_b_dist, d_b_pos, d_b_norm, blocks, d_res_pos, d_res_norm
            );

            cudaMemcpy(&out_snapped_pos, d_res_pos, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
            cudaMemcpy(&out_surface_normal, d_res_norm, sizeof(glm::vec3), cudaMemcpyDeviceToHost);

            cudaFree(d_secs);
            cudaFree(d_b_dist);
            cudaFree(d_b_pos);
            cudaFree(d_b_norm);
            cudaFree(d_res_pos);
            cudaFree(d_res_norm);

            auto t1 = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            return true;
        }
        if (d_secs) cudaFree(d_secs);
        if (d_b_dist) cudaFree(d_b_dist);
        if (d_b_pos) cudaFree(d_b_pos);
        if (d_b_norm) cudaFree(d_b_norm);
        if (d_res_pos) cudaFree(d_res_pos);
        if (d_res_norm) cudaFree(d_res_norm);
    }
#endif
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_mark_volumetric_sections_for_seed_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    int deleted_seed_idx,
    int* __restrict__ out_keep
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;

    int seed_slot = dev_find_vol_section_seed_slot(sections, num_sections, i);
    out_keep[i] = (seed_slot == deleted_seed_idx) ? 0 : 1;
}

__global__ void gpu_block_prefix_scan_sections_kernel(
    const int* __restrict__ in_flags,
    int n,
    int* __restrict__ out_local_scanned,
    int* __restrict__ out_block_sums
) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    s_data[tid] = (idx < n) ? in_flags[idx] : 0;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int temp = 0;
        if (tid >= offset) {
            temp = s_data[tid - offset];
        }
        __syncthreads();
        s_data[tid] += temp;
        __syncthreads();
    }

    if (idx < n) {
        out_local_scanned[idx] = s_data[tid];
    }
    if (tid == blockDim.x - 1) {
        out_block_sums[blockIdx.x] = s_data[tid];
    }
}

__global__ void gpu_single_block_scan_sections_kernel(
    int* __restrict__ io_block_sums,
    int num_blocks
) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    s_data[tid] = (tid < num_blocks) ? io_block_sums[tid] : 0;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int temp = 0;
        if (tid >= offset) {
            temp = s_data[tid - offset];
        }
        __syncthreads();
        s_data[tid] += temp;
        __syncthreads();
    }
    if (tid < num_blocks) {
        io_block_sums[tid] = s_data[tid];
    }
}

__global__ void gpu_add_scanned_block_offsets_sections_kernel(
    const int* __restrict__ local_scanned,
    const int* __restrict__ block_sums,
    const int* __restrict__ in_flags,
    int n,
    int* __restrict__ out_old_to_new_map
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (in_flags[idx] == 0) {
        out_old_to_new_map[idx] = -1;
    } else {
        int block_offset = (blockIdx.x > 0) ? block_sums[blockIdx.x - 1] : 0;
        out_old_to_new_map[idx] = (local_scanned[idx] - 1) + block_offset;
    }
}

__global__ void gpu_remap_and_compact_vol_sections_kernel(
    const CudaVolSection* __restrict__ in_sections,
    int num_sections,
    const int* __restrict__ old_to_new_map,
    int deleted_seed_idx,
    CudaVolSection* __restrict__ out_sections
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;
    int new_i = old_to_new_map[i];
    if (new_i < 0) return;

    CudaVolSection s = in_sections[i];
    if (s.parent_sec >= 0 && s.parent_sec < num_sections) {
        s.parent_sec = old_to_new_map[s.parent_sec];
    } else {
        s.parent_sec = -1;
    }

    if (s.parent_sec < 0) {
        if (s.root_id > deleted_seed_idx) {
            s.root_id -= 1;
        }
    }

    out_sections[new_i] = s;
}

__global__ void gpu_remap_and_compact_vol_tips_kernel(
    const CudaVolTip* __restrict__ in_tips,
    int num_tips,
    const int* __restrict__ old_to_new_map,
    int num_sections,
    CudaVolTip* __restrict__ out_tips,
    int* __restrict__ d_out_tip_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tips) return;

    CudaVolTip tip = in_tips[i];
    int old_sec = tip.sec_idx;
    if (old_sec < 0 || old_sec >= num_sections) return;

    int new_sec = old_to_new_map[old_sec];
    if (new_sec >= 0) {
        tip.sec_idx = new_sec;
        int slot = atomicAdd(d_out_tip_count, 1);
        out_tips[slot] = tip;
    }
}
#endif

bool CudaRasterizer::prune_volumetric_sections_by_seed_gpu(
    std::vector<CudaVolSection>& io_sections,
    std::vector<CudaVolTip>& io_tips,
    int deleted_seed_idx
) {
    if (io_sections.empty() || deleted_seed_idx < 0) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n_sec = static_cast<int>(io_sections.size());
        int n_tips = static_cast<int>(io_tips.size());

        CudaVolSection* d_in_secs = nullptr;
        CudaVolSection* d_out_secs = nullptr;
        int* d_keep = nullptr;
        int* d_local_scanned = nullptr;
        int* d_block_sums = nullptr;
        int* d_old_to_new = nullptr;
        CudaVolTip* d_in_tips = nullptr;
        CudaVolTip* d_out_tips = nullptr;
        int* d_tip_count = nullptr;

        const int threads = 256;
        const int blocks = (n_sec + threads - 1) / threads;

        bool ok = (cudaMalloc(&d_in_secs, n_sec * sizeof(CudaVolSection)) == cudaSuccess) &&
                  (cudaMalloc(&d_out_secs, n_sec * sizeof(CudaVolSection)) == cudaSuccess) &&
                  (cudaMalloc(&d_keep, n_sec * sizeof(int)) == cudaSuccess) &&
                  (cudaMalloc(&d_local_scanned, n_sec * sizeof(int)) == cudaSuccess) &&
                  (cudaMalloc(&d_block_sums, blocks * sizeof(int)) == cudaSuccess) &&
                  (cudaMalloc(&d_old_to_new, n_sec * sizeof(int)) == cudaSuccess);
        if (ok && n_tips > 0) {
            ok = (cudaMalloc(&d_in_tips, n_tips * sizeof(CudaVolTip)) == cudaSuccess) &&
                 (cudaMalloc(&d_out_tips, n_tips * sizeof(CudaVolTip)) == cudaSuccess) &&
                 (cudaMalloc(&d_tip_count, sizeof(int)) == cudaSuccess);
        }

        if (ok) {
            cudaMemcpy(d_in_secs, io_sections.data(), n_sec * sizeof(CudaVolSection), cudaMemcpyHostToDevice);
            if (n_tips > 0) {
                cudaMemcpy(d_in_tips, io_tips.data(), n_tips * sizeof(CudaVolTip), cudaMemcpyHostToDevice);
                cudaMemset(d_tip_count, 0, sizeof(int));
            }

            // 1. Mark keep (1) or discard (0) based on origin seed
            gpu_mark_volumetric_sections_for_seed_kernel<<<blocks, threads>>>(
                d_in_secs, n_sec, deleted_seed_idx, d_keep
            );

            // 2. Multi-block prefix scan
            size_t shared_bytes = threads * sizeof(int);
            gpu_block_prefix_scan_sections_kernel<<<blocks, threads, shared_bytes>>>(
                d_keep, n_sec, d_local_scanned, d_block_sums
            );
            if (blocks > 1) {
                gpu_single_block_scan_sections_kernel<<<1, 1024, 1024 * sizeof(int)>>>(
                    d_block_sums, blocks
                );
            }
            gpu_add_scanned_block_offsets_sections_kernel<<<blocks, threads>>>(
                d_local_scanned, d_block_sums, d_keep, n_sec, d_old_to_new
            );

            // 3. Compact and remap sections
            gpu_remap_and_compact_vol_sections_kernel<<<blocks, threads>>>(
                d_in_secs, n_sec, d_old_to_new, deleted_seed_idx, d_out_secs
            );

            int h_surviving_secs = 0;
            if (blocks > 0) {
                cudaMemcpy(&h_surviving_secs, &d_block_sums[blocks - 1], sizeof(int), cudaMemcpyDeviceToHost);
            }
            h_surviving_secs = std::clamp(h_surviving_secs, 0, n_sec);

            // 4. Compact and remap active tips
            int h_surviving_tips = 0;
            if (n_tips > 0) {
                int tip_blocks = (n_tips + threads - 1) / threads;
                gpu_remap_and_compact_vol_tips_kernel<<<tip_blocks, threads>>>(
                    d_in_tips, n_tips, d_old_to_new, n_sec, d_out_tips, d_tip_count
                );
                cudaMemcpy(&h_surviving_tips, d_tip_count, sizeof(int), cudaMemcpyDeviceToHost);
                h_surviving_tips = std::clamp(h_surviving_tips, 0, n_tips);
            }

            // 5. Copy back to host
            io_sections.resize(h_surviving_secs);
            if (h_surviving_secs > 0) {
                cudaMemcpy(io_sections.data(), d_out_secs, h_surviving_secs * sizeof(CudaVolSection), cudaMemcpyDeviceToHost);
            }

            io_tips.resize(h_surviving_tips);
            if (h_surviving_tips > 0) {
                cudaMemcpy(io_tips.data(), d_out_tips, h_surviving_tips * sizeof(CudaVolTip), cudaMemcpyDeviceToHost);
            }

            cudaFree(d_in_secs);
            cudaFree(d_out_secs);
            cudaFree(d_keep);
            cudaFree(d_local_scanned);
            cudaFree(d_block_sums);
            cudaFree(d_old_to_new);
            if (d_in_tips) cudaFree(d_in_tips);
            if (d_out_tips) cudaFree(d_out_tips);
            if (d_tip_count) cudaFree(d_tip_count);
            return true;
        }

        if (d_in_secs) cudaFree(d_in_secs);
        if (d_out_secs) cudaFree(d_out_secs);
        if (d_keep) cudaFree(d_keep);
        if (d_local_scanned) cudaFree(d_local_scanned);
        if (d_block_sums) cudaFree(d_block_sums);
        if (d_old_to_new) cudaFree(d_old_to_new);
        if (d_in_tips) cudaFree(d_in_tips);
        if (d_out_tips) cudaFree(d_out_tips);
        if (d_tip_count) cudaFree(d_tip_count);
    }
#endif
    return false;
}

#ifdef __CUDACC__

__global__ void gpu_generate_lateral_roots_kernel(
    const CudaVolSection* __restrict__ seminal_sections,
    int num_seminal_sections,
    const uint8_t* __restrict__ seed_has_lateral,
    const uint8_t* __restrict__ seed_snapped_to_root,
    int num_seeds,
    CudaVolGrowthParams params,
    CudaVolSection* __restrict__ out_sections,
    int* __restrict__ d_out_count,
    int max_out_sections
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_seminal_sections) return;

    int parent_idx = seminal_sections[i].parent_sec;
    if (parent_idx < 0 || parent_idx >= num_seminal_sections) return;

    int seed_slot = dev_find_vol_section_seed_slot(seminal_sections, num_seminal_sections, i);
    if (seed_slot < 0 || seed_slot >= num_seeds) return;
    if (seed_has_lateral == nullptr || seed_has_lateral[seed_slot] == 0) return;

    bool snap_to_surface = (seed_snapped_to_root != nullptr && seed_snapped_to_root[seed_slot] != 0);

    glm::vec3 p0 = seminal_sections[parent_idx].pos;
    glm::vec3 p1 = seminal_sections[i].pos;
    float r0 = seminal_sections[parent_idx].radius;
    float r1 = seminal_sections[i].radius;

    glm::vec3 seg = p1 - p0;
    float seg_len = glm::length(seg);
    if (seg_len < 1e-4f) return;

    glm::vec3 T = seg / seg_len; // Tangent along branch

    // Coordinate frame perpendicular to tangent T
    glm::vec3 up = (fabsf(T.z) < 0.9f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 B = dev_safe_norm(glm::cross(T, up), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::vec3 N = dev_safe_norm(glm::cross(B, T), glm::vec3(0.0f, 1.0f, 0.0f));

    float spacing = (params.lateral_density > 0.1f) ? params.lateral_density : 1.5f;
    int num_sprouts = dev_clamp(static_cast<int>(seg_len / spacing), 1, 6);

    float max_reach = (params.lateral_max_length > 0.5f) ? params.lateral_max_length : 8.0f;
    int lateral_steps = dev_clamp(static_cast<int>(max_reach / (params.growth_step > 0.2f ? params.growth_step : 0.8f)), 3, 10);
    float step_len = max_reach / static_cast<float>(lateral_steps);

    float lat_r_base = dev_clamp(params.lateral_trunk_caliber * 0.5f, 0.02f, 0.5f);
    float lat_r_tip = dev_clamp(params.lateral_tip_caliber * 0.5f, 0.01f, lat_r_base);

    for (int k = 0; k < num_sprouts; ++k) {
        float t = (static_cast<float>(k) + 0.5f) / static_cast<float>(num_sprouts);
        glm::vec3 c_center = p0 + seg * t;
        float c_radius = r0 + (r1 - r0) * t;

        // Golden spiral angle around circumference
        uint32_t seed_h = static_cast<uint32_t>(dev_hash_01(params.seed + static_cast<uint32_t>(i * 37 + k * 19)) * 10000.0f);
        float angle = static_cast<float>(k) * 2.39996323f + dev_hash_01(seed_h) * 1.5f;
        glm::vec3 radial_norm = cosf(angle) * B + sinf(angle) * N;

        // Spawn point on surface of seminal root
        glm::vec3 spawn_pos = c_center + radial_norm * (c_radius + lat_r_base);

        // Reserve space for lateral strand in output buffer.
        // TRACK C FIX 3: on overflow, write as many steps as fit instead of
        // abandoning the reservation. The old bare `return` still consumed
        // the atomicAdd reservation, leaving reserved-but-unwritten holes
        // inside [0, max_out_sections) that were later memcpy'd to the host
        // as uninitialized GPU garbage sections (NaN/huge positions poison
        // the rasterizer bounds). With no overflow, steps_to_write ==
        // lateral_steps and the loop is byte-identical to the old math.
        int base_slot = atomicAdd(d_out_count, lateral_steps);
        if (base_slot >= max_out_sections) {
            return;
        }
        const int steps_to_write = min(lateral_steps, max_out_sections - base_slot);

        glm::vec3 cur_pos = spawn_pos;
        glm::vec3 cur_dir;
        if (snap_to_surface) {
            // Surface-crawling: start tangent to root cylinder, slightly angled along branch
            glm::vec3 surf_tangent = dev_safe_norm(-sinf(angle) * B + cosf(angle) * N, B);
            cur_dir = dev_safe_norm(surf_tangent * 0.7f + T * 0.3f, surf_tangent);
        } else {
            // Outward sprouting: angle away from the surface normal
            float rad_deg = params.lateral_branch_angle * (3.14159265f / 180.0f);
            cur_dir = dev_safe_norm(radial_norm * cosf(rad_deg) + T * sinf(rad_deg), radial_norm);
        }

        int prev_idx = i; // Connects to the current seminal section for graph continuity
        for (int step = 0; step < steps_to_write; ++step) {
            float frac = static_cast<float>(step) / static_cast<float>(lateral_steps - 1);
            float cur_r = lat_r_base * (1.0f - frac) + lat_r_tip * frac;

            if (step > 0) {
                // Organic wander
                float w_u = dev_hash_01(seed_h + static_cast<uint32_t>(step * 13)) - 0.5f;
                float w_v = dev_hash_01(seed_h + static_cast<uint32_t>(step * 29)) - 0.5f;
                glm::vec3 wander_vec = (B * w_u + N * w_v) * params.lateral_wander * 0.5f;
                cur_dir = dev_safe_norm(cur_dir + wander_vec, cur_dir);

                if (snap_to_surface) {
                    // Pull back to cylinder surface at distance (c_radius + cur_r)
                    glm::vec3 to_cur = cur_pos - c_center;
                    float axial_dist = glm::dot(to_cur, T);
                    glm::vec3 rad_comp = to_cur - T * axial_dist;
                    float rad_len = glm::length(rad_comp);
                    if (rad_len > 1e-4f) {
                        glm::vec3 target_pos = c_center + T * axial_dist + (rad_comp / rad_len) * (c_radius + cur_r);
                        cur_pos = cur_pos * 0.4f + target_pos * 0.6f;
                    }
                }
                cur_pos += cur_dir * step_len;
            }

            CudaVolSection l_sec;
            l_sec.pos = cur_pos;
            l_sec.radius = cur_r;
            l_sec.parent_sec = prev_idx;
            l_sec.root_id = seminal_sections[i].root_id;
            l_sec.depth = seminal_sections[i].depth + step + 1;
            l_sec.pinned = (step > 0) ? 64 : 0;

            out_sections[base_slot + step] = l_sec;
            prev_idx = base_slot + step;
        }
    }
}

#endif // __CUDACC__

bool CudaRasterizer::generate_lateral_roots_gpu(
    const CudaVolSection* in_seminal_sections,
    int num_seminal_sections,
    const uint8_t* seed_has_lateral,
    const uint8_t* seed_snapped_to_root,
    int num_seeds,
    const CudaVolGrowthParams& params,
    std::vector<CudaVolSection>& out_lateral_sections,
    float& out_time_ms
) {
#ifdef __CUDACC__
    if (!in_seminal_sections || num_seminal_sections <= 0 || !seed_has_lateral || num_seeds <= 0) {
        return false;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    CudaVolSection* d_seminal = nullptr;
    uint8_t* d_has_lat = nullptr;
    uint8_t* d_snap = nullptr;
    CudaVolSection* d_out_sections = nullptr;
    int* d_out_count = nullptr;

    const int max_out_sections = num_seminal_sections * 48; // Generous ceiling for lateral rootlets

    cudaError_t err1 = cudaMalloc(&d_seminal, num_seminal_sections * sizeof(CudaVolSection));
    cudaError_t err2 = cudaMalloc(&d_has_lat, num_seeds * sizeof(uint8_t));
    cudaError_t err3 = cudaMalloc(&d_snap, num_seeds * sizeof(uint8_t));
    cudaError_t err4 = cudaMalloc(&d_out_sections, max_out_sections * sizeof(CudaVolSection));
    cudaError_t err5 = cudaMalloc(&d_out_count, sizeof(int));

    if (err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess && err4 == cudaSuccess && err5 == cudaSuccess) {
        cudaMemcpy(d_seminal, in_seminal_sections, num_seminal_sections * sizeof(CudaVolSection), cudaMemcpyHostToDevice);
        cudaMemcpy(d_has_lat, seed_has_lateral, num_seeds * sizeof(uint8_t), cudaMemcpyHostToDevice);
        if (seed_snapped_to_root) {
            cudaMemcpy(d_snap, seed_snapped_to_root, num_seeds * sizeof(uint8_t), cudaMemcpyHostToDevice);
        } else {
            cudaMemset(d_snap, 0, num_seeds * sizeof(uint8_t));
        }
        cudaMemset(d_out_count, 0, sizeof(int));

        int threads = 128;
        int blocks = (num_seminal_sections + threads - 1) / threads;
        gpu_generate_lateral_roots_kernel<<<blocks, threads>>>(
            d_seminal,
            num_seminal_sections,
            d_has_lat,
            d_snap,
            num_seeds,
            params,
            d_out_sections,
            d_out_count,
            max_out_sections
        );
        cudaDeviceSynchronize();

        int h_count = 0;
        cudaMemcpy(&h_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost);
        h_count = std::clamp(h_count, 0, max_out_sections);

        out_lateral_sections.resize(h_count);
        if (h_count > 0) {
            cudaMemcpy(out_lateral_sections.data(), d_out_sections, h_count * sizeof(CudaVolSection), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_seminal);
        cudaFree(d_has_lat);
        cudaFree(d_snap);
        cudaFree(d_out_sections);
        cudaFree(d_out_count);

        auto end_time = std::chrono::high_resolution_clock::now();
        out_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        return true;
    }

    if (d_seminal) cudaFree(d_seminal);
    if (d_has_lat) cudaFree(d_has_lat);
    if (d_snap) cudaFree(d_snap);
    if (d_out_sections) cudaFree(d_out_sections);
    if (d_out_count) cudaFree(d_out_count);
#endif
    return false;
}

} // namespace rhizome
