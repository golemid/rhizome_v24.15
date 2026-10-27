#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/cuda_common.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace rhizome {

#ifdef __CUDACC__

// --- 2. CUDA Kernel: Spore Search ---
__global__ void spore_search_kernel(
    CudaSporeData* __restrict__ spores,
    int num_spores,
    const CudaBranchNode* __restrict__ nodes,
    int num_nodes,
    float search_radius_sq,
    float kill_radius_sq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_spores) return;

    CudaSporeData sp = spores[idx];
    if (!sp.alive) return;

    float closest_dist_sq = search_radius_sq;
    int closest_idx = -1;
    bool kill = false;

    for (int n = 0; n < num_nodes; ++n) {
        CudaBranchNode node = nodes[n];
        glm::vec3 diff = sp.pos - node.pos;
        float d2 = glm::dot(diff, diff);

        float eff_kill_r = node.radius + sqrtf(kill_radius_sq);
        if (d2 <= eff_kill_r * eff_kill_r) {
            kill = true;
            break;
        }

        if (d2 < closest_dist_sq) {
            closest_dist_sq = d2;
            closest_idx = n;
        }
    }

    if (kill) {
        spores[idx].alive = false;
        spores[idx].closest_node_idx = -1;
        spores[idx].min_dist = 0.0f;
    } else {
        spores[idx].closest_node_idx = closest_idx;
        spores[idx].min_dist = sqrtf(closest_dist_sq);
    }
}

// v24.02 P1.2 DRAFT (2026-10-22) — UNVERIFIED UNTIL TRACK B:
// float atomicAdd accumulation replaced by fixed-point uint64 atomicAdd (Q = 2^24,
// quantum ~6e-8). CUDA exposes no signed 64-bit atomicAdd overload, so the shadow
// accumulators are unsigned long long: two's-complement addition mod 2^64 is
// associative and commutative, so the accumulated total is bit-identical
// regardless of thread scheduling (deterministic). The original float buffer is
// refilled by a dequantize pass for existing consumers.
// Track B fix 1 (2026-10-22): original draft used long long, which does not
// compile (no such atomicAdd overload) — flavor changed to uint64, semantics equal.
__global__ void gpu_accumulate_spore_forces_kernel(
    const CudaSporeData* __restrict__ spores,
    int num_spores,
    const CudaBranchNode* __restrict__ nodes,
    int num_nodes,
    unsigned long long* __restrict__ out_forces_q,
    int* __restrict__ out_counts
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_spores) return;

    CudaSporeData sp = spores[i];
    if (sp.alive && sp.closest_node_idx >= 0 && sp.closest_node_idx < num_nodes) {
        int ni = sp.closest_node_idx;
        glm::vec3 d = sp.pos - nodes[ni].pos;
        float len2 = glm::dot(d, d);
        if (len2 > 1e-12f) {
            glm::vec3 unit_d = d * rsqrtf(len2);
            atomicAdd(&out_forces_q[ni * 3 + 0], (unsigned long long)llroundf(unit_d.x * 16777216.0f));
            atomicAdd(&out_forces_q[ni * 3 + 1], (unsigned long long)llroundf(unit_d.y * 16777216.0f));
            atomicAdd(&out_forces_q[ni * 3 + 2], (unsigned long long)llroundf(unit_d.z * 16777216.0f));
            atomicAdd(&out_counts[ni], 1);
        }
    }
}

// v24.02 P1.2 DRAFT (2026-10-22) — UNVERIFIED UNTIL TRACK B:
// dequantize pass for the fixed-point uint64 accumulators (Q = 2^24). One thread
// per element; per-element conversion is deterministic by construction. The
// unsigned->signed conversion is well-defined modulo 2^64 under C++20
// (CMAKE_CUDA_STANDARD 20), so the two's-complement sum maps to the signed total.
__global__ void gpu_dequantize_ll_to_float_kernel(
    const unsigned long long* __restrict__ q,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<float>(static_cast<long long>(q[i])) * (1.0f / 16777216.0f);
}

#endif // __CUDACC__

// 2. Colonization Spore Search
bool CudaRasterizer::evaluate_spores(
    std::vector<CudaSporeData>& spores,
    const std::vector<CudaBranchNode>& nodes,
    float search_radius,
    float kill_radius,
    std::vector<glm::vec3>& out_attraction_forces,
    std::vector<int>& out_attraction_counts,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_attraction_forces.assign(nodes.size(), glm::vec3(0.0f));
    out_attraction_counts.assign(nodes.size(), 0);

    float search_r2 = search_radius * search_radius;
    float kill_r2 = kill_radius * kill_radius;

#ifdef __CUDACC__
    if (is_cuda_available() && !spores.empty() && !nodes.empty()) {
        CudaSporeData* d_spores = nullptr;
        CudaBranchNode* d_nodes = nullptr;
        float* d_forces = nullptr;
        int* d_counts = nullptr;
        unsigned long long* d_forces_q = nullptr; // v24.02 P1.2 DRAFT: quantized shadow accumulator (Q = 2^24, uint64)
        size_t num_spores = spores.size();
        size_t num_nodes = nodes.size();

        if (cudaMalloc(&d_spores, num_spores * sizeof(CudaSporeData)) == cudaSuccess &&
            cudaMalloc(&d_nodes, num_nodes * sizeof(CudaBranchNode)) == cudaSuccess &&
            cudaMalloc(&d_forces, num_nodes * 3 * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_counts, num_nodes * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_forces_q, num_nodes * 3 * sizeof(unsigned long long)) == cudaSuccess) {

            cudaMemcpy(d_spores, spores.data(), num_spores * sizeof(CudaSporeData), cudaMemcpyHostToDevice);
            cudaMemcpy(d_nodes, nodes.data(), num_nodes * sizeof(CudaBranchNode), cudaMemcpyHostToDevice);
            cudaMemset(d_forces, 0, num_nodes * 3 * sizeof(float));
            cudaMemset(d_counts, 0, num_nodes * sizeof(int));
            cudaMemset(d_forces_q, 0, num_nodes * 3 * sizeof(unsigned long long));

            int threads = 256;
            int blocks = (static_cast<int>(num_spores) + threads - 1) / threads;
            spore_search_kernel<<<blocks, threads>>>(
                d_spores, static_cast<int>(num_spores), d_nodes, static_cast<int>(num_nodes), search_r2, kill_r2
            );
            gpu_accumulate_spore_forces_kernel<<<blocks, threads>>>(
                d_spores, static_cast<int>(num_spores), d_nodes, static_cast<int>(num_nodes), d_forces_q, d_counts
            );
            // v24.02 P1.2 DRAFT (2026-10-22) — UNVERIFIED UNTIL TRACK B: dequantize pass
            int dq_blocks = (static_cast<int>(num_nodes * 3) + threads - 1) / threads;
            gpu_dequantize_ll_to_float_kernel<<<dq_blocks, threads>>>(
                d_forces_q, d_forces, static_cast<int>(num_nodes * 3)
            );

            cudaMemcpy(spores.data(), d_spores, num_spores * sizeof(CudaSporeData), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_attraction_forces.data(), d_forces, num_nodes * 3 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_attraction_counts.data(), d_counts, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_spores);
            cudaFree(d_nodes);
            cudaFree(d_forces);
            cudaFree(d_counts);
            cudaFree(d_forces_q);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_spores) cudaFree(d_spores);
        if (d_nodes) cudaFree(d_nodes);
        if (d_forces) cudaFree(d_forces);
        if (d_counts) cudaFree(d_counts);
        if (d_forces_q) cudaFree(d_forces_q);
    }
#endif

    std::cerr << "[CUDA Error] GPU acceleration required but unavailable or out of memory in evaluate_spores.\n";
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_sample_spores_kernel(
    const float* __restrict__ vessel_sdf,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    float voxel_size,
    float min_x, float min_y, float min_z,
    float span_x, float span_y, float span_z,
    float spore_margin,
    uint32_t seed,
    int attempts_per_thread,
    int target_count,
    glm::vec3* __restrict__ out_spores,
    int* __restrict__ d_valid_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandState_t rng;
    curand_init(seed + static_cast<uint32_t>(idx * 1664525u + 1013904223u), idx, 0, &rng);

    for (int step = 0; step < attempts_per_thread; ++step) {
        if (*d_valid_count >= target_count) break;

        float rx = curand_uniform(&rng);
        float ry = curand_uniform(&rng);
        float rz = curand_uniform(&rng);

        glm::vec3 cand(min_x + rx * span_x, min_y + ry * span_y, min_z + rz * span_z);
        float sdf = dev_sample_sdf_trilinear(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, cand);

        if (sdf <= spore_margin) {
            int slot = atomicAdd(d_valid_count, 1);
            if (slot < target_count) {
                out_spores[slot] = cand;
            } else {
                break;
            }
        }
    }
}
#endif

bool CudaRasterizer::sample_spores_gpu(
    const CudaVolumeGrid& vessel_grid,
    int target_spore_count,
    uint32_t seed,
    float spore_margin,
    std::vector<glm::vec3>& out_spores
) {
    if (!vessel_grid.is_valid() || target_spore_count <= 0) return false;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_grid = nullptr;
        size_t grid_bytes = vessel_grid.data.size() * sizeof(float);
        if (cudaMalloc(&d_grid, grid_bytes) == cudaSuccess) {
            cudaMemcpy(d_grid, vessel_grid.data.data(), grid_bytes, cudaMemcpyHostToDevice);

            glm::vec3 min_w = glm::vec3(vessel_grid.min_coord) * vessel_grid.voxel_size;
            glm::vec3 max_w = glm::vec3(vessel_grid.min_coord + vessel_grid.dims) * vessel_grid.voxel_size;
            glm::vec3 span = max_w - min_w;

            const int attempts_per_thread = 32;
            const int total_attempts = target_spore_count * 50;
            const int total_threads = std::max(256, (total_attempts + attempts_per_thread - 1) / attempts_per_thread);

            glm::vec3* d_out_spores = nullptr;
            int* d_valid_count = nullptr;

            if (cudaMalloc(&d_out_spores, target_spore_count * sizeof(glm::vec3)) == cudaSuccess &&
                cudaMalloc(&d_valid_count, sizeof(int)) == cudaSuccess) {

                cudaMemset(d_valid_count, 0, sizeof(int));

                int threads = 256;
                int blocks = (total_threads + threads - 1) / threads;
                gpu_sample_spores_kernel<<<blocks, threads>>>(
                    d_grid, vessel_grid.min_coord, vessel_grid.dims, vessel_grid.voxel_size,
                    min_w.x, min_w.y, min_w.z, span.x, span.y, span.z,
                    spore_margin, seed, attempts_per_thread, target_spore_count,
                    d_out_spores, d_valid_count
                );

                int h_count = 0;
                cudaMemcpy(&h_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
                h_count = std::min(h_count, target_spore_count);

                if (h_count > 0) {
                    out_spores.resize(h_count);
                    cudaMemcpy(out_spores.data(), d_out_spores, h_count * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
                }

                cudaFree(d_out_spores);
                cudaFree(d_valid_count);
                cudaFree(d_grid);
                return (h_count > 0);
            }

            if (d_out_spores) cudaFree(d_out_spores);
            if (d_valid_count) cudaFree(d_valid_count);
            cudaFree(d_grid);
        }
    }
#endif

    return !out_spores.empty();
}

#ifdef __CUDACC__

__global__ void gpu_compact_surviving_spores_kernel(
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

// --- 29. GPU Kernels for Classic Space Colonization ---

__global__ void gpu_init_seed_nodes_kernel(
    const glm::vec3* __restrict__ seeds,
    int num_seeds,
    float base_radius,
    GpuColonizationNodeLayout* __restrict__ out_nodes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_seeds) return;
    GpuColonizationNodeLayout n;
    n.position = seeds[i];
    n.parent_index = -1;
    n.radius = base_radius;
    n.depth = 0;
    n.energy = 1.0f;
    n.strand_id = i;
    out_nodes[i] = n;
}

// v24.02 P1.2 DRAFT (2026-10-22) — UNVERIFIED UNTIL TRACK B:
// same fixed-point uint64 substitution as gpu_accumulate_spore_forces_kernel
// (float atomicAdd is order-sensitive; uint64 atomicAdd at Q = 2^24 is not).
__global__ void gpu_classic_spore_eval_kernel(
    const glm::vec3* __restrict__ spores,
    int num_spores,
    const GpuColonizationNodeLayout* __restrict__ nodes,
    int num_nodes,
    float search_radius,
    float kill_radius,
    uint8_t* __restrict__ out_spore_alive,
    unsigned long long* __restrict__ out_node_dir_q,
    int* __restrict__ out_node_counts
) {
    int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (s_idx >= num_spores) return;

    glm::vec3 sp = spores[s_idx];
    float best_d = 1e30f;
    int best_node = -1;

    for (int n = 0; n < num_nodes; ++n) {
        // Only active (unfrozen) nodes compete to attract and consume spores
        if (nodes[n].energy < 0.0f) continue;
        glm::vec3 diff = sp - nodes[n].position;
        float d = sqrtf(glm::dot(diff, diff));
        if (d < best_d) {
            best_d = d;
            best_node = n;
        }
    }

    if (best_node >= 0 && best_d <= kill_radius) {
        out_spore_alive[s_idx] = 0;
        return;
    }

    out_spore_alive[s_idx] = 1;
    if (best_node >= 0 && best_d <= search_radius && best_d > 1e-6f) {
        glm::vec3 dir = (sp - nodes[best_node].position) / best_d;
        atomicAdd(&out_node_dir_q[best_node * 3 + 0], (unsigned long long)llroundf(dir.x * 16777216.0f));
        atomicAdd(&out_node_dir_q[best_node * 3 + 1], (unsigned long long)llroundf(dir.y * 16777216.0f));
        atomicAdd(&out_node_dir_q[best_node * 3 + 2], (unsigned long long)llroundf(dir.z * 16777216.0f));
        atomicAdd(&out_node_counts[best_node], 1);
    }
}

__global__ void gpu_classic_grow_nodes_kernel(
    GpuColonizationNodeLayout* __restrict__ io_nodes,
    const int* __restrict__ has_child,
    int num_existing_nodes,
    int max_nodes_capacity,
    const float* __restrict__ node_dir_xyz,
    const int* __restrict__ node_counts,
    const float* __restrict__ vessel_sdf,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    float voxel_size,
    float growth_step,
    bool variable_step,
    float min_growth_step,
    float max_growth_step,
    float branching_rate,
    float base_radius,
    float tip_radius,
    int total_classic_steps,
    uint32_t rng_seed,
    bool use_branching_prob,
    float wall_clearance_scale,
    bool variable_taper,
    float steering_cone,
    int* __restrict__ d_new_node_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_existing_nodes) return;
    if (node_counts[i] <= 0) return;

    bool is_lateral_branch = (io_nodes[i].parent_index >= 0 && has_child != nullptr && has_child[i] > 0);
    if (use_branching_prob && is_lateral_branch) {
        float prob = dev_hash_01(rng_seed + static_cast<uint32_t>(i * 7919 + 101));
        if (prob > branching_rate) return;
    }

    glm::vec3 avg_dir(
        node_dir_xyz[i * 3 + 0],
        node_dir_xyz[i * 3 + 1],
        node_dir_xyz[i * 3 + 2]
    );
    float len = sqrtf(glm::dot(avg_dir, avg_dir));
    if (len < 1e-6f) return;
    avg_dir /= len;

    // Apply forward momentum and steering cone constraint from parent branch
    glm::vec3 grow_dir = avg_dir;
    if (io_nodes[i].parent_index >= 0) {
        glm::vec3 in_dir = io_nodes[i].position - io_nodes[io_nodes[i].parent_index].position;
        float in_len = sqrtf(glm::dot(in_dir, in_dir));
        if (in_len > 1e-5f) {
            in_dir /= in_len;
            grow_dir = dev_clamp_to_cone(avg_dir, in_dir, steering_cone);
        }
    }

    int parent_gen = max(0, io_nodes[i].strand_id >> 16);
    int next_gen = is_lateral_branch ? min(parent_gen + 1, 8) : parent_gen;
    int branch_id = is_lateral_branch ? ((num_existing_nodes + i * 17 + 1) & 0xFFFF) : (io_nodes[i].strand_id & 0xFFFF);
    int next_strand_id = (next_gen << 16) | branch_id;
    int new_depth = io_nodes[i].depth + 1;

    float denom = use_branching_prob ? static_cast<float>(max(5, total_classic_steps)) : 40.0f;
    float t = fminf(1.0f, static_cast<float>(new_depth) / denom);

    float step_to_use = growth_step;
    if (variable_step) {
        float min_s = fmaxf(0.25f, fminf(min_growth_step, max_growth_step));
        float max_s = fmaxf(min_s, fmaxf(min_growth_step, max_growth_step));
        float wall_clear = 0.85f;
        if (vessel_sdf != nullptr && vessel_dims.x > 0) {
            float cur_sdf = dev_sample_sdf_trilinear(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, io_nodes[i].position);
            wall_clear = dev_clamp(-cur_sdf / fmaxf(max_s * 1.5f, 0.5f), 0.0f, 1.0f);
        }
        float gen_decay = powf(0.75f, static_cast<float>(next_gen));
        float branch_hash = dev_hash_01(rng_seed + static_cast<uint32_t>(branch_id * 7919 + next_gen * 1013 + 37));
        float branch_speed = (next_gen == 0) ? (0.80f + 0.20f * branch_hash) : (0.50f + 0.50f * branch_hash);
        float depth_decay = 1.0f - 0.25f * t;
        float t_step = dev_clamp(gen_decay * branch_speed * depth_decay * (0.40f + 0.60f * wall_clear), 0.0f, 1.0f);
        step_to_use = min_s + (max_s - min_s) * t_step;
    }
    float min_r = fminf(base_radius, tip_radius);
    float max_r = fmaxf(base_radius, tip_radius);
    float gen_decay = powf(0.74f, static_cast<float>(next_gen));
    float branch_hash = dev_hash_01(rng_seed + static_cast<uint32_t>(branch_id * 7919 + next_gen * 1013 + 53));
    float branch_scale = (next_gen == 0) ? (0.72f + 0.28f * branch_hash) : (0.48f + 0.52f * branch_hash);
    float branch_factor = dev_clamp(gen_decay * branch_scale, 0.08f, 1.0f);
    float branch_max_r = min_r + (max_r - min_r) * branch_factor;
    float new_r = branch_max_r + (tip_radius - branch_max_r) * t;
    float new_energy = fmaxf(0.0f, (1.0f - t) * branch_factor);
    if (variable_taper) {
        float wave = 0.5f + 0.5f * sinf(static_cast<float>(new_depth) * 0.65f + static_cast<float>(branch_id) * 2.17f);
        float jitter = dev_hash_01(rng_seed + static_cast<uint32_t>(i * 2971 + new_depth * 733));
        float t_var = dev_clamp(0.45f * branch_hash + 0.35f * wave + 0.20f * jitter, 0.0f, 1.0f) * dev_clamp(gen_decay, 0.15f, 1.0f);
        new_r = min_r + (max_r - min_r) * t_var;
        new_energy = t_var;
    }

    float local_step_floor = fmaxf(0.25f, fmaxf(io_nodes[i].radius * 1.12f, new_r * 1.12f));
    step_to_use = fmaxf(step_to_use, local_step_floor);

    glm::vec3 cand_pos = io_nodes[i].position + grow_dir * step_to_use;

    if (vessel_sdf != nullptr && vessel_dims.x > 0) {
        float target_wall_sdf = -fmaxf(0.05f, new_r * (wall_clearance_scale - 0.75f));
        float v_sdf = dev_sample_sdf_trilinear(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, cand_pos);
        if (use_branching_prob) {
            if (v_sdf > target_wall_sdf) {
                glm::vec3 n = dev_sample_sdf_grad(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, cand_pos);
                glm::vec3 tang = grow_dir - glm::dot(grow_dir, n) * n;
                float tlen = sqrtf(glm::dot(tang, tang));
                if (tlen > 1e-5f) {
                    cand_pos = io_nodes[i].position + dev_clamp_to_cone(tang / tlen, grow_dir, steering_cone) * step_to_use;
                    v_sdf = dev_sample_sdf_trilinear(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, cand_pos);
                }
                if (v_sdf > target_wall_sdf + 0.05f) {
                    glm::vec3 n2 = dev_sample_sdf_grad(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, cand_pos);
                    cand_pos -= n2 * (v_sdf - target_wall_sdf + 0.05f);
                }
            }
        } else {
            if (v_sdf > -0.05f) {
                glm::vec3 n = dev_sample_sdf_grad(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, cand_pos);
                cand_pos -= n * (v_sdf + 0.1f);
            }
        }
    }

    int add_slot = atomicAdd(d_new_node_count, 1);
    int out_idx = num_existing_nodes + add_slot;
    if (out_idx >= max_nodes_capacity) return;

    GpuColonizationNodeLayout new_n;
    new_n.position = cand_pos;
    new_n.parent_index = i;
    new_n.depth = new_depth;
    new_n.radius = new_r;
    new_n.energy = new_energy;
    new_n.strand_id = next_strand_id;
    io_nodes[out_idx] = new_n;
}

__global__ void gpu_reduce_max_node_depth_kernel(
    const GpuColonizationNodeLayout* __restrict__ nodes,
    int num_nodes,
    int* __restrict__ out_max_depth
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    atomicMax(out_max_depth, nodes[i].depth);
}

__global__ void gpu_taper_classic_nodes_kernel(
    GpuColonizationNodeLayout* __restrict__ nodes,
    int num_nodes,
    const int* __restrict__ d_max_depth,
    float base_radius,
    float tip_radius,
    bool variable_taper,
    uint32_t rng_seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    if (nodes[i].energy < 0.0f) return;
    if (nodes[i].parent_index < 0) {
        nodes[i].radius = base_radius;
        nodes[i].energy = 1.0f;
        return;
    }
    float max_d = static_cast<float>(max(1, *d_max_depth));
    float t = dev_clamp(static_cast<float>(nodes[i].depth) / max_d, 0.0f, 1.0f);
    int gen = max(0, nodes[i].strand_id >> 16);
    int branch_id = nodes[i].strand_id & 0xFFFF;
    float min_r = fminf(base_radius, tip_radius);
    float max_r = fmaxf(base_radius, tip_radius);
    float gen_decay = powf(0.74f, static_cast<float>(gen));
    float branch_hash = dev_hash_01(rng_seed + static_cast<uint32_t>(branch_id * 7919 + gen * 1013 + 53));
    if (variable_taper) {
        float wave = 0.5f + 0.5f * sinf(static_cast<float>(nodes[i].depth) * 0.65f + static_cast<float>(branch_id) * 2.17f);
        float jitter = dev_hash_01(rng_seed + static_cast<uint32_t>(i * 1999 + nodes[i].depth * 733 + branch_id * 37));
        float t_var = dev_clamp(0.45f * branch_hash + 0.35f * wave + 0.20f * jitter, 0.0f, 1.0f) * dev_clamp(gen_decay, 0.15f, 1.0f);
        nodes[i].radius = min_r + (max_r - min_r) * t_var;
        nodes[i].energy = t_var;
    } else {
        float branch_scale = (gen == 0) ? (0.72f + 0.28f * branch_hash) : (0.48f + 0.52f * branch_hash);
        float branch_factor = dev_clamp(gen_decay * branch_scale, 0.08f, 1.0f);
        float branch_max_r = min_r + (max_r - min_r) * branch_factor;
        nodes[i].radius = branch_max_r + (tip_radius - branch_max_r) * t;
        nodes[i].energy = fmaxf(0.0f, (1.0f - t) * branch_factor);
    }
}

__global__ void gpu_extract_root_seeds_kernel(
    const GpuColonizationNodeLayout* __restrict__ nodes,
    int num_nodes,
    glm::vec3* __restrict__ out_seeds,
    int* __restrict__ d_seed_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    if (nodes[i].parent_index < 0) {
        int slot = atomicAdd(d_seed_count, 1);
        out_seeds[slot] = nodes[i].position;
    }
}

__device__ inline int dev_find_classic_node_seed_slot(
    const GpuColonizationNodeLayout* __restrict__ nodes,
    int num_nodes,
    int idx
) {
    int cur = idx;
    for (int hop = 0; hop < 4096; ++hop) {
        int p = nodes[cur].parent_index;
        if (p < 0 || p >= num_nodes || p == cur) break;
        cur = p;
    }
    return (nodes[cur].strand_id & 0xFFFF);
}

__global__ void gpu_sync_classic_freeze_nodes_kernel(
    GpuColonizationNodeLayout* __restrict__ nodes,
    int num_nodes,
    const glm::vec3* __restrict__ seeds,
    const uint8_t* __restrict__ frozen_seeds,
    int num_seeds,
    int* __restrict__ out_seed_exists
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    int seed_slot = dev_find_classic_node_seed_slot(nodes, num_nodes, i);
    if (nodes[i].parent_index < 0 && seed_slot >= 0 && seed_slot < num_seeds) {
        out_seed_exists[seed_slot] = 1;
        // Keep seed root node position in sync with moved seed position
        if (frozen_seeds == nullptr || frozen_seeds[seed_slot] == 0) {
            nodes[i].position = seeds[seed_slot];
        }
    }
    bool is_frozen = (seed_slot >= 0 && seed_slot < num_seeds && frozen_seeds != nullptr && frozen_seeds[seed_slot] != 0);
    float abs_e = fmaxf(fabsf(nodes[i].energy), 0.001f);
    nodes[i].energy = is_frozen ? -abs_e : abs_e;
}

__global__ void gpu_spawn_missing_classic_seeds_kernel(
    const glm::vec3* __restrict__ seeds,
    const uint8_t* __restrict__ frozen_seeds,
    const int* __restrict__ seed_exists,
    int num_seeds,
    float base_radius,
    GpuColonizationNodeLayout* __restrict__ io_nodes,
    int* __restrict__ d_num_nodes,
    int max_nodes,
    int* __restrict__ d_spawned_count
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_seeds) return;
    if (seed_exists[s] != 0) return;

    int idx = atomicAdd(d_num_nodes, 1);
    if (idx >= max_nodes) return;

    bool is_frozen = (frozen_seeds[s] != 0);
    GpuColonizationNodeLayout n;
    n.position = seeds[s];
    n.parent_index = -1;
    n.radius = base_radius;
    n.depth = 0;
    n.energy = is_frozen ? -1.0f : 1.0f;
    n.strand_id = s;
    io_nodes[idx] = n;
    atomicAdd(d_spawned_count, 1);
}

#endif // __CUDACC__

bool CudaRasterizer::init_seed_nodes_gpu(
    const std::vector<glm::vec3>& seed_positions,
    float base_radius,
    void* out_nodes_data,
    size_t out_nodes_count
) {
    if (!out_nodes_data || out_nodes_count == 0) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n = static_cast<int>(out_nodes_count);
        glm::vec3* d_seeds = nullptr;
        GpuColonizationNodeLayout* d_nodes = nullptr;
        if (cudaMalloc(&d_seeds, n * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_nodes, n * sizeof(GpuColonizationNodeLayout)) == cudaSuccess) {
            cudaMemcpy(d_seeds, seed_positions.data(), n * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            int threads = 64;
            int blocks = (n + threads - 1) / threads;
            gpu_init_seed_nodes_kernel<<<blocks, threads>>>(d_seeds, n, base_radius, d_nodes);
            cudaMemcpy(out_nodes_data, d_nodes, n * sizeof(GpuColonizationNodeLayout), cudaMemcpyDeviceToHost);
            cudaFree(d_seeds);
            cudaFree(d_nodes);
            return true;
        }
        if (d_seeds) cudaFree(d_seeds);
        if (d_nodes) cudaFree(d_nodes);
    }
#endif
    return false;
}

bool CudaRasterizer::step_classic_colonization_gpu(
    const CudaVolumeGrid& vessel_grid,
    std::vector<glm::vec3>& io_spores,
    void* io_nodes_data,
    size_t& io_num_nodes,
    size_t max_nodes_capacity,
    float search_radius,
    float kill_radius,
    float growth_step,
    float branching_rate,
    float base_radius,
    float tip_radius,
    int total_classic_steps,
    uint32_t rng_seed,
    bool use_branching_prob,
    bool taper_all_nodes,
    bool variable_step,
    float min_growth_step,
    float max_growth_step,
    float wall_clearance_scale,
    bool variable_taper,
    float steering_cone
) {
    if (io_spores.empty() || !io_nodes_data || io_num_nodes == 0) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int num_spores = static_cast<int>(io_spores.size());
        int num_nodes = static_cast<int>(io_num_nodes);
        int max_cap = static_cast<int>(max_nodes_capacity);

        glm::vec3* d_spores = nullptr;
        glm::vec3* d_spores_compact = nullptr;
        uint8_t* d_spore_alive = nullptr;
        GpuColonizationNodeLayout* d_nodes = nullptr;
        int* d_has_child = nullptr;
        float* d_node_dir = nullptr;
        int* d_node_counts = nullptr;
        unsigned long long* d_node_dir_q = nullptr; // v24.02 P1.2 DRAFT: quantized shadow accumulator (Q = 2^24, uint64)
        float* d_vessel_sdf = nullptr;
        int* d_counters = nullptr; // [0]=surviving_spores, [1]=new_nodes, [2]=max_depth

        bool has_vessel = vessel_grid.is_valid() && !vessel_grid.data.empty();
        bool ok = (cudaMalloc(&d_spores, num_spores * sizeof(glm::vec3)) == cudaSuccess) &&
                  (cudaMalloc(&d_spores_compact, num_spores * sizeof(glm::vec3)) == cudaSuccess) &&
                  (cudaMalloc(&d_spore_alive, num_spores * sizeof(uint8_t)) == cudaSuccess) &&
                  (cudaMalloc(&d_nodes, max_cap * sizeof(GpuColonizationNodeLayout)) == cudaSuccess) &&
                  (cudaMalloc(&d_has_child, num_nodes * sizeof(int)) == cudaSuccess) &&
                  (cudaMalloc(&d_node_dir, num_nodes * 3 * sizeof(float)) == cudaSuccess) &&
                  (cudaMalloc(&d_node_dir_q, num_nodes * 3 * sizeof(unsigned long long)) == cudaSuccess) &&
                  (cudaMalloc(&d_node_counts, num_nodes * sizeof(int)) == cudaSuccess) &&
                  (cudaMalloc(&d_counters, 3 * sizeof(int)) == cudaSuccess);

        if (ok && has_vessel) {
            ok = (cudaMalloc(&d_vessel_sdf, vessel_grid.data.size() * sizeof(float)) == cudaSuccess);
        }

        if (ok) {
            cudaMemcpy(d_spores, io_spores.data(), num_spores * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            cudaMemcpy(d_nodes, io_nodes_data, num_nodes * sizeof(GpuColonizationNodeLayout), cudaMemcpyHostToDevice);
            cudaMemset(d_has_child, 0, num_nodes * sizeof(int));
            cudaMemset(d_node_dir, 0, num_nodes * 3 * sizeof(float));
            cudaMemset(d_node_counts, 0, num_nodes * sizeof(int));
            cudaMemset(d_node_dir_q, 0, num_nodes * 3 * sizeof(unsigned long long));
            int init_cnt[3] = { 0, 0, std::max(5, total_classic_steps) };
            cudaMemcpy(d_counters, init_cnt, 3 * sizeof(int), cudaMemcpyHostToDevice);
            if (has_vessel) {
                cudaMemcpy(d_vessel_sdf, vessel_grid.data.data(), vessel_grid.data.size() * sizeof(float), cudaMemcpyHostToDevice);
            }

            int threads = 256;
            int s_blocks = (num_spores + threads - 1) / threads;
            gpu_classic_spore_eval_kernel<<<s_blocks, threads>>>(
                d_spores, num_spores, d_nodes, num_nodes,
                search_radius, kill_radius,
                d_spore_alive, d_node_dir_q, d_node_counts
            );
            // v24.02 P1.2 DRAFT (2026-10-22) — UNVERIFIED UNTIL TRACK B: dequantize pass
            // (must land before gpu_classic_grow_nodes_kernel consumes d_node_dir)
            int dq_blocks = (num_nodes * 3 + threads - 1) / threads;
            gpu_dequantize_ll_to_float_kernel<<<dq_blocks, threads>>>(
                d_node_dir_q, d_node_dir, num_nodes * 3
            );

            gpu_compact_surviving_spores_kernel<<<s_blocks, threads>>>(
                d_spores, d_spore_alive, num_spores,
                d_spores_compact, &d_counters[0]
            );

            int n_blocks = (num_nodes + threads - 1) / threads;
            gpu_mark_node_children_kernel<<<n_blocks, threads>>>(d_nodes, num_nodes, d_has_child);

            gpu_classic_grow_nodes_kernel<<<n_blocks, threads>>>(
                d_nodes, d_has_child, num_nodes, max_cap,
                d_node_dir, d_node_counts,
                d_vessel_sdf,
                has_vessel ? vessel_grid.min_coord : glm::ivec3(0),
                has_vessel ? vessel_grid.dims : glm::ivec3(0),
                has_vessel ? vessel_grid.voxel_size : 1.0f,
                growth_step, variable_step, min_growth_step, max_growth_step,
                branching_rate, base_radius, tip_radius,
                total_classic_steps, rng_seed, use_branching_prob,
                wall_clearance_scale, variable_taper,
                steering_cone,
                &d_counters[1]
            );

            int h_cnt[3] = { 0, 0, 0 };
            cudaMemcpy(h_cnt, d_counters, 3 * sizeof(int), cudaMemcpyDeviceToHost);
            int surviving = std::min(h_cnt[0], num_spores);
            int added = std::min(h_cnt[1], max_cap - num_nodes);

            io_spores.resize(surviving);
            if (surviving > 0) {
                cudaMemcpy(io_spores.data(), d_spores_compact, surviving * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
            }

            int total_nodes = num_nodes + std::max(0, added);
            if (added > 0 && taper_all_nodes) {
                int t_blocks = (total_nodes + threads - 1) / threads;
                gpu_reduce_max_node_depth_kernel<<<t_blocks, threads>>>(d_nodes, total_nodes, &d_counters[2]);
                gpu_taper_classic_nodes_kernel<<<t_blocks, threads>>>(d_nodes, total_nodes, &d_counters[2], base_radius, tip_radius, variable_taper, rng_seed);
            }

            if (added > 0) {
                io_num_nodes = static_cast<size_t>(total_nodes);
                cudaMemcpy(io_nodes_data, d_nodes, total_nodes * sizeof(GpuColonizationNodeLayout), cudaMemcpyDeviceToHost);
            }

            cudaFree(d_spores);
            cudaFree(d_spores_compact);
            cudaFree(d_spore_alive);
            cudaFree(d_nodes);
            if (d_has_child) cudaFree(d_has_child);
            cudaFree(d_node_dir);
            cudaFree(d_node_counts);
            cudaFree(d_node_dir_q);
            if (d_vessel_sdf) cudaFree(d_vessel_sdf);
            cudaFree(d_counters);

            return added > 0;
        }
        if (d_spores) cudaFree(d_spores);
        if (d_spores_compact) cudaFree(d_spores_compact);
        if (d_spore_alive) cudaFree(d_spore_alive);
        if (d_nodes) cudaFree(d_nodes);
        if (d_has_child) cudaFree(d_has_child);
        if (d_node_dir) cudaFree(d_node_dir);
        if (d_node_counts) cudaFree(d_node_counts);
        if (d_node_dir_q) cudaFree(d_node_dir_q);
        if (d_vessel_sdf) cudaFree(d_vessel_sdf);
        if (d_counters) cudaFree(d_counters);
    }
#endif
    return false;
}

bool CudaRasterizer::extract_root_seeds_from_nodes_gpu(
    const void* nodes_data,
    size_t num_nodes,
    std::vector<glm::vec3>& out_seeds
) {
    out_seeds.clear();
    if (!nodes_data || num_nodes == 0) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n = static_cast<int>(num_nodes);
        GpuColonizationNodeLayout* d_nodes = nullptr;
        glm::vec3* d_seeds = nullptr;
        int* d_count = nullptr;
        if (cudaMalloc(&d_nodes, n * sizeof(GpuColonizationNodeLayout)) == cudaSuccess &&
            cudaMalloc(&d_seeds, n * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_count, sizeof(int)) == cudaSuccess) {

            cudaMemcpy(d_nodes, nodes_data, n * sizeof(GpuColonizationNodeLayout), cudaMemcpyHostToDevice);
            cudaMemset(d_count, 0, sizeof(int));

            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            gpu_extract_root_seeds_kernel<<<blocks, threads>>>(d_nodes, n, d_seeds, d_count);

            int h_count = 0;
            cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
            h_count = std::min(h_count, n);
            if (h_count > 0) {
                out_seeds.resize(h_count);
                cudaMemcpy(out_seeds.data(), d_seeds, h_count * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
            }

            cudaFree(d_nodes);
            cudaFree(d_seeds);
            cudaFree(d_count);
            return true;
        }
        if (d_nodes) cudaFree(d_nodes);
        if (d_seeds) cudaFree(d_seeds);
        if (d_count) cudaFree(d_count);
    }
#endif
    return false;
}

bool CudaRasterizer::sync_classic_seed_freeze_and_spawn_gpu(
    const std::vector<glm::vec3>& seed_positions,
    const std::vector<uint8_t>& frozen_seeds,
    float base_radius,
    void* io_nodes_data,
    size_t& io_num_nodes,
    size_t max_nodes_capacity,
    bool& out_spawned_new
) {
    out_spawned_new = false;
    if (!io_nodes_data || seed_positions.empty()) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int num_seeds = static_cast<int>(seed_positions.size());
        int num_nodes = static_cast<int>(io_num_nodes);
        int max_nodes = static_cast<int>(max_nodes_capacity);

        std::vector<uint8_t> h_frozen(num_seeds, 0);
        for (int s = 0; s < num_seeds && s < static_cast<int>(frozen_seeds.size()); ++s) {
            h_frozen[s] = frozen_seeds[s];
        }

        glm::vec3* d_seeds = nullptr;
        uint8_t* d_frozen = nullptr;
        int* d_seed_exists = nullptr;
        GpuColonizationNodeLayout* d_nodes = nullptr;
        int* d_cnt = nullptr; // [0]=num_nodes, [1]=spawned_count

        if (cudaMalloc(&d_seeds, num_seeds * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_frozen, num_seeds * sizeof(uint8_t)) == cudaSuccess &&
            cudaMalloc(&d_seed_exists, num_seeds * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_nodes, max_nodes * sizeof(GpuColonizationNodeLayout)) == cudaSuccess &&
            cudaMalloc(&d_cnt, 2 * sizeof(int)) == cudaSuccess) {

            cudaMemcpy(d_seeds, seed_positions.data(), num_seeds * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            cudaMemcpy(d_frozen, h_frozen.data(), num_seeds * sizeof(uint8_t), cudaMemcpyHostToDevice);
            cudaMemset(d_seed_exists, 0, num_seeds * sizeof(int));
            if (num_nodes > 0) {
                cudaMemcpy(d_nodes, io_nodes_data, num_nodes * sizeof(GpuColonizationNodeLayout), cudaMemcpyHostToDevice);
            }
            int init_cnt[2] = { num_nodes, 0 };
            cudaMemcpy(d_cnt, init_cnt, 2 * sizeof(int), cudaMemcpyHostToDevice);

            int threads = 256;
            if (num_nodes > 0) {
                int n_blocks = (num_nodes + threads - 1) / threads;
                gpu_sync_classic_freeze_nodes_kernel<<<n_blocks, threads>>>(
                    d_nodes, num_nodes, d_seeds, d_frozen, num_seeds, d_seed_exists
                );
            }

            int s_blocks = (num_seeds + 63) / 64;
            gpu_spawn_missing_classic_seeds_kernel<<<s_blocks, 64>>>(
                d_seeds, d_frozen, d_seed_exists, num_seeds, base_radius,
                d_nodes, &d_cnt[0], max_nodes, &d_cnt[1]
            );

            int out_cnt[2] = { num_nodes, 0 };
            cudaMemcpy(out_cnt, d_cnt, 2 * sizeof(int), cudaMemcpyDeviceToHost);
            int final_nodes = std::min(out_cnt[0], max_nodes);
            out_spawned_new = (out_cnt[1] > 0);
            io_num_nodes = static_cast<size_t>(final_nodes);
            if (final_nodes > 0) {
                cudaMemcpy(io_nodes_data, d_nodes, final_nodes * sizeof(GpuColonizationNodeLayout), cudaMemcpyDeviceToHost);
            }

            cudaFree(d_seeds);
            cudaFree(d_frozen);
            cudaFree(d_seed_exists);
            cudaFree(d_nodes);
            cudaFree(d_cnt);
            return true;
        }
        if (d_seeds) cudaFree(d_seeds);
        if (d_frozen) cudaFree(d_frozen);
        if (d_seed_exists) cudaFree(d_seed_exists);
        if (d_nodes) cudaFree(d_nodes);
        if (d_cnt) cudaFree(d_cnt);
    }
#endif
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_snap_seed_to_nodes_kernel(
    const GpuColonizationNodeLayout* __restrict__ nodes,
    int num_nodes,
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

    if (i < num_nodes) {
        GpuColonizationNodeLayout node = nodes[i];
        glm::vec3 seg_pt(0.0f);
        float seg_r = node.radius;

        if (node.parent_index >= 0 && node.parent_index < num_nodes) {
            GpuColonizationNodeLayout pnode = nodes[node.parent_index];
            glm::vec3 ba = node.position - pnode.position;
            float l2 = glm::dot(ba, ba);
            float t = (l2 > 1e-8f) ? dev_clamp(glm::dot(query_pos - pnode.position, ba) / l2, 0.0f, 1.0f) : 0.0f;
            seg_pt = pnode.position + t * ba;
            seg_r = pnode.radius + (node.radius - pnode.radius) * t;
        } else {
            seg_pt = node.position;
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

__global__ void gpu_reduce_snap_nodes_blocks_kernel(
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

bool CudaRasterizer::snap_seed_to_nodes_gpu(
    const void* nodes_data,
    size_t num_nodes,
    glm::vec3 query_pos,
    float target_offset,
    glm::vec3& out_snapped_pos,
    glm::vec3& out_surface_normal,
    float& out_time_ms
) {
    out_time_ms = 0.0f;
    out_snapped_pos = query_pos;
    out_surface_normal = glm::vec3(0.0f, 0.0f, 1.0f);
    if (!nodes_data || num_nodes == 0) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int n = static_cast<int>(num_nodes);
        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        GpuColonizationNodeLayout* d_nodes = nullptr;
        float* d_b_dist = nullptr;
        glm::vec3* d_b_pos = nullptr;
        glm::vec3* d_b_norm = nullptr;
        glm::vec3* d_res_pos = nullptr;
        glm::vec3* d_res_norm = nullptr;

        if (cudaMalloc(&d_nodes, n * sizeof(GpuColonizationNodeLayout)) == cudaSuccess &&
            cudaMalloc(&d_b_dist, blocks * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_b_pos, blocks * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_b_norm, blocks * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_res_pos, sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_res_norm, sizeof(glm::vec3)) == cudaSuccess) {

            cudaMemcpy(d_nodes, nodes_data, n * sizeof(GpuColonizationNodeLayout), cudaMemcpyHostToDevice);

            gpu_snap_seed_to_nodes_kernel<<<blocks, threads>>>(
                d_nodes, n, query_pos, target_offset, d_b_dist, d_b_pos, d_b_norm
            );

            gpu_reduce_snap_nodes_blocks_kernel<<<1, 1>>>(
                d_b_dist, d_b_pos, d_b_norm, blocks, d_res_pos, d_res_norm
            );

            cudaMemcpy(&out_snapped_pos, d_res_pos, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
            cudaMemcpy(&out_surface_normal, d_res_norm, sizeof(glm::vec3), cudaMemcpyDeviceToHost);

            cudaFree(d_nodes);
            cudaFree(d_b_dist);
            cudaFree(d_b_pos);
            cudaFree(d_b_norm);
            cudaFree(d_res_pos);
            cudaFree(d_res_norm);

            auto t1 = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            return true;
        }
        if (d_nodes) cudaFree(d_nodes);
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
__global__ void gpu_mark_classic_nodes_for_seed_kernel(
    const GpuColonizationNodeLayout* __restrict__ nodes,
    int num_nodes,
    int deleted_seed_idx,
    int* __restrict__ out_keep
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    int seed_slot = dev_find_classic_node_seed_slot(nodes, num_nodes, i);
    out_keep[i] = (seed_slot == deleted_seed_idx) ? 0 : 1;
}

__global__ void gpu_block_prefix_scan_nodes_kernel(
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

__global__ void gpu_single_block_scan_nodes_kernel(
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

__global__ void gpu_add_scanned_block_offsets_nodes_kernel(
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

__global__ void gpu_remap_and_compact_classic_nodes_kernel(
    const GpuColonizationNodeLayout* __restrict__ in_nodes,
    int num_nodes,
    const int* __restrict__ old_to_new_map,
    int deleted_seed_idx,
    GpuColonizationNodeLayout* __restrict__ out_nodes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    int new_i = old_to_new_map[i];
    if (new_i < 0) return;

    GpuColonizationNodeLayout n = in_nodes[i];
    if (n.parent_index >= 0 && n.parent_index < num_nodes) {
        n.parent_index = old_to_new_map[n.parent_index];
    } else {
        n.parent_index = -1;
    }

    int seed_slot = n.strand_id & 0xFFFF;
    int gen = n.strand_id >> 16;
    if (seed_slot > deleted_seed_idx) {
        seed_slot -= 1;
        n.strand_id = (gen << 16) | (seed_slot & 0xFFFF);
    }

    out_nodes[new_i] = n;
}
#endif

bool CudaRasterizer::prune_classic_nodes_by_seed_gpu(
    void* io_nodes_data,
    size_t& io_num_nodes,
    int deleted_seed_idx
) {
    if (!io_nodes_data || io_num_nodes == 0 || deleted_seed_idx < 0) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n_nodes = static_cast<int>(io_num_nodes);
        GpuColonizationNodeLayout* d_in_nodes = nullptr;
        GpuColonizationNodeLayout* d_out_nodes = nullptr;
        int* d_keep = nullptr;
        int* d_local_scanned = nullptr;
        int* d_block_sums = nullptr;
        int* d_old_to_new = nullptr;

        const int threads = 256;
        const int blocks = (n_nodes + threads - 1) / threads;

        bool ok = (cudaMalloc(&d_in_nodes, n_nodes * sizeof(GpuColonizationNodeLayout)) == cudaSuccess) &&
                  (cudaMalloc(&d_out_nodes, n_nodes * sizeof(GpuColonizationNodeLayout)) == cudaSuccess) &&
                  (cudaMalloc(&d_keep, n_nodes * sizeof(int)) == cudaSuccess) &&
                  (cudaMalloc(&d_local_scanned, n_nodes * sizeof(int)) == cudaSuccess) &&
                  (cudaMalloc(&d_block_sums, blocks * sizeof(int)) == cudaSuccess) &&
                  (cudaMalloc(&d_old_to_new, n_nodes * sizeof(int)) == cudaSuccess);

        if (ok) {
            cudaMemcpy(d_in_nodes, io_nodes_data, n_nodes * sizeof(GpuColonizationNodeLayout), cudaMemcpyHostToDevice);

            gpu_mark_classic_nodes_for_seed_kernel<<<blocks, threads>>>(
                d_in_nodes, n_nodes, deleted_seed_idx, d_keep
            );

            size_t shared_bytes = threads * sizeof(int);
            gpu_block_prefix_scan_nodes_kernel<<<blocks, threads, shared_bytes>>>(
                d_keep, n_nodes, d_local_scanned, d_block_sums
            );
            if (blocks > 1) {
                gpu_single_block_scan_nodes_kernel<<<1, 1024, 1024 * sizeof(int)>>>(
                    d_block_sums, blocks
                );
            }
            gpu_add_scanned_block_offsets_nodes_kernel<<<blocks, threads>>>(
                d_local_scanned, d_block_sums, d_keep, n_nodes, d_old_to_new
            );

            gpu_remap_and_compact_classic_nodes_kernel<<<blocks, threads>>>(
                d_in_nodes, n_nodes, d_old_to_new, deleted_seed_idx, d_out_nodes
            );

            int h_surviving_nodes = 0;
            if (blocks > 0) {
                cudaMemcpy(&h_surviving_nodes, &d_block_sums[blocks - 1], sizeof(int), cudaMemcpyDeviceToHost);
            }
            h_surviving_nodes = std::clamp(h_surviving_nodes, 0, n_nodes);

            if (h_surviving_nodes > 0) {
                cudaMemcpy(io_nodes_data, d_out_nodes, h_surviving_nodes * sizeof(GpuColonizationNodeLayout), cudaMemcpyDeviceToHost);
            }
            io_num_nodes = static_cast<size_t>(h_surviving_nodes);

            cudaFree(d_in_nodes);
            cudaFree(d_out_nodes);
            cudaFree(d_keep);
            cudaFree(d_local_scanned);
            cudaFree(d_block_sums);
            cudaFree(d_old_to_new);
            return true;
        }

        if (d_in_nodes) cudaFree(d_in_nodes);
        if (d_out_nodes) cudaFree(d_out_nodes);
        if (d_keep) cudaFree(d_keep);
        if (d_local_scanned) cudaFree(d_local_scanned);
        if (d_block_sums) cudaFree(d_block_sums);
        if (d_old_to_new) cudaFree(d_old_to_new);
    }
#endif
    return false;
}

} // namespace rhizome
