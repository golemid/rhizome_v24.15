#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/cuda_common.cuh"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <vector>

namespace rhizome {

#ifdef __CUDACC__
// --- Edge structures & kernels for Manifold Verification ---
struct GpuEdgeEntry {
    unsigned long long key; // 0ULL means empty slot
    uint32_t count;
    uint32_t dir_u;
    uint32_t dir_v;
    uint32_t conflicts;
};

__global__ void gpu_insert_edges_kernel(
    const uint32_t* __restrict__ indices,
    size_t num_tris,
    GpuEdgeEntry* __restrict__ table,
    size_t table_size
) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    uint32_t idx[3] = {
        indices[t * 3 + 0],
        indices[t * 3 + 1],
        indices[t * 3 + 2]
    };

    for (int e = 0; e < 3; ++e) {
        uint32_t u = idx[e];
        uint32_t v = idx[(e + 1) % 3];
        if (u == v) continue;

        uint32_t mn = (u < v) ? u : v;
        uint32_t mx = (u < v) ? v : u;
        unsigned long long key = ((unsigned long long)mn << 32) | (unsigned long long)mx;
        if (key == 0ULL) key = 1ULL;

        unsigned int slot = hash_u64(key) % table_size;
        for (size_t probe = 0; probe < table_size; ++probe) {
            unsigned int idx_probe = (slot + probe) % table_size;
            unsigned long long prev = atomicCAS(&table[idx_probe].key, 0ULL, key);
            if (prev == 0ULL || prev == key) {
                atomicAdd(&table[idx_probe].count, 1);
                if (prev == key) {
                    if (table[idx_probe].dir_u == u && table[idx_probe].dir_v == v) {
                        atomicAdd(&table[idx_probe].conflicts, 1);
                    }
                } else {
                    table[idx_probe].dir_u = u;
                    table[idx_probe].dir_v = v;
                }
                break;
            }
        }
    }
}

__global__ void gpu_count_manifold_stats_kernel(
    const GpuEdgeEntry* __restrict__ table,
    size_t table_size,
    unsigned int* __restrict__ out_boundary,
    unsigned int* __restrict__ out_non_manifold,
    unsigned int* __restrict__ out_conflicts
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= table_size) return;

    if (table[i].key != 0ULL) {
        uint32_t c = table[i].count;
        if (c == 1) {
            atomicAdd(out_boundary, 1);
        } else if (c > 2) {
            atomicAdd(out_non_manifold, 1);
        }
        if (table[i].conflicts > 0) {
            atomicAdd(out_conflicts, table[i].conflicts);
        }
    }
}

// --- Edge structures & kernels for Manifold Repair ---
struct GpuRepairEdgeSlot {
    unsigned long long key;
    uint32_t fwd_count;
    uint32_t rev_count;
    uint32_t fwd_tri[4];
    uint8_t  fwd_e[4];
    uint32_t rev_tri[4];
    uint8_t  rev_e[4];
};

__global__ void gpu_insert_repair_edges_kernel(
    const uint32_t* __restrict__ indices,
    int num_tris,
    GpuRepairEdgeSlot* __restrict__ table,
    size_t table_size
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    uint32_t idx[3] = {
        indices[t * 3 + 0],
        indices[t * 3 + 1],
        indices[t * 3 + 2]
    };

    for (uint8_t e = 0; e < 3; ++e) {
        uint32_t u = idx[e];
        uint32_t v = idx[(e + 1) % 3];
        if (u == v) continue;

        uint32_t mn = (u < v) ? u : v;
        uint32_t mx = (u < v) ? v : u;
        unsigned long long key = ((unsigned long long)mn << 32) | (unsigned long long)mx;
        if (key == 0ULL) key = 1ULL;

        bool is_fwd = (u < v);
        unsigned int slot = hash_u64(key) % table_size;
        for (size_t probe = 0; probe < table_size; ++probe) {
            size_t idx_probe = (slot + probe) % table_size;
            unsigned long long prev = atomicCAS(&table[idx_probe].key, 0ULL, key);
            if (prev == 0ULL || prev == key) {
                if (is_fwd) {
                    uint32_t pos = atomicAdd(&table[idx_probe].fwd_count, 1u);
                    if (pos < 4u) {
                        table[idx_probe].fwd_tri[pos] = static_cast<uint32_t>(t);
                        table[idx_probe].fwd_e[pos] = e;
                    }
                } else {
                    uint32_t pos = atomicAdd(&table[idx_probe].rev_count, 1u);
                    if (pos < 4u) {
                        table[idx_probe].rev_tri[pos] = static_cast<uint32_t>(t);
                        table[idx_probe].rev_e[pos] = e;
                    }
                }
                break;
            }
        }
    }
}

__global__ void gpu_resolve_non_manifold_edges_kernel(
    GpuRepairEdgeSlot* __restrict__ table,
    size_t table_size,
    float* __restrict__ vertices,
    int* __restrict__ d_num_verts,
    int max_verts,
    uint32_t* __restrict__ indices,
    uint8_t* __restrict__ out_remove_mask,
    int* __restrict__ d_any_non_manifold,
    int* __restrict__ d_any_removed
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= table_size) return;

    unsigned long long key = table[i].key;
    if (key == 0ULL) return;

    uint32_t fc = min(table[i].fwd_count, 4u);
    uint32_t rc = min(table[i].rev_count, 4u);
    if (fc + rc <= 2u) return;

    *d_any_non_manifold = 1;

    uint32_t f_tri[4];
    uint8_t  f_e[4];
    uint32_t r_tri[4];
    uint8_t  r_e[4];

    for (uint32_t k = 0; k < fc; ++k) {
        f_tri[k] = table[i].fwd_tri[k];
        f_e[k]   = table[i].fwd_e[k];
    }
    for (uint32_t k = 0; k < rc; ++k) {
        r_tri[k] = table[i].rev_tri[k];
        r_e[k]   = table[i].rev_e[k];
    }

    for (uint32_t a = 0; a < fc; ++a) {
        for (uint32_t b = a + 1; b < fc; ++b) {
            if (f_tri[b] < f_tri[a]) {
                uint32_t tt = f_tri[a]; f_tri[a] = f_tri[b]; f_tri[b] = tt;
                uint8_t  te = f_e[a];   f_e[a]   = f_e[b];   f_e[b]   = te;
            }
        }
    }
    for (uint32_t a = 0; a < rc; ++a) {
        for (uint32_t b = a + 1; b < rc; ++b) {
            if (r_tri[b] < r_tri[a]) {
                uint32_t tt = r_tri[a]; r_tri[a] = r_tri[b]; r_tri[b] = tt;
                uint8_t  te = r_e[a];   r_e[a]   = r_e[b];   r_e[b]   = te;
            }
        }
    }

    uint32_t u_base = static_cast<uint32_t>(key >> 32);
    uint32_t v_base = static_cast<uint32_t>(key & 0xFFFFFFFFULL);
    uint32_t pairs = min(fc, rc);

    for (uint32_t p = 1; p < pairs; ++p) {
        uint32_t ft = f_tri[p];
        uint8_t  fe = f_e[p];
        uint32_t rt = r_tri[p];
        uint8_t  re = r_e[p];

        uint32_t w_f = indices[ft * 3 + ((fe + 2) % 3)];
        uint32_t w_r = indices[rt * 3 + ((re + 2) % 3)];

        if (w_f != w_r && w_f != u_base && w_f != v_base && w_r != u_base && w_r != v_base) {
            indices[ft * 3 + 0] = u_base;
            indices[ft * 3 + 1] = w_r;
            indices[ft * 3 + 2] = w_f;

            indices[rt * 3 + 0] = v_base;
            indices[rt * 3 + 1] = w_f;
            indices[rt * 3 + 2] = w_r;
        } else {
            out_remove_mask[ft] = 1;
            out_remove_mask[rt] = 1;
            *d_any_removed = 1;
        }
    }

    for (uint32_t k = pairs; k < fc; ++k) {
        out_remove_mask[f_tri[k]] = 1;
        *d_any_removed = 1;
    }
    for (uint32_t k = pairs; k < rc; ++k) {
        out_remove_mask[r_tri[k]] = 1;
        *d_any_removed = 1;
    }
}

// gpu_compact_non_degenerate_tris_kernel: DUPLICATE BODY REMOVED in Phase 3
// (finding F13). This TU-local `static` copy was byte-identical to the
// externally-linked definition in engine/mesh/cuda_mesh_ops.cu (verified
// programmatically against the pristine v23.28 tree). The shared
// declaration now lives in engine/sdf/cuda_common.cuh; the two launch
// sites below compile against the single mesh_ops definition. The
// surviving kernel body is untouched — one kernel instead of two
// identical device functions with the same name.

// --- Signed Volume & Mesh Winding Orientation Kernels ---
__global__ void gpu_signed_volume_kernel(
    const float* __restrict__ vertices,
    const uint32_t* __restrict__ indices,
    size_t num_tris,
    float* __restrict__ out_vol6_sum
) {
    __shared__ float s_vol[256];
    int tid = threadIdx.x;
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    while (t < num_tris) {
        uint32_t i0 = indices[t * 3 + 0];
        uint32_t i1 = indices[t * 3 + 1];
        uint32_t i2 = indices[t * 3 + 2];

        glm::vec3 v0(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        glm::vec3 v1(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        glm::vec3 v2(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

        local_sum += glm::dot(v0, glm::cross(v1, v2));
        t += (size_t)gridDim.x * blockDim.x;
    }

    s_vol[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_vol[tid] += s_vol[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_vol6_sum, s_vol[0]);
    }
}

struct GpuOrientEdgeSlot {
    unsigned long long key;
    int tri_a;
    int tri_b;
    uint32_t u_a;
    uint32_t v_a;
    uint32_t u_b;
    uint32_t v_b;
    uint32_t count;
};

__global__ void gpu_build_orient_edges_kernel(
    const uint32_t* __restrict__ indices,
    int num_tris,
    GpuOrientEdgeSlot* __restrict__ table,
    size_t table_size
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    uint32_t idx[3] = {
        indices[t * 3 + 0],
        indices[t * 3 + 1],
        indices[t * 3 + 2]
    };

    for (int e = 0; e < 3; ++e) {
        uint32_t u = idx[e];
        uint32_t v = idx[(e + 1) % 3];
        if (u == v) continue;

        uint32_t mn = (u < v) ? u : v;
        uint32_t mx = (u < v) ? v : u;
        unsigned long long key = ((unsigned long long)mn << 32) | (unsigned long long)mx;
        if (key == 0ULL) key = 1ULL;

        unsigned int slot = hash_u64(key) % table_size;
        for (size_t probe = 0; probe < table_size; ++probe) {
            size_t idx_probe = (slot + probe) % table_size;
            unsigned long long prev = atomicCAS(&table[idx_probe].key, 0ULL, key);
            if (prev == 0ULL || prev == key) {
                uint32_t old_c = atomicAdd(&table[idx_probe].count, 1);
                if (old_c == 0) {
                    table[idx_probe].tri_a = t;
                    table[idx_probe].u_a = u;
                    table[idx_probe].v_a = v;
                } else if (old_c == 1) {
                    table[idx_probe].tri_b = t;
                    table[idx_probe].u_b = u;
                    table[idx_probe].v_b = v;
                }
                break;
            }
        }
    }
}

__global__ void gpu_BuildAdjFromEdgesKernel(
    const GpuOrientEdgeSlot* __restrict__ table,
    size_t table_size,
    int* __restrict__ adj_tri,
    int* __restrict__ adj_same_dir,
    int* __restrict__ adj_count
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= table_size) return;

    if (table[i].key != 0ULL && table[i].count == 2) {
        int ta = table[i].tri_a;
        int tb = table[i].tri_b;
        if (ta >= 0 && tb >= 0 && ta != tb) {
            int same = (table[i].u_a == table[i].u_b && table[i].v_a == table[i].v_b) ? 1 : 0;
            int sa = atomicAdd(&adj_count[ta], 1);
            if (sa < 3) {
                adj_tri[ta * 3 + sa] = tb;
                adj_same_dir[ta * 3 + sa] = same;
            }
            int sb = atomicAdd(&adj_count[tb], 1);
            if (sb < 3) {
                adj_tri[tb * 3 + sb] = ta;
                adj_same_dir[tb * 3 + sb] = same;
            }
        }
    }
}

__global__ void gpu_init_orient_labels_kernel(
    int num_tris,
    int* __restrict__ comp_id,
    int* __restrict__ flip_flag
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;
    comp_id[t] = t;
    flip_flag[t] = 0;
}

__global__ void gpu_propagate_orient_step_kernel(
    int num_tris,
    const int* __restrict__ adj_tri,
    const int* __restrict__ adj_same_dir,
    const int* __restrict__ adj_count,
    int* __restrict__ comp_id,
    int* __restrict__ flip_flag,
    int* __restrict__ d_changed,
    int* __restrict__ d_had_inconsistent,
    int* __restrict__ d_non_orientable
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    int my_comp = comp_id[t];
    int my_flip = flip_flag[t];
    int cnt = min(adj_count[t], 3);

    for (int k = 0; k < cnt; ++k) {
        int nb = adj_tri[t * 3 + k];
        if (nb < 0) continue;
        int same = adj_same_dir[t * 3 + k];
        if (same) {
            *d_had_inconsistent = 1;
        }
        int nb_comp = comp_id[nb];
        int nb_flip = flip_flag[nb];
        int req_flip = nb_flip ^ same;

        if (nb_comp < my_comp) {
            my_comp = nb_comp;
            my_flip = req_flip;
            comp_id[t] = my_comp;
            flip_flag[t] = my_flip;
            *d_changed = 1;
        } else if (nb_comp == my_comp && my_flip != req_flip) {
            *d_non_orientable = 1;
        }
    }
}

__global__ void gpu_apply_flip_flags_kernel(
    uint32_t* __restrict__ indices,
    int num_tris,
    const int* __restrict__ flip_flag
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;
    if (flip_flag[t]) {
        uint32_t tmp = indices[t * 3 + 1];
        indices[t * 3 + 1] = indices[t * 3 + 2];
        indices[t * 3 + 2] = tmp;
    }
}

__global__ void gpu_invert_all_tris_kernel(
    uint32_t* __restrict__ indices,
    int num_tris
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;
    uint32_t tmp = indices[t * 3 + 1];
    indices[t * 3 + 1] = indices[t * 3 + 2];
    indices[t * 3 + 2] = tmp;
}
#endif // __CUDACC__

bool CudaRasterizer::verify_manifold_gpu(
    const TriangleMesh& mesh,
    size_t& out_boundary_edges,
    size_t& out_non_manifold_edges,
    size_t& out_conflicts,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_boundary_edges = 0;
    out_non_manifold_edges = 0;
    out_conflicts = 0;
    size_t num_tris = mesh.triangle_count();
    if (num_tris == 0) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        uint32_t* d_indices = nullptr;
        GpuEdgeEntry* d_table = nullptr;
        unsigned int* d_boundary = nullptr;
        unsigned int* d_non_manifold = nullptr;
        unsigned int* d_conflicts = nullptr;

        size_t table_size = num_tris * 6 + 1024;
        size_t idx_bytes = num_tris * 3 * sizeof(uint32_t);
        size_t table_bytes = table_size * sizeof(GpuEdgeEntry);

        if (cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_table, table_bytes) == cudaSuccess &&
            cudaMalloc(&d_boundary, sizeof(unsigned int)) == cudaSuccess &&
            cudaMalloc(&d_non_manifold, sizeof(unsigned int)) == cudaSuccess &&
            cudaMalloc(&d_conflicts, sizeof(unsigned int)) == cudaSuccess) {

            cudaMemcpy(d_indices, mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);
            cudaMemset(d_table, 0, table_bytes);
            cudaMemset(d_boundary, 0, sizeof(unsigned int));
            cudaMemset(d_non_manifold, 0, sizeof(unsigned int));
            cudaMemset(d_conflicts, 0, sizeof(unsigned int));

            int threads = 256;
            int blocks = static_cast<int>((num_tris + threads - 1) / threads);
            gpu_insert_edges_kernel<<<blocks, threads>>>(d_indices, num_tris, d_table, table_size);

            int t_blocks = static_cast<int>((table_size + threads - 1) / threads);
            gpu_count_manifold_stats_kernel<<<t_blocks, threads>>>(d_table, table_size, d_boundary, d_non_manifold, d_conflicts);

            unsigned int h_b = 0, h_nm = 0, h_c = 0;
            cudaMemcpy(&h_b, d_boundary, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_nm, d_non_manifold, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_c, d_conflicts, sizeof(unsigned int), cudaMemcpyDeviceToHost);

            out_boundary_edges = h_b;
            out_non_manifold_edges = h_nm;
            out_conflicts = h_c;

            cudaFree(d_indices);
            cudaFree(d_table);
            cudaFree(d_boundary);
            cudaFree(d_non_manifold);
            cudaFree(d_conflicts);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_indices) cudaFree(d_indices);
        if (d_table) cudaFree(d_table);
        if (d_boundary) cudaFree(d_boundary);
        if (d_non_manifold) cudaFree(d_non_manifold);
        if (d_conflicts) cudaFree(d_conflicts);
    }
#endif
    return false;
}

bool CudaRasterizer::repair_mesh_manifold_gpu(
    TriangleMesh& inout_mesh,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    int num_tris = static_cast<int>(inout_mesh.triangle_count());
    int num_verts = static_cast<int>(inout_mesh.vertex_count());
    if (num_tris <= 0 || num_verts <= 0) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        int max_verts = num_verts + std::max(num_tris, 1024);
        size_t table_size = static_cast<size_t>(num_tris) * 6 + 1024;

        float* d_verts = nullptr;
        uint32_t* d_indices_a = nullptr;
        uint32_t* d_indices_b = nullptr;
        uint8_t* d_remove_mask = nullptr;
        GpuRepairEdgeSlot* d_table = nullptr;
        int* d_counts = nullptr; // [0]=tri_count, [1]=vert_count, [2]=any_non_manifold, [3]=any_removed

        if (cudaMalloc(&d_verts, static_cast<size_t>(max_verts) * 3 * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_indices_a, static_cast<size_t>(num_tris) * 3 * sizeof(uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_indices_b, static_cast<size_t>(num_tris) * 3 * sizeof(uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_remove_mask, static_cast<size_t>(num_tris) * sizeof(uint8_t)) == cudaSuccess &&
            cudaMalloc(&d_table, table_size * sizeof(GpuRepairEdgeSlot)) == cudaSuccess &&
            cudaMalloc(&d_counts, 4 * sizeof(int)) == cudaSuccess) {

            cudaMemcpy(d_verts, inout_mesh.vertices.data(), static_cast<size_t>(num_verts) * 3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices_a, inout_mesh.indices.data(), static_cast<size_t>(num_tris) * 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);

            int threads = 256;
            int tri_blocks = (num_tris + threads - 1) / threads;

            cudaMemset(d_counts, 0, sizeof(int));
            gpu_compact_non_degenerate_tris_kernel<<<tri_blocks, threads>>>(
                d_indices_a, num_tris, nullptr, d_indices_b, &d_counts[0]
            );
            cudaMemcpy(&num_tris, &d_counts[0], sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_indices_a, d_indices_b);

            cudaMemcpy(&d_counts[1], &num_verts, sizeof(int), cudaMemcpyHostToDevice);

            int tab_blocks = static_cast<int>((table_size + threads - 1) / threads);
            for (int iter = 0; iter < 4 && num_tris > 0; ++iter) {
                tri_blocks = (num_tris + threads - 1) / threads;
                cudaMemset(d_table, 0, table_size * sizeof(GpuRepairEdgeSlot));
                cudaMemset(d_remove_mask, 0, static_cast<size_t>(num_tris) * sizeof(uint8_t));
                cudaMemset(&d_counts[2], 0, 2 * sizeof(int));

                gpu_insert_repair_edges_kernel<<<tri_blocks, threads>>>(
                    d_indices_a, num_tris, d_table, table_size
                );

                gpu_resolve_non_manifold_edges_kernel<<<tab_blocks, threads>>>(
                    d_table, table_size,
                    d_verts, &d_counts[1], max_verts,
                    d_indices_a, d_remove_mask,
                    &d_counts[2], &d_counts[3]
                );

                int h_flags[2] = {0, 0};
                cudaMemcpy(h_flags, &d_counts[2], 2 * sizeof(int), cudaMemcpyDeviceToHost);
                int any_non_manifold = h_flags[0];
                int any_removed = h_flags[1];

                if (any_removed) {
                    cudaMemset(&d_counts[0], 0, sizeof(int));
                    gpu_compact_non_degenerate_tris_kernel<<<tri_blocks, threads>>>(
                        d_indices_a, num_tris, d_remove_mask, d_indices_b, &d_counts[0]
                    );
                    cudaMemcpy(&num_tris, &d_counts[0], sizeof(int), cudaMemcpyDeviceToHost);
                    std::swap(d_indices_a, d_indices_b);
                }

                if (!any_non_manifold) break;
            }

            cudaMemcpy(&num_verts, &d_counts[1], sizeof(int), cudaMemcpyDeviceToHost);
            num_verts = std::min(num_verts, max_verts);

            if (num_tris > 0 && num_verts > 0) {
                inout_mesh.vertices.resize(static_cast<size_t>(num_verts) * 3);
                inout_mesh.indices.resize(static_cast<size_t>(num_tris) * 3);
                cudaMemcpy(inout_mesh.vertices.data(), d_verts, inout_mesh.vertices.size() * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(inout_mesh.indices.data(), d_indices_a, inout_mesh.indices.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            } else {
                inout_mesh.indices.clear();
            }

            cudaFree(d_verts);
            cudaFree(d_indices_a);
            cudaFree(d_indices_b);
            cudaFree(d_remove_mask);
            cudaFree(d_table);
            cudaFree(d_counts);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_verts) cudaFree(d_verts);
        if (d_indices_a) cudaFree(d_indices_a);
        if (d_indices_b) cudaFree(d_indices_b);
        if (d_remove_mask) cudaFree(d_remove_mask);
        if (d_table) cudaFree(d_table);
        if (d_counts) cudaFree(d_counts);
    }
#endif

    auto end = std::chrono::high_resolution_clock::now();
    out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    return true;
}

bool CudaRasterizer::compute_signed_volume_gpu(
    const TriangleMesh& mesh,
    float& out_signed_volume_mm3,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_signed_volume_mm3 = 0.0f;
    size_t num_tris = mesh.triangle_count();
    if (num_tris == 0 || mesh.vertices.empty()) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        float* d_vol6 = nullptr;

        size_t vert_bytes = mesh.vertices.size() * sizeof(float);
        size_t idx_bytes = mesh.indices.size() * sizeof(uint32_t);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_vol6, sizeof(float)) == cudaSuccess) {

            cudaMemcpy(d_verts, mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);
            cudaMemset(d_vol6, 0, sizeof(float));

            int threads = 256;
            int blocks = std::min(1024, static_cast<int>((num_tris + threads - 1) / threads));
            gpu_signed_volume_kernel<<<blocks, threads>>>(d_verts, d_indices, num_tris, d_vol6);

            float h_vol6 = 0.0f;
            cudaMemcpy(&h_vol6, d_vol6, sizeof(float), cudaMemcpyDeviceToHost);
            out_signed_volume_mm3 = h_vol6 / 6.0f;

            cudaFree(d_verts);
            cudaFree(d_indices);
            cudaFree(d_vol6);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
        if (d_vol6) cudaFree(d_vol6);
    }
#endif
    return false;
}

bool CudaRasterizer::orient_mesh_gpu(
    TriangleMesh& inout_mesh,
    bool& out_is_orientable,
    bool& out_had_inconsistent_winding,
    bool& out_was_inverted,
    float& out_signed_volume,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_is_orientable = true;
    out_had_inconsistent_winding = false;
    out_was_inverted = false;
    out_signed_volume = 0.0f;

    int num_tris = static_cast<int>(inout_mesh.triangle_count());
    if (num_tris <= 0 || inout_mesh.vertices.empty()) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        size_t table_size = static_cast<size_t>(num_tris) * 6 + 1024;
        GpuOrientEdgeSlot* d_table = nullptr;
        int* d_adj_tri = nullptr;
        int* d_adj_same = nullptr;
        int* d_adj_cnt = nullptr;
        int* d_comp = nullptr;
        int* d_flip = nullptr;
        int* d_flags = nullptr;
        float* d_vol6 = nullptr;

        size_t vert_bytes = inout_mesh.vertices.size() * sizeof(float);
        size_t idx_bytes = inout_mesh.indices.size() * sizeof(uint32_t);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_table, table_size * sizeof(GpuOrientEdgeSlot)) == cudaSuccess &&
            cudaMalloc(&d_adj_tri, static_cast<size_t>(num_tris) * 3 * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_adj_same, static_cast<size_t>(num_tris) * 3 * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_adj_cnt, static_cast<size_t>(num_tris) * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_comp, static_cast<size_t>(num_tris) * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_flip, static_cast<size_t>(num_tris) * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_flags, 3 * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_vol6, sizeof(float)) == cudaSuccess) {

            cudaMemcpy(d_verts, inout_mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, inout_mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);
            cudaMemset(d_table, 0, table_size * sizeof(GpuOrientEdgeSlot));
            cudaMemset(d_adj_tri, 0xFF, static_cast<size_t>(num_tris) * 3 * sizeof(int));
            cudaMemset(d_adj_same, 0, static_cast<size_t>(num_tris) * 3 * sizeof(int));
            cudaMemset(d_adj_cnt, 0, static_cast<size_t>(num_tris) * sizeof(int));
            cudaMemset(d_flags, 0, 3 * sizeof(int));

            int threads = 256;
            int tri_blocks = (num_tris + threads - 1) / threads;
            int tab_blocks = static_cast<int>((table_size + threads - 1) / threads);

            gpu_build_orient_edges_kernel<<<tri_blocks, threads>>>(d_indices, num_tris, d_table, table_size);
            gpu_BuildAdjFromEdgesKernel<<<tab_blocks, threads>>>(d_table, table_size, d_adj_tri, d_adj_same, d_adj_cnt);
            gpu_init_orient_labels_kernel<<<tri_blocks, threads>>>(num_tris, d_comp, d_flip);

            int max_iters = std::min(num_tris, 256);
            for (int it = 0; it < max_iters; ++it) {
                cudaMemset(d_flags, 0, sizeof(int));
                gpu_propagate_orient_step_kernel<<<tri_blocks, threads>>>(
                    num_tris, d_adj_tri, d_adj_same, d_adj_cnt,
                    d_comp, d_flip, &d_flags[0], &d_flags[1], &d_flags[2]
                );
                int h_changed = 0;
                cudaMemcpy(&h_changed, &d_flags[0], sizeof(int), cudaMemcpyDeviceToHost);
                if (h_changed == 0) break;
            }

            int h_flags[3] = {0, 0, 0};
            cudaMemcpy(h_flags, d_flags, 3 * sizeof(int), cudaMemcpyDeviceToHost);
            out_had_inconsistent_winding = (h_flags[1] != 0);
            out_is_orientable = (h_flags[2] == 0);

            if (out_had_inconsistent_winding) {
                gpu_apply_flip_flags_kernel<<<tri_blocks, threads>>>(d_indices, num_tris, d_flip);
            }

            cudaMemset(d_vol6, 0, sizeof(float));
            int red_blocks = std::min(1024, tri_blocks);
            gpu_signed_volume_kernel<<<red_blocks, threads>>>(d_verts, d_indices, static_cast<size_t>(num_tris), d_vol6);

            float h_vol6 = 0.0f;
            cudaMemcpy(&h_vol6, d_vol6, sizeof(float), cudaMemcpyDeviceToHost);
            float vol = h_vol6 / 6.0f;

            if (vol < -1e-5f) {
                gpu_invert_all_tris_kernel<<<tri_blocks, threads>>>(d_indices, num_tris);
                vol = -vol;
                out_was_inverted = true;
            }
            out_signed_volume = vol;

            if (out_had_inconsistent_winding || out_was_inverted) {
                cudaMemcpy(inout_mesh.indices.data(), d_indices, idx_bytes, cudaMemcpyDeviceToHost);
            }

            cudaFree(d_verts);
            cudaFree(d_indices);
            cudaFree(d_table);
            cudaFree(d_adj_tri);
            cudaFree(d_adj_same);
            cudaFree(d_adj_cnt);
            cudaFree(d_comp);
            cudaFree(d_flip);
            cudaFree(d_flags);
            cudaFree(d_vol6);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
        if (d_table) cudaFree(d_table);
        if (d_adj_tri) cudaFree(d_adj_tri);
        if (d_adj_same) cudaFree(d_adj_same);
        if (d_adj_cnt) cudaFree(d_adj_cnt);
        if (d_comp) cudaFree(d_comp);
        if (d_flip) cudaFree(d_flip);
        if (d_flags) cudaFree(d_flags);
        if (d_vol6) cudaFree(d_vol6);
    }
#endif
    return false;
}

} // namespace rhizome
