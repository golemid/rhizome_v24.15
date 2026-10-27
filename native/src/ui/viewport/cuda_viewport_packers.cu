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

__device__ inline glm::mat4 gpu_mat4_mul(const glm::mat4& a, const glm::mat4& b) {
    glm::mat4 res;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) {
                s += a[k][r] * b[c][k];
            }
            res[c][r] = s;
        }
    }
    return res;
}

__device__ inline glm::mat4 gpu_mat4_inverse(const glm::mat4& m) {
    float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3];
    float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3];
    float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3];
    float a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3];

    float b00 = a00 * a11 - a01 * a10;
    float b01 = a00 * a12 - a02 * a10;
    float b02 = a00 * a13 - a03 * a10;
    float b03 = a01 * a12 - a02 * a11;
    float b04 = a01 * a13 - a03 * a11;
    float b05 = a02 * a13 - a03 * a12;
    float b06 = a20 * a31 - a21 * a30;
    float b07 = a20 * a32 - a22 * a30;
    float b08 = a20 * a33 - a23 * a30;
    float b09 = a21 * a32 - a22 * a31;
    float b10 = a21 * a33 - a23 * a31;
    float b11 = a22 * a33 - a23 * a32;

    float det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (fabsf(det) < 1e-12f) return glm::mat4(1.0f);
    float inv_det = 1.0f / det;

    glm::mat4 inv;
    inv[0][0] = (a11 * b11 - a12 * b10 + a13 * b09) * inv_det;
    inv[0][1] = (-a01 * b11 + a02 * b10 - a03 * b09) * inv_det;
    inv[0][2] = (a31 * b05 - a32 * b04 + a33 * b03) * inv_det;
    inv[0][3] = (-a21 * b05 + a22 * b04 - a23 * b03) * inv_det;

    inv[1][0] = (-a10 * b11 + a12 * b08 - a13 * b07) * inv_det;
    inv[1][1] = (a00 * b11 - a02 * b08 + a03 * b07) * inv_det;
    inv[1][2] = (-a30 * b05 + a32 * b02 - a33 * b01) * inv_det;
    inv[1][3] = (a20 * b05 - a22 * b02 + a23 * b01) * inv_det;

    inv[2][0] = (a10 * b10 - a11 * b08 + a13 * b06) * inv_det;
    inv[2][1] = (-a00 * b10 + a01 * b08 - a03 * b06) * inv_det;
    inv[2][2] = (a30 * b04 - a31 * b02 + a33 * b00) * inv_det;
    inv[2][3] = (-a20 * b04 + a21 * b02 - a23 * b00) * inv_det;

    inv[3][0] = (-a10 * b09 + a11 * b07 - a12 * b06) * inv_det;
    inv[3][1] = (a00 * b09 - a01 * b07 + a02 * b06) * inv_det;
    inv[3][2] = (-a30 * b03 + a31 * b01 - a32 * b00) * inv_det;
    inv[3][3] = (a20 * b03 - a21 * b01 + a22 * b00) * inv_det;

    return inv;
}

__global__ void gpu_compute_camera_matrices_kernel(
    float fov_rad,
    float aspect,
    float near_z,
    float far_z,
    glm::vec3 eye,
    glm::vec3 target,
    glm::vec3 world_up,
    glm::mat4* __restrict__ out_view,
    glm::mat4* __restrict__ out_proj,
    glm::mat4* __restrict__ out_view_proj,
    glm::mat4* __restrict__ out_inv_view_proj
) {
    __shared__ float s_V[4][4];
    __shared__ float s_P[4][4];
    __shared__ float s_VP[4][4];

    int tid = threadIdx.x;
    if (tid >= 16) return;

    int c = tid / 4;
    int r = tid % 4;

    glm::vec3 f_dir = target - eye;
    float f_len = glm::length(f_dir);
    glm::vec3 f = (f_len > 1e-6f) ? (f_dir / f_len) : glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 right_v = glm::cross(f, world_up);
    float r_len = glm::length(right_v);
    right_v = (r_len > 1e-6f) ? (right_v / r_len) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 u = glm::cross(right_v, f);

    // Parallel construction of View matrix entry (c, r)
    float v_val = 0.0f;
    if (r == 0) {
        v_val = (c == 0) ? right_v.x : (c == 1) ? right_v.y : (c == 2) ? right_v.z : -glm::dot(right_v, eye);
    } else if (r == 1) {
        v_val = (c == 0) ? u.x : (c == 1) ? u.y : (c == 2) ? u.z : -glm::dot(u, eye);
    } else if (r == 2) {
        v_val = (c == 0) ? -f.x : (c == 1) ? -f.y : (c == 2) ? -f.z : glm::dot(f, eye);
    } else {
        v_val = (c == 3) ? 1.0f : 0.0f;
    }
    s_V[c][r] = v_val;

    // Parallel construction of Projection matrix entry (c, r)
    float tan_half = tanf(fov_rad * 0.5f);
    float p_val = 0.0f;
    if (c == 0 && r == 0) p_val = 1.0f / (aspect * tan_half);
    else if (c == 1 && r == 1) p_val = 1.0f / tan_half;
    else if (c == 2 && r == 2) p_val = -(far_z + near_z) / (far_z - near_z);
    else if (c == 2 && r == 3) p_val = -1.0f;
    else if (c == 3 && r == 2) p_val = -(2.0f * far_z * near_z) / (far_z - near_z);
    s_P[c][r] = p_val;
    __syncthreads();

    // Parallel 16-thread matrix multiplication VP[c][r] = sum_k P[k][r] * V[c][k]
    float vp_val = s_P[0][r] * s_V[c][0] +
                   s_P[1][r] * s_V[c][1] +
                   s_P[2][r] * s_V[c][2] +
                   s_P[3][r] * s_V[c][3];
    s_VP[c][r] = vp_val;

    reinterpret_cast<float*>(out_view)[tid] = v_val;
    reinterpret_cast<float*>(out_proj)[tid] = p_val;
    reinterpret_cast<float*>(out_view_proj)[tid] = vp_val;
    __syncthreads();

    // Parallel cofactor & inverse evaluation per (c, r) thread
    int r0 = (c == 0) ? 1 : 0;
    int r1 = (c <= 1) ? 2 : 1;
    int r2 = (c <= 2) ? 3 : 2;

    int c0 = (r == 0) ? 1 : 0;
    int c1 = (r <= 1) ? 2 : 1;
    int c2 = (r <= 2) ? 3 : 2;

    // Cofactor of transpose entry (r, c) for adjugate[c][r]
    float m00 = s_VP[c0][r0], m01 = s_VP[c1][r0], m02 = s_VP[c2][r0];
    float m10 = s_VP[c0][r1], m11 = s_VP[c1][r1], m12 = s_VP[c2][r1];
    float m20 = s_VP[c0][r2], m21 = s_VP[c1][r2], m22 = s_VP[c2][r2];

    float minor_det = m00 * (m11 * m22 - m12 * m21)
                    - m01 * (m10 * m22 - m12 * m20)
                    + m02 * (m10 * m21 - m11 * m20);
    float sign = ((c + r) & 1) ? -1.0f : 1.0f;
    float adj_cr = sign * minor_det;

    __shared__ float s_adj[4][4];
    s_adj[c][r] = adj_cr;
    __syncthreads();

    float det = s_VP[0][0] * s_adj[0][0] +
                s_VP[0][1] * s_adj[1][0] +
                s_VP[0][2] * s_adj[2][0] +
                s_VP[0][3] * s_adj[3][0];

    float inv_val = (fabsf(det) > 1e-12f) ? (adj_cr / det) : ((c == r) ? 1.0f : 0.0f);
    reinterpret_cast<float*>(out_inv_view_proj)[tid] = inv_val;
}
#endif

bool CudaRasterizer::compute_camera_matrices_gpu(
    float fov_rad,
    float aspect,
    float near_z,
    float far_z,
    glm::vec3 eye,
    glm::vec3 target,
    glm::vec3 world_up,
    glm::mat4& out_view,
    glm::mat4& out_proj,
    glm::mat4& out_view_proj,
    glm::mat4& out_inv_view_proj,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();

#ifdef __CUDACC__
    if (is_cuda_available()) {
        glm::mat4* d_view = nullptr;
        glm::mat4* d_proj = nullptr;
        glm::mat4* d_vp = nullptr;
        glm::mat4* d_inv_vp = nullptr;

        if (cudaMalloc(&d_view, sizeof(glm::mat4)) == cudaSuccess &&
            cudaMalloc(&d_proj, sizeof(glm::mat4)) == cudaSuccess &&
            cudaMalloc(&d_vp, sizeof(glm::mat4)) == cudaSuccess &&
            cudaMalloc(&d_inv_vp, sizeof(glm::mat4)) == cudaSuccess) {

            gpu_compute_camera_matrices_kernel<<<1, 16>>>(
                fov_rad, aspect, near_z, far_z,
                eye, target, world_up,
                d_view, d_proj, d_vp, d_inv_vp
            );

            cudaMemcpy(&out_view, d_view, sizeof(glm::mat4), cudaMemcpyDeviceToHost);
            cudaMemcpy(&out_proj, d_proj, sizeof(glm::mat4), cudaMemcpyDeviceToHost);
            cudaMemcpy(&out_view_proj, d_vp, sizeof(glm::mat4), cudaMemcpyDeviceToHost);
            cudaMemcpy(&out_inv_view_proj, d_inv_vp, sizeof(glm::mat4), cudaMemcpyDeviceToHost);

            cudaFree(d_view);
            cudaFree(d_proj);
            cudaFree(d_vp);
            cudaFree(d_inv_vp);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_view) cudaFree(d_view);
        if (d_proj) cudaFree(d_proj);
        if (d_vp) cudaFree(d_vp);
        if (d_inv_vp) cudaFree(d_inv_vp);
    }
#endif
    return false;
}

#ifdef __CUDACC__
// --- 25. GPU Kernels for Interior Seed Placement & 3D Seed Ray Picking ---
__global__ void gpu_find_best_seed_voxel_kernel(
    const float* __restrict__ sdf_data,
    int dim_x,
    int dim_y,
    int dim_z,
    int min_coord_x,
    int min_coord_y,
    int min_coord_z,
    float voxel_size,
    glm::vec3 min_p,
    glm::vec3 center,
    float effective_margin,
    unsigned long long* __restrict__ out_best_packed
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(dim_x) * dim_y * dim_z;
    if (idx >= total) return;

    float d = sdf_data[idx];
    if (d > effective_margin) return;

    int x = static_cast<int>(idx % dim_x);
    int y = static_cast<int>((idx / dim_x) % dim_y);
    int z = static_cast<int>(idx / (static_cast<size_t>(dim_x) * dim_y));

    glm::vec3 p(
        (min_coord_x + x) * voxel_size,
        (min_coord_y + y) * voxel_size,
        (min_coord_z + z) * voxel_size
    );

    float xy_dist = glm::length(glm::vec2(p.x - center.x, p.y - center.y));
    float z_height = fmaxf(0.0f, p.z - min_p.z);

    // Prioritize lowest vertical position (floor of vessel/cavity) with smooth central alignment tie-breaker
    float score = z_height * 10.0f + xy_dist * 0.5f;

    unsigned long long key = (static_cast<unsigned long long>(dev_float_to_ordered_uint(score)) << 32) |
                             static_cast<unsigned long long>(idx & 0xFFFFFFFFULL);
    atomicMin(out_best_packed, key);
}

__global__ void gpu_pick_seed_ray_kernel(
    const glm::vec3* __restrict__ seeds,
    int num_seeds,
    glm::vec3 r_orig,
    glm::vec3 r_dir,
    float pick_radius,
    unsigned long long* __restrict__ out_best_hit
) {
    int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= num_seeds) return;

    glm::vec3 sc = seeds[si];
    glm::vec3 to_sc = sc - r_orig;
    float proj_len = glm::dot(to_sc, r_dir);
    if (proj_len > 0.0f) {
        glm::vec3 pt_on_ray = r_orig + proj_len * r_dir;
        float d = glm::length(sc - pt_on_ray);
        if (d < pick_radius) {
            unsigned long long key = (static_cast<unsigned long long>(dev_float_to_ordered_uint(proj_len)) << 32) |
                                     static_cast<unsigned long long>(static_cast<unsigned int>(si));
            atomicMin(out_best_hit, key);
        }
    }
}
#endif

bool CudaRasterizer::find_default_seed_gpu(
    const CudaVolumeGrid& grid,
    glm::vec3 min_p,
    glm::vec3 max_p,
    float desired_margin,
    glm::vec3& out_seed,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    glm::vec3 center = (min_p + max_p) * 0.5f;
    out_seed = glm::vec3(center.x, center.y, min_p.z + 2.0f * grid.voxel_size);

    if (!grid.is_valid() || grid.data.empty()) {
        out_time_ms = 0.0f;
        return false;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        size_t total_voxels = grid.data.size();
        float* d_sdf = nullptr;
        float* d_min_sdf = nullptr;
        unsigned long long* d_best = nullptr;

        if (cudaMalloc(&d_sdf, total_voxels * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_min_sdf, sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_best, sizeof(unsigned long long)) == cudaSuccess) {

            cudaMemcpy(d_sdf, grid.data.data(), total_voxels * sizeof(float), cudaMemcpyHostToDevice);
            float init_min = 1e30f;
            cudaMemcpy(d_min_sdf, &init_min, sizeof(float), cudaMemcpyHostToDevice);

            int threads = 256;
            int red_blocks = std::min(1024, static_cast<int>((total_voxels + threads - 1) / threads));
            gpu_reduce_min_float_kernel<<<red_blocks, threads>>>(d_sdf, total_voxels, d_min_sdf);

            float h_min_sdf = 1e30f;
            cudaMemcpy(&h_min_sdf, d_min_sdf, sizeof(float), cudaMemcpyDeviceToHost);

            float effective_margin = (h_min_sdf < 0.0f)
                ? std::max(desired_margin, h_min_sdf * 0.65f)
                : (h_min_sdf + grid.voxel_size);
            unsigned long long init_best = 0xFFFFFFFFFFFFFFFFULL;
            cudaMemcpy(d_best, &init_best, sizeof(unsigned long long), cudaMemcpyHostToDevice);

                int vox_blocks = static_cast<int>((total_voxels + threads - 1) / threads);
                gpu_find_best_seed_voxel_kernel<<<vox_blocks, threads>>>(
                    d_sdf,
                    grid.dims.x, grid.dims.y, grid.dims.z,
                    grid.min_coord.x, grid.min_coord.y, grid.min_coord.z,
                    grid.voxel_size,
                    min_p, center, effective_margin, d_best
                );

                unsigned long long h_best = 0xFFFFFFFFFFFFFFFFULL;
                cudaMemcpy(&h_best, d_best, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                if (h_best != 0xFFFFFFFFFFFFFFFFULL) {
                    size_t best_idx = static_cast<size_t>(h_best & 0xFFFFFFFFULL);
                    if (best_idx < total_voxels) {
                        int x = static_cast<int>(best_idx % grid.dims.x);
                        int y = static_cast<int>((best_idx / grid.dims.x) % grid.dims.y);
                        int z = static_cast<int>(best_idx / (static_cast<size_t>(grid.dims.x) * grid.dims.y));
                        out_seed = glm::vec3(
                            (grid.min_coord.x + x) * grid.voxel_size,
                            (grid.min_coord.y + y) * grid.voxel_size,
                            (grid.min_coord.z + z) * grid.voxel_size
                        );
                    }
                }

            cudaFree(d_sdf);
            cudaFree(d_min_sdf);
            cudaFree(d_best);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_sdf) cudaFree(d_sdf);
        if (d_min_sdf) cudaFree(d_min_sdf);
        if (d_best) cudaFree(d_best);
    }
#endif
    return false;
}

bool CudaRasterizer::pick_seed_ray_gpu(
    const std::vector<glm::vec3>& seed_points,
    glm::vec3 ray_origin,
    glm::vec3 ray_dir,
    float pick_radius,
    int& out_hit_index,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_hit_index = -1;
    if (seed_points.empty()) {
        out_time_ms = 0.0f;
        return false;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        int num_seeds = static_cast<int>(seed_points.size());
        glm::vec3* d_seeds = nullptr;
        unsigned long long* d_best = nullptr;

        if (cudaMalloc(&d_seeds, num_seeds * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_best, sizeof(unsigned long long)) == cudaSuccess) {

            cudaMemcpy(d_seeds, seed_points.data(), num_seeds * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            unsigned long long init_best = 0xFFFFFFFFFFFFFFFFULL;
            cudaMemcpy(d_best, &init_best, sizeof(unsigned long long), cudaMemcpyHostToDevice);

            int threads = 64;
            int blocks = (num_seeds + threads - 1) / threads;
            gpu_pick_seed_ray_kernel<<<blocks, threads>>>(
                d_seeds, num_seeds, ray_origin, ray_dir, pick_radius, d_best
            );

            unsigned long long h_best = 0xFFFFFFFFFFFFFFFFULL;
            cudaMemcpy(&h_best, d_best, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            if (h_best != 0xFFFFFFFFFFFFFFFFULL) {
                out_hit_index = static_cast<int>(h_best & 0xFFFFFFFFULL);
            }

            cudaFree(d_seeds);
            cudaFree(d_best);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return out_hit_index >= 0;
        }
        if (d_seeds) cudaFree(d_seeds);
        if (d_best) cudaFree(d_best);
    }
#endif
    return false;
}

#ifdef __CUDACC__
// --- 26. GPU Kernels for OpenGL VBO Buffer Packing ---
__global__ void gpu_pack_gl_mesh_vbo_kernel(
    const float* __restrict__ vertices,
    const float* __restrict__ normals,
    size_t num_verts,
    bool has_normals,
    float* __restrict__ out_vbo
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_verts) return;

    size_t v_off = i * 3;
    size_t out_off = i * 6;
    out_vbo[out_off + 0] = vertices[v_off + 0];
    out_vbo[out_off + 1] = vertices[v_off + 1];
    out_vbo[out_off + 2] = vertices[v_off + 2];

    if (has_normals && normals != nullptr) {
        out_vbo[out_off + 3] = normals[v_off + 0];
        out_vbo[out_off + 4] = normals[v_off + 1];
        out_vbo[out_off + 5] = normals[v_off + 2];
    } else {
        out_vbo[out_off + 3] = 0.0f;
        out_vbo[out_off + 4] = 1.0f;
        out_vbo[out_off + 5] = 0.0f;
    }
}

__global__ void gpu_pack_branch_skeleton_vbo_kernel(
    const GpuColonizationNodeLayout* __restrict__ nodes,
    int num_nodes,
    bool colorize_by_root,
    float* __restrict__ out_vbo,
    unsigned int* __restrict__ out_segment_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    int p = nodes[i].parent_index;
    if (p >= 0 && p < num_nodes) {
        unsigned int seg_idx = atomicAdd(out_segment_count, 1u);
        if (seg_idx < static_cast<unsigned int>(num_nodes)) {
            size_t off = static_cast<size_t>(seg_idx) * 12;

            glm::vec3 a = nodes[p].position;
            glm::vec3 b = nodes[i].position;

            int color_id = colorize_by_root
                ? dev_trace_node_root_seed_index(nodes, num_nodes, i)
                : (nodes[i].strand_id & 0xFFFF);
            glm::vec3 col = dev_get_strand_color(color_id);

            out_vbo[off + 0]  = a.x;   out_vbo[off + 1]  = a.y;   out_vbo[off + 2]  = a.z;
            out_vbo[off + 3]  = col.x; out_vbo[off + 4]  = col.y; out_vbo[off + 5]  = col.z;
            out_vbo[off + 6]  = b.x;   out_vbo[off + 7]  = b.y;   out_vbo[off + 8]  = b.z;
            out_vbo[off + 9]  = col.x; out_vbo[off + 10] = col.y; out_vbo[off + 11] = col.z;
        }
    }
}

__global__ void gpu_pack_spore_cloud_vbo_kernel(
    const glm::vec3* __restrict__ spores,
    size_t num_spores,
    float* __restrict__ out_vbo
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_spores) return;

    glm::vec3 sp = spores[i];
    size_t off = i * 6;
    out_vbo[off + 0] = sp.x;
    out_vbo[off + 1] = sp.y;
    out_vbo[off + 2] = sp.z;
    out_vbo[off + 3] = 0.0f;
    out_vbo[off + 4] = 0.0f;
    out_vbo[off + 5] = 1.0f;
}

__global__ void gpu_pack_floor_grid_vbo_kernel(
    int n,
    float half_size,
    float step,
    float* __restrict__ out_vbo
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int num_non_zero = 2 * n;
    if (k >= num_non_zero) return;

    // Map k in [0 .. 2*n - 1] to i in [-n .. -1, 1 .. n]
    int i = (k < n) ? (-n + k) : (1 + (k - n));
    float c = static_cast<float>(i) * step;

    size_t off = static_cast<size_t>(k) * 24;
    // Line parallel to Y
    out_vbo[off + 0]  = c;          out_vbo[off + 1]  = -half_size; out_vbo[off + 2]  = 0.0f;
    out_vbo[off + 3]  = 0.0f;       out_vbo[off + 4]  = 0.0f;       out_vbo[off + 5]  = 1.0f;
    out_vbo[off + 6]  = c;          out_vbo[off + 7]  = half_size;  out_vbo[off + 8]  = 0.0f;
    out_vbo[off + 9]  = 0.0f;       out_vbo[off + 10] = 0.0f;       out_vbo[off + 11] = 1.0f;

    // Line parallel to X
    out_vbo[off + 12] = -half_size; out_vbo[off + 13] = c;          out_vbo[off + 14] = 0.0f;
    out_vbo[off + 15] = 0.0f;       out_vbo[off + 16] = 0.0f;       out_vbo[off + 17] = 1.0f;
    out_vbo[off + 18] = half_size;  out_vbo[off + 19] = c;          out_vbo[off + 20] = 0.0f;
    out_vbo[off + 21] = 0.0f;       out_vbo[off + 22] = 0.0f;       out_vbo[off + 23] = 1.0f;

    if (k == 0) {
        size_t axis_off = static_cast<size_t>(num_non_zero) * 24;
        // Primary X axis line (y=0, z=0)
        out_vbo[axis_off + 0]  = -half_size; out_vbo[axis_off + 1]  = 0.0f;       out_vbo[axis_off + 2]  = 0.0f;
        out_vbo[axis_off + 3]  = 0.0f;       out_vbo[axis_off + 4]  = 0.0f;       out_vbo[axis_off + 5]  = 1.0f;
        out_vbo[axis_off + 6]  = half_size;  out_vbo[axis_off + 7]  = 0.0f;       out_vbo[axis_off + 8]  = 0.0f;
        out_vbo[axis_off + 9]  = 0.0f;       out_vbo[axis_off + 10] = 0.0f;       out_vbo[axis_off + 11] = 1.0f;

        // Primary Y axis line (x=0, z=0)
        out_vbo[axis_off + 12] = 0.0f;       out_vbo[axis_off + 13] = -half_size; out_vbo[axis_off + 14] = 0.0f;
        out_vbo[axis_off + 15] = 0.0f;       out_vbo[axis_off + 16] = 0.0f;       out_vbo[axis_off + 17] = 1.0f;
        out_vbo[axis_off + 18] = 0.0f;       out_vbo[axis_off + 19] = half_size;  out_vbo[axis_off + 20] = 0.0f;
        out_vbo[axis_off + 21] = 0.0f;       out_vbo[axis_off + 22] = 0.0f;       out_vbo[axis_off + 23] = 1.0f;
    }
}
#endif

bool CudaRasterizer::pack_gl_mesh_vbo_gpu(
    const TriangleMesh& mesh,
    bool compute_normals,
    std::vector<float>& out_vbo,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_vbo.clear();
    size_t num_verts = mesh.vertex_count();
    size_t num_tris = mesh.triangle_count();
    if (num_verts == 0) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        float* d_normals = nullptr;
        float* d_vbo = nullptr;

        size_t vert_bytes = mesh.vertices.size() * sizeof(float);
        size_t idx_bytes = mesh.indices.size() * sizeof(uint32_t);
        size_t vbo_floats = num_verts * 6;
        size_t vbo_bytes = vbo_floats * sizeof(float);

        bool ok = (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess) &&
                  (cudaMalloc(&d_vbo, vbo_bytes) == cudaSuccess);
        if (ok && compute_normals && num_tris > 0) {
            ok = (cudaMalloc(&d_indices, idx_bytes) == cudaSuccess) &&
                 (cudaMalloc(&d_normals, vert_bytes) == cudaSuccess);
        }

        if (ok) {
            cudaMemcpy(d_verts, mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            int threads = 256;
            int v_blocks = static_cast<int>((num_verts + threads - 1) / threads);

            if (compute_normals && num_tris > 0) {
                cudaMemcpy(d_indices, mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);
                cudaMemset(d_normals, 0, vert_bytes);
                int t_blocks = static_cast<int>((num_tris + threads - 1) / threads);
                gpu_accumulate_face_normals_kernel<<<t_blocks, threads>>>(
                    d_verts, static_cast<int>(num_verts), d_indices, static_cast<int>(num_tris), d_normals
                );
                gpu_normalize_normals_kernel<<<v_blocks, threads>>>(d_normals, static_cast<int>(num_verts));
            }

            gpu_pack_gl_mesh_vbo_kernel<<<v_blocks, threads>>>(
                d_verts, d_normals, num_verts, compute_normals && (num_tris > 0), d_vbo
            );

            out_vbo.resize(vbo_floats);
            cudaMemcpy(out_vbo.data(), d_vbo, vbo_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            if (d_indices) cudaFree(d_indices);
            if (d_normals) cudaFree(d_normals);
            cudaFree(d_vbo);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
        if (d_normals) cudaFree(d_normals);
        if (d_vbo) cudaFree(d_vbo);
    }
#endif
    return false;
}

bool CudaRasterizer::pack_branch_skeleton_vbo_gpu(
    const void* nodes_data,
    size_t num_nodes,
    std::vector<float>& out_vbo,
    float& out_time_ms,
    bool colorize_by_root
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_vbo.clear();
    if (!nodes_data || num_nodes == 0) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        GpuColonizationNodeLayout* d_nodes = nullptr;
        float* d_vbo = nullptr;
        unsigned int* d_seg_count = nullptr;

        size_t nodes_bytes = num_nodes * sizeof(GpuColonizationNodeLayout);
        size_t max_vbo_floats = num_nodes * 12;

        if (cudaMalloc(&d_nodes, nodes_bytes) == cudaSuccess &&
            cudaMalloc(&d_vbo, max_vbo_floats * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_seg_count, sizeof(unsigned int)) == cudaSuccess) {

            cudaMemcpy(d_nodes, nodes_data, nodes_bytes, cudaMemcpyHostToDevice);
            cudaMemset(d_seg_count, 0, sizeof(unsigned int));

            int threads = 256;
            int blocks = static_cast<int>((num_nodes + threads - 1) / threads);
            gpu_pack_branch_skeleton_vbo_kernel<<<blocks, threads>>>(
                d_nodes, static_cast<int>(num_nodes), colorize_by_root, d_vbo, d_seg_count
            );

            unsigned int h_seg_count = 0;
            cudaMemcpy(&h_seg_count, d_seg_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (h_seg_count > 0) {
                if (h_seg_count > static_cast<unsigned int>(num_nodes)) {
                    h_seg_count = static_cast<unsigned int>(num_nodes);
                }
                size_t valid_floats = static_cast<size_t>(h_seg_count) * 12;
                out_vbo.resize(valid_floats);
                cudaMemcpy(out_vbo.data(), d_vbo, valid_floats * sizeof(float), cudaMemcpyDeviceToHost);
            }

            cudaFree(d_nodes);
            cudaFree(d_vbo);
            cudaFree(d_seg_count);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_nodes) cudaFree(d_nodes);
        if (d_vbo) cudaFree(d_vbo);
        if (d_seg_count) cudaFree(d_seg_count);
    }
#endif
    return false;
}

bool CudaRasterizer::pack_spore_cloud_vbo_gpu(
    const std::vector<glm::vec3>& spores,
    std::vector<float>& out_vbo,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_vbo.clear();
    size_t num_spores = spores.size();
    if (num_spores == 0) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        glm::vec3* d_spores = nullptr;
        float* d_vbo = nullptr;
        size_t vbo_floats = num_spores * 6;

        if (cudaMalloc(&d_spores, num_spores * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_vbo, vbo_floats * sizeof(float)) == cudaSuccess) {

            cudaMemcpy(d_spores, spores.data(), num_spores * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            int threads = 256;
            int blocks = static_cast<int>((num_spores + threads - 1) / threads);
            gpu_pack_spore_cloud_vbo_kernel<<<blocks, threads>>>(d_spores, num_spores, d_vbo);

            out_vbo.resize(vbo_floats);
            cudaMemcpy(out_vbo.data(), d_vbo, vbo_floats * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_spores);
            cudaFree(d_vbo);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_spores) cudaFree(d_spores);
        if (d_vbo) cudaFree(d_vbo);
    }
#endif
    return false;
}

bool CudaRasterizer::pack_floor_grid_vbo_gpu(
    float half_size,
    float step,
    std::vector<float>& out_vbo,
    size_t& out_grid_line_count,
    size_t& out_axis_line_count,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_vbo.clear();
    out_grid_line_count = 0;
    out_axis_line_count = 0;

    int n = std::max(1, static_cast<int>(half_size / std::max(step, 0.001f)));
    int num_non_zero = 2 * n;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        size_t total_floats = static_cast<size_t>(num_non_zero + 1) * 24;
        float* d_vbo = nullptr;
        if (cudaMalloc(&d_vbo, total_floats * sizeof(float)) == cudaSuccess) {
            int threads = 64;
            int blocks = (num_non_zero + threads - 1) / threads;
            gpu_pack_floor_grid_vbo_kernel<<<blocks, threads>>>(n, half_size, step, d_vbo);

            out_vbo.resize(total_floats);
            cudaMemcpy(out_vbo.data(), d_vbo, total_floats * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_vbo);

            out_grid_line_count = static_cast<size_t>(num_non_zero) * 4;
            out_axis_line_count = 4;

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
    }
#endif
    return false;
}

} // namespace rhizome
