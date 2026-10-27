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
// Parallel GPU Normal Accumulation and Normalization
__global__ void gpu_accumulate_face_normals_kernel(
    const float* __restrict__ vertices,
    int num_vertices,
    const uint32_t* __restrict__ indices,
    int num_triangles,
    float* __restrict__ out_normals
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    uint32_t i0 = indices[t * 3 + 0];
    uint32_t i1 = indices[t * 3 + 1];
    uint32_t i2 = indices[t * 3 + 2];

    if (i0 >= num_vertices || i1 >= num_vertices || i2 >= num_vertices) return;

    glm::vec3 v0(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
    glm::vec3 v1(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
    glm::vec3 v2(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

    glm::vec3 fn = glm::cross(v1 - v0, v2 - v0);
    float len = glm::length(fn);
    if (len > 1e-10f) {
        fn /= len;
        atomicAdd(&out_normals[i0 * 3 + 0], fn.x);
        atomicAdd(&out_normals[i0 * 3 + 1], fn.y);
        atomicAdd(&out_normals[i0 * 3 + 2], fn.z);

        atomicAdd(&out_normals[i1 * 3 + 0], fn.x);
        atomicAdd(&out_normals[i1 * 3 + 1], fn.y);
        atomicAdd(&out_normals[i1 * 3 + 2], fn.z);

        atomicAdd(&out_normals[i2 * 3 + 0], fn.x);
        atomicAdd(&out_normals[i2 * 3 + 1], fn.y);
        atomicAdd(&out_normals[i2 * 3 + 2], fn.z);
    }
}

__global__ void gpu_normalize_normals_kernel(
    float* __restrict__ normals,
    int num_vertices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;

    glm::vec3 n(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]);
    float len = glm::length(n);
    if (len > 1e-6f) {
        n /= len;
    } else {
        n = glm::vec3(0.0f, 0.0f, 1.0f);
    }
    normals[i * 3 + 0] = n.x;
    normals[i * 3 + 1] = n.y;
    normals[i * 3 + 2] = n.z;
}

__global__ void gpu_reduce_mesh_bbox_kernel(
    const float* __restrict__ vertices,
    int num_vertices,
    float* __restrict__ out_min_xyz,
    float* __restrict__ out_max_xyz
) {
    __shared__ float s_min_x[256], s_min_y[256], s_min_z[256];
    __shared__ float s_max_x[256], s_max_y[256], s_max_z[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float min_x = 1e30f, min_y = 1e30f, min_z = 1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;

    while (i < num_vertices) {
        float vx = vertices[i * 3 + 0];
        float vy = vertices[i * 3 + 1];
        float vz = vertices[i * 3 + 2];

        if (vx < min_x) min_x = vx;
        if (vy < min_y) min_y = vy;
        if (vz < min_z) min_z = vz;

        if (vx > max_x) max_x = vx;
        if (vy > max_y) max_y = vy;
        if (vz > max_z) max_z = vz;

        i += gridDim.x * blockDim.x;
    }

    s_min_x[tid] = min_x; s_min_y[tid] = min_y; s_min_z[tid] = min_z;
    s_max_x[tid] = max_x; s_max_y[tid] = max_y; s_max_z[tid] = max_z;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min_x[tid + s] < s_min_x[tid]) s_min_x[tid] = s_min_x[tid + s];
            if (s_min_y[tid + s] < s_min_y[tid]) s_min_y[tid] = s_min_y[tid + s];
            if (s_min_z[tid + s] < s_min_z[tid]) s_min_z[tid] = s_min_z[tid + s];

            if (s_max_x[tid + s] > s_max_x[tid]) s_max_x[tid] = s_max_x[tid + s];
            if (s_max_y[tid + s] > s_max_y[tid]) s_max_y[tid] = s_max_y[tid + s];
            if (s_max_z[tid + s] > s_max_z[tid]) s_max_z[tid] = s_max_z[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dev_atomicMin_float(&out_min_xyz[0], s_min_x[0]);
        dev_atomicMin_float(&out_min_xyz[1], s_min_y[0]);
        dev_atomicMin_float(&out_min_xyz[2], s_min_z[0]);

        dev_atomicMax_float(&out_max_xyz[0], s_max_x[0]);
        dev_atomicMax_float(&out_max_xyz[1], s_max_y[0]);
        dev_atomicMax_float(&out_max_xyz[2], s_max_z[0]);
    }
}
#endif // __CUDACC__

bool CudaRasterizer::compute_vertex_normals_gpu(
    const TriangleMesh& mesh,
    std::vector<float>& out_normals_xyz,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t num_verts = mesh.vertex_count();
    size_t num_tris = mesh.triangle_count();

    out_normals_xyz.assign(num_verts * 3, 0.0f);
    if (num_verts == 0 || num_tris == 0) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        float* d_normals = nullptr;

        size_t vert_bytes = mesh.vertices.size() * sizeof(float);
        size_t idx_bytes = mesh.indices.size() * sizeof(uint32_t);
        size_t norm_bytes = out_normals_xyz.size() * sizeof(float);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_normals, norm_bytes) == cudaSuccess) {

            cudaMemcpy(d_verts, mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);
            cudaMemset(d_normals, 0, norm_bytes);

            int tri_threads = 256;
            int tri_blocks = (static_cast<int>(num_tris) + tri_threads - 1) / tri_threads;
            gpu_accumulate_face_normals_kernel<<<tri_blocks, tri_threads>>>(
                d_verts, static_cast<int>(num_verts), d_indices, static_cast<int>(num_tris), d_normals
            );

            int vert_threads = 256;
            int vert_blocks = (static_cast<int>(num_verts) + vert_threads - 1) / vert_threads;
            gpu_normalize_normals_kernel<<<vert_blocks, vert_threads>>>(
                d_normals, static_cast<int>(num_verts)
            );

            cudaMemcpy(out_normals_xyz.data(), d_normals, norm_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            cudaFree(d_indices);
            cudaFree(d_normals);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
        if (d_normals) cudaFree(d_normals);
    }
#endif

    out_time_ms = 0.0f;
    return true;
}

bool CudaRasterizer::compute_bounding_box_gpu(
    const TriangleMesh& mesh,
    glm::vec3& out_min,
    glm::vec3& out_max,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t num_verts = mesh.vertex_count();

    if (num_verts == 0) {
        out_min = glm::vec3(0.0f);
        out_max = glm::vec3(0.0f);
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        float* d_min = nullptr;
        float* d_max = nullptr;

        size_t vert_bytes = mesh.vertices.size() * sizeof(float);
        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_min, 3 * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_max, 3 * sizeof(float)) == cudaSuccess) {

            glm::vec3 init_min(1e30f);
            glm::vec3 init_max(-1e30f);
            cudaMemcpy(d_min, &init_min, 3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_max, &init_max, 3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_verts, mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (static_cast<int>(num_verts) + threads - 1) / threads;
            gpu_reduce_mesh_bbox_kernel<<<blocks, threads>>>(
                d_verts, static_cast<int>(num_verts), d_min, d_max
            );

            cudaMemcpy(&out_min, d_min, 3 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&out_max, d_max, 3 * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            cudaFree(d_min);
            cudaFree(d_max);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_verts) cudaFree(d_verts);
        if (d_min) cudaFree(d_min);
        if (d_max) cudaFree(d_max);
    }
#endif

    std::cerr << "[CUDA Error] GPU acceleration required for compute_bounding_box_gpu.\n";
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_compact_non_degenerate_tris_kernel(
    const uint32_t* __restrict__ in_indices,
    int num_tris,
    const uint8_t* __restrict__ remove_mask,
    uint32_t* __restrict__ out_indices,
    int* __restrict__ out_tri_count
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;
    if (remove_mask && remove_mask[t]) return;

    uint32_t i0 = in_indices[t * 3 + 0];
    uint32_t i1 = in_indices[t * 3 + 1];
    uint32_t i2 = in_indices[t * 3 + 2];
    if (i0 == i1 || i1 == i2 || i2 == i0) return;

    int slot = atomicAdd(out_tri_count, 1);
    out_indices[slot * 3 + 0] = i0;
    out_indices[slot * 3 + 1] = i1;
    out_indices[slot * 3 + 2] = i2;
}

__global__ void gpu_adaptive_curvature_accum_kernel(
    const float* __restrict__ vertices,
    int num_verts,
    const uint32_t* __restrict__ indices,
    int num_tris,
    float* __restrict__ out_norm_sum,
    float* __restrict__ out_area_sum
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    uint32_t i0 = indices[t * 3 + 0];
    uint32_t i1 = indices[t * 3 + 1];
    uint32_t i2 = indices[t * 3 + 2];
    if (i0 >= static_cast<uint32_t>(num_verts) ||
        i1 >= static_cast<uint32_t>(num_verts) ||
        i2 >= static_cast<uint32_t>(num_verts)) return;

    glm::vec3 v0(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
    glm::vec3 v1(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
    glm::vec3 v2(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

    glm::vec3 cr = glm::cross(v1 - v0, v2 - v0);
    float area2 = glm::length(cr);
    if (area2 > 1e-10f) {
        atomicAdd(&out_norm_sum[i0 * 3 + 0], cr.x);
        atomicAdd(&out_norm_sum[i0 * 3 + 1], cr.y);
        atomicAdd(&out_norm_sum[i0 * 3 + 2], cr.z);
        atomicAdd(&out_area_sum[i0], area2);

        atomicAdd(&out_norm_sum[i1 * 3 + 0], cr.x);
        atomicAdd(&out_norm_sum[i1 * 3 + 1], cr.y);
        atomicAdd(&out_norm_sum[i1 * 3 + 2], cr.z);
        atomicAdd(&out_area_sum[i1], area2);

        atomicAdd(&out_norm_sum[i2 * 3 + 0], cr.x);
        atomicAdd(&out_norm_sum[i2 * 3 + 1], cr.y);
        atomicAdd(&out_norm_sum[i2 * 3 + 2], cr.z);
        atomicAdd(&out_area_sum[i2], area2);
    }
}

__global__ void gpu_adaptive_propose_collapse_kernel(
    const float* __restrict__ vertices,
    int num_verts,
    const uint32_t* __restrict__ indices,
    int num_tris,
    const float* __restrict__ norm_sum,
    const float* __restrict__ area_sum,
    float min_vox,
    float max_vox,
    int* __restrict__ io_parent
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    uint32_t tri_v[3] = {
        indices[t * 3 + 0],
        indices[t * 3 + 1],
        indices[t * 3 + 2]
    };

    for (int e = 0; e < 3; ++e) {
        int u = static_cast<int>(tri_v[e]);
        int v = static_cast<int>(tri_v[(e + 1) % 3]);
        if (u == v || u < 0 || v < 0 || u >= num_verts || v >= num_verts) continue;

        glm::vec3 pu(vertices[u * 3 + 0], vertices[u * 3 + 1], vertices[u * 3 + 2]);
        glm::vec3 pv(vertices[v * 3 + 0], vertices[v * 3 + 1], vertices[v * 3 + 2]);
        float edge_len = glm::length(pv - pu);

        glm::vec3 nu_sum(norm_sum[u * 3 + 0], norm_sum[u * 3 + 1], norm_sum[u * 3 + 2]);
        glm::vec3 nv_sum(norm_sum[v * 3 + 0], norm_sum[v * 3 + 1], norm_sum[v * 3 + 2]);
        float au = fmaxf(area_sum[u], 1e-8f);
        float av = fmaxf(area_sum[v], 1e-8f);

        float flat_u = fminf(fmaxf(glm::length(nu_sum) / au, 0.0f), 1.0f);
        float flat_v = fminf(fmaxf(glm::length(nv_sum) / av, 0.0f), 1.0f);
        glm::vec3 nu = (glm::length(nu_sum) > 1e-8f) ? glm::normalize(nu_sum) : glm::vec3(0.0f, 0.0f, 1.0f);
        glm::vec3 nv = (glm::length(nv_sum) > 1e-8f) ? glm::normalize(nv_sum) : glm::vec3(0.0f, 0.0f, 1.0f);
        float align_uv = fminf(fmaxf(glm::dot(nu, nv), 0.0f), 1.0f);

        float flatness = flat_u * flat_v * align_uv;
        // High flatness (~1.0) allows coarsening up to max_vox; high curvature (~0.0) preserves min_vox
        if (flatness > 0.86f) {
            float t_flat = fminf(fmaxf((flatness - 0.86f) / 0.14f, 0.0f), 1.0f);
            float target_edge = min_vox + (max_vox - min_vox) * (t_flat * t_flat) * 0.82f;
            if (edge_len < target_edge) {
                int hi = (u > v) ? u : v;
                int lo = (u > v) ? v : u;
                // Only collapse if both vertices are still independent roots in this pass (star matching)
                if (io_parent[lo] == lo && io_parent[hi] == hi) {
                    atomicCAS(&io_parent[hi], hi, lo);
                }
            }
        }
    }
}

__global__ void gpu_adaptive_apply_collapse_kernel(
    float* __restrict__ vertices,
    int num_verts,
    int* __restrict__ io_parent,
    int* __restrict__ d_any_collapsed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_verts) return;

    // Resolve 1-hop chain safely so mapping is strictly depth-1
    int p = io_parent[i];
    if (p != i) {
        int gp = io_parent[p];
        if (gp != p) {
            // Cancel chained hop so star matching stays disjoint
            io_parent[i] = i;
        } else {
            *d_any_collapsed = 1;
        }
    }
}

__global__ void gpu_adaptive_remap_indices_kernel(
    uint32_t* __restrict__ indices,
    int num_tris,
    const int* __restrict__ parent
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    uint32_t i0 = static_cast<uint32_t>(parent[indices[t * 3 + 0]]);
    uint32_t i1 = static_cast<uint32_t>(parent[indices[t * 3 + 1]]);
    uint32_t i2 = static_cast<uint32_t>(parent[indices[t * 3 + 2]]);
    indices[t * 3 + 0] = i0;
    indices[t * 3 + 1] = i1;
    indices[t * 3 + 2] = i2;
}

__global__ void gpu_init_identity_parent_kernel(int* __restrict__ parent, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) parent[i] = i;
}
#endif

bool CudaRasterizer::decimate_mesh_adaptive_gpu(
    TriangleMesh& inout_mesh,
    float min_voxel_size,
    float max_voxel_size,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    int num_verts = static_cast<int>(inout_mesh.vertex_count());
    int num_tris  = static_cast<int>(inout_mesh.triangle_count());
    if (num_verts <= 0 || num_tris <= 0 || max_voxel_size <= min_voxel_size + 1e-4f) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices_a = nullptr;
        uint32_t* d_indices_b = nullptr;
        float* d_norm_sum = nullptr;
        float* d_area_sum = nullptr;
        int* d_parent = nullptr;
        int* d_counts = nullptr; // [0]=tri_count, [1]=any_collapsed

        size_t vert_bytes = static_cast<size_t>(num_verts) * 3 * sizeof(float);
        size_t idx_bytes  = static_cast<size_t>(num_tris) * 3 * sizeof(uint32_t);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices_a, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices_b, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_norm_sum, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_area_sum, static_cast<size_t>(num_verts) * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_parent, static_cast<size_t>(num_verts) * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_counts, 2 * sizeof(int)) == cudaSuccess) {

            cudaMemcpy(d_verts, inout_mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices_a, inout_mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int vert_blocks = (num_verts + threads - 1) / threads;

            for (int pass = 0; pass < 3 && num_tris > 0; ++pass) {
                int tri_blocks = (num_tris + threads - 1) / threads;
                cudaMemset(d_norm_sum, 0, vert_bytes);
                cudaMemset(d_area_sum, 0, static_cast<size_t>(num_verts) * sizeof(float));
                cudaMemset(d_counts, 0, 2 * sizeof(int));

                gpu_init_identity_parent_kernel<<<vert_blocks, threads>>>(d_parent, num_verts);

                gpu_adaptive_curvature_accum_kernel<<<tri_blocks, threads>>>(
                    d_verts, num_verts, d_indices_a, num_tris, d_norm_sum, d_area_sum
                );

                gpu_adaptive_propose_collapse_kernel<<<tri_blocks, threads>>>(
                    d_verts, num_verts, d_indices_a, num_tris,
                    d_norm_sum, d_area_sum, min_voxel_size, max_voxel_size, d_parent
                );

                gpu_adaptive_apply_collapse_kernel<<<vert_blocks, threads>>>(
                    d_verts, num_verts, d_parent, &d_counts[1]
                );

                int any_collapsed = 0;
                cudaMemcpy(&any_collapsed, &d_counts[1], sizeof(int), cudaMemcpyDeviceToHost);
                if (!any_collapsed) break;

                gpu_adaptive_remap_indices_kernel<<<tri_blocks, threads>>>(
                    d_indices_a, num_tris, d_parent
                );

                gpu_compact_non_degenerate_tris_kernel<<<tri_blocks, threads>>>(
                    d_indices_a, num_tris, nullptr, d_indices_b, &d_counts[0]
                );
                cudaMemcpy(&num_tris, &d_counts[0], sizeof(int), cudaMemcpyDeviceToHost);
                std::swap(d_indices_a, d_indices_b);
            }

            if (num_tris > 0) {
                inout_mesh.indices.resize(static_cast<size_t>(num_tris) * 3);
                cudaMemcpy(inout_mesh.indices.data(), d_indices_a, inout_mesh.indices.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            } else {
                inout_mesh.indices.clear();
            }

            cudaFree(d_verts);
            cudaFree(d_indices_a);
            cudaFree(d_indices_b);
            cudaFree(d_norm_sum);
            cudaFree(d_area_sum);
            cudaFree(d_parent);
            cudaFree(d_counts);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_verts) cudaFree(d_verts);
        if (d_indices_a) cudaFree(d_indices_a);
        if (d_indices_b) cudaFree(d_indices_b);
        if (d_norm_sum) cudaFree(d_norm_sum);
        if (d_area_sum) cudaFree(d_area_sum);
        if (d_parent) cudaFree(d_parent);
        if (d_counts) cudaFree(d_counts);
    }
#endif

    out_time_ms = 0.0f;
    return true;
}

#ifdef __CUDACC__
__global__ void gpu_raycast_mesh_kernel(
    const float* __restrict__ vertices,
    const uint32_t* __restrict__ indices,
    size_t num_tris,
    float ro_x, float ro_y, float ro_z,
    float rd_x, float rd_y, float rd_z,
    unsigned long long* __restrict__ out_min_packed
) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    uint32_t i0 = indices[t * 3 + 0];
    uint32_t i1 = indices[t * 3 + 1];
    uint32_t i2 = indices[t * 3 + 2];

    glm::vec3 v0(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
    glm::vec3 v1(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
    glm::vec3 v2(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

    glm::vec3 ro(ro_x, ro_y, ro_z);
    glm::vec3 rd(rd_x, rd_y, rd_z);

    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;
    glm::vec3 pvec = glm::cross(rd, e2);
    float det = glm::dot(e1, pvec);

    if (fabsf(det) < 1e-8f) return;
    float inv_det = 1.0f / det;

    glm::vec3 tvec = ro - v0;
    float u = glm::dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return;

    glm::vec3 qvec = glm::cross(tvec, e1);
    float v = glm::dot(rd, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return;

    float dist_t = glm::dot(e2, qvec) * inv_det;
    if (dist_t <= 1e-4f) return;

    unsigned int dist_bits = __float_as_uint(dist_t);
    unsigned long long pack = ((static_cast<unsigned long long>(dist_bits)) << 32) |
                              (static_cast<unsigned long long>(static_cast<uint32_t>(t)));
    atomicMin(out_min_packed, pack);
}
#endif

bool CudaRasterizer::raycast_mesh_gpu(
    const TriangleMesh& mesh,
    const glm::vec3& ray_origin,
    const glm::vec3& ray_dir,
    glm::vec3& out_hit_point,
    glm::vec3& out_hit_normal,
    float& out_hit_t,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_hit_t = -1.0f;
    size_t num_tris = mesh.triangle_count();
    if (num_tris == 0 || mesh.vertices.empty() || mesh.indices.empty()) {
        out_time_ms = 0.0f;
        return false;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_vertices = nullptr;
        uint32_t* d_indices = nullptr;
        unsigned long long* d_min_packed = nullptr;

        size_t vert_bytes = mesh.vertices.size() * sizeof(float);
        size_t idx_bytes = mesh.indices.size() * sizeof(uint32_t);

        if (cudaMalloc(&d_vertices, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_min_packed, sizeof(unsigned long long)) == cudaSuccess) {

            cudaMemcpy(d_vertices, mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);

            float inf_f = 1e9f;
            unsigned int inf_bits = *reinterpret_cast<unsigned int*>(&inf_f);
            unsigned long long init_packed = ((static_cast<unsigned long long>(inf_bits)) << 32) | 0xFFFFFFFFULL;
            cudaMemcpy(d_min_packed, &init_packed, sizeof(unsigned long long), cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = static_cast<int>((num_tris + threads - 1) / threads);

            gpu_raycast_mesh_kernel<<<blocks, threads>>>(
                d_vertices, d_indices, num_tris,
                ray_origin.x, ray_origin.y, ray_origin.z,
                ray_dir.x, ray_dir.y, ray_dir.z,
                d_min_packed
            );

            unsigned long long h_packed = 0xFFFFFFFFULL;
            cudaMemcpy(&h_packed, d_min_packed, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

            cudaFree(d_vertices); d_vertices = nullptr;
            cudaFree(d_indices);  d_indices = nullptr;
            cudaFree(d_min_packed); d_min_packed = nullptr;

            unsigned int h_bits = static_cast<unsigned int>(h_packed >> 32);
            uint32_t raw_tri = static_cast<uint32_t>(h_packed & 0xFFFFFFFFULL);
            int h_tri = (raw_tri == 0xFFFFFFFFU) ? -1 : static_cast<int>(raw_tri);
            float hit_t = *reinterpret_cast<float*>(&h_bits);

            if (h_tri >= 0 && hit_t > 0.0f && hit_t < 1e8f) {
                out_hit_t = hit_t;
                out_hit_point = ray_origin + hit_t * ray_dir;

                glm::vec3 v0, v1, v2;
                mesh.get_triangle(static_cast<size_t>(h_tri), v0, v1, v2);
                glm::vec3 n = glm::cross(v1 - v0, v2 - v0);
                float len = glm::length(n);
                out_hit_normal = (len > 1e-6f) ? (n / len) : glm::vec3(0.0f, 0.0f, 1.0f);

                auto end = std::chrono::high_resolution_clock::now();
                out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
                return true;
            }
        }

        if (d_vertices) cudaFree(d_vertices);
        if (d_indices) cudaFree(d_indices);
        if (d_min_packed) cudaFree(d_min_packed);
    }
#endif
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_query_point_distances_kernel(
    const float* __restrict__ vertices,
    const uint32_t* __restrict__ indices,
    size_t num_tris,
    const float* __restrict__ query_points,
    size_t num_points,
    float* __restrict__ out_distances
) {
    size_t p_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_idx >= num_points) return;

    glm::vec3 p(
        query_points[p_idx * 3 + 0],
        query_points[p_idx * 3 + 1],
        query_points[p_idx * 3 + 2]
    );

    float min_d2 = 1e18f;

    for (size_t t = 0; t < num_tris; ++t) {
        uint32_t i0 = indices[t * 3 + 0];
        uint32_t i1 = indices[t * 3 + 1];
        uint32_t i2 = indices[t * 3 + 2];

        glm::vec3 v0(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        glm::vec3 v1(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        glm::vec3 v2(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

        glm::vec3 closest_pt;
        float d2 = point_triangle_distance_sq(p, v0, v1, v2, closest_pt);
        if (d2 < min_d2) {
            min_d2 = d2;
        }
    }

    out_distances[p_idx] = sqrtf(min_d2);
}
#endif

bool CudaRasterizer::query_distance_batch_gpu(
    const TriangleMesh& mesh,
    const std::vector<glm::vec3>& query_points,
    std::vector<float>& out_distances,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_distances.resize(query_points.size(), 0.0f);
    if (query_points.empty() || mesh.indices.empty()) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        size_t num_tris = mesh.triangle_count();
        size_t num_pts = query_points.size();

        float* d_vertices = nullptr;
        uint32_t* d_indices = nullptr;
        float* d_query_points = nullptr;
        float* d_out_distances = nullptr;

        size_t vert_bytes = mesh.vertices.size() * sizeof(float);
        size_t idx_bytes = mesh.indices.size() * sizeof(uint32_t);
        size_t pts_bytes = num_pts * 3 * sizeof(float);
        size_t dist_bytes = num_pts * sizeof(float);

        if (cudaMalloc(&d_vertices, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_query_points, pts_bytes) == cudaSuccess &&
            cudaMalloc(&d_out_distances, dist_bytes) == cudaSuccess) {

            cudaMemcpy(d_vertices, mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_query_points, query_points.data(), pts_bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = static_cast<int>((num_pts + threads - 1) / threads);

            gpu_query_point_distances_kernel<<<blocks, threads>>>(
                d_vertices, d_indices, num_tris,
                d_query_points, num_pts, d_out_distances
            );

            cudaMemcpy(out_distances.data(), d_out_distances, dist_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_vertices);
            cudaFree(d_indices);
            cudaFree(d_query_points);
            cudaFree(d_out_distances);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_vertices) cudaFree(d_vertices);
        if (d_indices) cudaFree(d_indices);
        if (d_query_points) cudaFree(d_query_points);
        if (d_out_distances) cudaFree(d_out_distances);
    }
#endif
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_compute_winding_numbers_kernel(
    const float* __restrict__ vertices,
    const uint32_t* __restrict__ indices,
    size_t num_tris,
    const float* __restrict__ query_points,
    size_t num_points,
    float* __restrict__ out_winding_numbers
) {
    size_t p_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (p_idx >= num_points) return;

    glm::vec3 p(
        query_points[p_idx * 3 + 0],
        query_points[p_idx * 3 + 1],
        query_points[p_idx * 3 + 2]
    );

    const float inv_4pi = 0.07957747154594767f;
    float total_solid_angle = 0.0f;

    for (size_t t = 0; t < num_tris; ++t) {
        uint32_t i0 = indices[t * 3 + 0];
        uint32_t i1 = indices[t * 3 + 1];
        uint32_t i2 = indices[t * 3 + 2];

        glm::vec3 a(vertices[i0 * 3 + 0] - p.x, vertices[i0 * 3 + 1] - p.y, vertices[i0 * 3 + 2] - p.z);
        glm::vec3 b(vertices[i1 * 3 + 0] - p.x, vertices[i1 * 3 + 1] - p.y, vertices[i1 * 3 + 2] - p.z);
        glm::vec3 c(vertices[i2 * 3 + 0] - p.x, vertices[i2 * 3 + 1] - p.y, vertices[i2 * 3 + 2] - p.z);

        float la = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
        float lb = sqrtf(b.x * b.x + b.y * b.y + b.z * b.z);
        float lc = sqrtf(c.x * c.x + c.y * c.y + c.z * c.z);

        if (la < 1e-7f || lb < 1e-7f || lc < 1e-7f) continue;

        float det = a.x * (b.y * c.z - b.z * c.y) -
                    a.y * (b.x * c.z - b.z * c.x) +
                    a.z * (b.x * c.y - b.y * c.x);

        float dot_ab = a.x * b.x + a.y * b.y + a.z * b.z;
        float dot_bc = b.x * c.x + b.y * c.y + b.z * c.z;
        float dot_ca = c.x * a.x + c.y * a.y + c.z * a.z;

        float denom = la * lb * lc + dot_ab * lc + dot_bc * la + dot_ca * lb;
        float omega = 2.0f * atan2f(det, denom);
        total_solid_angle += omega;
    }

    out_winding_numbers[p_idx] = total_solid_angle * inv_4pi;
}
#endif

bool CudaRasterizer::compute_winding_numbers_gpu(
    const TriangleMesh& mesh,
    const std::vector<glm::vec3>& query_points,
    std::vector<float>& out_winding_numbers,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_winding_numbers.resize(query_points.size(), 0.0f);
    if (query_points.empty() || mesh.indices.empty()) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        size_t num_tris = mesh.triangle_count();
        size_t num_pts = query_points.size();

        float* d_vertices = nullptr;
        uint32_t* d_indices = nullptr;
        float* d_query_points = nullptr;
        float* d_out_wn = nullptr;

        size_t vert_bytes = mesh.vertices.size() * sizeof(float);
        size_t idx_bytes = mesh.indices.size() * sizeof(uint32_t);
        size_t pts_bytes = num_pts * 3 * sizeof(float);
        size_t wn_bytes = num_pts * sizeof(float);

        if (cudaMalloc(&d_vertices, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_query_points, pts_bytes) == cudaSuccess &&
            cudaMalloc(&d_out_wn, wn_bytes) == cudaSuccess) {

            cudaMemcpy(d_vertices, mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_query_points, query_points.data(), pts_bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = static_cast<int>((num_pts + threads - 1) / threads);

            gpu_compute_winding_numbers_kernel<<<blocks, threads>>>(
                d_vertices, d_indices, num_tris,
                d_query_points, num_pts, d_out_wn
            );
            cudaDeviceSynchronize();

            cudaMemcpy(out_winding_numbers.data(), d_out_wn, wn_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_vertices);
            cudaFree(d_indices);
            cudaFree(d_query_points);
            cudaFree(d_out_wn);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_vertices) cudaFree(d_vertices);
        if (d_indices) cudaFree(d_indices);
        if (d_query_points) cudaFree(d_query_points);
        if (d_out_wn) cudaFree(d_out_wn);
    }
#endif
    return false;
}

#ifdef __CUDACC__
// --- 21. GPU Kernels for Z-Up Alignment & Floor Placement ---
__global__ void gpu_rotate_yup_to_zup_kernel(
    float* __restrict__ vertices,
    int num_vertices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;
    float vx = vertices[i * 3 + 0];
    float vy = vertices[i * 3 + 1];
    float vz = vertices[i * 3 + 2];
    vertices[i * 3 + 0] = vx;
    vertices[i * 3 + 1] = -vz;
    vertices[i * 3 + 2] = vy;
}

__global__ void gpu_translate_vertices_z_kernel(
    float* __restrict__ vertices,
    int num_vertices,
    float z_offset
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;
    vertices[i * 3 + 2] += z_offset;
}

// --- 22. GPU Kernel for Mesh Edge Metrics ---
__global__ void gpu_mesh_edge_metrics_kernel(
    const float* __restrict__ vertices,
    const uint32_t* __restrict__ indices,
    int num_tris,
    float* __restrict__ out_sum_len,
    float* __restrict__ out_min_len,
    float* __restrict__ out_max_len
) {
    __shared__ float s_sum[256];
    __shared__ float s_min[256];
    __shared__ float s_max[256];

    int tid = threadIdx.x;
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    float local_min = 1e30f;
    float local_max = 0.0f;

    while (t < num_tris) {
        uint32_t i0 = indices[t * 3 + 0];
        uint32_t i1 = indices[t * 3 + 1];
        uint32_t i2 = indices[t * 3 + 2];

        glm::vec3 v0(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        glm::vec3 v1(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        glm::vec3 v2(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

        float l0 = glm::length(v1 - v0);
        float l1 = glm::length(v2 - v1);
        float l2 = glm::length(v0 - v2);

        local_sum += (l0 + l1 + l2);
        local_min = fminf(local_min, fminf(l0, fminf(l1, l2)));
        local_max = fmaxf(local_max, fmaxf(l0, fmaxf(l1, l2)));

        t += gridDim.x * blockDim.x;
    }

    s_sum[tid] = local_sum;
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            if (s_min[tid + s] < s_min[tid]) s_min[tid] = s_min[tid + s];
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_sum_len, s_sum[0]);
        dev_atomicMin_float(out_min_len, s_min[0]);
        dev_atomicMax_float(out_max_len, s_max[0]);
    }
}
#endif // __CUDACC__

bool CudaRasterizer::align_mesh_z_up_gpu(
    TriangleMesh& inout_mesh,
    glm::vec3& out_min,
    glm::vec3& out_max,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    int num_verts = static_cast<int>(inout_mesh.vertex_count());
    if (num_verts <= 0) {
        out_min = glm::vec3(0.0f);
        out_max = glm::vec3(0.0f);
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        float* d_min = nullptr;
        float* d_max = nullptr;
        size_t vert_bytes = inout_mesh.vertices.size() * sizeof(float);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_min, 3 * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_max, 3 * sizeof(float)) == cudaSuccess) {

            cudaMemcpy(d_verts, inout_mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (num_verts + threads - 1) / threads;
            int red_blocks = std::min(1024, blocks);

            auto run_bbox = [&](glm::vec3& bmin, glm::vec3& bmax) {
                glm::vec3 init_min(1e30f), init_max(-1e30f);
                cudaMemcpy(d_min, &init_min, 3 * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_max, &init_max, 3 * sizeof(float), cudaMemcpyHostToDevice);
                gpu_reduce_mesh_bbox_kernel<<<red_blocks, threads>>>(d_verts, num_verts, d_min, d_max);
                cudaMemcpy(&bmin, d_min, 3 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&bmax, d_max, 3 * sizeof(float), cudaMemcpyDeviceToHost);
            };

            glm::vec3 min_p(0.0f), max_p(0.0f);
            run_bbox(min_p, max_p);
            glm::vec3 sz = max_p - min_p;

            if (sz.y > sz.z * 1.25f) {
                gpu_rotate_yup_to_zup_kernel<<<blocks, threads>>>(d_verts, num_verts);
                run_bbox(min_p, max_p);
            }

            float z_offset = -min_p.z;
            if (std::abs(z_offset) > 1e-4f) {
                gpu_translate_vertices_z_kernel<<<blocks, threads>>>(d_verts, num_verts, z_offset);
                min_p.z += z_offset;
                max_p.z += z_offset;
            }

            cudaMemcpy(inout_mesh.vertices.data(), d_verts, vert_bytes, cudaMemcpyDeviceToHost);
            out_min = min_p;
            out_max = max_p;

            cudaFree(d_verts);
            cudaFree(d_min);
            cudaFree(d_max);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_verts) cudaFree(d_verts);
        if (d_min) cudaFree(d_min);
        if (d_max) cudaFree(d_max);
    }
#endif
    return false;
}

bool CudaRasterizer::compute_mesh_metrics_gpu(
    const TriangleMesh& mesh,
    float& out_avg_edge_mm,
    float& out_min_edge_mm,
    float& out_max_edge_mm,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_avg_edge_mm = 0.0f;
    out_min_edge_mm = 0.0f;
    out_max_edge_mm = 0.0f;

    int num_tris = static_cast<int>(mesh.triangle_count());
    if (num_tris <= 0 || mesh.vertices.empty()) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        float* d_sum = nullptr;
        float* d_min = nullptr;
        float* d_max = nullptr;

        size_t vert_bytes = mesh.vertices.size() * sizeof(float);
        size_t idx_bytes = mesh.indices.size() * sizeof(uint32_t);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_sum, sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_min, sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_max, sizeof(float)) == cudaSuccess) {

            cudaMemcpy(d_verts, mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);

            float init_sum = 0.0f, init_min = 1e30f, init_max = 0.0f;
            cudaMemcpy(d_sum, &init_sum, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = std::min(1024, (num_tris + threads - 1) / threads);
            gpu_mesh_edge_metrics_kernel<<<blocks, threads>>>(d_verts, d_indices, num_tris, d_sum, d_min, d_max);

            float h_sum = 0.0f, h_min = 0.0f, h_max = 0.0f;
            cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

            out_avg_edge_mm = h_sum / static_cast<float>(num_tris * 3);
            out_min_edge_mm = (h_min > 1e20f) ? 0.0f : h_min;
            out_max_edge_mm = h_max;

            cudaFree(d_verts);
            cudaFree(d_indices);
            cudaFree(d_sum);
            cudaFree(d_min);
            cudaFree(d_max);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
        if (d_sum) cudaFree(d_sum);
        if (d_min) cudaFree(d_min);
        if (d_max) cudaFree(d_max);
    }
#endif
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_generate_bark_texture_2d_kernel(
    int width,
    int height,
    float scale,
    float stretch_y,
    float furrow_depth,
    int octaves,
    float persistence,
    uint32_t seed,
    float* __restrict__ out_heightmap,
    uint32_t* __restrict__ out_rgba
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float u = static_cast<float>(x) / static_cast<float>(width);
    float v = static_cast<float>(y) / static_cast<float>(height);

    float sy = (stretch_y > 0.01f) ? stretch_y : 1.0f;
    glm::vec3 p(u * scale, (v * scale) / sy, static_cast<float>(seed % 1000u) * 0.173f);

    float sum = 0.0f;
    float amp = 1.0f;
    float max_amp = 0.0f;
    float freq = 1.0f;

    for (int oct = 0; oct < octaves; ++oct) {
        glm::vec3 sample_p = p * freq;
        float n = dev_simplex_noise_3d(sample_p);
        if (oct == 0) {
            float furrow = 1.0f - fabsf(n);
            furrow = furrow * furrow;
            n = (1.0f - furrow_depth) * n + furrow_depth * (furrow * 2.0f - 1.0f);
        }
        sum += n * amp;
        max_amp += amp;
        amp *= persistence;
        freq *= 2.0f;
    }

    float norm_val = (max_amp > 1e-5f) ? (sum / max_amp) : 0.0f;
    // Continuous grayscale gradient mapping centered at 0.5 mid-gray
    float gray = 0.5f + 0.5f * tanhf(norm_val * 1.5f);
    gray = fminf(1.0f, fmaxf(0.0f, gray));

    out_heightmap[idx] = gray;
    if (out_rgba) {
        uint32_t byte_val = static_cast<uint32_t>(gray * 255.0f + 0.5f);
        if (byte_val > 255u) byte_val = 255u;
        out_rgba[idx] = 0xFF000000u | (byte_val << 16) | (byte_val << 8) | byte_val;
    }
}

__global__ void gpu_displace_mesh_from_bark_texture_kernel(
    float* __restrict__ vertices,
    int num_vertices,
    const float* __restrict__ normals,
    const float* __restrict__ heightmap,
    int tex_width,
    int tex_height,
    float min_z,
    float max_z,
    float depth_mm,
    float tile_u,
    float tile_v
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;

    float vx = vertices[i * 3 + 0];
    float vy = vertices[i * 3 + 1];
    float vz = vertices[i * 3 + 2];

    float nx = normals[i * 3 + 0];
    float ny = normals[i * 3 + 1];
    float nz = normals[i * 3 + 2];

    // Cylindrical UV projection around growth axis (Z)
    float angle = atan2f(vy, vx);
    float u = (angle + 3.141592653589793f) * (0.5f / 3.141592653589793f) * tile_u;
    float z_range = fmaxf(max_z - min_z, 1e-4f);
    float v = ((vz - min_z) / z_range) * tile_v;

    // Fractional wrap for seamless repeating
    u = u - floorf(u);
    v = v - floorf(v);

    // Bilinear continuous filtering from grayscale heightmap
    float fx = u * static_cast<float>(tex_width - 1);
    float fy = v * static_cast<float>(tex_height - 1);
    int x0 = static_cast<int>(floorf(fx));
    int y0 = static_cast<int>(floorf(fy));
    int x1 = (x0 + 1) % tex_width;
    int y1 = (y0 + 1) % tex_height;
    float tx = fx - static_cast<float>(x0);
    float ty = fy - static_cast<float>(y0);

    float s00 = heightmap[y0 * tex_width + x0];
    float s10 = heightmap[y0 * tex_width + x1];
    float s01 = heightmap[y1 * tex_width + x0];
    float s11 = heightmap[y1 * tex_width + x1];

    float sample = (1.0f - tx) * (1.0f - ty) * s00 +
                   tx * (1.0f - ty) * s10 +
                   (1.0f - tx) * ty * s01 +
                   tx * ty * s11;

    // Natural gradual embossing (>0.5) and debossing (<0.5) centered at 0.5
    float offset_mm = (sample - 0.50f) * depth_mm;

    vertices[i * 3 + 0] = vx + nx * offset_mm;
    vertices[i * 3 + 1] = vy + ny * offset_mm;
    vertices[i * 3 + 2] = vz + nz * offset_mm;
}
#endif

bool CudaRasterizer::generate_bark_texture_2d_gpu(
    const BarkTexture2DParams& params,
    std::vector<float>& out_heightmap,
    std::vector<uint32_t>& out_rgba_preview,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_time_ms = 0.0f;

    int w = std::max(64, params.width);
    int h = std::max(64, params.height);
    size_t total_pixels = static_cast<size_t>(w) * static_cast<size_t>(h);

    out_heightmap.resize(total_pixels, 0.5f);
    out_rgba_preview.resize(total_pixels, 0xFF808080u);

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_heightmap = nullptr;
        uint32_t* d_rgba = nullptr;
        size_t float_bytes = total_pixels * sizeof(float);
        size_t rgba_bytes = total_pixels * sizeof(uint32_t);

        if (cudaMalloc(&d_heightmap, float_bytes) == cudaSuccess &&
            cudaMalloc(&d_rgba, rgba_bytes) == cudaSuccess) {

            dim3 threads(16, 16);
            dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);

            gpu_generate_bark_texture_2d_kernel<<<blocks, threads>>>(
                w, h,
                params.scale,
                params.stretch_y,
                params.furrow_depth,
                params.octaves,
                params.persistence,
                params.seed,
                d_heightmap,
                d_rgba
            );

            cudaMemcpy(out_heightmap.data(), d_heightmap, float_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(out_rgba_preview.data(), d_rgba, rgba_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_heightmap);
            cudaFree(d_rgba);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_heightmap) cudaFree(d_heightmap);
        if (d_rgba) cudaFree(d_rgba);
    }
#endif
    return false;
}

bool CudaRasterizer::displace_mesh_from_bark_texture_gpu(
    TriangleMesh& inout_mesh,
    const std::vector<float>& in_heightmap,
    int tex_width,
    int tex_height,
    const MeshDisplacementParams& params,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_time_ms = 0.0f;

    size_t num_verts = inout_mesh.vertex_count();
    size_t num_tris = inout_mesh.triangle_count();
    if (num_verts == 0 || in_heightmap.empty() || tex_width <= 0 || tex_height <= 0) {
        return false;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        float* d_normals = nullptr;
        float* d_heightmap = nullptr;
        uint32_t* d_indices = nullptr;

        size_t vert_bytes = inout_mesh.vertices.size() * sizeof(float);
        size_t height_bytes = in_heightmap.size() * sizeof(float);
        size_t idx_bytes = inout_mesh.indices.size() * sizeof(uint32_t);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_normals, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_heightmap, height_bytes) == cudaSuccess) {

            cudaMemcpy(d_verts, inout_mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_heightmap, in_heightmap.data(), height_bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int vert_blocks = (static_cast<int>(num_verts) + threads - 1) / threads;

            // Compute vertex normals directly in VRAM from triangle faces
            if (num_tris > 0 && cudaMalloc(&d_indices, idx_bytes) == cudaSuccess) {
                cudaMemcpy(d_indices, inout_mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);
                cudaMemset(d_normals, 0, vert_bytes);

                int tri_blocks = (static_cast<int>(num_tris) + threads - 1) / threads;
                gpu_accumulate_face_normals_kernel<<<tri_blocks, threads>>>(
                    d_verts, static_cast<int>(num_verts), d_indices, static_cast<int>(num_tris), d_normals
                );
                gpu_normalize_normals_kernel<<<vert_blocks, threads>>>(
                    d_normals, static_cast<int>(num_verts)
                );
            } else {
                cudaMemset(d_normals, 0, vert_bytes);
            }

            // Compute Z bounds on GPU
            glm::vec3 b_min, b_max;
            float bbox_ms = 0.0f;
            compute_bounding_box_gpu(inout_mesh, b_min, b_max, bbox_ms);

            gpu_displace_mesh_from_bark_texture_kernel<<<vert_blocks, threads>>>(
                d_verts,
                static_cast<int>(num_verts),
                d_normals,
                d_heightmap,
                tex_width,
                tex_height,
                b_min.z,
                b_max.z,
                params.displacement_depth_mm,
                params.tile_repeat_u,
                params.tile_repeat_v
            );

            cudaMemcpy(inout_mesh.vertices.data(), d_verts, vert_bytes, cudaMemcpyDeviceToHost);

            if (d_indices) cudaFree(d_indices);
            cudaFree(d_verts);
            cudaFree(d_normals);
            cudaFree(d_heightmap);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_verts) cudaFree(d_verts);
        if (d_normals) cudaFree(d_normals);
        if (d_heightmap) cudaFree(d_heightmap);
    }
#endif
    return false;
}

} // namespace rhizome
