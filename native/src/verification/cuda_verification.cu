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
__global__ void gpu_translate_vertices_xyz_kernel(
    float* __restrict__ vertices,
    int num_vertices,
    glm::vec3 shift
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;
    vertices[i * 3 + 0] += shift.x;
    vertices[i * 3 + 1] += shift.y;
    vertices[i * 3 + 2] += shift.z;
}
#endif

bool CudaRasterizer::position_mesh_at_x_gpu(
    TriangleMesh& inout_mesh,
    float target_x,
    glm::vec3& out_orig_min,
    glm::vec3& out_orig_max,
    glm::vec3& out_shifted_min,
    glm::vec3& out_shifted_max,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    int num_verts = static_cast<int>(inout_mesh.vertex_count());
    if (num_verts <= 0) {
        out_orig_min = glm::vec3(0.0f);
        out_orig_max = glm::vec3(0.0f);
        out_shifted_min = glm::vec3(0.0f);
        out_shifted_max = glm::vec3(0.0f);
        out_time_ms = 0.0f;
        return false;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        size_t vert_bytes = inout_mesh.vertices.size() * sizeof(float);

        float bbox_ms = 0.0f;
        if (!compute_bounding_box_gpu(inout_mesh, out_orig_min, out_orig_max, bbox_ms)) {
            out_time_ms = 0.0f;
            return false;
        }

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess) {
            cudaMemcpy(d_verts, inout_mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (num_verts + threads - 1) / threads;

            glm::vec3 center = (out_orig_min + out_orig_max) * 0.5f;
            glm::vec3 shift(target_x - center.x, -center.y, -out_orig_min.z);

            gpu_translate_vertices_xyz_kernel<<<blocks, threads>>>(d_verts, num_verts, shift);
            cudaMemcpy(inout_mesh.vertices.data(), d_verts, vert_bytes, cudaMemcpyDeviceToHost);

            out_shifted_min = out_orig_min + shift;
            out_shifted_max = out_orig_max + shift;

            cudaFree(d_verts);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_verts) cudaFree(d_verts);
    }
#endif
    return false;
}

#ifdef __CUDACC__
// --- 30. GPU Kernels for 16-Test Verification & Analytical Suite ---

__device__ inline float dev_exact_tapered_seg_dist(
    const glm::vec3& p,
    const glm::vec3& a,
    const glm::vec3& b,
    float ra,
    float rb
) {
    glm::vec3 ba = b - a;
    glm::vec3 pa = p - a;
    float l2 = glm::dot(ba, ba);
    if (l2 < 1e-12f) {
        return sqrtf(glm::dot(pa, pa)) - fmaxf(ra, rb);
    }
    float t = dev_clamp(glm::dot(pa, ba) / l2, 0.0f, 1.0f);
    glm::vec3 c = a + t * ba;
    float r = ra + (rb - ra) * t;
    glm::vec3 d = p - c;
    return sqrtf(glm::dot(d, d)) - r;
}

__device__ inline float dev_eval_analytical_shape_sdf(
    const glm::vec3& p,
    int shape_type,
    float p0,
    float p1,
    glm::vec3 box_half_extents
) {
    if (shape_type == 0) {
        return sqrtf(glm::dot(p, p)) - p0;
    } else if (shape_type == 1) {
        float qx = sqrtf(p.x * p.x + p.y * p.y) - p0;
        return sqrtf(qx * qx + p.z * p.z) - p1;
    } else if (shape_type == 2) {
        glm::vec3 b = box_half_extents - glm::vec3(p0);
        glm::vec3 q = glm::abs(p) - b;
        glm::vec3 qmax = glm::max(q, glm::vec3(0.0f));
        float d_out = sqrtf(glm::dot(qmax, qmax));
        float d_in = fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
        return d_out + d_in - p0;
    } else {
        float d0 = dev_exact_tapered_seg_dist(p, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 15.0f), 3.0f, 2.5f);
        float d1 = dev_exact_tapered_seg_dist(p, glm::vec3(0.0f, 0.0f, 15.0f), glm::vec3(10.0f, 0.0f, 25.0f), 2.5f, 2.0f);
        float d2 = dev_exact_tapered_seg_dist(p, glm::vec3(10.0f, 0.0f, 25.0f), glm::vec3(20.0f, 10.0f, 30.0f), 2.0f, 1.5f);
        return fminf(d0, fminf(d1, d2));
    }
}

__global__ void gpu_eval_vertex_deviations_kernel(
    const float* __restrict__ vertices,
    int num_verts,
    int shape_type,
    float p0,
    float p1,
    glm::vec3 box_half_extents,
    float* __restrict__ out_max_vert_dev,
    double* __restrict__ out_sum_dev,
    double* __restrict__ out_sum_sq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_verts) return;
    glm::vec3 v(vertices[i * 3 + 0], vertices[i * 3 + 1], vertices[i * 3 + 2]);
    float dev = fabsf(dev_eval_analytical_shape_sdf(v, shape_type, p0, p1, box_half_extents));
    dev_atomicMax_float(out_max_vert_dev, dev);
    atomicAdd(out_sum_dev, static_cast<double>(dev));
    atomicAdd(out_sum_sq, static_cast<double>(dev) * static_cast<double>(dev));
}

__global__ void gpu_eval_midpoint_deviations_kernel(
    const float* __restrict__ vertices,
    const uint32_t* __restrict__ indices,
    int num_tris,
    int shape_type,
    float p0,
    float p1,
    glm::vec3 box_half_extents,
    float* __restrict__ out_max_mid_dev,
    double* __restrict__ out_sum_dev,
    double* __restrict__ out_sum_sq
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    uint32_t i0 = indices[t * 3 + 0];
    uint32_t i1 = indices[t * 3 + 1];
    uint32_t i2 = indices[t * 3 + 2];

    glm::vec3 v0(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
    glm::vec3 v1(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
    glm::vec3 v2(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

    glm::vec3 mid01 = (v0 + v1) * 0.5f;
    glm::vec3 mid12 = (v1 + v2) * 0.5f;
    glm::vec3 mid20 = (v2 + v0) * 0.5f;
    glm::vec3 center = (v0 + v1 + v2) * (1.0f / 3.0f);

    float d01 = fabsf(dev_eval_analytical_shape_sdf(mid01, shape_type, p0, p1, box_half_extents));
    float d12 = fabsf(dev_eval_analytical_shape_sdf(mid12, shape_type, p0, p1, box_half_extents));
    float d20 = fabsf(dev_eval_analytical_shape_sdf(mid20, shape_type, p0, p1, box_half_extents));
    float dc  = fabsf(dev_eval_analytical_shape_sdf(center, shape_type, p0, p1, box_half_extents));

    float m = fmaxf(fmaxf(d01, d12), fmaxf(d20, dc));
    dev_atomicMax_float(out_max_mid_dev, m);

    double s = static_cast<double>(d01) + d12 + d20 + dc;
    double sq = static_cast<double>(d01) * d01 + static_cast<double>(d12) * d12 +
                static_cast<double>(d20) * d20 + static_cast<double>(dc) * dc;
    atomicAdd(out_sum_dev, s);
    atomicAdd(out_sum_sq, sq);
}

__global__ void gpu_flip_triangle_windings_kernel(
    uint32_t* __restrict__ indices,
    size_t start_tri,
    size_t end_tri
) {
    size_t t = start_tri + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= end_tri) return;
    uint32_t tmp = indices[t * 3 + 1];
    indices[t * 3 + 1] = indices[t * 3 + 2];
    indices[t * 3 + 2] = tmp;
}

__global__ void gpu_generate_gwn_test_points_kernel(
    int test_mode,
    uint32_t rng_seed,
    glm::vec3* __restrict__ out_pts
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 10000) return;

    uint32_t s0 = rng_seed + static_cast<uint32_t>(i * 9973 + 17);
    if (test_mode == 0) {
        if (i < 5000) {
            float x = -8.5f + 17.0f * dev_hash_01(s0 + 1u);
            float y = -8.5f + 17.0f * dev_hash_01(s0 + 2u);
            float z = -8.5f + 11.5f * dev_hash_01(s0 + 3u);
            out_pts[i] = glm::vec3(x, y, z);
        } else {
            for (int att = 0; att < 64; ++att) {
                float x = -35.0f + 70.0f * dev_hash_01(s0 + static_cast<uint32_t>(att * 3 + 1));
                float y = -35.0f + 70.0f * dev_hash_01(s0 + static_cast<uint32_t>(att * 3 + 2));
                float z = -35.0f + 70.0f * dev_hash_01(s0 + static_cast<uint32_t>(att * 3 + 3));
                if (!(fabsf(x) <= 12.0f && fabsf(y) <= 12.0f && z >= -12.0f && z <= 12.0f)) {
                    out_pts[i] = glm::vec3(x, y, z);
                    return;
                }
            }
            out_pts[i] = glm::vec3(25.0f, 25.0f, 25.0f);
        }
    } else if (test_mode == 1) {
        float r = (i < 5000)
            ? (13.5f * dev_hash_01(s0 + 1u))
            : (16.5f + 18.5f * dev_hash_01(s0 + 1u));
        float u = -1.0f + 2.0f * dev_hash_01(s0 + 2u);
        float theta = 6.2831853f * dev_hash_01(s0 + 3u);
        float s = sqrtf(fmaxf(0.0f, 1.0f - u * u));
        out_pts[i] = r * glm::vec3(s * cosf(theta), s * sinf(theta), u);
    } else {
        if (i < 5000) {
            float x = -8.5f + 25.0f * dev_hash_01(s0 + 1u);
            float y = -8.5f + 17.0f * dev_hash_01(s0 + 2u);
            float z = -8.5f + 17.0f * dev_hash_01(s0 + 3u);
            out_pts[i] = glm::vec3(x, y, z);
        } else {
            for (int att = 0; att < 64; ++att) {
                float x = -35.0f + 70.0f * dev_hash_01(s0 + static_cast<uint32_t>(att * 3 + 1));
                float y = -35.0f + 70.0f * dev_hash_01(s0 + static_cast<uint32_t>(att * 3 + 2));
                float z = -35.0f + 70.0f * dev_hash_01(s0 + static_cast<uint32_t>(att * 3 + 3));
                bool in_band = (x >= -12.0f && x <= 20.0f && fabsf(y) <= 12.0f && fabsf(z) <= 12.0f);
                if (!in_band) {
                    out_pts[i] = glm::vec3(x, y, z);
                    return;
                }
            }
            out_pts[i] = glm::vec3(28.0f, 28.0f, 28.0f);
        }
    }
}

__global__ void gpu_score_gwn_classifications_kernel(
    const float* __restrict__ wn,
    int test_mode,
    int* __restrict__ out_correct
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 10000) return;

    float in_thresh = (test_mode == 1) ? 0.50f : 0.45f;
    float out_thresh = (test_mode == 0) ? 0.35f : 0.25f;

    bool ok = (i < 5000) ? (wn[i] >= in_thresh) : (wn[i] < out_thresh);
    if (ok) {
        atomicAdd(out_correct, 1);
    }
}

__global__ void gpu_eval_containment_overshoot_kernel(
    const float* __restrict__ vertices,
    int num_verts,
    const float* __restrict__ vessel_sdf,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    float voxel_size,
    float sphere_radius,
    float* __restrict__ out_max_overshoot
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_verts) return;
    glm::vec3 v(vertices[i * 3 + 0], vertices[i * 3 + 1], vertices[i * 3 + 2]);
    float over = 0.0f;
    if (vessel_sdf != nullptr && vessel_dims.x > 0) {
        over = dev_sample_sdf_trilinear(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, v);
    } else {
        over = sqrtf(glm::dot(v, v)) - sphere_radius;
    }
    if (over > 0.0f) {
        dev_atomicMax_float(out_max_overshoot, over);
    }
}

__global__ void gpu_build_round_trip_queries_kernel(
    const float* __restrict__ vertices,
    int num_verts,
    const uint32_t* __restrict__ indices,
    int num_tris,
    glm::vec3* __restrict__ out_queries
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_verts + num_tris * 3;
    if (idx >= total) return;

    if (idx < num_verts) {
        out_queries[idx] = glm::vec3(vertices[idx * 3 + 0], vertices[idx * 3 + 1], vertices[idx * 3 + 2]);
    } else {
        int e_idx = idx - num_verts;
        int t = e_idx / 3;
        int e = e_idx % 3;
        uint32_t i0 = indices[t * 3 + e];
        uint32_t i1 = indices[t * 3 + ((e + 1) % 3)];
        glm::vec3 v0(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
        glm::vec3 v1(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
        out_queries[idx] = (v0 + v1) * 0.5f;
    }
}

__global__ void gpu_reduce_round_trip_devs_kernel(
    const float* __restrict__ dists,
    int num_verts,
    int total_queries,
    float* __restrict__ out_max_vert,
    float* __restrict__ out_max_mid,
    double* __restrict__ out_sum_vert
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_queries) return;
    float d = dists[idx];
    if (idx < num_verts) {
        dev_atomicMax_float(out_max_vert, d);
        atomicAdd(out_sum_vert, static_cast<double>(d));
    } else {
        dev_atomicMax_float(out_max_mid, d);
    }
}

__global__ void gpu_fill_analytical_sdf_grid_kernel(
    float* __restrict__ data,
    glm::ivec3 min_coord,
    glm::ivec3 dims,
    float voxel_size,
    int field_type,
    float p0,
    float p1,
    glm::vec3 origin
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(dims.x) * dims.y * dims.z;
    if (idx >= total) return;

    int x = static_cast<int>(idx % dims.x);
    int y = static_cast<int>((idx / dims.x) % dims.y);
    int z = static_cast<int>(idx / (static_cast<size_t>(dims.x) * dims.y));

    glm::vec3 p = (field_type == 3)
        ? (origin + glm::vec3(x, y, z) * voxel_size)
        : glm::vec3((min_coord.x + x) * voxel_size, (min_coord.y + y) * voxel_size, (min_coord.z + z) * voxel_size);

    float d = 0.0f;
    if (field_type == 0 || field_type == 3) {
        d = sqrtf(glm::dot(p, p)) - p0;
    } else if (field_type == 1) {
        glm::vec3 cA(-p1, 0.0f, 0.0f);
        glm::vec3 cB(+p1, 0.0f, 0.0f);
        float dA = sqrtf(glm::dot(p - cA, p - cA)) - p0;
        float dB = sqrtf(glm::dot(p - cB, p - cB)) - p0;
        d = fminf(dA, dB);
    } else {
        glm::vec3 a0(-10.0f, 0.0f, 0.0f), a1(10.0f, 0.0f, 0.0f);
        glm::vec3 b0(0.0f, -10.0f, 0.0f), b1(0.0f, 10.0f, 0.0f);
        float dA = dev_exact_tapered_seg_dist(p, a0, a1, p0, p0);
        float dB = dev_exact_tapered_seg_dist(p, b0, b1, p0, p0);
        float h_blend = fmaxf(p1 - fabsf(dA - dB), 0.0f) / p1;
        d = fminf(dA, dB) - h_blend * h_blend * p1 * 0.25f;
    }
    data[idx] = d;
}

__global__ void gpu_evaluate_eikonal_field_kernel(
    const float* __restrict__ sdf_data,
    glm::ivec3 min_coord,
    glm::ivec3 dims,
    float dx,
    int sample_mode,
    float p0,
    float p1,
    uint32_t rng_seed,
    int num_samples,
    unsigned int* __restrict__ out_eval_count,
    unsigned int* __restrict__ out_excl_count,
    unsigned int* __restrict__ out_in_tol_count,
    float* __restrict__ out_max_diff,
    double* __restrict__ out_sum_diff,
    double* __restrict__ out_sum_sq_diff
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_samples) return;

    uint32_t s0 = rng_seed + static_cast<uint32_t>(i * 19997 + 31);
    glm::vec3 p(0.0f);
    bool excluded = false;

    if (sample_mode == 0 || sample_mode == 4) {
        float th = 6.2831853f * dev_hash_01(s0 + 1u);
        float cos_p = -1.0f + 2.0f * dev_hash_01(s0 + 2u);
        float sin_p = sqrtf(fmaxf(0.0f, 1.0f - cos_p * cos_p));
        float r = p0 + (-3.0f * dx + 6.0f * dx * dev_hash_01(s0 + 3u));
        p = (sample_mode == 4)
            ? glm::vec3(r * sin_p * cosf(th), r * sin_p * sinf(th), r * cos_p)
            : glm::vec3(r * sin_p * cosf(th), r * cos_p, r * sin_p * sinf(th));
    } else if (sample_mode == 1) {
        for (int att = 0; att < 64; ++att) {
            uint32_t sa = s0 + static_cast<uint32_t>(att * 17);
            glm::vec3 cand(
                -5.0f + 25.0f * dev_hash_01(sa + 1u),
                -5.0f + 19.0f * dev_hash_01(sa + 2u),
                -15.0f + 42.0f * dev_hash_01(sa + 3u)
            );
            float f_val = dev_sample_sdf_trilinear(sdf_data, min_coord, dims, dx, cand);
            if (fabsf(f_val) <= 3.0f * dx) {
                p = cand;
                float d0 = dev_exact_tapered_seg_dist(p, glm::vec3(0.0f, 0.0f, -10.0f), glm::vec3(0.0f, 0.0f, 4.0f), 3.0f, 2.6f);
                float d1 = dev_exact_tapered_seg_dist(p, glm::vec3(0.0f, 0.0f, 4.0f), glm::vec3(8.0f, 0.0f, 14.0f), 2.6f, 2.2f);
                float d2 = dev_exact_tapered_seg_dist(p, glm::vec3(8.0f, 0.0f, 14.0f), glm::vec3(14.0f, 8.0f, 22.0f), 2.2f, 1.8f);
                float m0 = fminf(d0, fminf(d1, d2));
                float m1 = (m0 == d0) ? fminf(d1, d2) : ((m0 == d1) ? fminf(d0, d2) : fminf(d0, d1));
                excluded = (fabsf(m0 - m1) <= 1.0f * dx);
                break;
            }
        }
    } else if (sample_mode == 2) {
        glm::vec3 cA(-p1, 0.0f, 0.0f);
        glm::vec3 cB(+p1, 0.0f, 0.0f);
        for (int att = 0; att < 64; ++att) {
            uint32_t sa = s0 + static_cast<uint32_t>(att * 19);
            glm::vec3 cand(
                -22.0f + 44.0f * dev_hash_01(sa + 1u),
                -16.0f + 32.0f * dev_hash_01(sa + 2u),
                -16.0f + 32.0f * dev_hash_01(sa + 3u)
            );
            float dA = sqrtf(glm::dot(cand - cA, cand - cA)) - p0;
            float dB = sqrtf(glm::dot(cand - cB, cand - cB)) - p0;
            float d_true = fminf(dA, dB);
            if (fabsf(d_true) <= 3.0f * dx) {
                p = cand;
                excluded = (fabsf(dA - dB) <= 1.0f * dx);
                break;
            }
        }
    } else {
        for (int att = 0; att < 64; ++att) {
            uint32_t sa = s0 + static_cast<uint32_t>(att * 23);
            glm::vec3 cand(
                -13.0f + 26.0f * dev_hash_01(sa + 1u),
                -13.0f + 26.0f * dev_hash_01(sa + 2u),
                -6.0f + 12.0f * dev_hash_01(sa + 3u)
            );
            float f_val = dev_sample_sdf_trilinear(sdf_data, min_coord, dims, dx, cand);
            if (fabsf(f_val) <= 3.0f * dx) {
                p = cand;
                break;
            }
        }
    }

    if (excluded) {
        atomicAdd(out_excl_count, 1u);
        return;
    }

    atomicAdd(out_eval_count, 1u);
    float h = 0.5f * dx;
    float inv_2h = 1.0f / (2.0f * h);
    float fx_p = dev_sample_sdf_trilinear(sdf_data, min_coord, dims, dx, glm::vec3(p.x + h, p.y, p.z));
    float fx_m = dev_sample_sdf_trilinear(sdf_data, min_coord, dims, dx, glm::vec3(p.x - h, p.y, p.z));
    float fy_p = dev_sample_sdf_trilinear(sdf_data, min_coord, dims, dx, glm::vec3(p.x, p.y + h, p.z));
    float fy_m = dev_sample_sdf_trilinear(sdf_data, min_coord, dims, dx, glm::vec3(p.x, p.y - h, p.z));
    float fz_p = dev_sample_sdf_trilinear(sdf_data, min_coord, dims, dx, glm::vec3(p.x, p.y, p.z + h));
    float fz_m = dev_sample_sdf_trilinear(sdf_data, min_coord, dims, dx, glm::vec3(p.x, p.y, p.z - h));

    float gx = (fx_p - fx_m) * inv_2h;
    float gy = (fy_p - fy_m) * inv_2h;
    float gz = (fz_p - fz_m) * inv_2h;
    float g_mag = sqrtf(gx * gx + gy * gy + gz * gz);
    float diff = fabsf(g_mag - 1.0f);

    dev_atomicMax_float(out_max_diff, diff);
    atomicAdd(out_sum_diff, static_cast<double>(diff));
    atomicAdd(out_sum_sq_diff, static_cast<double>(diff) * static_cast<double>(diff));

    float tol = (sample_mode == 4) ? 0.001f : 0.02f;
    if (diff <= tol) {
        atomicAdd(out_in_tol_count, 1u);
    }
}

__global__ void gpu_eval_sdf_grid_error_kernel(
    const float* __restrict__ gpu_dists,
    glm::ivec3 min_coord,
    int dim,
    float voxel_size,
    float radius,
    float* __restrict__ out_max_err
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim * dim * dim;
    if (idx >= total) return;

    int x = idx % dim;
    int y = (idx / dim) % dim;
    int z = idx / (dim * dim);

    glm::vec3 p((min_coord.x + x) * voxel_size, (min_coord.y + y) * voxel_size, (min_coord.z + z) * voxel_size);
    float analytical_d = sqrtf(glm::dot(p, p)) - radius;
    float err = fabsf(analytical_d - gpu_dists[idx]);
    dev_atomicMax_float(out_max_err, err);
}

__global__ void gpu_init_test_spores_and_nodes_kernel(
    CudaSporeData* __restrict__ spores,
    int num_spores,
    CudaBranchNode* __restrict__ nodes,
    int num_nodes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_spores) {
        uint32_t s = 42u + static_cast<uint32_t>(i * 7919);
        spores[i].pos = glm::vec3(
            -20.0f + 40.0f * dev_hash_01(s + 1u),
            -20.0f + 40.0f * dev_hash_01(s + 2u),
            -20.0f + 40.0f * dev_hash_01(s + 3u)
        );
        spores[i].alive = true;
        spores[i].closest_node_idx = -1;
        spores[i].min_dist = 1e6f;
    }
    if (i < num_nodes) {
        uint32_t s = 1337u + static_cast<uint32_t>(i * 4999);
        nodes[i].pos = glm::vec3(
            -10.0f + 20.0f * dev_hash_01(s + 1u),
            -10.0f + 20.0f * dev_hash_01(s + 2u),
            -10.0f + 20.0f * dev_hash_01(s + 3u)
        );
        nodes[i].radius = 1.0f;
    }
}

__global__ void gpu_count_alive_spores_kernel(
    const CudaSporeData* __restrict__ spores,
    int num_spores,
    unsigned int* __restrict__ out_alive
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_spores) return;
    if (spores[i].alive) {
        atomicAdd(out_alive, 1u);
    }
}

__global__ void gpu_init_test_50_capsules_kernel(
    CudaCapsule* __restrict__ capsules,
    int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    CudaCapsule c;
    c.ax = -20.0f + i * 0.8f;
    c.ay = 0.0f;
    c.az = 0.0f;
    c.bx = -20.0f + (i + 1) * 0.8f;
    c.by = sinf(i * 0.2f) * 3.0f;
    c.bz = cosf(i * 0.2f) * 3.0f;
    c.ra = 1.5f;
    c.rb = 1.2f;
    c.min_x = fminf(c.ax, c.bx) - 2.0f;
    c.max_x = fmaxf(c.ax, c.bx) + 2.0f;
    c.min_y = fminf(c.ay, c.by) - 2.0f;
    c.max_y = fmaxf(c.ay, c.by) + 2.0f;
    c.min_z = fminf(c.az, c.bz) - 2.0f;
    c.max_z = fmaxf(c.az, c.bz) + 2.0f;
    capsules[i] = c;
}

__global__ void gpu_verification_build_cuda_triangles_kernel(
    const float* __restrict__ vertices,
    int num_verts,
    const uint32_t* __restrict__ indices,
    int num_tris,
    CudaTriangle* __restrict__ out_tris,
    int* __restrict__ out_valid_count
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;
    CudaTriangle tri;
    if (dev_try_build_cuda_triangle(vertices, num_verts, indices, t, tri)) {
        int slot = atomicAdd(out_valid_count, 1);
        out_tris[slot] = tri;
    }
}
#endif

bool CudaRasterizer::evaluate_mesh_deviations_gpu(
    const TriangleMesh& mesh,
    int shape_type,
    float p0,
    float p1,
    glm::vec3 box_half_extents,
    size_t& out_sample_count,
    float& out_vertex_max,
    float& out_midpoint_max,
    float& out_mean,
    float& out_rms
) {
    out_sample_count = 0;
    out_vertex_max = 0.0f;
    out_midpoint_max = 0.0f;
    out_mean = 0.0f;
    out_rms = 0.0f;

    int num_verts = static_cast<int>(mesh.vertex_count());
    int num_tris = static_cast<int>(mesh.triangle_count());
    if (num_verts == 0 || num_tris == 0) return false;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        float* d_max_vals = nullptr;
        double* d_sums = nullptr;

        if (cudaMalloc(&d_verts, mesh.vertices.size() * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_indices, mesh.indices.size() * sizeof(uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_max_vals, 2 * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_sums, 2 * sizeof(double)) == cudaSuccess) {

            cudaMemcpy(d_verts, mesh.vertices.data(), mesh.vertices.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemset(d_max_vals, 0, 2 * sizeof(float));
            cudaMemset(d_sums, 0, 2 * sizeof(double));

            int threads = 256;
            int v_blocks = (num_verts + threads - 1) / threads;
            int t_blocks = (num_tris + threads - 1) / threads;

            gpu_eval_vertex_deviations_kernel<<<v_blocks, threads>>>(
                d_verts, num_verts, shape_type, p0, p1, box_half_extents,
                &d_max_vals[0], &d_sums[0], &d_sums[1]
            );
            gpu_eval_midpoint_deviations_kernel<<<t_blocks, threads>>>(
                d_verts, d_indices, num_tris, shape_type, p0, p1, box_half_extents,
                &d_max_vals[1], &d_sums[0], &d_sums[1]
            );

            float h_max[2] = { 0.0f, 0.0f };
            double h_sums[2] = { 0.0, 0.0 };
            cudaMemcpy(h_max, d_max_vals, 2 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_sums, d_sums, 2 * sizeof(double), cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            cudaFree(d_indices);
            cudaFree(d_max_vals);
            cudaFree(d_sums);

            out_sample_count = static_cast<size_t>(num_verts) + static_cast<size_t>(num_tris) * 4;
            out_vertex_max = h_max[0];
            out_midpoint_max = h_max[1];
            out_mean = static_cast<float>(h_sums[0] / static_cast<double>(out_sample_count));
            out_rms = static_cast<float>(std::sqrt(h_sums[1] / static_cast<double>(out_sample_count)));
            return true;
        }
        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
        if (d_max_vals) cudaFree(d_max_vals);
        if (d_sums) cudaFree(d_sums);
    }
#endif
    return false;
}

bool CudaRasterizer::flip_triangle_windings_gpu(
    TriangleMesh& inout_mesh,
    size_t start_tri,
    size_t end_tri
) {
    size_t n_tris = inout_mesh.triangle_count();
    end_tri = std::min(end_tri, n_tris);
    if (start_tri >= end_tri) return true;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        uint32_t* d_indices = nullptr;
        size_t bytes = inout_mesh.indices.size() * sizeof(uint32_t);
        if (cudaMalloc(&d_indices, bytes) == cudaSuccess) {
            cudaMemcpy(d_indices, inout_mesh.indices.data(), bytes, cudaMemcpyHostToDevice);
            int count = static_cast<int>(end_tri - start_tri);
            int threads = 256;
            int blocks = (count + threads - 1) / threads;
            gpu_flip_triangle_windings_kernel<<<blocks, threads>>>(d_indices, start_tri, end_tri);
            cudaMemcpy(inout_mesh.indices.data(), d_indices, bytes, cudaMemcpyDeviceToHost);
            cudaFree(d_indices);
            return true;
        }
    }
#endif
    return false;
}

bool CudaRasterizer::evaluate_gwn_classification_gpu(
    const TriangleMesh& mesh,
    int test_mode,
    uint32_t rng_seed,
    int& out_correct_count
) {
    out_correct_count = 0;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        glm::vec3* d_pts = nullptr;
        if (cudaMalloc(&d_pts, 10000 * sizeof(glm::vec3)) == cudaSuccess) {
            int threads = 256;
            int blocks = (10000 + threads - 1) / threads;
            gpu_generate_gwn_test_points_kernel<<<blocks, threads>>>(test_mode, rng_seed, d_pts);

            std::vector<glm::vec3> pts(10000);
            cudaMemcpy(pts.data(), d_pts, 10000 * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
            cudaFree(d_pts);

            std::vector<float> wn;
            float gwn_ms = 0.0f;
            if (!compute_winding_numbers_gpu(mesh, pts, wn, gwn_ms)) return false;

            float* d_wn = nullptr;
            int* d_correct = nullptr;
            if (cudaMalloc(&d_wn, 10000 * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_correct, sizeof(int)) == cudaSuccess) {
                cudaMemcpy(d_wn, wn.data(), 10000 * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemset(d_correct, 0, sizeof(int));
                gpu_score_gwn_classifications_kernel<<<blocks, threads>>>(d_wn, test_mode, d_correct);
                cudaMemcpy(&out_correct_count, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
                cudaFree(d_wn);
                cudaFree(d_correct);
                return true;
            }
            if (d_wn) cudaFree(d_wn);
            if (d_correct) cudaFree(d_correct);
        }
    }
#endif
    return false;
}

bool CudaRasterizer::evaluate_containment_overshoot_gpu(
    const TriangleMesh& mesh,
    const CudaVolumeGrid* vessel_grid,
    float sphere_radius,
    float& out_max_overshoot
) {
    out_max_overshoot = 0.0f;
    int num_verts = static_cast<int>(mesh.vertex_count());
    if (num_verts == 0) return true;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        float* d_sdf = nullptr;
        float* d_max = nullptr;

        bool has_grid = (vessel_grid != nullptr && vessel_grid->is_valid());
        bool ok = (cudaMalloc(&d_verts, mesh.vertices.size() * sizeof(float)) == cudaSuccess) &&
                  (cudaMalloc(&d_max, sizeof(float)) == cudaSuccess);
        if (ok && has_grid) {
            ok = (cudaMalloc(&d_sdf, vessel_grid->data.size() * sizeof(float)) == cudaSuccess);
        }

        if (ok) {
            cudaMemcpy(d_verts, mesh.vertices.data(), mesh.vertices.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_max, 0, sizeof(float));
            if (has_grid) {
                cudaMemcpy(d_sdf, vessel_grid->data.data(), vessel_grid->data.size() * sizeof(float), cudaMemcpyHostToDevice);
            }

            int threads = 256;
            int blocks = (num_verts + threads - 1) / threads;
            gpu_eval_containment_overshoot_kernel<<<blocks, threads>>>(
                d_verts, num_verts,
                d_sdf,
                has_grid ? vessel_grid->min_coord : glm::ivec3(0),
                has_grid ? vessel_grid->dims : glm::ivec3(0),
                has_grid ? vessel_grid->voxel_size : 1.0f,
                sphere_radius,
                d_max
            );

            cudaMemcpy(&out_max_overshoot, d_max, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_verts);
            if (d_sdf) cudaFree(d_sdf);
            cudaFree(d_max);
            return true;
        }
        if (d_verts) cudaFree(d_verts);
        if (d_sdf) cudaFree(d_sdf);
        if (d_max) cudaFree(d_max);
    }
#endif
    return false;
}

bool CudaRasterizer::evaluate_round_trip_precision_gpu(
    const TriangleMesh& reference_mesh,
    const TriangleMesh& test_mesh,
    float& out_max_vert_dev,
    float& out_max_mid_dev,
    float& out_mean_vert_dev,
    size_t& out_total_samples
) {
    out_max_vert_dev = 0.0f;
    out_max_mid_dev = 0.0f;
    out_mean_vert_dev = 0.0f;
    out_total_samples = 0;

    int num_verts = static_cast<int>(test_mesh.vertex_count());
    int num_tris = static_cast<int>(test_mesh.triangle_count());
    int total_queries = num_verts + num_tris * 3;
    if (num_verts == 0 || total_queries == 0) return false;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        glm::vec3* d_queries = nullptr;

        if (cudaMalloc(&d_verts, test_mesh.vertices.size() * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_indices, test_mesh.indices.size() * sizeof(uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_queries, total_queries * sizeof(glm::vec3)) == cudaSuccess) {

            cudaMemcpy(d_verts, test_mesh.vertices.data(), test_mesh.vertices.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, test_mesh.indices.data(), test_mesh.indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

            int threads = 256;
            int q_blocks = (total_queries + threads - 1) / threads;
            gpu_build_round_trip_queries_kernel<<<q_blocks, threads>>>(d_verts, num_verts, d_indices, num_tris, d_queries);

            std::vector<glm::vec3> query_pts(total_queries);
            cudaMemcpy(query_pts.data(), d_queries, total_queries * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            cudaFree(d_indices);
            cudaFree(d_queries);

            std::vector<float> gpu_distances;
            float q_ms = 0.0f;
            if (!query_distance_batch_gpu(reference_mesh, query_pts, gpu_distances, q_ms)) return false;

            float* d_dists = nullptr;
            float* d_max = nullptr;
            double* d_sum = nullptr;
            if (cudaMalloc(&d_dists, total_queries * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_max, 2 * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_sum, sizeof(double)) == cudaSuccess) {

                cudaMemcpy(d_dists, gpu_distances.data(), total_queries * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemset(d_max, 0, 2 * sizeof(float));
                cudaMemset(d_sum, 0, sizeof(double));

                gpu_reduce_round_trip_devs_kernel<<<q_blocks, threads>>>(
                    d_dists, num_verts, total_queries, &d_max[0], &d_max[1], d_sum
                );

                float h_max[2] = { 0.0f, 0.0f };
                double h_sum = 0.0;
                cudaMemcpy(h_max, d_max, 2 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

                cudaFree(d_dists);
                cudaFree(d_max);
                cudaFree(d_sum);

                out_max_vert_dev = h_max[0];
                out_max_mid_dev = h_max[1];
                out_mean_vert_dev = static_cast<float>(h_sum / static_cast<double>(num_verts));
                out_total_samples = static_cast<size_t>(total_queries);
                return true;
            }
            if (d_dists) cudaFree(d_dists);
            if (d_max) cudaFree(d_max);
            if (d_sum) cudaFree(d_sum);
        }
        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
        if (d_queries) cudaFree(d_queries);
    }
#endif
    return false;
}

bool CudaRasterizer::fill_analytical_sdf_grid_gpu(
    CudaVolumeGrid& grid,
    int field_type,
    float p0,
    float p1,
    glm::vec3 origin
) {
    size_t total = static_cast<size_t>(grid.dims.x) * grid.dims.y * grid.dims.z;
    if (total == 0) return false;
    grid.data.resize(total);
#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_data = nullptr;
        if (cudaMalloc(&d_data, total * sizeof(float)) == cudaSuccess) {
            int threads = 256;
            int blocks = static_cast<int>((total + threads - 1) / threads);
            gpu_fill_analytical_sdf_grid_kernel<<<blocks, threads>>>(
                d_data, grid.min_coord, grid.dims, grid.voxel_size, field_type, p0, p1, origin
            );
            cudaMemcpy(grid.data.data(), d_data, total * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_data);
            return true;
        }
    }
#endif
    return false;
}

bool CudaRasterizer::evaluate_eikonal_field_gpu(
    const CudaVolumeGrid& grid,
    int sample_mode,
    float p0,
    float p1,
    uint32_t rng_seed,
    size_t num_target_samples,
    size_t& out_evaluated_count,
    size_t& out_excluded_count,
    size_t& out_in_tol_count,
    float& out_fraction_in_tol,
    float& out_max_diff,
    float& out_mean_diff,
    float& out_rms_diff
) {
    out_evaluated_count = 0;
    out_excluded_count = 0;
    out_in_tol_count = 0;
    out_fraction_in_tol = 0.0f;
    out_max_diff = 0.0f;
    out_mean_diff = 0.0f;
    out_rms_diff = 0.0f;
    if (!grid.is_valid() || num_target_samples == 0) return false;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_sdf = nullptr;
        unsigned int* d_cnt = nullptr;
        float* d_max = nullptr;
        double* d_sums = nullptr;

        if (cudaMalloc(&d_sdf, grid.data.size() * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_cnt, 3 * sizeof(unsigned int)) == cudaSuccess &&
            cudaMalloc(&d_max, sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_sums, 2 * sizeof(double)) == cudaSuccess) {

            cudaMemcpy(d_sdf, grid.data.data(), grid.data.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_cnt, 0, 3 * sizeof(unsigned int));
            cudaMemset(d_max, 0, sizeof(float));
            cudaMemset(d_sums, 0, 2 * sizeof(double));

            int n = static_cast<int>(num_target_samples);
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            gpu_evaluate_eikonal_field_kernel<<<blocks, threads>>>(
                d_sdf, grid.min_coord, grid.dims, grid.voxel_size,
                sample_mode, p0, p1, rng_seed, n,
                &d_cnt[0], &d_cnt[1], &d_cnt[2], d_max, &d_sums[0], &d_sums[1]
            );

            unsigned int h_cnt[3] = { 0, 0, 0 };
            float h_max = 0.0f;
            double h_sums[2] = { 0.0, 0.0 };
            cudaMemcpy(h_cnt, d_cnt, 3 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_sums, d_sums, 2 * sizeof(double), cudaMemcpyDeviceToHost);

            cudaFree(d_sdf);
            cudaFree(d_cnt);
            cudaFree(d_max);
            cudaFree(d_sums);

            out_evaluated_count = h_cnt[0];
            out_excluded_count = h_cnt[1];
            out_in_tol_count = h_cnt[2];
            out_max_diff = h_max;
            if (out_evaluated_count > 0) {
                out_fraction_in_tol = static_cast<float>(out_in_tol_count) / static_cast<float>(out_evaluated_count);
                out_mean_diff = static_cast<float>(h_sums[0] / static_cast<double>(out_evaluated_count));
                out_rms_diff = static_cast<float>(std::sqrt(h_sums[1] / static_cast<double>(out_evaluated_count)));
            }
            return true;
        }
        if (d_sdf) cudaFree(d_sdf);
        if (d_cnt) cudaFree(d_cnt);
        if (d_max) cudaFree(d_max);
        if (d_sums) cudaFree(d_sums);
    }
#endif
    return false;
}

bool CudaRasterizer::evaluate_cuda_5point_helpers_gpu(
    const TriangleMesh& sphere_mesh,
    float& out_sdf_max_err,
    float& out_sdf_time_ms,
    size_t& out_active_spores,
    float& out_spore_time_ms,
    float& out_capsule_min_d,
    float& out_capsule_time_ms,
    size_t& out_capsule_voxels
) {
    out_sdf_max_err = 0.0f;
    out_sdf_time_ms = 0.0f;
    out_active_spores = 0;
    out_spore_time_ms = 0.0f;
    out_capsule_min_d = 0.0f;
    out_capsule_time_ms = 0.0f;
    out_capsule_voxels = 0;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        int num_verts = static_cast<int>(sphere_mesh.vertex_count());
        int num_tris = static_cast<int>(sphere_mesh.triangle_count());
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        CudaTriangle* d_tris = nullptr;
        int* d_valid_count = nullptr;

        std::vector<CudaTriangle> h_tris;
        if (cudaMalloc(&d_verts, sphere_mesh.vertices.size() * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_indices, sphere_mesh.indices.size() * sizeof(uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_tris, num_tris * sizeof(CudaTriangle)) == cudaSuccess &&
            cudaMalloc(&d_valid_count, sizeof(int)) == cudaSuccess) {

            cudaMemcpy(d_verts, sphere_mesh.vertices.data(), sphere_mesh.vertices.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, sphere_mesh.indices.data(), sphere_mesh.indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemset(d_valid_count, 0, sizeof(int));

            int threads = 256;
            int t_blocks = (num_tris + threads - 1) / threads;
            gpu_verification_build_cuda_triangles_kernel<<<t_blocks, threads>>>(d_verts, num_verts, d_indices, num_tris, d_tris, d_valid_count);

            int h_valid = 0;
            cudaMemcpy(&h_valid, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_valid > 0) {
                h_tris.resize(h_valid);
                cudaMemcpy(h_tris.data(), d_tris, static_cast<size_t>(h_valid) * sizeof(CudaTriangle), cudaMemcpyDeviceToHost);
            }

            cudaFree(d_verts);
            cudaFree(d_indices);
            cudaFree(d_tris);
            cudaFree(d_valid_count);
        } else {
            if (d_verts) cudaFree(d_verts);
            if (d_indices) cudaFree(d_indices);
            if (d_tris) cudaFree(d_tris);
            if (d_valid_count) cudaFree(d_valid_count);
        }

        std::vector<float> distances;
        glm::ivec3 min_coord(-18, -18, -18);
        glm::ivec3 max_coord(18, 18, 18);
        compute_mesh_sdf(h_tris, 1.0f, min_coord, max_coord, distances, out_sdf_time_ms);

        if (!distances.empty()) {
            float* d_dists = nullptr;
            float* d_err = nullptr;
            if (cudaMalloc(&d_dists, distances.size() * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&d_err, sizeof(float)) == cudaSuccess) {
                cudaMemcpy(d_dists, distances.data(), distances.size() * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemset(d_err, 0, sizeof(float));
                int threads = 256;
                int blocks = static_cast<int>((distances.size() + threads - 1) / threads);
                gpu_eval_sdf_grid_error_kernel<<<blocks, threads>>>(d_dists, min_coord, 37, 1.0f, 15.0f, d_err);
                cudaMemcpy(&out_sdf_max_err, d_err, sizeof(float), cudaMemcpyDeviceToHost);
                cudaFree(d_dists);
                cudaFree(d_err);
            }
        }

        const int num_spores = 25000;
        const int num_nodes = 200;
        std::vector<CudaSporeData> spores(num_spores);
        std::vector<CudaBranchNode> nodes(num_nodes);
        {
            CudaSporeData* d_sp = nullptr;
            CudaBranchNode* d_nd = nullptr;
            if (cudaMalloc(&d_sp, num_spores * sizeof(CudaSporeData)) == cudaSuccess &&
                cudaMalloc(&d_nd, num_nodes * sizeof(CudaBranchNode)) == cudaSuccess) {
                int threads = 256;
                int blocks = (num_spores + threads - 1) / threads;
                gpu_init_test_spores_and_nodes_kernel<<<blocks, threads>>>(d_sp, num_spores, d_nd, num_nodes);
                cudaMemcpy(spores.data(), d_sp, num_spores * sizeof(CudaSporeData), cudaMemcpyDeviceToHost);
                cudaMemcpy(nodes.data(), d_nd, num_nodes * sizeof(CudaBranchNode), cudaMemcpyDeviceToHost);
                cudaFree(d_sp);
                cudaFree(d_nd);
            }
        }

        std::vector<glm::vec3> attraction_forces;
        std::vector<int> attraction_counts;
        evaluate_spores(spores, nodes, 12.0f, 2.0f, attraction_forces, attraction_counts, out_spore_time_ms);
        {
            CudaSporeData* d_sp = nullptr;
            unsigned int* d_alive = nullptr;
            if (cudaMalloc(&d_sp, num_spores * sizeof(CudaSporeData)) == cudaSuccess &&
                cudaMalloc(&d_alive, sizeof(unsigned int)) == cudaSuccess) {
                cudaMemcpy(d_sp, spores.data(), num_spores * sizeof(CudaSporeData), cudaMemcpyHostToDevice);
                cudaMemset(d_alive, 0, sizeof(unsigned int));
                int threads = 256;
                int blocks = (num_spores + threads - 1) / threads;
                gpu_count_alive_spores_kernel<<<blocks, threads>>>(d_sp, num_spores, d_alive);
                unsigned int h_alive = 0;
                cudaMemcpy(&h_alive, d_alive, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                out_active_spores = h_alive;
                cudaFree(d_sp);
                cudaFree(d_alive);
            }
        }

        std::vector<CudaCapsule> capsules(50);
        {
            CudaCapsule* d_caps = nullptr;
            if (cudaMalloc(&d_caps, 50 * sizeof(CudaCapsule)) == cudaSuccess) {
                gpu_init_test_50_capsules_kernel<<<1, 64>>>(d_caps, 50);
                cudaMemcpy(capsules.data(), d_caps, 50 * sizeof(CudaCapsule), cudaMemcpyDeviceToHost);
                cudaFree(d_caps);
            }
        }
        CudaVolumeGrid cap_grid;
        cap_grid.voxel_size = 0.5f;
        cap_grid.min_coord = glm::ivec3(-25, -10, -10);
        cap_grid.max_coord = glm::ivec3(25, 10, 10);
        cap_grid.dims = cap_grid.max_coord - cap_grid.min_coord + glm::ivec3(1);
        rasterize(capsules, 0.5f, 100.0f, cap_grid.min_coord, cap_grid.max_coord, cap_grid.data, out_capsule_time_ms);
        out_capsule_voxels = cap_grid.data.size();
        float min_ms = 0.0f;
        compute_min_sdf_gpu(cap_grid, out_capsule_min_d, min_ms);

        return true;
    }
#endif
    return false;
}

} // namespace rhizome
