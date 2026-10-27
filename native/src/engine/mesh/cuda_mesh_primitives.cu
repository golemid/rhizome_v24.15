#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/cuda_common.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

namespace rhizome {

#ifdef __CUDACC__
// GPU Sphere Kernels
__global__ void gpu_sphere_vertices_kernel(
    float* vertices, float radius, int rings, int sectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_interior = (rings - 1) * sectors;
    if (idx >= total_interior) return;

    int r = 1 + idx / sectors;
    int s = idx % sectors;

    float phi = 3.141592653589793f * (static_cast<float>(r) / static_cast<float>(rings));
    float theta = 2.0f * 3.141592653589793f * (static_cast<float>(s) / static_cast<float>(sectors));

    int out_v = 1 + idx;
    vertices[out_v * 3 + 0] = radius * sinf(phi) * cosf(theta);
    vertices[out_v * 3 + 1] = radius * cosf(phi);
    vertices[out_v * 3 + 2] = radius * sinf(phi) * sinf(theta);
}

__global__ void gpu_sphere_indices_kernel(
    uint32_t* indices, int rings, int sectors, uint32_t north_idx, uint32_t south_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // North pole fan: sectors triangles
    if (idx < sectors) {
        int s = idx;
        uint32_t next = (s + 1) % sectors;
        indices[s * 3 + 0] = north_idx;
        indices[s * 3 + 1] = 1 + next;
        indices[s * 3 + 2] = 1 + s;
        return;
    }

    // Body quads: (rings - 2) * sectors quads (each 2 triangles)
    int body_idx = idx - sectors;
    int total_body_quads = (rings - 2) * sectors;
    if (body_idx < total_body_quads) {
        int r = body_idx / sectors;
        int s = body_idx % sectors;
        uint32_t s_next = (s + 1) % sectors;
        uint32_t r0 = 1 + r * sectors;
        uint32_t r1 = 1 + (r + 1) * sectors;

        int tri_base = sectors + body_idx * 2;
        indices[tri_base * 3 + 0] = r0 + s;
        indices[tri_base * 3 + 1] = r0 + s_next;
        indices[tri_base * 3 + 2] = r1 + s;

        indices[(tri_base + 1) * 3 + 0] = r0 + s_next;
        indices[(tri_base + 1) * 3 + 1] = r1 + s_next;
        indices[(tri_base + 1) * 3 + 2] = r1 + s;
        return;
    }

    // South pole fan: sectors triangles
    int south_fan_idx = idx - (sectors + total_body_quads);
    if (south_fan_idx < sectors) {
        int s = south_fan_idx;
        uint32_t next = (s + 1) % sectors;
        uint32_t last_ring_start = 1 + (rings - 2) * sectors;
        int tri_base = sectors + total_body_quads * 2 + s;
        indices[tri_base * 3 + 0] = south_idx;
        indices[tri_base * 3 + 1] = last_ring_start + s;
        indices[tri_base * 3 + 2] = last_ring_start + next;
    }
}

// GPU Torus Kernels
__global__ void gpu_torus_vertices_kernel(
    float* vertices, float major_r, float minor_r, int rings, int sectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_v = (rings + 1) * (sectors + 1);
    if (idx >= total_v) return;

    int r = idx / (sectors + 1);
    int s = idx % (sectors + 1);

    float u = 2.0f * 3.141592653589793f * (static_cast<float>(r) / static_cast<float>(rings));
    float v = 2.0f * 3.141592653589793f * (static_cast<float>(s) / static_cast<float>(sectors));

    vertices[idx * 3 + 0] = (major_r + minor_r * cosf(v)) * cosf(u);
    vertices[idx * 3 + 1] = (major_r + minor_r * cosf(v)) * sinf(u);
    vertices[idx * 3 + 2] = minor_r * sinf(v);
}

__global__ void gpu_torus_indices_kernel(
    uint32_t* indices, int rings, int sectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_quads = rings * sectors;
    if (idx >= total_quads) return;

    int r = idx / sectors;
    int s = idx % sectors;

    uint32_t cur = r * (sectors + 1) + s;
    uint32_t next = cur + sectors + 1;

    int tri_base = idx * 2;
    indices[tri_base * 3 + 0] = cur;
    indices[tri_base * 3 + 1] = next;
    indices[tri_base * 3 + 2] = cur + 1;

    indices[(tri_base + 1) * 3 + 0] = cur + 1;
    indices[(tri_base + 1) * 3 + 1] = next;
    indices[(tri_base + 1) * 3 + 2] = next + 1;
}

// GPU Filleted Box Kernels
__global__ void gpu_box_vertices_kernel(
    float* vertices, float hx, float hy, float hz, float fillet_r, int subd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int verts_per_face = (subd + 1) * (subd + 1);
    int total_verts = 6 * verts_per_face;
    if (idx >= total_verts) return;

    int face = idx / verts_per_face;
    int face_v = idx % verts_per_face;
    int i = face_v / (subd + 1);
    int j = face_v % (subd + 1);

    float u = (static_cast<float>(i) / static_cast<float>(subd)) * 2.0f - 1.0f;
    float v = (static_cast<float>(j) / static_cast<float>(subd)) * 2.0f - 1.0f;

    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (face == 0) { px =  1.0f; py =  u;    pz =  v; }
    if (face == 1) { px = -1.0f; py = -u;    pz =  v; }
    if (face == 2) { px = -u;    py =  1.0f; pz =  v; }
    if (face == 3) { px =  u;    py = -1.0f; pz =  v; }
    if (face == 4) { px =  u;    py =  v;    pz =  1.0f; }
    if (face == 5) { px = -u;    py =  v;    pz = -1.0f; }

    float raw_x = px * hx;
    float raw_y = py * hy;
    float raw_z = pz * hz;

    float bx = hx - fillet_r;
    float by = hy - fillet_r;
    float bz = hz - fillet_r;

    float cx = fmaxf(-bx, fminf(bx, raw_x));
    float cy = fmaxf(-by, fminf(by, raw_y));
    float cz = fmaxf(-bz, fminf(bz, raw_z));

    float diff_x = raw_x - cx;
    float diff_y = raw_y - cy;
    float diff_z = raw_z - cz;
    float d_len = sqrtf(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

    if (d_len > 1e-6f) {
        vertices[idx * 3 + 0] = cx + (diff_x / d_len) * fillet_r;
        vertices[idx * 3 + 1] = cy + (diff_y / d_len) * fillet_r;
        vertices[idx * 3 + 2] = cz + (diff_z / d_len) * fillet_r;
    } else {
        vertices[idx * 3 + 0] = raw_x;
        vertices[idx * 3 + 1] = raw_y;
        vertices[idx * 3 + 2] = raw_z;
    }
}

__global__ void gpu_box_indices_kernel(
    uint32_t* indices, int subd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int quads_per_face = subd * subd;
    int total_quads = 6 * quads_per_face;
    if (idx >= total_quads) return;

    int face = idx / quads_per_face;
    int quad_idx = idx % quads_per_face;
    int i = quad_idx / subd;
    int j = quad_idx % subd;

    uint32_t base_v = face * (subd + 1) * (subd + 1);
    uint32_t i0 = base_v + i * (subd + 1) + j;
    uint32_t i1 = base_v + (i + 1) * (subd + 1) + j;
    uint32_t i2 = base_v + (i + 1) * (subd + 1) + (j + 1);
    uint32_t i3 = base_v + i * (subd + 1) + (j + 1);

    int tri_base = idx * 2;
    indices[tri_base * 3 + 0] = i0;
    indices[tri_base * 3 + 1] = i1;
    indices[tri_base * 3 + 2] = i2;

    indices[(tri_base + 1) * 3 + 0] = i0;
    indices[(tri_base + 1) * 3 + 1] = i2;
    indices[(tri_base + 1) * 3 + 2] = i3;
}
#endif

TriangleMesh CudaRasterizer::generate_sphere_gpu(float radius, int rings, int sectors) {
    TriangleMesh mesh;
    size_t total_v = 2 + (rings - 1) * sectors;
    size_t total_tris = sectors * 2 + (rings - 2) * sectors * 2;

    mesh.vertices.resize(total_v * 3);
    mesh.indices.resize(total_tris * 3);

    mesh.vertices[0] = 0.0f;
    mesh.vertices[1] = radius;
    mesh.vertices[2] = 0.0f;

    uint32_t south_idx = static_cast<uint32_t>(total_v - 1);
    mesh.vertices[south_idx * 3 + 0] = 0.0f;
    mesh.vertices[south_idx * 3 + 1] = -radius;
    mesh.vertices[south_idx * 3 + 2] = 0.0f;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        size_t vert_bytes = total_v * 3 * sizeof(float);
        size_t idx_bytes = total_tris * 3 * sizeof(uint32_t);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess) {

            int interior_v = (rings - 1) * sectors;
            int threads = 256;
            int blocks = (interior_v + threads - 1) / threads;
            gpu_sphere_vertices_kernel<<<blocks, threads>>>(d_verts, radius, rings, sectors);

            int total_quad_items = sectors + (rings - 2) * sectors + sectors;
            int idx_blocks = (total_quad_items + threads - 1) / threads;
            gpu_sphere_indices_kernel<<<idx_blocks, threads>>>(d_indices, rings, sectors, 0, south_idx);

            cudaMemcpy(mesh.vertices.data() + 3, d_verts + 3, (total_v - 2) * 3 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(mesh.indices.data(), d_indices, idx_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            cudaFree(d_indices);
            return mesh;
        }
        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
    }
#endif
    return mesh;
}

TriangleMesh CudaRasterizer::generate_torus_gpu(float major_r, float minor_r, int rings, int sectors) {
    TriangleMesh mesh;
    size_t total_v = (rings + 1) * (sectors + 1);
    size_t total_tris = rings * sectors * 2;

    mesh.vertices.resize(total_v * 3);
    mesh.indices.resize(total_tris * 3);

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        size_t vert_bytes = total_v * 3 * sizeof(float);
        size_t idx_bytes = total_tris * 3 * sizeof(uint32_t);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess) {

            int threads = 256;
            int blocks_v = (static_cast<int>(total_v) + threads - 1) / threads;
            gpu_torus_vertices_kernel<<<blocks_v, threads>>>(d_verts, major_r, minor_r, rings, sectors);

            int total_quads = rings * sectors;
            int blocks_idx = (total_quads + threads - 1) / threads;
            gpu_torus_indices_kernel<<<blocks_idx, threads>>>(d_indices, rings, sectors);

            cudaMemcpy(mesh.vertices.data(), d_verts, vert_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(mesh.indices.data(), d_indices, idx_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            cudaFree(d_indices);
            return mesh;
        }
        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
    }
#endif
    return mesh;
}

TriangleMesh CudaRasterizer::generate_filleted_box_gpu(glm::vec3 half_extents, float fillet_r, int subd) {
    TriangleMesh mesh;
    size_t total_v = 6 * (subd + 1) * (subd + 1);
    size_t total_tris = 6 * subd * subd * 2;

    mesh.vertices.resize(total_v * 3);
    mesh.indices.resize(total_tris * 3);

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        size_t vert_bytes = total_v * 3 * sizeof(float);
        size_t idx_bytes = total_tris * 3 * sizeof(uint32_t);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess) {

            int threads = 256;
            int blocks_v = (static_cast<int>(total_v) + threads - 1) / threads;
            gpu_box_vertices_kernel<<<blocks_v, threads>>>(d_verts, half_extents.x, half_extents.y, half_extents.z, fillet_r, subd);

            int total_quads = 6 * subd * subd;
            int blocks_idx = (total_quads + threads - 1) / threads;
            gpu_box_indices_kernel<<<blocks_idx, threads>>>(d_indices, subd);

            cudaMemcpy(mesh.vertices.data(), d_verts, vert_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(mesh.indices.data(), d_indices, idx_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            cudaFree(d_indices);
            return mesh;
        }
        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
    }
#endif
    return mesh;
}

#ifdef __CUDACC__
__global__ void gpu_generate_vase_vertices_kernel(
    float radius,
    float height,
    int rings,
    int sectors,
    float* __restrict__ out_verts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_verts = 2 + (rings + 1) * sectors;
    if (idx >= total_verts) return;

    if (idx == 0) {
        out_verts[0] = 0.0f; out_verts[1] = 0.0f; out_verts[2] = -10.0f;
        return;
    }
    if (idx == total_verts - 1) {
        out_verts[idx * 3 + 0] = 0.0f; out_verts[idx * 3 + 1] = 0.0f; out_verts[idx * 3 + 2] = 30.0f;
        return;
    }

    int body_idx = idx - 1;
    int r = body_idx / sectors;
    int s = body_idx % sectors;
    float t = static_cast<float>(r) / static_cast<float>(rings);
    float z = -10.0f + t * height;
    float rad = radius + 2.0f * sinf(t * 3.14159265f);
    float phi = static_cast<float>(s) * (2.0f * 3.14159265f / static_cast<float>(sectors));

    out_verts[idx * 3 + 0] = rad * cosf(phi);
    out_verts[idx * 3 + 1] = rad * sinf(phi);
    out_verts[idx * 3 + 2] = z;
}

__global__ void gpu_generate_vase_indices_kernel(
    int rings,
    int sectors,
    uint32_t top_center_idx,
    uint32_t* __restrict__ out_indices
) {
    int s_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_slots = sectors + rings * sectors + sectors;
    if (s_id >= total_slots) return;

    if (s_id < sectors) {
        int s = s_id;
        uint32_t next = static_cast<uint32_t>((s + 1) % sectors);
        int off = s * 3;
        out_indices[off + 0] = 0u;
        out_indices[off + 1] = 1u + next;
        out_indices[off + 2] = 1u + static_cast<uint32_t>(s);
    } else if (s_id < sectors + rings * sectors) {
        int body = s_id - sectors;
        int r = body / sectors;
        int s = body % sectors;
        uint32_t r0 = static_cast<uint32_t>(1 + r * sectors);
        uint32_t r1 = static_cast<uint32_t>(1 + (r + 1) * sectors);
        uint32_t s_next = static_cast<uint32_t>((s + 1) % sectors);
        int off = sectors * 3 + body * 6;
        out_indices[off + 0] = r0 + static_cast<uint32_t>(s);
        out_indices[off + 1] = r0 + s_next;
        out_indices[off + 2] = r1 + s_next;
        out_indices[off + 3] = r0 + static_cast<uint32_t>(s);
        out_indices[off + 4] = r1 + s_next;
        out_indices[off + 5] = r1 + static_cast<uint32_t>(s);
    } else {
        int s = s_id - (sectors + rings * sectors);
        uint32_t top_ring = static_cast<uint32_t>(1 + rings * sectors);
        uint32_t next = static_cast<uint32_t>((s + 1) % sectors);
        int off = sectors * 3 + (rings * sectors) * 6 + s * 3;
        out_indices[off + 0] = top_center_idx;
        out_indices[off + 1] = top_ring + static_cast<uint32_t>(s);
        out_indices[off + 2] = top_ring + next;
    }
}
#endif

bool CudaRasterizer::generate_cylinder_vase_gpu(
    float radius,
    float height,
    int rings,
    int sectors,
    TriangleMesh& out_mesh
) {
    out_mesh.vertices.clear();
    out_mesh.indices.clear();
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int total_verts = 2 + (rings + 1) * sectors;
        int total_tris = sectors + rings * sectors * 2 + sectors;
        int total_slots = sectors + rings * sectors + sectors;
        uint32_t top_center_idx = static_cast<uint32_t>(total_verts - 1);

        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        if (cudaMalloc(&d_verts, total_verts * 3 * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_indices, total_tris * 3 * sizeof(uint32_t)) == cudaSuccess) {

            int threads = 256;
            int v_blocks = (total_verts + threads - 1) / threads;
            int i_blocks = (total_slots + threads - 1) / threads;

            gpu_generate_vase_vertices_kernel<<<v_blocks, threads>>>(radius, height, rings, sectors, d_verts);
            gpu_generate_vase_indices_kernel<<<i_blocks, threads>>>(rings, sectors, top_center_idx, d_indices);

            out_mesh.vertices.resize(static_cast<size_t>(total_verts) * 3);
            out_mesh.indices.resize(static_cast<size_t>(total_tris) * 3);
            cudaMemcpy(out_mesh.vertices.data(), d_verts, out_mesh.vertices.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_mesh.indices.data(), d_indices, out_mesh.indices.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            cudaFree(d_indices);
            return true;
        }
        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
    }
#endif
    return false;
}

} // namespace rhizome
