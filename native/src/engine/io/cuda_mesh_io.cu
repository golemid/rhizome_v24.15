#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/cuda_common.cuh"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace rhizome {

#ifdef __CUDACC__
#pragma pack(push, 1)
struct GpuStlFacet {
    float normal[3];
    float v0[3];
    float v1[3];
    float v2[3];
    uint16_t attr;
};
#pragma pack(pop)

__global__ void gpu_format_binary_stl_kernel(
    const float* __restrict__ vertices,
    const uint32_t* __restrict__ indices,
    size_t num_triangles,
    GpuStlFacet* __restrict__ facets
) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    uint32_t i0 = indices[t * 3 + 0];
    uint32_t i1 = indices[t * 3 + 1];
    uint32_t i2 = indices[t * 3 + 2];

    float p0x = vertices[i0 * 3 + 0];
    float p0y = vertices[i0 * 3 + 1];
    float p0z = vertices[i0 * 3 + 2];

    float p1x = vertices[i1 * 3 + 0];
    float p1y = vertices[i1 * 3 + 1];
    float p1z = vertices[i1 * 3 + 2];

    float p2x = vertices[i2 * 3 + 0];
    float p2y = vertices[i2 * 3 + 1];
    float p2z = vertices[i2 * 3 + 2];

    float e1x = p1x - p0x, e1y = p1y - p0y, e1z = p1z - p0z;
    float e2x = p2x - p0x, e2y = p2y - p0y, e2z = p2z - p0z;
    float nx = e1y * e2z - e1z * e2y;
    float ny = e1z * e2x - e1x * e2z;
    float nz = e1x * e2y - e1y * e2x;
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (len > 1e-8f) {
        nx /= len; ny /= len; nz /= len;
    } else {
        nx = 0.0f; ny = 0.0f; nz = 1.0f;
    }

    GpuStlFacet f;
    f.normal[0] = nx;
    f.normal[1] = ny;
    f.normal[2] = nz;
    f.v0[0] = p0x;
    f.v0[1] = p0y;
    f.v0[2] = p0z;
    f.v1[0] = p1x;
    f.v1[1] = p1y;
    f.v1[2] = p1z;
    f.v2[0] = p2x;
    f.v2[1] = p2y;
    f.v2[2] = p2z;
    f.attr = 0;

    facets[t] = f;
}
#endif

bool CudaRasterizer::format_binary_stl_gpu(
    const TriangleMesh& mesh,
    std::vector<uint8_t>& out_formatted_buffer,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t num_tris = mesh.triangle_count();
    if (num_tris == 0 || mesh.vertices.empty() || num_tris > 0xFFFFFFFFULL) return false;
    if (num_tris > (size_t)(~0ULL - 84ULL) / sizeof(GpuStlFacet)) return false;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        GpuStlFacet* d_facets = nullptr;

        size_t vert_bytes = mesh.vertices.size() * sizeof(float);
        size_t idx_bytes = mesh.indices.size() * sizeof(uint32_t);
        size_t facet_bytes = num_tris * sizeof(GpuStlFacet);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_facets, facet_bytes) == cudaSuccess) {

            cudaMemcpy(d_verts, mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = static_cast<int>((num_tris + threads - 1) / threads);
            gpu_format_binary_stl_kernel<<<blocks, threads>>>(d_verts, d_indices, num_tris, d_facets);

            out_formatted_buffer.resize(84 + facet_bytes);
            
            // 80 bytes header
            std::memset(out_formatted_buffer.data(), 0, 80);
            const char* header_tag = "Rhizome Native GPU Formatted Binary STL Model";
            std::memcpy(out_formatted_buffer.data(), header_tag, std::min<size_t>(std::strlen(header_tag), 79));

            // 4 bytes triangle count
            uint32_t tri_count_u32 = static_cast<uint32_t>(num_tris);
            std::memcpy(out_formatted_buffer.data() + 80, &tri_count_u32, sizeof(uint32_t));

            // Copy formatted 50-byte facets directly from GPU
            cudaMemcpy(out_formatted_buffer.data() + 84, d_facets, facet_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            cudaFree(d_indices);
            cudaFree(d_facets);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
        if (d_facets) cudaFree(d_facets);
    }
#endif
    return false;
}

#ifdef __CUDACC__
#pragma pack(push, 1)
struct GpuPlyVertex {
    float x, y, z;
    float nx, ny, nz;
};
struct GpuPlyFace {
    uint8_t count;
    uint32_t i0, i1, i2;
};
#pragma pack(pop)

__global__ void gpu_pack_ply_vertices_kernel(
    const float* __restrict__ vertices,
    const float* __restrict__ normals,
    size_t num_verts,
    GpuPlyVertex* __restrict__ out_ply_verts
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_verts) return;

    GpuPlyVertex v;
    v.x = vertices[i * 3 + 0];
    v.y = vertices[i * 3 + 1];
    v.z = vertices[i * 3 + 2];
    v.nx = normals[i * 3 + 0];
    v.ny = normals[i * 3 + 1];
    v.nz = normals[i * 3 + 2];
    out_ply_verts[i] = v;
}

__global__ void gpu_pack_ply_faces_kernel(
    const uint32_t* __restrict__ indices,
    size_t num_tris,
    GpuPlyFace* __restrict__ out_ply_faces
) {
    size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    GpuPlyFace f;
    f.count = 3;
    f.i0 = indices[t * 3 + 0];
    f.i1 = indices[t * 3 + 1];
    f.i2 = indices[t * 3 + 2];
    out_ply_faces[t] = f;
}
#endif

bool CudaRasterizer::format_binary_ply_gpu(
    const TriangleMesh& mesh,
    std::vector<uint8_t>& out_formatted_buffer,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t num_verts = mesh.vertex_count();
    size_t num_tris = mesh.triangle_count();
    if (num_verts == 0 || num_tris == 0 || num_verts > 0xFFFFFFFFULL || num_tris > 0xFFFFFFFFULL) {
        out_time_ms = 0.0f;
        return false;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        // 1. Compute vertex normals on GPU first
        std::vector<float> h_normals;
        float norm_time = 0.0f;
        if (!compute_vertex_normals_gpu(mesh, h_normals, norm_time)) {
            return false;
        }

        // 2. Prepare ASCII Header
        std::string header =
            "ply\n"
            "format binary_little_endian 1.0\n"
            "comment Rhizome Native GPU Export\n"
            "element vertex " + std::to_string(num_verts) + "\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "element face " + std::to_string(num_tris) + "\n"
            "property list uchar uint vertex_indices\n"
            "end_header\n";

        size_t header_len = header.size();
        size_t verts_bytes = num_verts * sizeof(GpuPlyVertex);
        size_t faces_bytes = num_tris * sizeof(GpuPlyFace);
        size_t total_bytes = header_len + verts_bytes + faces_bytes;

        float* d_verts = nullptr;
        float* d_normals = nullptr;
        uint32_t* d_indices = nullptr;
        GpuPlyVertex* d_ply_verts = nullptr;
        GpuPlyFace* d_ply_faces = nullptr;

        size_t raw_vert_bytes = num_verts * 3 * sizeof(float);
        size_t raw_idx_bytes = num_tris * 3 * sizeof(uint32_t);

        if (cudaMalloc(&d_verts, raw_vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_normals, raw_vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, raw_idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_ply_verts, verts_bytes) == cudaSuccess &&
            cudaMalloc(&d_ply_faces, faces_bytes) == cudaSuccess) {

            cudaMemcpy(d_verts, mesh.vertices.data(), raw_vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_normals, h_normals.data(), raw_vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, mesh.indices.data(), raw_idx_bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int v_blocks = static_cast<int>((num_verts + threads - 1) / threads);
            int f_blocks = static_cast<int>((num_tris + threads - 1) / threads);

            gpu_pack_ply_vertices_kernel<<<v_blocks, threads>>>(d_verts, d_normals, num_verts, d_ply_verts);
            gpu_pack_ply_faces_kernel<<<f_blocks, threads>>>(d_indices, num_tris, d_ply_faces);

            out_formatted_buffer.resize(total_bytes);
            std::memcpy(out_formatted_buffer.data(), header.data(), header_len);

            cudaMemcpy(out_formatted_buffer.data() + header_len, d_ply_verts, verts_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(out_formatted_buffer.data() + header_len + verts_bytes, d_ply_faces, faces_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            cudaFree(d_normals);
            cudaFree(d_indices);
            cudaFree(d_ply_verts);
            cudaFree(d_ply_faces);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_verts) cudaFree(d_verts);
        if (d_normals) cudaFree(d_normals);
        if (d_indices) cudaFree(d_indices);
        if (d_ply_verts) cudaFree(d_ply_verts);
        if (d_ply_faces) cudaFree(d_ply_faces);
    }
#endif
    return false;
}

#ifdef __CUDACC__
__device__ inline int dev_write_uint_str(char* buf, uint32_t val, int max_len = 32) {
    if (max_len <= 0) return 0;
    if (val == 0) {
        buf[0] = '0';
        return 1;
    }
    char tmp[12];
    int n = 0;
    while (val > 0 && n < 12) {
        tmp[n++] = static_cast<char>('0' + (val % 10));
        val /= 10;
    }
    int count = (n < max_len) ? n : max_len;
    for (int i = 0; i < count; ++i) {
        buf[i] = tmp[n - 1 - i];
    }
    return count;
}

__device__ inline int dev_write_float6_str(char* buf, float val, int max_len = 32) {
    if (max_len <= 0) return 0;
    int pos = 0;
    if (isnan(val) || isinf(val)) {
        val = 0.0f;
    }
    if (val < 0.0f) {
        if (pos < max_len) buf[pos++] = '-';
        val = -val;
    }
    double scaled = static_cast<double>(val) * 1000000.0 + 0.5;
    unsigned long long total_micro = static_cast<unsigned long long>(scaled);
    uint32_t int_part = static_cast<uint32_t>(total_micro / 1000000ULL);
    uint32_t frac_part = static_cast<uint32_t>(total_micro % 1000000ULL);

    pos += dev_write_uint_str(buf + pos, int_part, max_len - pos);
    if (pos < max_len) {
        buf[pos++] = '.';
    }
    for (int d = 5; d >= 0; --d) {
        if (pos + d < max_len) {
            buf[pos + d] = static_cast<char>('0' + (frac_part % 10));
        }
        frac_part /= 10;
    }
    pos += 6;
    if (pos > max_len) pos = max_len;
    return pos;
}

__global__ void gpu_format_obj_vertices_kernel(
    const float* __restrict__ vertices,
    size_t num_verts,
    char* __restrict__ out_lines
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_verts) return;

    char* line = out_lines + i * 64;
    int p = 0;
    line[p++] = 'v';
    line[p++] = ' ';
    p += dev_write_float6_str(line + p, vertices[i * 3 + 0], 62 - p);
    if (p < 62) line[p++] = ' ';
    p += dev_write_float6_str(line + p, vertices[i * 3 + 1], 62 - p);
    if (p < 62) line[p++] = ' ';
    p += dev_write_float6_str(line + p, vertices[i * 3 + 2], 62 - p);
    while (p < 63) {
        line[p++] = ' ';
    }
    line[63] = '\n';
}

__global__ void gpu_format_obj_faces_kernel(
    const uint32_t* __restrict__ indices,
    size_t num_tris,
    char* __restrict__ out_lines
) {
    size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    char* line = out_lines + t * 48;
    int p = 0;
    line[p++] = 'f';
    line[p++] = ' ';
    p += dev_write_uint_str(line + p, indices[t * 3 + 0] + 1u, 46 - p);
    if (p < 46) line[p++] = ' ';
    p += dev_write_uint_str(line + p, indices[t * 3 + 1] + 1u, 46 - p);
    if (p < 46) line[p++] = ' ';
    p += dev_write_uint_str(line + p, indices[t * 3 + 2] + 1u, 46 - p);
    while (p < 47) {
        line[p++] = ' ';
    }
    line[47] = '\n';
}

__global__ void gpu_offset_indices_kernel(
    const uint32_t* __restrict__ src_indices,
    size_t num_indices,
    uint32_t vertex_offset,
    uint32_t* __restrict__ dst_indices
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_indices) return;
    dst_indices[i] = src_indices[i] + vertex_offset;
}
#endif

bool CudaRasterizer::format_obj_gpu(
    const TriangleMesh& mesh,
    std::vector<char>& out_formatted_buffer,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t num_verts = mesh.vertex_count();
    size_t num_tris = mesh.triangle_count();
    if (num_verts == 0 || num_tris == 0) return false;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        const char* header = "# Rhizome Native GPU Accelerated OBJ Exporter\n# Unit: Millimeters (1.0 = 1.0mm)\n\n";
        size_t header_len = std::strlen(header);
        size_t verts_text_bytes = num_verts * 64;
        size_t faces_text_bytes = num_tris * 48;
        size_t total_text_bytes = header_len + verts_text_bytes + 1 + faces_text_bytes;

        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        char* d_text = nullptr;

        size_t vert_bytes = mesh.vertices.size() * sizeof(float);
        size_t idx_bytes = mesh.indices.size() * sizeof(uint32_t);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_text, total_text_bytes) == cudaSuccess) {

            cudaMemcpy(d_verts, mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_text, header, header_len, cudaMemcpyHostToDevice);
            char newline_ch = '\n';
            cudaMemcpy(d_text + header_len + verts_text_bytes, &newline_ch, 1, cudaMemcpyHostToDevice);

            int threads = 256;
            int v_blocks = static_cast<int>((num_verts + threads - 1) / threads);
            int f_blocks = static_cast<int>((num_tris + threads - 1) / threads);

            gpu_format_obj_vertices_kernel<<<v_blocks, threads>>>(
                d_verts, num_verts, d_text + header_len
            );
            gpu_format_obj_faces_kernel<<<f_blocks, threads>>>(
                d_indices, num_tris, d_text + header_len + verts_text_bytes + 1
            );

            out_formatted_buffer.resize(total_text_bytes);
            cudaMemcpy(out_formatted_buffer.data(), d_text, total_text_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_verts);
            cudaFree(d_indices);
            cudaFree(d_text);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_verts) cudaFree(d_verts);
        if (d_indices) cudaFree(d_indices);
        if (d_text) cudaFree(d_text);
    }
#endif
    return false;
}

bool CudaRasterizer::merge_meshes_gpu(
    const TriangleMesh& mesh_a,
    const TriangleMesh& mesh_b,
    TriangleMesh& out_merged,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_merged.vertices.clear();
    out_merged.indices.clear();

    if (mesh_a.vertex_count() == 0) {
        out_merged = mesh_b;
        out_time_ms = 0.0f;
        return true;
    }
    if (mesh_b.vertex_count() == 0) {
        out_merged = mesh_a;
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        size_t va_floats = mesh_a.vertices.size();
        size_t vb_floats = mesh_b.vertices.size();
        size_t ia_count = mesh_a.indices.size();
        size_t ib_count = mesh_b.indices.size();

        uint32_t* d_ib_src = nullptr;
        uint32_t* d_ib_dst = nullptr;
        size_t ib_bytes = ib_count * sizeof(uint32_t);

        if (cudaMalloc(&d_ib_src, ib_bytes) == cudaSuccess &&
            cudaMalloc(&d_ib_dst, ib_bytes) == cudaSuccess) {

            cudaMemcpy(d_ib_src, mesh_b.indices.data(), ib_bytes, cudaMemcpyHostToDevice);
            uint32_t v_offset = static_cast<uint32_t>(mesh_a.vertex_count());

            int threads = 256;
            int blocks = static_cast<int>((ib_count + threads - 1) / threads);
            gpu_offset_indices_kernel<<<blocks, threads>>>(d_ib_src, ib_count, v_offset, d_ib_dst);

            out_merged.vertices.resize(va_floats + vb_floats);
            std::memcpy(out_merged.vertices.data(), mesh_a.vertices.data(), va_floats * sizeof(float));
            std::memcpy(out_merged.vertices.data() + va_floats, mesh_b.vertices.data(), vb_floats * sizeof(float));

            out_merged.indices.resize(ia_count + ib_count);
            std::memcpy(out_merged.indices.data(), mesh_a.indices.data(), ia_count * sizeof(uint32_t));
            cudaMemcpy(out_merged.indices.data() + ia_count, d_ib_dst, ib_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_ib_src);
            cudaFree(d_ib_dst);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_ib_src) cudaFree(d_ib_src);
        if (d_ib_dst) cudaFree(d_ib_dst);
    }
#endif
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_decode_binary_stl_kernel(
    const uint8_t* __restrict__ payload,
    uint32_t num_tris,
    float* __restrict__ raw_vertices,
    float* __restrict__ raw_normals,
    uint32_t* __restrict__ raw_indices
) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    const uint8_t* rec = payload + (size_t)t * 50;

    // 1. Decode triangle surface normal (float nx, ny, nz at bytes 0..11)
    float nx, ny, nz;
    memcpy(&nx, rec + 0, sizeof(float));
    memcpy(&ny, rec + 4, sizeof(float));
    memcpy(&nz, rec + 8, sizeof(float));

    // 2. Decode 3 vertices (each 12 bytes at bytes 12..47)
    float v[3][3];
    for (int k = 0; k < 3; ++k) {
        memcpy(&v[k][0], rec + 12 + k * 12 + 0, sizeof(float));
        memcpy(&v[k][1], rec + 12 + k * 12 + 4, sizeof(float));
        memcpy(&v[k][2], rec + 12 + k * 12 + 8, sizeof(float));
    }

    // Robust normal normalization and geometric fallback
    float norm_len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (norm_len > 1e-7f) {
        nx /= norm_len;
        ny /= norm_len;
        nz /= norm_len;
    } else {
        float e1x = v[1][0] - v[0][0], e1y = v[1][1] - v[0][1], e1z = v[1][2] - v[0][2];
        float e2x = v[2][0] - v[0][0], e2y = v[2][1] - v[0][1], e2z = v[2][2] - v[0][2];
        nx = e1y * e2z - e1z * e2y;
        ny = e1z * e2x - e1x * e2z;
        nz = e1x * e2y - e1y * e2x;
        float geom_len = sqrtf(nx * nx + ny * ny + nz * nz);
        if (geom_len > 1e-7f) {
            nx /= geom_len;
            ny /= geom_len;
            nz /= geom_len;
        } else {
            nx = 0.0f; ny = 0.0f; nz = 1.0f;
        }
    }

    if (raw_normals != nullptr) {
        raw_normals[t * 3 + 0] = nx;
        raw_normals[t * 3 + 1] = ny;
        raw_normals[t * 3 + 2] = nz;
    }

    for (int k = 0; k < 3; ++k) {
        size_t v_idx = (size_t)t * 3 + k;
        if (raw_vertices != nullptr) {
            raw_vertices[v_idx * 3 + 0] = v[k][0];
            raw_vertices[v_idx * 3 + 1] = v[k][1];
            raw_vertices[v_idx * 3 + 2] = v[k][2];
        }
        if (raw_indices != nullptr) {
            raw_indices[v_idx] = static_cast<uint32_t>(v_idx);
        }
    }
}
#endif

bool CudaRasterizer::parse_binary_stl_gpu(
    const uint8_t* raw_bytes,
    size_t byte_count,
    TriangleMesh& out_mesh,
    float& out_time_ms,
    std::vector<float>* out_normals
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_mesh.vertices.clear();
    out_mesh.indices.clear();
    if (out_normals) out_normals->clear();

    if (byte_count < 84) {
        out_time_ms = 0.0f;
        return false;
    }

    uint32_t num_tris = 0;
    std::memcpy(&num_tris, raw_bytes + 80, sizeof(uint32_t));
    if (num_tris == 0) {
        out_time_ms = 0.0f;
        return true;
    }

    if (byte_count < 84 + (size_t)num_tris * 50) {
        out_time_ms = 0.0f;
        return false;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        const uint8_t* payload = raw_bytes + 84;
        size_t payload_bytes = (size_t)num_tris * 50;

        uint8_t* d_payload = nullptr;
        float* d_raw_verts = nullptr;
        float* d_raw_normals = nullptr;
        uint32_t* d_raw_indices = nullptr;

        size_t total_v = (size_t)num_tris * 3;
        size_t vert_bytes = total_v * 3 * sizeof(float);
        size_t norm_bytes = (size_t)num_tris * 3 * sizeof(float);
        size_t idx_bytes = total_v * sizeof(uint32_t);

        if (cudaMalloc(&d_payload, payload_bytes) == cudaSuccess &&
            cudaMalloc(&d_raw_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_raw_normals, norm_bytes) == cudaSuccess &&
            cudaMalloc(&d_raw_indices, idx_bytes) == cudaSuccess) {

            cudaMemcpy(d_payload, payload, payload_bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (static_cast<int>(num_tris) + threads - 1) / threads;
            gpu_decode_binary_stl_kernel<<<blocks, threads>>>(
                d_payload, num_tris, d_raw_verts, d_raw_normals, d_raw_indices
            );
            cudaDeviceSynchronize();

            std::vector<float> h_raw_verts(total_v * 3);
            cudaMemcpy(h_raw_verts.data(), d_raw_verts, vert_bytes, cudaMemcpyDeviceToHost);

            if (out_normals != nullptr) {
                out_normals->resize((size_t)num_tris * 3);
                cudaMemcpy(out_normals->data(), d_raw_normals, norm_bytes, cudaMemcpyDeviceToHost);
            }

            cudaFree(d_payload);
            cudaFree(d_raw_verts);
            cudaFree(d_raw_normals);
            cudaFree(d_raw_indices);

            struct Vec3Key {
                int32_t x, y, z;
                bool operator==(const Vec3Key& o) const noexcept {
                    return x == o.x && y == o.y && z == o.z;
                }
            };
            struct Vec3Hash {
                size_t operator()(const Vec3Key& k) const noexcept {
                    return static_cast<size_t>((k.x * 73856093) ^ (k.y * 19349663) ^ (k.z * 83492791));
                }
            };

            std::unordered_map<Vec3Key, uint32_t, Vec3Hash> v_map;
            v_map.reserve(total_v);
            out_mesh.vertices.reserve(total_v * 3);
            out_mesh.indices.reserve(total_v);

            for (size_t t = 0; t < num_tris; ++t) {
                uint32_t tri_idx[3];
                for (int k = 0; k < 3; ++k) {
                    float x = h_raw_verts[(t * 3 + k) * 3 + 0];
                    float y = h_raw_verts[(t * 3 + k) * 3 + 1];
                    float z = h_raw_verts[(t * 3 + k) * 3 + 2];
                    Vec3Key key{
                        static_cast<int32_t>(std::round(x * 10000.0f)),
                        static_cast<int32_t>(std::round(y * 10000.0f)),
                        static_cast<int32_t>(std::round(z * 10000.0f))
                    };
                    auto it = v_map.find(key);
                    if (it != v_map.end()) {
                        tri_idx[k] = it->second;
                    } else {
                        uint32_t new_id = static_cast<uint32_t>(out_mesh.vertex_count());
                        out_mesh.vertices.push_back(x);
                        out_mesh.vertices.push_back(y);
                        out_mesh.vertices.push_back(z);
                        v_map[key] = new_id;
                        tri_idx[k] = new_id;
                    }
                }
                if (tri_idx[0] != tri_idx[1] && tri_idx[1] != tri_idx[2] && tri_idx[2] != tri_idx[0]) {
                    out_mesh.indices.push_back(tri_idx[0]);
                    out_mesh.indices.push_back(tri_idx[1]);
                    out_mesh.indices.push_back(tri_idx[2]);
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_payload) cudaFree(d_payload);
        if (d_raw_verts) cudaFree(d_raw_verts);
        if (d_raw_normals) cudaFree(d_raw_normals);
        if (d_raw_indices) cudaFree(d_raw_indices);
    }
#endif
    return false;
}

bool CudaRasterizer::decode_binary_stl_gpu(
    const uint8_t* raw_bytes,
    size_t byte_count,
    std::vector<float>& out_vertices,
    std::vector<float>& out_normals,
    std::vector<uint32_t>& out_indices,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_vertices.clear();
    out_normals.clear();
    out_indices.clear();

    if (byte_count < 84) {
        out_time_ms = 0.0f;
        return false;
    }

    uint32_t num_tris = 0;
    std::memcpy(&num_tris, raw_bytes + 80, sizeof(uint32_t));
    if (num_tris == 0) {
        out_time_ms = 0.0f;
        return true;
    }

    if (byte_count < 84 + (size_t)num_tris * 50) {
        out_time_ms = 0.0f;
        return false;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        const uint8_t* payload = raw_bytes + 84;
        size_t payload_bytes = (size_t)num_tris * 50;

        uint8_t* d_payload = nullptr;
        float* d_raw_verts = nullptr;
        float* d_raw_normals = nullptr;
        uint32_t* d_raw_indices = nullptr;

        size_t total_v = (size_t)num_tris * 3;
        size_t vert_bytes = total_v * 3 * sizeof(float);
        size_t norm_bytes = (size_t)num_tris * 3 * sizeof(float);
        size_t idx_bytes = total_v * sizeof(uint32_t);

        if (cudaMalloc(&d_payload, payload_bytes) == cudaSuccess &&
            cudaMalloc(&d_raw_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_raw_normals, norm_bytes) == cudaSuccess &&
            cudaMalloc(&d_raw_indices, idx_bytes) == cudaSuccess) {

            cudaMemcpy(d_payload, payload, payload_bytes, cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (static_cast<int>(num_tris) + threads - 1) / threads;
            gpu_decode_binary_stl_kernel<<<blocks, threads>>>(
                d_payload, num_tris, d_raw_verts, d_raw_normals, d_raw_indices
            );
            cudaDeviceSynchronize();

            out_vertices.resize(total_v * 3);
            out_normals.resize((size_t)num_tris * 3);
            out_indices.resize(total_v);

            cudaMemcpy(out_vertices.data(), d_raw_verts, vert_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(out_normals.data(), d_raw_normals, norm_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(out_indices.data(), d_raw_indices, idx_bytes, cudaMemcpyDeviceToHost);

            cudaFree(d_payload);
            cudaFree(d_raw_verts);
            cudaFree(d_raw_normals);
            cudaFree(d_raw_indices);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }

        if (d_payload) cudaFree(d_payload);
        if (d_raw_verts) cudaFree(d_raw_verts);
        if (d_raw_normals) cudaFree(d_raw_normals);
        if (d_raw_indices) cudaFree(d_raw_indices);
    }
#endif
    return false;
}

#ifdef __CUDACC__
__device__ inline float device_fast_atof(const char*& p) {
    while (*p == ' ' || *p == '\t') p++;
    float sign = 1.0f;
    if (*p == '-') { sign = -1.0f; p++; }
    else if (*p == '+') { p++; }

    float val = 0.0f;
    while (*p >= '0' && *p <= '9') {
        val = val * 10.0f + static_cast<float>(*p - '0');
        p++;
    }
    if (*p == '.') {
        p++;
        float factor = 0.1f;
        while (*p >= '0' && *p <= '9') {
            val += static_cast<float>(*p - '0') * factor;
            factor *= 0.1f;
            p++;
        }
    }
    if (*p == 'e' || *p == 'E') {
        p++;
        float exp_sign = 1.0f;
        if (*p == '-') { exp_sign = -1.0f; p++; }
        else if (*p == '+') { p++; }
        int exp_val = 0;
        while (*p >= '0' && *p <= '9') {
            exp_val = exp_val * 10 + (*p - '0');
            p++;
        }
        val *= powf(10.0f, exp_sign * static_cast<float>(exp_val));
    }
    return sign * val;
}

__device__ inline int device_fast_atoi(const char*& p) {
    while (*p == ' ' || *p == '\t') p++;
    int sign = 1;
    if (*p == '-') { sign = -1; p++; }
    else if (*p == '+') { p++; }
    int val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    return sign * val;
}

__global__ void gpu_find_obj_lines_kernel(
    const char* __restrict__ text,
    size_t text_len,
    size_t* __restrict__ out_v_offsets,
    unsigned int* __restrict__ out_v_count,
    size_t max_v,
    size_t* __restrict__ out_f_offsets,
    unsigned int* __restrict__ out_f_count,
    size_t max_f
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= text_len) return;

    bool is_line_start = (i == 0) || (text[i - 1] == '\n');
    if (!is_line_start) return;

    if (i + 1 < text_len) {
        if (text[i] == 'v' && text[i + 1] == ' ') {
            unsigned int idx = atomicAdd(out_v_count, 1);
            if (idx < max_v) {
                out_v_offsets[idx] = i + 2;
            }
        } else if (text[i] == 'f' && text[i + 1] == ' ') {
            unsigned int idx = atomicAdd(out_f_count, 1);
            if (idx < max_f) {
                out_f_offsets[idx] = i + 2;
            }
        }
    }
}

__global__ void gpu_parse_obj_vertices_kernel(
    const char* __restrict__ text,
    const size_t* __restrict__ v_offsets,
    unsigned int v_count,
    float* __restrict__ out_vertices
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= v_count) return;

    const char* p = text + v_offsets[idx];
    float x = device_fast_atof(p);
    float y = device_fast_atof(p);
    float z = device_fast_atof(p);

    out_vertices[idx * 3 + 0] = x;
    out_vertices[idx * 3 + 1] = y;
    out_vertices[idx * 3 + 2] = z;
}

__global__ void gpu_parse_obj_faces_kernel(
    const char* __restrict__ text,
    const size_t* __restrict__ f_offsets,
    unsigned int f_count,
    unsigned int total_verts,
    uint32_t* __restrict__ out_indices,
    unsigned int* __restrict__ out_tri_count,
    unsigned int max_tris
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= f_count) return;

    const char* p = text + f_offsets[idx];
    int face_idx[4];
    int count = 0;

    while (*p != '\n' && *p != '\r' && *p != '\0' && count < 4) {
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\n' || *p == '\r' || *p == '\0') break;

        int vi = device_fast_atoi(p);
        while (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r' && *p != '\0') p++;

        if (vi < 0) {
            vi = static_cast<int>(total_verts) + vi + 1;
        }
        if (vi > 0) {
            face_idx[count++] = vi - 1; // 1-based to 0-based
        }
    }

    if (count == 3) {
        unsigned int t = atomicAdd(out_tri_count, 1);
        if (t < max_tris) {
            out_indices[t * 3 + 0] = static_cast<uint32_t>(face_idx[0]);
            out_indices[t * 3 + 1] = static_cast<uint32_t>(face_idx[1]);
            out_indices[t * 3 + 2] = static_cast<uint32_t>(face_idx[2]);
        }
    } else if (count == 4) {
        unsigned int t = atomicAdd(out_tri_count, 2);
        if (t + 1 < max_tris) {
            out_indices[t * 3 + 0] = static_cast<uint32_t>(face_idx[0]);
            out_indices[t * 3 + 1] = static_cast<uint32_t>(face_idx[1]);
            out_indices[t * 3 + 2] = static_cast<uint32_t>(face_idx[2]);

            out_indices[(t + 1) * 3 + 0] = static_cast<uint32_t>(face_idx[0]);
            out_indices[(t + 1) * 3 + 1] = static_cast<uint32_t>(face_idx[2]);
            out_indices[(t + 1) * 3 + 2] = static_cast<uint32_t>(face_idx[3]);
        }
    }
}
#endif

bool CudaRasterizer::parse_obj_text_gpu(
    const char* text_buffer,
    size_t text_length,
    TriangleMesh& out_mesh,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_mesh.vertices.clear();
    out_mesh.indices.clear();

    if (text_length == 0 || !text_buffer) {
        out_time_ms = 0.0f;
        return false;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        char* d_text = nullptr;
        size_t* d_v_offsets = nullptr;
        size_t* d_f_offsets = nullptr;
        unsigned int* d_v_count = nullptr;
        unsigned int* d_f_count = nullptr;

        size_t max_lines = text_length / 10 + 1024;
        if (cudaMalloc(&d_text, text_length + 1) == cudaSuccess &&
            cudaMalloc(&d_v_offsets, max_lines * sizeof(size_t)) == cudaSuccess &&
            cudaMalloc(&d_f_offsets, max_lines * sizeof(size_t)) == cudaSuccess &&
            cudaMalloc(&d_v_count, sizeof(unsigned int)) == cudaSuccess &&
            cudaMalloc(&d_f_count, sizeof(unsigned int)) == cudaSuccess) {

            cudaMemcpy(d_text, text_buffer, text_length, cudaMemcpyHostToDevice);
            char zero_byte = '\0';
            cudaMemcpy(d_text + text_length, &zero_byte, 1, cudaMemcpyHostToDevice);
            cudaMemset(d_v_count, 0, sizeof(unsigned int));
            cudaMemset(d_f_count, 0, sizeof(unsigned int));

            int threads = 256;
            int blocks = static_cast<int>((text_length + threads - 1) / threads);
            gpu_find_obj_lines_kernel<<<blocks, threads>>>(
                d_text, text_length,
                d_v_offsets, d_v_count, max_lines,
                d_f_offsets, d_f_count, max_lines
            );

            unsigned int h_v_count = 0, h_f_count = 0;
            cudaMemcpy(&h_v_count, d_v_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_f_count, d_f_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

            h_v_count = std::min<unsigned int>(h_v_count, static_cast<unsigned int>(max_lines));
            h_f_count = std::min<unsigned int>(h_f_count, static_cast<unsigned int>(max_lines));

            if (h_v_count > 0 && h_f_count > 0) {
                // Sort offsets to strictly preserve 1-based sequential file order
                std::vector<size_t> h_v_offsets(h_v_count);
                std::vector<size_t> h_f_offsets(h_f_count);
                cudaMemcpy(h_v_offsets.data(), d_v_offsets, h_v_count * sizeof(size_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_f_offsets.data(), d_f_offsets, h_f_count * sizeof(size_t), cudaMemcpyDeviceToHost);

                std::sort(h_v_offsets.begin(), h_v_offsets.end());
                std::sort(h_f_offsets.begin(), h_f_offsets.end());

                cudaMemcpy(d_v_offsets, h_v_offsets.data(), h_v_count * sizeof(size_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_f_offsets, h_f_offsets.data(), h_f_count * sizeof(size_t), cudaMemcpyHostToDevice);

                float* d_vertices = nullptr;
                uint32_t* d_indices = nullptr;
                unsigned int* d_tri_count = nullptr;

                unsigned int max_tris = h_f_count * 2;
                size_t vert_bytes = static_cast<size_t>(h_v_count) * 3 * sizeof(float);
                size_t idx_bytes = static_cast<size_t>(max_tris) * 3 * sizeof(uint32_t);

                if (cudaMalloc(&d_vertices, vert_bytes) == cudaSuccess &&
                    cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
                    cudaMalloc(&d_tri_count, sizeof(unsigned int)) == cudaSuccess) {

                    cudaMemset(d_tri_count, 0, sizeof(unsigned int));

                    int v_blocks = static_cast<int>((h_v_count + threads - 1) / threads);
                    gpu_parse_obj_vertices_kernel<<<v_blocks, threads>>>(
                        d_text, d_v_offsets, h_v_count, d_vertices
                    );

                    int f_blocks = static_cast<int>((h_f_count + threads - 1) / threads);
                    gpu_parse_obj_faces_kernel<<<f_blocks, threads>>>(
                        d_text, d_f_offsets, h_f_count, h_v_count,
                        d_indices, d_tri_count, max_tris
                    );

                    unsigned int h_tri_count = 0;
                    cudaMemcpy(&h_tri_count, d_tri_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                    h_tri_count = std::min(h_tri_count, max_tris);

                    if (h_tri_count > 0) {
                        out_mesh.vertices.resize(static_cast<size_t>(h_v_count) * 3);
                        out_mesh.indices.resize(static_cast<size_t>(h_tri_count) * 3);

                        cudaMemcpy(out_mesh.vertices.data(), d_vertices, out_mesh.vertices.size() * sizeof(float), cudaMemcpyDeviceToHost);
                        cudaMemcpy(out_mesh.indices.data(), d_indices, out_mesh.indices.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    }

                    cudaFree(d_vertices);
                    cudaFree(d_indices);
                    cudaFree(d_tri_count);
                }
            }

            cudaFree(d_text);
            cudaFree(d_v_offsets);
            cudaFree(d_f_offsets);
            cudaFree(d_v_count);
            cudaFree(d_f_count);

            if (!out_mesh.indices.empty()) {
                auto end = std::chrono::high_resolution_clock::now();
                out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
                return true;
            }
        }

        if (d_text) cudaFree(d_text);
        if (d_v_offsets) cudaFree(d_v_offsets);
        if (d_f_offsets) cudaFree(d_f_offsets);
        if (d_v_count) cudaFree(d_v_count);
        if (d_f_count) cudaFree(d_f_count);
    }
#endif
    return false;
}

#ifdef __CUDACC__
// --- 23. GPU Kernels for Parallel ASCII STL Parsing & Vertex Welding ---
__global__ void gpu_find_ascii_stl_vertices_kernel(
    const char* __restrict__ text,
    size_t text_len,
    size_t* __restrict__ out_v_offsets,
    unsigned int* __restrict__ out_v_count,
    size_t max_v
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + 7 >= text_len) return;

    bool is_token_start = (i == 0) || (text[i - 1] == ' ' || text[i - 1] == '\t' || text[i - 1] == '\n' || text[i - 1] == '\r');
    if (!is_token_start) return;

    if (text[i + 0] == 'v' &&
        text[i + 1] == 'e' &&
        text[i + 2] == 'r' &&
        text[i + 3] == 't' &&
        text[i + 4] == 'e' &&
        text[i + 5] == 'x' &&
        (text[i + 6] == ' ' || text[i + 6] == '\t')) {
        unsigned int slot = atomicAdd(out_v_count, 1);
        if (slot < max_v) {
            out_v_offsets[slot] = i + 6;
        }
    }
}

__global__ void gpu_sort_offsets_bitonic_step_kernel(
    size_t* __restrict__ data,
    unsigned int n_pow2,
    unsigned int j,
    unsigned int k
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pow2) return;
    unsigned int ixj = i ^ j;
    if (ixj > i) {
        bool ascending = ((i & k) == 0);
        size_t a = data[i];
        size_t b = data[ixj];
        if ((ascending && a > b) || (!ascending && a < b)) {
            data[i] = b;
            data[ixj] = a;
        }
    }
}

struct GpuWeldSlot {
    unsigned long long key;
    uint32_t out_idx;
};

__device__ inline unsigned long long dev_quantize_vec3_key(float x, float y, float z) {
    int32_t qx = static_cast<int32_t>(roundf(x * 10000.0f));
    int32_t qy = static_cast<int32_t>(roundf(y * 10000.0f));
    int32_t qz = static_cast<int32_t>(roundf(z * 10000.0f));
    uint64_t ux = static_cast<uint64_t>(static_cast<uint32_t>(qx));
    uint64_t uy = static_cast<uint64_t>(static_cast<uint32_t>(qy));
    uint64_t uz = static_cast<uint64_t>(static_cast<uint32_t>(qz));
    unsigned long long h = (ux * 73856093ULL) ^ (uy * 19349663ULL) ^ (uz * 83492791ULL) ^ ((ux & 0xFFFFF) << 42) ^ ((uy & 0xFFFFF) << 21) ^ (uz & 0x1FFFFF);
    return (h == 0ULL) ? 1ULL : h;
}

__global__ void gpu_weld_raw_vertices_kernel(
    const float* __restrict__ raw_vertices,
    unsigned int num_raw_verts,
    GpuWeldSlot* __restrict__ table,
    size_t table_size,
    float* __restrict__ out_unique_vertices,
    unsigned int* __restrict__ d_unique_count,
    uint32_t* __restrict__ out_mapped_indices
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_raw_verts) return;

    float x = raw_vertices[i * 3 + 0];
    float y = raw_vertices[i * 3 + 1];
    float z = raw_vertices[i * 3 + 2];

    unsigned long long key = dev_quantize_vec3_key(x, y, z);
    unsigned int slot = hash_u64(key) % table_size;

    for (size_t probe = 0; probe < table_size; ++probe) {
        size_t idx_probe = (slot + probe) % table_size;
        unsigned long long prev = atomicCAS(&table[idx_probe].key, 0ULL, key);
        if (prev == 0ULL) {
            unsigned int new_id = atomicAdd(d_unique_count, 1);
            out_unique_vertices[new_id * 3 + 0] = x;
            out_unique_vertices[new_id * 3 + 1] = y;
            out_unique_vertices[new_id * 3 + 2] = z;
            atomicExch(&table[idx_probe].out_idx, new_id + 1u);
            out_mapped_indices[i] = new_id;
            return;
        } else if (prev == key) {
            uint32_t existing = table[idx_probe].out_idx;
            int spin = 0;
            while (existing == 0u && spin < 10000) {
                existing = atomicAdd(&table[idx_probe].out_idx, 0u);
                spin++;
            }
            out_mapped_indices[i] = (existing == 0u) ? 0u : (existing - 1u);
            return;
        }
    }
}
#endif // __CUDACC__

bool CudaRasterizer::parse_ascii_stl_gpu(
    const char* text_buffer,
    size_t text_length,
    TriangleMesh& out_mesh,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_mesh.vertices.clear();
    out_mesh.indices.clear();
    if (!text_buffer || text_length < 15) {
        out_time_ms = 0.0f;
        return false;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        char* d_text = nullptr;
        size_t* d_v_offsets = nullptr;
        unsigned int* d_v_count = nullptr;

        size_t max_verts = text_length / 12 + 1024;
        unsigned int pow2_cap = 1024;
        while (pow2_cap < max_verts && pow2_cap < (1u << 26)) pow2_cap <<= 1;

        if (cudaMalloc(&d_text, text_length + 1) == cudaSuccess &&
            cudaMalloc(&d_v_offsets, pow2_cap * sizeof(size_t)) == cudaSuccess &&
            cudaMalloc(&d_v_count, sizeof(unsigned int)) == cudaSuccess) {

            cudaMemcpy(d_text, text_buffer, text_length, cudaMemcpyHostToDevice);
            char zero_byte = '\0';
            cudaMemcpy(d_text + text_length, &zero_byte, 1, cudaMemcpyHostToDevice);
            cudaMemset(d_v_offsets, 0xFF, pow2_cap * sizeof(size_t));
            cudaMemset(d_v_count, 0, sizeof(unsigned int));

            int threads = 256;
            int blocks = static_cast<int>((text_length + threads - 1) / threads);
            gpu_find_ascii_stl_vertices_kernel<<<blocks, threads>>>(
                d_text, text_length, d_v_offsets, d_v_count, pow2_cap
            );

            unsigned int h_v_count = 0;
            cudaMemcpy(&h_v_count, d_v_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            h_v_count = std::min(h_v_count, pow2_cap);
            unsigned int num_tris = h_v_count / 3;
            h_v_count = num_tris * 3;

            if (num_tris > 0) {
                unsigned int sort_n = 1;
                while (sort_n < h_v_count) sort_n <<= 1;
                int sort_blocks = static_cast<int>((sort_n + threads - 1) / threads);
                for (unsigned int k = 2; k <= sort_n; k <<= 1) {
                    for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                        gpu_sort_offsets_bitonic_step_kernel<<<sort_blocks, threads>>>(d_v_offsets, sort_n, j, k);
                    }
                }

                float* d_raw_verts = nullptr;
                float* d_unique_verts = nullptr;
                unsigned int* d_unique_count = nullptr;
                uint32_t* d_mapped_indices = nullptr;
                GpuWeldSlot* d_weld_table = nullptr;
                size_t weld_table_size = static_cast<size_t>(h_v_count) * 4 + 1024;

                if (cudaMalloc(&d_raw_verts, static_cast<size_t>(h_v_count) * 3 * sizeof(float)) == cudaSuccess &&
                    cudaMalloc(&d_unique_verts, static_cast<size_t>(h_v_count) * 3 * sizeof(float)) == cudaSuccess &&
                    cudaMalloc(&d_unique_count, sizeof(unsigned int)) == cudaSuccess &&
                    cudaMalloc(&d_mapped_indices, static_cast<size_t>(h_v_count) * sizeof(uint32_t)) == cudaSuccess &&
                    cudaMalloc(&d_weld_table, weld_table_size * sizeof(GpuWeldSlot)) == cudaSuccess) {

                    int v_blocks = static_cast<int>((h_v_count + threads - 1) / threads);
                    gpu_parse_obj_vertices_kernel<<<v_blocks, threads>>>(
                        d_text, d_v_offsets, h_v_count, d_raw_verts
                    );

                    cudaMemset(d_unique_count, 0, sizeof(unsigned int));
                    cudaMemset(d_weld_table, 0xFF, weld_table_size * sizeof(GpuWeldSlot));
                    // Clear keys to 0ULL while keeping out_idx = 0xFFFFFFFFu
                    cudaMemset(d_weld_table, 0, weld_table_size * sizeof(GpuWeldSlot));

                    gpu_weld_raw_vertices_kernel<<<v_blocks, threads>>>(
                        d_raw_verts, h_v_count, d_weld_table, weld_table_size,
                        d_unique_verts, d_unique_count, d_mapped_indices
                    );

                    unsigned int h_unique = 0;
                    cudaMemcpy(&h_unique, d_unique_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                    if (h_unique > 0) {
                        out_mesh.vertices.resize(static_cast<size_t>(h_unique) * 3);
                        out_mesh.indices.resize(h_v_count);
                        cudaMemcpy(out_mesh.vertices.data(), d_unique_verts, out_mesh.vertices.size() * sizeof(float), cudaMemcpyDeviceToHost);
                        cudaMemcpy(out_mesh.indices.data(), d_mapped_indices, out_mesh.indices.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    }

                    cudaFree(d_raw_verts);
                    cudaFree(d_unique_verts);
                    cudaFree(d_unique_count);
                    cudaFree(d_mapped_indices);
                    cudaFree(d_weld_table);
                } else {
                    if (d_raw_verts) cudaFree(d_raw_verts);
                    if (d_unique_verts) cudaFree(d_unique_verts);
                    if (d_unique_count) cudaFree(d_unique_count);
                    if (d_mapped_indices) cudaFree(d_mapped_indices);
                    if (d_weld_table) cudaFree(d_weld_table);
                }
            }

            cudaFree(d_text);
            cudaFree(d_v_offsets);
            cudaFree(d_v_count);

            if (!out_mesh.indices.empty()) {
                auto end = std::chrono::high_resolution_clock::now();
                out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
                return true;
            }
        }
        if (d_text) cudaFree(d_text);
        if (d_v_offsets) cudaFree(d_v_offsets);
        if (d_v_count) cudaFree(d_v_count);
    }
#endif
    return false;
}

} // namespace rhizome
