#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/cuda_common.cuh"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

namespace rhizome {

// =========================================================================
// v24.04/v24.05 TRACK D — FAST SDF PATH (0.1 mm default detail level)
// =========================================================================
// The v23.x mesh_to_sdf_kernel is exact but O(voxels x triangles): every
// voxel evaluates every triangle's solid angle (winding number) plus the
// point-triangle distance. On mug.obj (3,106 tris) at 0.10 mm that is
// 699M x 3,106 = 2.17 trillion solid-angle evaluations (~66 s on a 12 GB
// GPU; measured 2026-10-23, scaling 3.26x per voxel-size halving step —
// pure O(V x T)). VRAM is NOT the constraint (16% used) — the algorithm is.
//
// The fast path replaces it for LARGE grids with a pipeline that is EXACT
// where it matters and eikonal-approximate (O(h), h = voxel) in the far
// field:
//
//   [A] Band splat (per triangle): exact point-triangle distance written
//       via atomicMin into every voxel within RHI_SDF_BAND_VOXELS of the
//       surface. Unsigned distances in the band are bit-identical to the
//       legacy kernel (same point_triangle_distance_sq; min over exact
//       values is order-independent). Scales with surface area, not volume.
//
//   [B] Scanline parity signs (per z-column): one thread gathers the
//       z-intersections of the mesh along its (sub-voxel jittered) column
//       ray, sorts them, and alternates outside/inside down the column.
//       For watertight meshes parity classification is mathematically
//       identical to the winding-number sign (Jordan-Brouwer). Columns
//       with an odd or overflowing hit count (leaks / degeneracies) are
//       deferred to the exact per-voxel winding fixup [C].
//
//   [C] Exact winding fixup: full-volume early-out pass recomputing the
//       legacy exact winding number only for voxels marked unknown by [B].
//
//   [D] FSM eikonal re-distancing (6 directional sweeps per round, in
//       place): propagates the exact band distances into the far field
//       with the same Godunov update used by gpu_eikonal_update_kernel
//       (see Sec "19. Eikonal" in CODE_MAP.md). Far-field |d| is the
//       viscosity solution (error O(h); typically well below 0.1 mm for
//       vessel-class geometry).
//
//   [E] Sampled exact-winding verification (v24.05 Track D R2): 4096
//       stratified voxel centers per probe are re-classified with the
//       legacy exact winding sum and compared against the parity signs.
//       A small number of mismatched sites (compact divergences: handle/
//       wall self-overlap junctions, sub-voxel wall slivers) is PATCHED
//       in place — a (2r+1)^3 box around each site is marked unknown and
//       recomputed with the exact winding fixup [C] — then re-probed with
//       fresh stratifications (up to 3 generations). Accept only on a
//       clean probe; systematic divergence (a probe above the per-probe
//       site cap, or persistence across all generations) still aborts and
//       falls back to the untouched legacy kernel — the fast path can
//       never silently ship a classification the exact path would have
//       made differently at any probed site.
//
//   [F] Sign apply: signed = inside ? -|d| : +|d| (same convention as the
//       legacy kernel: |w| >= 0.5 -> inside -> negative).
//
// Gate: the fast path engages only when total_voxels >=
// RHI_FAST_SDF_MIN_VOXELS (32M — every verification-suite grid is below
// it, so --verify results are bit-identical to v24.03 by construction;
// interactive grids at 0.25 mm or finer on mug-class specimens are above).
// =========================================================================

constexpr size_t RHI_FAST_SDF_MIN_VOXELS = 32ull * 1024ull * 1024ull; // 32M voxels (~320^3)
constexpr int RHI_SDF_BAND_VOXELS = 8;      // exact-distance band half-width, in voxels
constexpr int RHI_PARITY_MAX_HITS = 128;    // max z-crossings per column before deferring to fixup
constexpr float RHI_SDF_FAR_SENTINEL = 1e30f; // "unreached" far-field marker before re-distancing
// v24.05 Track D R2 — iterative exact-winding patch policy for the [E] gate:
constexpr int RHI_VERIFY_SAMPLES = 4096;               // stratified samples per probe
constexpr int RHI_VERIFY_PROBE_GENS = 3;               // independent probe generations
constexpr int RHI_VERIFY_MAX_SITES_PER_PROBE = 64;     // above this, a probe means systematic divergence
constexpr int RHI_SDF_PATCH_RADIUS_VOXELS = 24;        // exact-fixup box half-width per mismatch site

#ifdef __CUDACC__

// RASTER_TILE_SIZE now lives in core/math_utils.cuh (v23.29 Phase 2,
// AGENTS.md Sec A.2 single-source-of-truth fix, finding F6).

struct CudaTileHeader {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    float center_x, center_y, center_z;
    float area_x, area_y, area_z;
    float radius_sq;
};

__global__ void mesh_to_sdf_kernel(
    const CudaTriangle* __restrict__ triangles,
    const CudaTileHeader* __restrict__ tile_headers,
    int num_triangles,
    int num_tiles,
    float voxel_size,
    int min_x, int min_y, int min_z,
    int dim_x, int dim_y, int dim_z,
    float* __restrict__ out_grid
) {
    __shared__ CudaTriangle s_triangles[RASTER_TILE_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int linear_tid = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int num_block_threads = blockDim.x * blockDim.y * blockDim.z;

    bool valid_voxel = (x < dim_x && y < dim_y && z < dim_z);
    size_t idx = valid_voxel ? ((size_t)z * dim_y * dim_x + (size_t)y * dim_x + (size_t)x) : 0;
    glm::vec3 p((min_x + x) * voxel_size, (min_y + y) * voxel_size, (min_z + z) * voxel_size);

    float min_dist_sq = 1e12f;
    float solid_angle_sum = 0.0f;

    for (int t = 0; t < num_tiles; ++t) {
        int tile_start = t * RASTER_TILE_SIZE;
        int tile_count = (num_triangles - tile_start < RASTER_TILE_SIZE) ? (num_triangles - tile_start) : RASTER_TILE_SIZE;

        CudaTileHeader th = tile_headers[t];

        for (int load_idx = linear_tid; load_idx < tile_count; load_idx += num_block_threads) {
            s_triangles[load_idx] = triangles[tile_start + load_idx];
        }
        __syncthreads();

        if (valid_voxel) {
            float bdx = (p.x < th.min_x) ? (th.min_x - p.x) : ((p.x > th.max_x) ? (p.x - th.max_x) : 0.0f);
            float bdy = (p.y < th.min_y) ? (th.min_y - p.y) : ((p.y > th.max_y) ? (p.y - th.max_y) : 0.0f);
            float bdz = (p.z < th.min_z) ? (th.min_z - p.z) : ((p.z > th.max_z) ? (p.z - th.max_z) : 0.0f);
            float box_d2 = bdx * bdx + bdy * bdy + bdz * bdz;

            bool can_contain_closer = (box_d2 < min_dist_sq);

            float cx = th.center_x - p.x;
            float cy = th.center_y - p.y;
            float cz = th.center_z - p.z;
            float c_dist2 = cx * cx + cy * cy + cz * cz;

            // DISABLED: Cluster dipole approximation caused sign errors on thin/curved geometry.
            // Now always using exact triangle_solid_angle for robust inside/outside classification.
            if (false && c_dist2 > 16.0f * th.radius_sq && th.radius_sq > 1e-4f) {
                float c_len = sqrtf(c_dist2);
                float inv_c3 = 1.0f / (c_dist2 * c_len);
                solid_angle_sum += (th.area_x * cx + th.area_y * cy + th.area_z * cz) * inv_c3;
                if (can_contain_closer) {
                    for (int i = 0; i < tile_count; ++i) {
                        const CudaTriangle& tri = s_triangles[i];
                        glm::vec3 closest;
                        float d2 = point_triangle_distance_sq(p, tri.v0, tri.v1, tri.v2, closest);
                        if (d2 < min_dist_sq) {
                            min_dist_sq = d2;
                        }
                    }
                }
            } else {
                for (int i = 0; i < tile_count; ++i) {
                    const CudaTriangle& tri = s_triangles[i];
                    if (can_contain_closer) {
                        glm::vec3 closest;
                        float d2 = point_triangle_distance_sq(p, tri.v0, tri.v1, tri.v2, closest);
                        if (d2 < min_dist_sq) {
                            min_dist_sq = d2;
                        }
                    }
                    solid_angle_sum += triangle_solid_angle(p, tri.v0, tri.v1, tri.v2);
                }
            }
        }
        __syncthreads();
    }

    if (valid_voxel) {
        float winding_number = solid_angle_sum * 0.07957747154594767f; // 1 / (4 * pi)
        float sign = (fabsf(winding_number) >= 0.5f) ? -1.0f : 1.0f;
        out_grid[idx] = sign * sqrtf(min_dist_sq);
    }
}

// =========================================================================
// v24.04 TRACK D — FAST SDF PATH KERNELS
// (see the Track D header comment above for the pipeline description)
// =========================================================================

// [init] Grid-stride fill (the far sentinel is a float pattern; cudaMemset
// cannot write it, so this is a dedicated kernel).
__global__ void gpu_fill_float_kernel(
    float* __restrict__ data,
    size_t n,
    float value
) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        data[i] = value;
    }
}

// [A] Band splat: one thread per (triangle, z-layer) pair. Threads whose
// layer is outside the triangle's padded AABB exit immediately; the rest
// walk the (x,y) footprint and write the EXACT point-triangle distance
// via atomicMin wherever it lands inside the band. Deterministic: the min
// over exact float values is order-independent, so band values are
// bit-identical to the legacy per-voxel kernel.
__global__ void gpu_splat_band_distances_kernel(
    const CudaTriangle* __restrict__ triangles,
    int num_triangles,
    float voxel_size,
    float band_width,
    int min_x, int min_y, int min_z,
    int dim_x, int dim_y, int dim_z,
    float* __restrict__ io_dist
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_triangles * dim_z;
    if (tid >= total) return;
    int t = tid / dim_z;
    int z = tid % dim_z;

    const CudaTriangle& tri = triangles[t];

    float tx0 = fminf(tri.v0.x, fminf(tri.v1.x, tri.v2.x));
    float tx1 = fmaxf(tri.v0.x, fmaxf(tri.v1.x, tri.v2.x));
    float ty0 = fminf(tri.v0.y, fminf(tri.v1.y, tri.v2.y));
    float ty1 = fmaxf(tri.v0.y, fmaxf(tri.v1.y, tri.v2.y));
    float tz0 = fminf(tri.v0.z, fminf(tri.v1.z, tri.v2.z));
    float tz1 = fmaxf(tri.v0.z, fmaxf(tri.v1.z, tri.v2.z));

    float wz = (min_z + z) * voxel_size;
    if (wz < tz0 - band_width || wz > tz1 + band_width) return;

    float band_sq = band_width * band_width;
    float bdz = (wz < tz0) ? (tz0 - wz) : ((wz > tz1) ? (wz - tz1) : 0.0f);

    int x0 = (int)ceilf((tx0 - band_width) / voxel_size) - min_x;
    int x1 = (int)floorf((tx1 + band_width) / voxel_size) - min_x;
    int y0 = (int)ceilf((ty0 - band_width) / voxel_size) - min_y;
    int y1 = (int)floorf((ty1 + band_width) / voxel_size) - min_y;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > dim_x - 1) x1 = dim_x - 1;
    if (y1 > dim_y - 1) y1 = dim_y - 1;
    if (x1 < x0 || y1 < y0) return;

    size_t zstride = (size_t)dim_y * (size_t)dim_x;
    for (int y = y0; y <= y1; ++y) {
        float py = (min_y + y) * voxel_size;
        float bdy = (py < ty0) ? (ty0 - py) : ((py > ty1) ? (py - ty1) : 0.0f);
        size_t row_base = (size_t)z * zstride + (size_t)y * (size_t)dim_x + (size_t)x0;
        for (int x = x0; x <= x1; ++x, ++row_base) {
            float px = (min_x + x) * voxel_size;
            float bdx = (px < tx0) ? (tx0 - px) : ((px > tx1) ? (px - tx1) : 0.0f);
            if (bdx * bdx + bdy * bdy + bdz * bdz > band_sq) continue; // cheap box reject
            glm::vec3 p(px, py, wz);
            glm::vec3 closest;
            float d2 = point_triangle_distance_sq(p, tri.v0, tri.v1, tri.v2, closest);
            if (d2 <= band_sq) {
                dev_atomicMin_float(io_dist + row_base, sqrtf(d2));
            }
        }
    }
}

// [B] Scanline parity signs: one thread per (x,y) column. The column ray
// is jittered sub-voxel (fixed constants) to dodge ray/vertex and
// ray/edge degeneracies — the classic measure-zero failure of parity
// casting. Candidate triangles come from a host-built 4x4-voxel supercell
// CSR (XY); the kernel rejects non-overlapping candidates via the
// per-triangle XY AABB. Projected-edge sign tests skip degenerate
// (zero-area / edge-on) projections automatically: for a collinear 2D
// projection d0+d1+d2 == 0, which can never be all-positive or
// all-negative.
__global__ void gpu_parity_sign_columns_kernel(
    const CudaTriangle* __restrict__ triangles,
    const float4* __restrict__ tri_xyab,   // (minx, miny, maxx, maxy)
    const int* __restrict__ sup_start,     // CSR starts, per XY supercell
    const int* __restrict__ sup_tris,      // CSR triangle ids
    float voxel_size,
    int min_x, int min_y, int min_z,
    int dim_x, int dim_y, int dim_z,
    int sup_dim_x,
    uint8_t* __restrict__ out_sign,        // 0 outside, 1 inside, 2 unknown
    int* __restrict__ out_unknown_flag
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cols = dim_x * dim_y;
    if (col >= num_cols) return;

    int cx = col % dim_x;
    int cy = col / dim_x;

    const float JX = 0.037f;
    const float JY = 0.051f;
    float rx = (min_x + cx + JX) * voxel_size;
    float ry = (min_y + cy + JY) * voxel_size;

    float hit_z[RHI_PARITY_MAX_HITS];
    int n_hits = 0;
    bool bad_column = false;

    int sup = (cy >> 2) * sup_dim_x + (cx >> 2);
    for (int e = sup_start[sup]; e < sup_start[sup + 1]; ++e) {
        int ti = sup_tris[e];
        float4 ab = tri_xyab[ti];
        if (rx < ab.x || rx > ab.z || ry < ab.y || ry > ab.w) continue;

        const CudaTriangle& tri = triangles[ti];
        float d0 = (tri.v1.x - tri.v0.x) * (ry - tri.v0.y) - (tri.v1.y - tri.v0.y) * (rx - tri.v0.x);
        float d1 = (tri.v2.x - tri.v1.x) * (ry - tri.v1.y) - (tri.v2.y - tri.v1.y) * (rx - tri.v1.x);
        float d2 = (tri.v0.x - tri.v2.x) * (ry - tri.v2.y) - (tri.v0.y - tri.v2.y) * (rx - tri.v2.x);
        bool pos = (d0 > 0.0f && d1 > 0.0f && d2 > 0.0f);
        bool neg = (d0 < 0.0f && d1 < 0.0f && d2 < 0.0f);
        if (!pos && !neg) continue; // ray outside the projected triangle
        float area2 = d0 + d1 + d2; // 2 * signed projected area
        if (fabsf(area2) < 1e-12f) continue;
        // Barycentric interpolation of the hit height: vertex i's weight is
        // the edge function of the OPPOSITE edge (v0<->d1, v1<->d2, v2<->d0).
        float z_hit = (d1 * tri.v0.z + d2 * tri.v1.z + d0 * tri.v2.z) / area2;
        if (n_hits < RHI_PARITY_MAX_HITS) {
            hit_z[n_hits++] = z_hit;
        } else {
            bad_column = true;
            break;
        }
    }

    size_t col_base = (size_t)cy * (size_t)dim_x + (size_t)cx;
    size_t zstride = (size_t)dim_y * (size_t)dim_x;

    if (bad_column || (n_hits & 1) != 0) {
        // Overflowing or odd hit count: this column leaks (open boundary)
        // or clipped a degenerate configuration — defer to the exact fixup.
        for (int z = 0; z < dim_z; ++z) {
            out_sign[(size_t)z * zstride + col_base] = 2;
        }
        *out_unknown_flag = 1;
        return;
    }

    // Insertion sort (a vessel column crosses the shell a handful of times).
    for (int i = 1; i < n_hits; ++i) {
        float key = hit_z[i];
        int j = i - 1;
        while (j >= 0 && hit_z[j] > key) {
            hit_z[j + 1] = hit_z[j];
            --j;
        }
        hit_z[j + 1] = key;
    }

    // Alternate outside/inside between consecutive crossings.
    int h = 0;
    bool inside = false;
    for (int z = 0; z < dim_z; ++z) {
        float zc = (min_z + z) * voxel_size;
        while (h < n_hits && hit_z[h] < zc) {
            inside = !inside;
            ++h;
        }
        out_sign[(size_t)z * zstride + col_base] = inside ? 1 : 0;
    }
}

// [C] Exact winding fixup for voxels the parity pass could not classify.
// Full-volume grid-stride launch with an early-out: only the (rare)
// unknown voxels pay the O(triangles) exact sum. Same classification
// threshold as the legacy kernel (|w| >= 0.5 -> inside).
__global__ void gpu_exact_winding_fixup_kernel(
    const CudaTriangle* __restrict__ triangles,
    int num_triangles,
    float voxel_size,
    int min_x, int min_y, int min_z,
    int dim_x, int dim_y, int dim_z,
    uint8_t* __restrict__ io_sign,
    int* __restrict__ out_fixed_count
) {
    size_t n = (size_t)dim_x * (size_t)dim_y * (size_t)dim_z;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (io_sign[i] != 2) continue;
        int x = (int)(i % (size_t)dim_x);
        int y = (int)((i / (size_t)dim_x) % (size_t)dim_y);
        int z = (int)(i / ((size_t)dim_x * (size_t)dim_y));
        glm::vec3 p((min_x + x) * voxel_size, (min_y + y) * voxel_size, (min_z + z) * voxel_size);
        float solid_angle_sum = 0.0f;
        for (int t = 0; t < num_triangles; ++t) {
            const CudaTriangle& tri = triangles[t];
            solid_angle_sum += triangle_solid_angle(p, tri.v0, tri.v1, tri.v2);
        }
        float winding = solid_angle_sum * 0.07957747154594767f; // 1 / (4*pi)
        io_sign[i] = (fabsf(winding) >= 0.5f) ? 1 : 0;
        atomicAdd(out_fixed_count, 1);
    }
}

// [D] Godunov upwind eikonal update from the per-axis neighbor minima —
// identical arithmetic to gpu_eikonal_update_kernel below (v23.29), factored
// out for the directional FSM sweeps.
__device__ __forceinline__ float rhi_eikonal_godunov(float ax, float ay, float az, float h) {
    float s0 = fminf(ax, fminf(ay, az));
    float s2 = fmaxf(ax, fmaxf(ay, az));
    float s1 = ax + ay + az - s0 - s2;

    float updated = s0 + h;
    if (updated > s1) {
        float disc = 2.0f * h * h - (s1 - s0) * (s1 - s0);
        if (disc >= 0.0f) {
            updated = (s0 + s1 + sqrtf(disc)) * 0.5f;
            if (updated > s2) {
                float disc3 = 3.0f * h * h -
                    ((s1 - s0) * (s1 - s0) + (s2 - s1) * (s2 - s1) + (s2 - s0) * (s2 - s0)) * 0.5f;
                if (disc3 >= 0.0f) {
                    updated = (s0 + s1 + s2 + sqrtf(disc3)) / 3.0f;
                }
            }
        }
    }
    return updated;
}

// [D] One directional fast-sweeping pass. One thread per line along the two
// non-sweep axes; the swept axis is walked SEQUENTIALLY inside the thread
// (classical FSM column decomposition — the along-axis dependency is
// in-thread, so no inter-block ordering is needed for it). Values only ever
// decrease (monotone relaxation), so the asynchronous cross-axis reads are
// race-free in outcome and the loop terminates on the no-change flag.
__global__ void gpu_eikonal_fsm_sweep_kernel(
    float* __restrict__ grid,
    int dim_x, int dim_y, int dim_z,
    float h,
    int axis,  // 0 = sweep along x, 1 = along y, 2 = along z
    int dir,   // +1 ascending, -1 descending
    int* __restrict__ out_changed
) {
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    size_t xstride = 1;
    size_t ystride = (size_t)dim_x;
    size_t zstride = (size_t)dim_y * (size_t)dim_x;

    if (axis == 0) {
        int n_lines = dim_y * dim_z;
        if (line >= n_lines) return;
        int y = line % dim_y;
        int z = line / dim_y;
        size_t base = (size_t)z * zstride + (size_t)y * ystride;
        for (int s = 0; s < dim_x; ++s) {
            int x = (dir > 0) ? s : (dim_x - 1 - s);
            size_t idx = base + (size_t)x * xstride;
            float cur = grid[idx];
            float ax = fminf((x > 0) ? grid[idx - xstride] : RHI_SDF_FAR_SENTINEL,
                             (x < dim_x - 1) ? grid[idx + xstride] : RHI_SDF_FAR_SENTINEL);
            float ay = fminf((y > 0) ? grid[idx - ystride] : RHI_SDF_FAR_SENTINEL,
                             (y < dim_y - 1) ? grid[idx + ystride] : RHI_SDF_FAR_SENTINEL);
            float az = fminf((z > 0) ? grid[idx - zstride] : RHI_SDF_FAR_SENTINEL,
                             (z < dim_z - 1) ? grid[idx + zstride] : RHI_SDF_FAR_SENTINEL);
            float updated = rhi_eikonal_godunov(ax, ay, az, h);
            if (updated < cur) {
                grid[idx] = updated;
                *out_changed = 1;
            }
        }
    } else if (axis == 1) {
        int n_lines = dim_x * dim_z;
        if (line >= n_lines) return;
        int x = line % dim_x;
        int z = line / dim_x;
        size_t base = (size_t)z * zstride + (size_t)x * xstride;
        for (int s = 0; s < dim_y; ++s) {
            int y = (dir > 0) ? s : (dim_y - 1 - s);
            size_t idx = base + (size_t)y * ystride;
            float cur = grid[idx];
            float ax = fminf((x > 0) ? grid[idx - xstride] : RHI_SDF_FAR_SENTINEL,
                             (x < dim_x - 1) ? grid[idx + xstride] : RHI_SDF_FAR_SENTINEL);
            float ay = fminf((y > 0) ? grid[idx - ystride] : RHI_SDF_FAR_SENTINEL,
                             (y < dim_y - 1) ? grid[idx + ystride] : RHI_SDF_FAR_SENTINEL);
            float az = fminf((z > 0) ? grid[idx - zstride] : RHI_SDF_FAR_SENTINEL,
                             (z < dim_z - 1) ? grid[idx + zstride] : RHI_SDF_FAR_SENTINEL);
            float updated = rhi_eikonal_godunov(ax, ay, az, h);
            if (updated < cur) {
                grid[idx] = updated;
                *out_changed = 1;
            }
        }
    } else {
        int n_lines = dim_x * dim_y;
        if (line >= n_lines) return;
        int x = line % dim_x;
        int y = line / dim_x;
        size_t base = (size_t)y * ystride + (size_t)x * xstride;
        for (int s = 0; s < dim_z; ++s) {
            int z = (dir > 0) ? s : (dim_z - 1 - s);
            size_t idx = base + (size_t)z * zstride;
            float cur = grid[idx];
            float ax = fminf((x > 0) ? grid[idx - xstride] : RHI_SDF_FAR_SENTINEL,
                             (x < dim_x - 1) ? grid[idx + xstride] : RHI_SDF_FAR_SENTINEL);
            float ay = fminf((y > 0) ? grid[idx - ystride] : RHI_SDF_FAR_SENTINEL,
                             (y < dim_y - 1) ? grid[idx + ystride] : RHI_SDF_FAR_SENTINEL);
            float az = fminf((z > 0) ? grid[idx - zstride] : RHI_SDF_FAR_SENTINEL,
                             (z < dim_z - 1) ? grid[idx + zstride] : RHI_SDF_FAR_SENTINEL);
            float updated = rhi_eikonal_godunov(ax, ay, az, h);
            if (updated < cur) {
                grid[idx] = updated;
                *out_changed = 1;
            }
        }
    }
}

// [D2] (v24.06 Track D R3) Far-sentinel audit: counts voxels still at or
// above the 1e29 "unreached" threshold after the eikonal rounds. A nonzero
// count means the FSM cap truncated the wavefront; those voxels would
// otherwise be clamped to ±1e6 by [F] and shipped, and the viewport
// raymarcher steps 0.85 × 1e6 per sample there — whole rays leap out of
// the volume and the rotating model appears partially trimmed by
// plane-like cuts (the v24.05 report). The host uses this count to extend
// the sweeps adaptively and, as a final safety net, to clamp leftovers.
__global__ void gpu_count_far_sentinels_kernel(
    const float* __restrict__ grid,
    size_t n,
    int* __restrict__ out_count
) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (grid[i] >= 1e29f) {
            atomicAdd(out_count, 1);
        }
    }
}

// [D3] (v24.06 Track D R3) Final safety clamp: any voxel still unreached
// after all extension rounds is written a conservative, sign-aware
// band-edge magnitude (inside -> -band_width, outside -> +band_width).
// The sign comes from the parity/winding classification, so no zero
// crossing can be created or destroyed; the true surface keeps its exact
// [A] band values. Every consumer (viewport raymarcher, marching cubes,
// clearance heuristics) sees a conservative underestimate of |d| — never
// the ±1e6 far-sentinel junk.
__global__ void gpu_clamp_far_sentinels_kernel(
    float* __restrict__ io_dist,
    const uint8_t* __restrict__ signs,
    size_t n,
    float band_width
) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (io_dist[i] >= 1e29f) {
            io_dist[i] = (signs[i] == 1) ? -band_width : band_width;
        }
    }
}

// [E] Sampled verification (v24.05 Track D R2): deterministic stratified
// voxel centers are re-classified with the legacy EXACT winding sum and
// compared with the parity signs. Mismatched sample voxels are RECORDED
// (out_sites, capacity num_samples) so the host can patch exact winding
// around them. Probe generation g selects the stratification offset
// {1/2, 1/6, 5/6} — generation 0 is bit-identical to the v24.04 probe;
// generations 1 and 2 interleave fresh positions, so the union of the
// probes is a 3x denser stratification of the volume.
__global__ void gpu_verify_sign_samples_kernel(
    const uint8_t* __restrict__ signs,
    const CudaTriangle* __restrict__ triangles,
    int num_triangles,
    float voxel_size,
    int min_x, int min_y, int min_z,
    int dim_x, int dim_y, int dim_z,
    size_t total_voxels,
    int num_samples,
    int probe_gen,
    int* __restrict__ out_mismatches,
    int* __restrict__ out_sites
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_samples) return;

    double off = (probe_gen == 0) ? 0.5
               : (probe_gen == 1) ? (1.0 / 6.0)
                                  : (5.0 / 6.0);
    size_t vidx = (size_t)(((double)i + off) * (double)total_voxels / (double)num_samples);
    if (vidx >= total_voxels) vidx = total_voxels - 1;

    uint8_t s = signs[vidx];
    if (s == 2) return; // already exact via the fixup

    int x = (int)(vidx % (size_t)dim_x);
    int y = (int)((vidx / (size_t)dim_x) % (size_t)dim_y);
    int z = (int)(vidx / ((size_t)dim_x * (size_t)dim_y));
    glm::vec3 p((min_x + x) * voxel_size, (min_y + y) * voxel_size, (min_z + z) * voxel_size);

    float solid_angle_sum = 0.0f;
    for (int t = 0; t < num_triangles; ++t) {
        const CudaTriangle& tri = triangles[t];
        solid_angle_sum += triangle_solid_angle(p, tri.v0, tri.v1, tri.v2);
    }
    float winding = solid_angle_sum * 0.07957747154594767f; // 1 / (4*pi)
    bool exact_inside = (fabsf(winding) >= 0.5f);
    if ((s == 1) != exact_inside) {
        int pos = atomicAdd(out_mismatches, 1);
        if (pos < num_samples) out_sites[pos] = (int)vidx;
    }
}

// [E2] Patch marking (v24.05 Track D R2): one BLOCK per mismatch site;
// the block's threads stride the site's (2r+1)^3 box and mark every voxel
// in it unknown (sign = 2) so the exact winding fixup [C] recomputes it
// with the legacy per-voxel sum. Overlapping boxes write the same value
// (benign); a box may cover already-exact voxels — recomputing them is
// idempotent (same sequential triangle order, same floats).
__global__ void gpu_mark_patch_unknown_kernel(
    uint8_t* __restrict__ io_sign,
    const int* __restrict__ sites,
    int num_sites,
    int radius,
    int dim_x, int dim_y, int dim_z
) {
    int s = blockIdx.x;
    if (s >= num_sites) return;
    long long v = (long long)sites[s];
    int cx = (int)((size_t)v % (size_t)dim_x);
    int cy = (int)(((size_t)v / (size_t)dim_x) % (size_t)dim_y);
    int cz = (int)((size_t)v / ((size_t)dim_x * (size_t)dim_y));
    int x0 = cx - radius; if (x0 < 0) x0 = 0;
    int y0 = cy - radius; if (y0 < 0) y0 = 0;
    int z0 = cz - radius; if (z0 < 0) z0 = 0;
    int x1 = (cx + radius < dim_x - 1) ? (cx + radius) : (dim_x - 1);
    int y1 = (cy + radius < dim_y - 1) ? (cy + radius) : (dim_y - 1);
    int z1 = (cz + radius < dim_z - 1) ? (cz + radius) : (dim_z - 1);
    long long nx = x1 - x0 + 1;
    long long ny = y1 - y0 + 1;
    long long box = nx * ny * (long long)(z1 - z0 + 1);
    size_t ystride = (size_t)dim_x;
    size_t zstride = (size_t)dim_y * (size_t)dim_x;
    for (long long k = threadIdx.x; k < box; k += blockDim.x) {
        int tz = (int)(k / (nx * ny));
        long long rem = k - (long long)tz * nx * ny;
        int ty = (int)(rem / nx);
        int tx = (int)(rem - (long long)ty * nx);
        io_sign[(size_t)(z0 + tz) * zstride + (size_t)(y0 + ty) * ystride + (size_t)(x0 + tx)] = 2;
    }
}

// [F] Sign apply: inside -> negative (legacy convention), plus a defensive
// clamp of any unreached far sentinel to the legacy 1e6 init constant.
__global__ void gpu_apply_signs_kernel(
    float* __restrict__ io_dist,
    const uint8_t* __restrict__ signs,
    size_t n
) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float d = io_dist[i];
        if (d > 1e29f) d = 1e6f;
        io_dist[i] = (signs[i] == 1) ? -d : d;
    }
}

// --- 3. CUDA Kernel: Capsule Rasterizer with Shared Memory Tiling ---
__global__ void rasterize_capsules_kernel(
    const CudaCapsule* __restrict__ capsules,
    int num_capsules,
    float voxel_size,
    float background,
    float smooth_k,
    int min_x, int min_y, int min_z,
    int dim_x, int dim_y, int dim_z,
    float* __restrict__ out_grid
) {
    __shared__ CudaCapsule s_capsules[RASTER_TILE_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int linear_tid = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int num_block_threads = blockDim.x * blockDim.y * blockDim.z;

    bool valid_voxel = (x < dim_x && y < dim_y && z < dim_z);
    size_t idx = valid_voxel ? ((size_t)z * dim_y * dim_x + (size_t)y * dim_x + (size_t)x) : 0;

    float px = (min_x + x) * voxel_size;
    float py = (min_y + y) * voxel_size;
    float pz = (min_z + z) * voxel_size;

    float min_d = background;
    bool found_any = false;

    for (int tile_start = 0; tile_start < num_capsules; tile_start += RASTER_TILE_SIZE) {
        int tile_count = (num_capsules - tile_start < RASTER_TILE_SIZE) ? (num_capsules - tile_start) : RASTER_TILE_SIZE;

        for (int load_idx = linear_tid; load_idx < tile_count; load_idx += num_block_threads) {
            s_capsules[load_idx] = capsules[tile_start + load_idx];
        }
        __syncthreads();

        if (valid_voxel) {
            for (int i = 0; i < tile_count; ++i) {
                const CudaCapsule& c = s_capsules[i];
                if (px < c.min_x || px > c.max_x ||
                    py < c.min_y || py > c.max_y ||
                    pz < c.min_z || pz > c.max_z) {
                    continue;
                }

                float bax = c.bx - c.ax;
                float bay = c.by - c.ay;
                float baz = c.bz - c.az;
                float pax = px - c.ax;
                float pay = py - c.ay;
                float paz = pz - c.az;

                float l2 = bax * bax + bay * bay + baz * baz;
                float d;
                if (l2 < 1e-8f) {
                    d = sqrtf(pax * pax + pay * pay + paz * paz) - c.ra;
                } else {
                    float t = (pax * bax + pay * bay + paz * baz) / l2;
                    t = fminf(fmaxf(t, 0.0f), 1.0f);
                    float proj_x = c.ax + t * bax;
                    float proj_y = c.ay + t * bay;
                    float proj_z = c.az + t * baz;
                    float r = c.ra + t * (c.rb - c.ra);
                    float dx = px - proj_x;
                    float dy = py - proj_y;
                    float dz = pz - proj_z;
                    d = sqrtf(dx * dx + dy * dy + dz * dz) - r;
                }

                if (smooth_k > 1e-5f) {
                    if (!found_any) {
                        min_d = fminf(min_d, d);
                        found_any = true;
                    } else {
                        float hk = fmaxf(smooth_k - fabsf(min_d - d), 0.0f) / smooth_k;
                        min_d = fminf(min_d, d) - hk * hk * smooth_k * 0.25f;
                    }
                } else {
                    if (d < min_d) {
                        min_d = d;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (valid_voxel) {
        out_grid[idx] = min_d;
    }
}

#endif // __CUDACC__

bool CudaRasterizer::is_cuda_available() {
#ifdef __CUDACC__
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

bool CudaRasterizer::query_gpu_memory(size_t& out_free_bytes, size_t& out_total_bytes) {
#ifdef __CUDACC__
    if (!is_cuda_available()) {
        out_free_bytes = 0;
        out_total_bytes = 0;
        return false;
    }
    cudaError_t err = cudaMemGetInfo(&out_free_bytes, &out_total_bytes);
    return (err == cudaSuccess);
#else
    out_free_bytes = 0;
    out_total_bytes = 0;
    return false;
#endif
}

// =========================================================================
// v24.04 TRACK D — FAST SDF HOST ORCHESTRATION
// =========================================================================
// Contract: on success fills out_distances completely and returns true
// (out_time_ms = fast-path wall time). On failure leaves out_distances
// UNTOUCHED and returns false — the caller falls back to the legacy exact
// kernel, so the worst case is exactly the v24.03 behavior.
// (Guarded like every other CUDA-touching host block in this TU: the file
// also compiles without nvcc, where none of this exists.)
// =========================================================================
#ifdef __CUDACC__
static bool rhi_compute_mesh_sdf_fast(
    const std::vector<CudaTriangle>& triangles,
    float voxel_size,
    glm::ivec3 min_coord,
    glm::ivec3 max_coord,
    std::vector<float>& out_distances,
    float& out_time_ms
) {
    auto t_start = std::chrono::high_resolution_clock::now();

    int dim_x = max_coord.x - min_coord.x + 1;
    int dim_y = max_coord.y - min_coord.y + 1;
    int dim_z = max_coord.z - min_coord.z + 1;
    size_t total_voxels = (size_t)dim_x * (size_t)dim_y * (size_t)dim_z;
    int num_tris = static_cast<int>(triangles.size());
    if (dim_x <= 0 || dim_y <= 0 || dim_z <= 0 || num_tris <= 0 || total_voxels == 0) {
        return false;
    }
    // Thread/line counts must stay inside int range for the 1D launches.
    // v24.05 R2: d_sites stores int voxel indices — keep the grid in int range.
    if ((long long)total_voxels > 2000000000LL) return false;
    if ((long long)num_tris * (long long)dim_z > 2000000000LL) return false;
    if ((long long)dim_x * (long long)dim_y > 2000000000LL) return false;
    if ((long long)dim_x * (long long)dim_z > 2000000000LL) return false;
    if ((long long)dim_y * (long long)dim_z > 2000000000LL) return false;

    float band_width = (float)RHI_SDF_BAND_VOXELS * voxel_size;

    // ---- Host-side XY supercell CSR (4x4 voxel cells) --------------------
    const int SUP = 4;
    int sup_dim_x = (dim_x + SUP - 1) / SUP;
    int sup_dim_y = (dim_y + SUP - 1) / SUP;
    size_t num_sup = (size_t)sup_dim_x * (size_t)sup_dim_y;

    std::vector<float4> tri_xyab(num_tris);
    std::vector<int> sup_start(num_sup + 1, 0);
    std::vector<int> tri_sx0(num_tris), tri_sx1(num_tris);
    std::vector<int> tri_sy0(num_tris), tri_sy1(num_tris);

    long long total_entries = 0;
    for (int t = 0; t < num_tris; ++t) {
        const CudaTriangle& tri = triangles[t];
        float minx = std::min(tri.v0.x, std::min(tri.v1.x, tri.v2.x));
        float maxx = std::max(tri.v0.x, std::max(tri.v1.x, tri.v2.x));
        float miny = std::min(tri.v0.y, std::min(tri.v1.y, tri.v2.y));
        float maxy = std::max(tri.v0.y, std::max(tri.v1.y, tri.v2.y));
        tri_xyab[t] = make_float4(minx, miny, maxx, maxy);

        // Conservative column coverage (+-1 column slack; the kernel's
        // per-triangle AABB test rejects non-overlapping candidates).
        int cx0 = (int)std::floor(minx / voxel_size) - min_coord.x - 1;
        int cx1 = (int)std::ceil(maxx / voxel_size) - min_coord.x + 1;
        int cy0 = (int)std::floor(miny / voxel_size) - min_coord.y - 1;
        int cy1 = (int)std::ceil(maxy / voxel_size) - min_coord.y + 1;

        int sx0 = std::max(0, cx0 / SUP);
        int sx1 = std::min(sup_dim_x - 1, cx1 / SUP);
        int sy0 = std::max(0, cy0 / SUP);
        int sy1 = std::min(sup_dim_y - 1, cy1 / SUP);

        tri_sx0[t] = sx0; tri_sx1[t] = sx1;
        tri_sy0[t] = sy0; tri_sy1[t] = sy1;

        if (sx1 < sx0 || sy1 < sy0) continue;
        for (int sy = sy0; sy <= sy1; ++sy) {
            for (int sx = sx0; sx <= sx1; ++sx) {
                ++sup_start[sy * sup_dim_x + sx + 1];
            }
        }
        total_entries += (long long)(sx1 - sx0 + 1) * (sy1 - sy0 + 1);
    }
    if (total_entries <= 0 || total_entries > 100000000LL) {
        return false; // pathological triangle spread -> legacy path
    }
    for (size_t i = 1; i <= num_sup; ++i) sup_start[i] += sup_start[i - 1];

    std::vector<int> sup_tris((size_t)total_entries);
    {
        std::vector<int> cursor(sup_start.begin(), sup_start.end() - 1);
        for (int t = 0; t < num_tris; ++t) {
            for (int sy = tri_sy0[t]; sy <= tri_sy1[t]; ++sy) {
                for (int sx = tri_sx0[t]; sx <= tri_sx1[t]; ++sx) {
                    sup_tris[cursor[sy * sup_dim_x + sx]++] = t;
                }
            }
        }
    }

    // ---- Device buffers ---------------------------------------------------
    CudaTriangle* d_tris = nullptr;
    float4* d_xyab = nullptr;
    int* d_sup_start = nullptr;
    int* d_sup_tris = nullptr;
    float* d_dist = nullptr;
    uint8_t* d_sign = nullptr;
    int* d_flag = nullptr; // reused: unknown flag / fixed count / changed / mismatches
    int* d_sites = nullptr; // v24.05 R2: mismatched sample voxel ids (verify -> patch)

    auto fast_cleanup = [&]() {
        if (d_tris) cudaFree(d_tris);
        if (d_xyab) cudaFree(d_xyab);
        if (d_sup_start) cudaFree(d_sup_start);
        if (d_sup_tris) cudaFree(d_sup_tris);
        if (d_dist) cudaFree(d_dist);
        if (d_sign) cudaFree(d_sign);
        if (d_flag) cudaFree(d_flag);
        if (d_sites) cudaFree(d_sites);
    };

    bool alloc_ok = true;
    alloc_ok = alloc_ok && cudaMalloc(&d_tris, (size_t)num_tris * sizeof(CudaTriangle)) == cudaSuccess;
    alloc_ok = alloc_ok && cudaMalloc(&d_xyab, (size_t)num_tris * sizeof(float4)) == cudaSuccess;
    alloc_ok = alloc_ok && cudaMalloc(&d_sup_start, (num_sup + 1) * sizeof(int)) == cudaSuccess;
    alloc_ok = alloc_ok && cudaMalloc(&d_sup_tris, (size_t)total_entries * sizeof(int)) == cudaSuccess;
    alloc_ok = alloc_ok && cudaMalloc(&d_dist, total_voxels * sizeof(float)) == cudaSuccess;
    alloc_ok = alloc_ok && cudaMalloc(&d_sign, total_voxels * sizeof(uint8_t)) == cudaSuccess;
    alloc_ok = alloc_ok && cudaMalloc(&d_flag, sizeof(int)) == cudaSuccess;
    alloc_ok = alloc_ok && cudaMalloc(&d_sites, (size_t)RHI_VERIFY_SAMPLES * sizeof(int)) == cudaSuccess;
    if (!alloc_ok) {
        fast_cleanup();
        RhizomeLog("CUDA", "Fast SDF: VRAM allocation failed - falling back to exact winding kernel");
        return false;
    }

    cudaMemcpy(d_tris, triangles.data(), (size_t)num_tris * sizeof(CudaTriangle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xyab, tri_xyab.data(), (size_t)num_tris * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sup_start, sup_start.data(), (num_sup + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sup_tris, sup_tris.data(), (size_t)total_entries * sizeof(int), cudaMemcpyHostToDevice);

    const int THREADS = 256;

    // [init] far-sentinel fill (float pattern -> dedicated kernel)
    {
        size_t blocks = std::min<size_t>((total_voxels + THREADS - 1) / THREADS, 32768);
        gpu_fill_float_kernel<<<(unsigned)blocks, THREADS>>>(d_dist, total_voxels, RHI_SDF_FAR_SENTINEL);
    }

    // [A] exact band distances (per-triangle splat)
    {
        long long total_threads = (long long)num_tris * (long long)dim_z;
        unsigned blocks = (unsigned)((total_threads + THREADS - 1) / THREADS);
        gpu_splat_band_distances_kernel<<<blocks, THREADS>>>(
            d_tris, num_tris, voxel_size, band_width,
            min_coord.x, min_coord.y, min_coord.z,
            dim_x, dim_y, dim_z, d_dist);
    }

    // [B] scanline parity signs
    int h_unknown = 0;
    {
        long long num_cols = (long long)dim_x * (long long)dim_y;
        unsigned blocks = (unsigned)((num_cols + THREADS - 1) / THREADS);
        cudaMemset(d_flag, 0, sizeof(int));
        gpu_parity_sign_columns_kernel<<<blocks, THREADS>>>(
            d_tris, d_xyab, d_sup_start, d_sup_tris,
            voxel_size, min_coord.x, min_coord.y, min_coord.z,
            dim_x, dim_y, dim_z, sup_dim_x, d_sign, d_flag);
        cudaMemcpy(&h_unknown, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    }
    auto t_parity_done = std::chrono::high_resolution_clock::now();

    // [C] exact winding fixup (rare: leaking/degenerate columns only)
    int fixup_voxels = 0;
    if (h_unknown != 0) {
        cudaMemset(d_flag, 0, sizeof(int));
        size_t blocks = std::min<size_t>((total_voxels + THREADS - 1) / THREADS, 32768);
        gpu_exact_winding_fixup_kernel<<<(unsigned)blocks, THREADS>>>(
            d_tris, num_tris, voxel_size,
            min_coord.x, min_coord.y, min_coord.z,
            dim_x, dim_y, dim_z, d_sign, d_flag);
        cudaMemcpy(&fixup_voxels, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    }

    // [D] FSM eikonal re-distancing: 6 directional sweeps per round until
    // nothing changes (band seeds surround every far voxel, so typical
    // grids converge in 2-3 rounds).
    //
    // v24.06 Track D R3: reporting the cap ("eikonal 8 rounds") was
    // ambiguous — the same number is printed when the field converged
    // exactly at the cap AND when the wavefront was truncated with far
    // voxels still at the 1e30 sentinel. In the truncated case [F] clamps
    // those voxels to ±1e6 and ships them; the viewport raymarcher then
    // advances 0.85 × 1e6 per sample, whole rays leap out of the volume,
    // and the rotating model appears partially trimmed by plane-like cuts
    // (user report on the first v24.05 build, mug.obj at the 0.10 mm
    // default: "partially trimmed by random 'plane'"). R3 adds a sentinel
    // AUDIT after the rounds: a nonzero residual drives up to 16 extra
    // sweep rounds (audit-gated, so the ulp-level tail that keeps the
    // changed flag hot cannot waste rounds), and any true leftover is
    // safe-clamped to sign-aware band-edge magnitudes. A converged field
    // costs exactly one extra audit pass over d_dist.
    int eikonal_rounds = 0;
    long long far_residual = 0;   // R3: unreached voxels after all rounds
    int extra_rounds = 0;        // R3: audit-gated extension rounds run
    long long safe_clamped = 0;  // R3: voxels clamped by the final net
    {
        auto launch_sweep = [&](int axis, int dir) {
            int n_lines = 1;
            if (axis == 0) n_lines = dim_y * dim_z;
            else if (axis == 1) n_lines = dim_x * dim_z;
            else n_lines = dim_x * dim_y;
            unsigned blocks = (unsigned)((n_lines + THREADS - 1) / THREADS);
            gpu_eikonal_fsm_sweep_kernel<<<blocks, THREADS>>>(
                d_dist, dim_x, dim_y, dim_z, voxel_size, axis, dir, d_flag);
        };
        auto audit_sentinels = [&]() -> long long {
            cudaMemset(d_flag, 0, sizeof(int));
            size_t blocks = std::min<size_t>((total_voxels + THREADS - 1) / THREADS, 32768);
            gpu_count_far_sentinels_kernel<<<(unsigned)blocks, THREADS>>>(
                d_dist, total_voxels, d_flag);
            int cnt = 0;
            cudaMemcpy(&cnt, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
            return static_cast<long long>(cnt);
        };
        const int MAX_ROUNDS = 8;
        for (int r = 0; r < MAX_ROUNDS; ++r) {
            cudaMemset(d_flag, 0, sizeof(int));
            launch_sweep(0, +1);
            launch_sweep(0, -1);
            launch_sweep(1, +1);
            launch_sweep(1, -1);
            launch_sweep(2, +1);
            launch_sweep(2, -1);
            int changed = 1;
            cudaMemcpy(&changed, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
            eikonal_rounds = r + 1;
            if (changed == 0) break;
        }
        // R3: the changed flag cannot distinguish "converged at the cap"
        // from "truncated at the cap"; the residual count can.
        far_residual = audit_sentinels();
        const int MAX_EXTRA_ROUNDS = 16;
        while (far_residual > 0 && extra_rounds < MAX_EXTRA_ROUNDS) {
            launch_sweep(0, +1);
            launch_sweep(0, -1);
            launch_sweep(1, +1);
            launch_sweep(1, -1);
            launch_sweep(2, +1);
            launch_sweep(2, -1);
            ++extra_rounds;
            far_residual = audit_sentinels();
        }
        eikonal_rounds += extra_rounds;
        if (far_residual > 0) {
            char emsg[224];
            snprintf(emsg, sizeof(emsg),
                     "Fast SDF eikonal: %lld far voxels still unreached after %d rounds (cap reached) - applying safe clamp",
                     far_residual, eikonal_rounds);
            RhizomeLog("CUDA", emsg);
        } else if (extra_rounds > 0) {
            char emsg[224];
            snprintf(emsg, sizeof(emsg),
                     "Fast SDF eikonal: %d extra audit-gated rounds completed the far field (residual now 0)",
                     extra_rounds);
            RhizomeLog("CUDA", emsg);
        }
        if (far_residual > 0) {
            // Final safety net (pathological geometry only): sign-aware
            // clamp to the band-edge magnitude. No zero crossing can be
            // created or destroyed (signs untouched); the true surface
            // keeps its exact [A] band values; every consumer sees a
            // conservative underestimate of |d| instead of ±1e6 junk.
            safe_clamped = far_residual;
            size_t blocks = std::min<size_t>((total_voxels + THREADS - 1) / THREADS, 32768);
            gpu_clamp_far_sentinels_kernel<<<(unsigned)blocks, THREADS>>>(
                d_dist, d_sign, total_voxels, band_width);
            far_residual = audit_sentinels();  // expect 0 after the clamp
        }
    }
    auto t_eikonal_done = std::chrono::high_resolution_clock::now();

    // [E] sampled exact-winding verification — iterative patch policy
    // (v24.05 Track D R2). Generation 0 probes the exact v24.04 sample
    // positions; a small number of mismatched sites (compact divergences:
    // handle/wall self-overlap junctions, sub-voxel wall slivers) is
    // patched in place — a (2r+1)^3 box around each site is marked
    // unknown and recomputed with the exact winding fixup [C] — then
    // re-probed with fresh stratifications. Accepted only on a clean
    // probe; systematic divergence (a probe above the per-probe site cap,
    // or persistence across all generations) still falls back to the
    // legacy kernel, so the worst case remains exactly the v24.04
    // behavior (fast attempt discarded, legacy rebuild).
    int verify_probes_used = 0;
    int total_sites = 0;
    long long patched_voxels_total = 0;
    int patch_rounds = 0;
    int last_probe_m = 0;
    bool verify_ok = false;
    {
        const int NUM_SAMPLES = RHI_VERIFY_SAMPLES;
        unsigned blocks = (unsigned)((NUM_SAMPLES + THREADS - 1) / THREADS);
        size_t fixup_blocks = std::min<size_t>((total_voxels + THREADS - 1) / THREADS, 32768);
        for (int gen = 0; gen < RHI_VERIFY_PROBE_GENS; ++gen) {
            int m = 0;
            cudaMemset(d_flag, 0, sizeof(int));
            gpu_verify_sign_samples_kernel<<<blocks, THREADS>>>(
                d_sign, d_tris, num_tris, voxel_size,
                min_coord.x, min_coord.y, min_coord.z,
                dim_x, dim_y, dim_z, total_voxels, NUM_SAMPLES, gen, d_flag, d_sites);
            cudaMemcpy(&m, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
            ++verify_probes_used;
            last_probe_m = m;
            if (m == 0) { verify_ok = true; break; }
            if (m > NUM_SAMPLES) m = NUM_SAMPLES; // clamp to the recorded-site capacity
            if (m > RHI_VERIFY_MAX_SITES_PER_PROBE) break;  // systematic divergence
            if (gen == RHI_VERIFY_PROBE_GENS - 1) break;    // persistent after 2 patches
            char pmsg[192];
            snprintf(pmsg, sizeof(pmsg),
                     "Fast SDF verify: %d/%d sign mismatches (probe gen %d) - patching exact winding boxes (r=%d)",
                     m, NUM_SAMPLES, gen, RHI_SDF_PATCH_RADIUS_VOXELS);
            RhizomeLog("CUDA", pmsg);
            // Patch: mark a (2r+1)^3 box around every mismatched site as
            // unknown, then let the exact fixup recompute just those.
            gpu_mark_patch_unknown_kernel<<<m, THREADS>>>(
                d_sign, d_sites, m, RHI_SDF_PATCH_RADIUS_VOXELS,
                dim_x, dim_y, dim_z);
            cudaMemset(d_flag, 0, sizeof(int));
            gpu_exact_winding_fixup_kernel<<<(unsigned)fixup_blocks, THREADS>>>(
                d_tris, num_tris, voxel_size,
                min_coord.x, min_coord.y, min_coord.z,
                dim_x, dim_y, dim_z, d_sign, d_flag);
            int fixed = 0;
            cudaMemcpy(&fixed, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
            patched_voxels_total += fixed;
            total_sites += m;
            ++patch_rounds;
        }
    }
    if (!verify_ok) {
        fast_cleanup();
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Fast SDF verify: %d/%d sign mismatches persist (probe gen %d; open or self-overlapping mesh?) - rebuilding with exact winding kernel",
                 last_probe_m, RHI_VERIFY_SAMPLES, verify_probes_used - 1);
        RhizomeLog("CUDA", msg);
        return false;
    }

    // [F] apply signs, then download the finished signed field
    {
        size_t blocks = std::min<size_t>((total_voxels + THREADS - 1) / THREADS, 32768);
        gpu_apply_signs_kernel<<<(unsigned)blocks, THREADS>>>(d_dist, d_sign, total_voxels);
    }
    cudaMemcpy(out_distances.data(), d_dist, total_voxels * sizeof(float), cudaMemcpyDeviceToHost);

    fast_cleanup();

    auto t_end = std::chrono::high_resolution_clock::now();
    out_time_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    {
        float ms_to_parity = std::chrono::duration<float, std::milli>(t_parity_done - t_start).count();
        float ms_eikonal = std::chrono::duration<float, std::milli>(t_eikonal_done - t_parity_done).count();
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "Fast SDF path OK: %.0f ms total (splat+parity %.0f ms | eikonal %d rounds %.0f ms | verify+sign+copy %.0f ms) | %dx%dx%d voxels, %d tris, band %d vox, fixup %d | verify %d probes, %d sites, %lld voxels patched in %d rounds | far residual %lld, extra rounds %d, safe-clamped %lld",
                 out_time_ms, ms_to_parity, eikonal_rounds, ms_eikonal,
                 out_time_ms - ms_to_parity - ms_eikonal,
                 dim_x, dim_y, dim_z, num_tris, RHI_SDF_BAND_VOXELS, fixup_voxels,
                 verify_probes_used, total_sites, patched_voxels_total, patch_rounds,
                 far_residual, extra_rounds, safe_clamped);
        RhizomeLog("CUDA", msg);
    }
    return true;
}
#endif // __CUDACC__ (v24.04 Track D fast SDF host orchestration)

// 1. Mesh to SDF calculation
bool CudaRasterizer::compute_mesh_sdf(
    const std::vector<CudaTriangle>& triangles,
    float voxel_size,
    glm::ivec3 min_coord,
    glm::ivec3 max_coord,
    std::vector<float>& out_distances,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    int dim_x = max_coord.x - min_coord.x + 1;
    int dim_y = max_coord.y - min_coord.y + 1;
    int dim_z = max_coord.z - min_coord.z + 1;
    if (dim_x <= 0 || dim_y <= 0 || dim_z <= 0) {
        out_time_ms = 0.0f;
        return false;
    }
    size_t total_voxels = (size_t)dim_x * dim_y * dim_z;
    out_distances.resize(total_voxels, 1e6f);

#ifdef __CUDACC__
    if (is_cuda_available() && !triangles.empty()) {
        // v24.04 Track D: fast path for LARGE grids (band splat + parity
        // signs + eikonal re-distancing, exact-winding verified; see the
        // Track D header above). Grids below RHI_FAST_SDF_MIN_VOXELS —
        // which includes every verification-suite grid — take the legacy
        // exact kernel below, bit-identical to v24.03.
        if (total_voxels >= RHI_FAST_SDF_MIN_VOXELS) {
            if (rhi_compute_mesh_sdf_fast(triangles, voxel_size, min_coord, max_coord, out_distances, out_time_ms)) {
                return true;
            }
            // Fast path declined (VRAM / systematic divergence / pathological
            // CSR): out_distances is untouched — fall through to the exact kernel.
        }

        int num_tris = static_cast<int>(triangles.size());
        int num_tiles = (num_tris + RASTER_TILE_SIZE - 1) / RASTER_TILE_SIZE;
        std::vector<CudaTileHeader> tile_headers(num_tiles);

        for (int t = 0; t < num_tiles; ++t) {
            int start_idx = t * RASTER_TILE_SIZE;
            int end_idx = std::min(start_idx + RASTER_TILE_SIZE, num_tris);
            glm::vec3 bmin(1e30f);
            glm::vec3 bmax(-1e30f);
            glm::vec3 c_sum(0.0f);
            glm::vec3 a_sum(0.0f);

            for (int i = start_idx; i < end_idx; ++i) {
                const auto& tri = triangles[i];
                bmin = glm::min(bmin, glm::min(tri.v0, glm::min(tri.v1, tri.v2)));
                bmax = glm::max(bmax, glm::max(tri.v0, glm::max(tri.v1, tri.v2)));
                glm::vec3 tri_c = (tri.v0 + tri.v1 + tri.v2) * 0.33333333f;
                c_sum += tri_c;
                glm::vec3 ab = tri.v1 - tri.v0;
                glm::vec3 ac = tri.v2 - tri.v0;
                a_sum += 0.5f * glm::cross(ab, ac);
            }
            int count = end_idx - start_idx;
            glm::vec3 center = (count > 0) ? (c_sum / static_cast<float>(count)) : glm::vec3(0.0f);
            float r_sq = 0.0f;
            for (int i = start_idx; i < end_idx; ++i) {
                const auto& tri = triangles[i];
                r_sq = std::max(r_sq, glm::dot(tri.v0 - center, tri.v0 - center));
                r_sq = std::max(r_sq, glm::dot(tri.v1 - center, tri.v1 - center));
                r_sq = std::max(r_sq, glm::dot(tri.v2 - center, tri.v2 - center));
            }

            auto& th = tile_headers[t];
            th.min_x = bmin.x; th.min_y = bmin.y; th.min_z = bmin.z;
            th.max_x = bmax.x; th.max_y = bmax.y; th.max_z = bmax.z;
            th.center_x = center.x; th.center_y = center.y; th.center_z = center.z;
            th.area_x = a_sum.x; th.area_y = a_sum.y; th.area_z = a_sum.z;
            th.radius_sq = r_sq;
        }

        CudaTriangle* d_tris = nullptr;
        CudaTileHeader* d_headers = nullptr;
        float* d_grid = nullptr;
        size_t headers_bytes = tile_headers.size() * sizeof(CudaTileHeader);
        if (cudaMalloc(&d_tris, triangles.size() * sizeof(CudaTriangle)) == cudaSuccess &&
            cudaMalloc(&d_headers, headers_bytes) == cudaSuccess &&
            cudaMalloc(&d_grid, total_voxels * sizeof(float)) == cudaSuccess) {
            cudaMemcpy(d_tris, triangles.data(), triangles.size() * sizeof(CudaTriangle), cudaMemcpyHostToDevice);
            cudaMemcpy(d_headers, tile_headers.data(), headers_bytes, cudaMemcpyHostToDevice);
            dim3 block(8, 8, 8);
            dim3 grid((dim_x + 7) / 8, (dim_y + 7) / 8, (dim_z + 7) / 8);
            mesh_to_sdf_kernel<<<grid, block>>>(d_tris, d_headers, num_tris, num_tiles, voxel_size, min_coord.x, min_coord.y, min_coord.z, dim_x, dim_y, dim_z, d_grid);
            cudaMemcpy(out_distances.data(), d_grid, total_voxels * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_tris);
            cudaFree(d_headers);
            cudaFree(d_grid);
            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_tris) cudaFree(d_tris);
        if (d_headers) cudaFree(d_headers);
        if (d_grid) cudaFree(d_grid);
    }
#endif

    std::cerr << "[CUDA Error] GPU acceleration required but unavailable or out of memory in build_from_mesh.\n";
    return false;
}

// 3. Capsule Rasterizer
bool CudaRasterizer::rasterize(
    const std::vector<CudaCapsule>& capsules,
    float voxel_size,
    float background,
    glm::ivec3 min_coord,
    glm::ivec3 max_coord,
    std::vector<float>& out_distances,
    float& out_time_ms,
    float smooth_k
) {
    auto start = std::chrono::high_resolution_clock::now();
    int dim_x = max_coord.x - min_coord.x + 1;
    int dim_y = max_coord.y - min_coord.y + 1;
    int dim_z = max_coord.z - min_coord.z + 1;
    if (dim_x <= 0 || dim_y <= 0 || dim_z <= 0) {
        out_time_ms = 0.0f;
        return false;
    }
    size_t total_voxels = (size_t)dim_x * dim_y * dim_z;
    out_distances.assign(total_voxels, background);

    if (capsules.empty()) {
        out_time_ms = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        CudaCapsule* d_capsules = nullptr;
        float* d_grid = nullptr;
        if (cudaMalloc(&d_capsules, capsules.size() * sizeof(CudaCapsule)) == cudaSuccess &&
            cudaMalloc(&d_grid, total_voxels * sizeof(float)) == cudaSuccess) {
            cudaMemcpy(d_capsules, capsules.data(), capsules.size() * sizeof(CudaCapsule), cudaMemcpyHostToDevice);
            dim3 block(8, 8, 8);
            dim3 grid((dim_x + 7) / 8, (dim_y + 7) / 8, (dim_z + 7) / 8);
            rasterize_capsules_kernel<<<grid, block>>>(d_capsules, (int)capsules.size(), voxel_size, background, smooth_k, min_coord.x, min_coord.y, min_coord.z, dim_x, dim_y, dim_z, d_grid);
            cudaMemcpy(out_distances.data(), d_grid, total_voxels * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_capsules);
            cudaFree(d_grid);
            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_capsules) cudaFree(d_capsules);
        if (d_grid) cudaFree(d_grid);
    }
#endif

    std::cerr << "[CUDA Error] GPU acceleration required but unavailable or out of memory in rasterize.\n";
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_build_cuda_triangles_kernel(
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

std::shared_ptr<CudaVolumeGrid> CudaRasterizer::build_from_mesh(
    const TriangleMesh& mesh,
    float voxel_size,
    float& out_time_ms
) {
    if (mesh.vertex_count() == 0 || mesh.triangle_count() == 0) {
        return nullptr;
    }

    auto volume = std::make_shared<CudaVolumeGrid>();
    volume->voxel_size = voxel_size;

    glm::vec3 min_p(0.0f), max_p(0.0f);
    float bbox_ms = 0.0f;
    if (!compute_bounding_box_gpu(mesh, min_p, max_p, bbox_ms)) {
        return nullptr;
    }

    float padding = voxel_size * 10.0f;
    min_p -= glm::vec3(padding);
    max_p += glm::vec3(padding);

    volume->min_coord = glm::ivec3(
        static_cast<int>(std::floor(min_p.x / voxel_size)),
        static_cast<int>(std::floor(min_p.y / voxel_size)),
        static_cast<int>(std::floor(min_p.z / voxel_size))
    );
    volume->max_coord = glm::ivec3(
        static_cast<int>(std::ceil(max_p.x / voxel_size)),
        static_cast<int>(std::ceil(max_p.y / voxel_size)),
        static_cast<int>(std::ceil(max_p.z / voxel_size))
    );
    volume->dims = volume->max_coord - volume->min_coord + glm::ivec3(1);

    std::vector<CudaTriangle> triangles;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int num_verts = static_cast<int>(mesh.vertex_count());
        int num_tris = static_cast<int>(mesh.triangle_count());
        float* d_verts = nullptr;
        uint32_t* d_indices = nullptr;
        CudaTriangle* d_tris = nullptr;
        int* d_valid_count = nullptr;

        size_t vert_bytes = mesh.vertices.size() * sizeof(float);
        size_t idx_bytes = mesh.indices.size() * sizeof(uint32_t);

        if (cudaMalloc(&d_verts, vert_bytes) == cudaSuccess &&
            cudaMalloc(&d_indices, idx_bytes) == cudaSuccess &&
            cudaMalloc(&d_tris, static_cast<size_t>(num_tris) * sizeof(CudaTriangle)) == cudaSuccess &&
            cudaMalloc(&d_valid_count, sizeof(int)) == cudaSuccess) {

            cudaMemcpy(d_verts, mesh.vertices.data(), vert_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, mesh.indices.data(), idx_bytes, cudaMemcpyHostToDevice);
            cudaMemset(d_valid_count, 0, sizeof(int));

            int threads = 256;
            int blocks = (num_tris + threads - 1) / threads;
            gpu_build_cuda_triangles_kernel<<<blocks, threads>>>(
                d_verts, num_verts, d_indices, num_tris, d_tris, d_valid_count
            );

            int h_valid = 0;
            cudaMemcpy(&h_valid, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_valid > 0) {
                triangles.resize(h_valid);
                cudaMemcpy(triangles.data(), d_tris, static_cast<size_t>(h_valid) * sizeof(CudaTriangle), cudaMemcpyDeviceToHost);
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
    }
#endif

    bool ok = compute_mesh_sdf(
        triangles,
        voxel_size,
        volume->min_coord,
        volume->max_coord,
        volume->data,
        out_time_ms
    );

    if (!ok) return nullptr;
    static uint64_t s_next_vol_version = 2;
    volume->version = ++s_next_vol_version;
    return volume;
}

#ifdef __CUDACC__
__global__ void gpu_csg_intersect_kernel(
    float* __restrict__ root_grid,
    glm::ivec3 root_min_coord,
    glm::ivec3 root_dims,
    float voxel_size,
    const float* __restrict__ vessel_grid,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    float wall_cut_smooth
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= root_dims.x || y >= root_dims.y || z >= root_dims.z) return;

    size_t root_idx = static_cast<size_t>(z) * root_dims.y * root_dims.x + static_cast<size_t>(y) * root_dims.x + static_cast<size_t>(x);
    glm::vec3 world_pos(
        (root_min_coord.x + x) * voxel_size,
        (root_min_coord.y + y) * voxel_size,
        (root_min_coord.z + z) * voxel_size
    );

    float v_dist = dev_sample_sdf_trilinear(vessel_grid, vessel_min_coord, vessel_dims, voxel_size, world_pos);
    float r_dist = root_grid[root_idx];
    float d = fmaxf(r_dist, v_dist);
    if (wall_cut_smooth > 1e-5f) {
        float hk = fmaxf(wall_cut_smooth - fabsf(r_dist - v_dist), 0.0f) / wall_cut_smooth;
        d += hk * hk * wall_cut_smooth * 0.25f;
    }
    // TRACK C FIX 1: clamp to the narrow-band background. Both rasterizer
    // paths (rasterize_root_nodes_gpu / rasterize_volumetric_sections_gpu)
    // store background = 4 * voxel_size. Outside the vessel grid the
    // trilinear sampler returns its 1000.0f out-of-bounds sentinel; storing
    // that here made the viewport sphere-tracer advance ~850 units per
    // sample (d * 0.85f) and leap past the entire root volume, so the baked
    // roots visually vanished whenever the root grid extended beyond the
    // vessel grid — e.g. lateral rootlets (Growth > Seeding > Lateral Roots)
    // sprouting up to lateral_max_length past the specimen while the vessel
    // grid only pads 10 voxels. Clamping far-field magnitudes never moves a
    // zero crossing, so marching-cubes meshes, exports and verification
    // results are unaffected.
    d = fminf(d, 4.0f * voxel_size);
    if (x == 0 || x == root_dims.x - 1 ||
        y == 0 || y == root_dims.y - 1 ||
        z == 0 || z == root_dims.z - 1) {
        d = fmaxf(d, voxel_size * 0.25f);
    }
    root_grid[root_idx] = d;
}
#endif

bool CudaRasterizer::intersect_volumes_gpu(
    CudaVolumeGrid& inout_root_grid,
    const CudaVolumeGrid& vessel_grid,
    float wall_cut_smooth
) {
    if (!inout_root_grid.is_valid() || !vessel_grid.is_valid()) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_root = nullptr;
        float* d_vessel = nullptr;

        size_t root_bytes = inout_root_grid.data.size() * sizeof(float);
        size_t vessel_bytes = vessel_grid.data.size() * sizeof(float);

        if (cudaMalloc(&d_root, root_bytes) == cudaSuccess &&
            cudaMalloc(&d_vessel, vessel_bytes) == cudaSuccess) {

            cudaMemcpy(d_root, inout_root_grid.data.data(), root_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_vessel, vessel_grid.data.data(), vessel_bytes, cudaMemcpyHostToDevice);

            dim3 block(8, 8, 8);
            dim3 grid(
                (inout_root_grid.dims.x + 7) / 8,
                (inout_root_grid.dims.y + 7) / 8,
                (inout_root_grid.dims.z + 7) / 8
            );

            gpu_csg_intersect_kernel<<<grid, block>>>(
                d_root, inout_root_grid.min_coord, inout_root_grid.dims, inout_root_grid.voxel_size,
                d_vessel, vessel_grid.min_coord, vessel_grid.dims, wall_cut_smooth
            );

            cudaMemcpy(inout_root_grid.data.data(), d_root, root_bytes, cudaMemcpyDeviceToHost);
            inout_root_grid.version++;
            cudaFree(d_root);
            cudaFree(d_vessel);
            return true;
        }
        if (d_root) cudaFree(d_root);
        if (d_vessel) cudaFree(d_vessel);
    }
#endif
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_eikonal_update_kernel(
    float* __restrict__ grid_data,
    int dim_x, int dim_y, int dim_z,
    float h,
    int parity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x <= 0 || x >= dim_x - 1 || y <= 0 || y >= dim_y - 1 || z <= 0 || z >= dim_z - 1) return;
    if (((x + y + z) & 1) != parity) return;

    size_t idx = (size_t)z * dim_y * dim_x + (size_t)y * dim_x + (size_t)x;
    float current = fabsf(grid_data[idx]);
    float sign = (grid_data[idx] >= 0.0f) ? 1.0f : -1.0f;

    float a_x = fminf(fabsf(grid_data[idx - 1]), fabsf(grid_data[idx + 1]));
    float a_y = fminf(fabsf(grid_data[idx - dim_x]), fabsf(grid_data[idx + dim_x]));
    float a_z = fminf(fabsf(grid_data[idx - (size_t)dim_y * dim_x]), fabsf(grid_data[idx + (size_t)dim_y * dim_x]));

    float s0 = fminf(a_x, fminf(a_y, a_z));
    float s2 = fmaxf(a_x, fmaxf(a_y, a_z));
    float s1 = a_x + a_y + a_z - s0 - s2;

    float updated = s0 + h;
    if (updated > s1) {
        float disc = 2.0f * h * h - (s1 - s0) * (s1 - s0);
        if (disc >= 0.0f) {
            updated = (s0 + s1 + sqrtf(disc)) * 0.5f;
            if (updated > s2) {
                float disc3 = 3.0f * h * h - ((s1 - s0) * (s1 - s0) + (s2 - s1) * (s2 - s1) + (s2 - s0) * (s2 - s0)) * 0.5f;
                if (disc3 >= 0.0f) {
                    updated = (s0 + s1 + s2 + sqrtf(disc3)) / 3.0f;
                }
            }
        }
    }

    if (updated < current) {
        grid_data[idx] = updated * sign;
    }
}
#endif

bool CudaRasterizer::fast_sweeping_eikonal_gpu(
    CudaVolumeGrid& grid,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    if (!grid.is_valid()) {
        out_time_ms = 0.0f;
        return false;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_grid = nullptr;
        size_t total_voxels = grid.data.size();
        size_t grid_bytes = total_voxels * sizeof(float);

        if (cudaMalloc(&d_grid, grid_bytes) == cudaSuccess) {
            cudaMemcpy(d_grid, grid.data.data(), grid_bytes, cudaMemcpyHostToDevice);

            dim3 threads(8, 8, 8);
            dim3 blocks(
                (grid.dims.x + threads.x - 1) / threads.x,
                (grid.dims.y + threads.y - 1) / threads.y,
                (grid.dims.z + threads.z - 1) / threads.z
            );

            // Red-black Gauss-Seidel iterations for Eikonal propagation
            for (int iter = 0; iter < 8; ++iter) {
                gpu_eikonal_update_kernel<<<blocks, threads>>>(
                    d_grid, grid.dims.x, grid.dims.y, grid.dims.z, grid.voxel_size, 0
                );
                gpu_eikonal_update_kernel<<<blocks, threads>>>(
                    d_grid, grid.dims.x, grid.dims.y, grid.dims.z, grid.voxel_size, 1
                );
            }
            cudaDeviceSynchronize();

            cudaMemcpy(grid.data.data(), d_grid, grid_bytes, cudaMemcpyDeviceToHost);
            cudaFree(d_grid);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_grid) cudaFree(d_grid);
    }
#endif
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_reduce_min_float_kernel(
    const float* __restrict__ data,
    size_t n,
    float* __restrict__ out_min
) {
    __shared__ float s_min[256];
    int tid = threadIdx.x;
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    float local_min = 1e30f;
    while (i < n) {
        float v = data[i];
        if (v < local_min) local_min = v;
        i += static_cast<size_t>(gridDim.x) * blockDim.x;
    }

    s_min[tid] = local_min;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min[tid + s] < s_min[tid]) s_min[tid] = s_min[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dev_atomicMin_float(out_min, s_min[0]);
    }
}
#endif

bool CudaRasterizer::compute_min_sdf_gpu(
    const CudaVolumeGrid& grid,
    float& out_min_sdf,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_min_sdf = 0.0f;
    if (!grid.is_valid() || grid.data.empty()) {
        out_time_ms = 0.0f;
        return false;
    }

#ifdef __CUDACC__
    if (is_cuda_available()) {
        size_t total_voxels = grid.data.size();
        float* d_sdf = nullptr;
        float* d_min_sdf = nullptr;

        if (cudaMalloc(&d_sdf, total_voxels * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_min_sdf, sizeof(float)) == cudaSuccess) {

            cudaMemcpy(d_sdf, grid.data.data(), total_voxels * sizeof(float), cudaMemcpyHostToDevice);
            float init_min = 1e30f;
            cudaMemcpy(d_min_sdf, &init_min, sizeof(float), cudaMemcpyHostToDevice);

            int threads = 256;
            int red_blocks = std::min(1024, static_cast<int>((total_voxels + threads - 1) / threads));
            gpu_reduce_min_float_kernel<<<red_blocks, threads>>>(d_sdf, total_voxels, d_min_sdf);

            cudaMemcpy(&out_min_sdf, d_min_sdf, sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_sdf);
            cudaFree(d_min_sdf);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_sdf) cudaFree(d_sdf);
        if (d_min_sdf) cudaFree(d_min_sdf);
    }
#endif
    return false;
}

float CudaVolumeGrid::get_min_sdf() const {
    if (!is_valid()) return 0.0f;
    if (cached_min_sdf_count_ == data.size()) {
        return cached_min_sdf_;
    }
    float min_v = 0.0f, ms = 0.0f;
    CudaRasterizer::compute_min_sdf_gpu(*this, min_v, ms);
    cached_min_sdf_ = min_v;
    cached_min_sdf_count_ = data.size();
    return cached_min_sdf_;
}

#ifdef __CUDACC__
__global__ void gpu_sample_volume_point_kernel(
    const float* __restrict__ sdf_data,
    glm::ivec3 min_coord,
    glm::ivec3 dims,
    float voxel_size,
    glm::vec3 world_pos,
    int mode, // 0 = nearest, 1 = trilinear
    float* __restrict__ out_val
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (mode == 0) {
            *out_val = dev_sample_sdf_nearest(sdf_data, min_coord, dims, voxel_size, world_pos);
        } else {
            *out_val = dev_sample_sdf_trilinear(sdf_data, min_coord, dims, voxel_size, world_pos);
        }
    }
}
#endif

float CudaVolumeGrid::sample_nearest(const glm::vec3& world_pos) const {
    if (!is_valid()) return 1000.0f;
    float h_val = 1000.0f;
#ifdef __CUDACC__
    if (CudaRasterizer::is_cuda_available()) {
        float* d_sdf = nullptr;
        float* d_out = nullptr;
        size_t bytes = data.size() * sizeof(float);
        if (cudaMalloc(&d_sdf, bytes) == cudaSuccess &&
            cudaMalloc(&d_out, sizeof(float)) == cudaSuccess) {
            cudaMemcpy(d_sdf, data.data(), bytes, cudaMemcpyHostToDevice);
            gpu_sample_volume_point_kernel<<<1, 1>>>(d_sdf, min_coord, dims, voxel_size, world_pos, 0, d_out);
            cudaMemcpy(&h_val, d_out, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_sdf);
            cudaFree(d_out);
        } else {
            if (d_sdf) cudaFree(d_sdf);
            if (d_out) cudaFree(d_out);
        }
    }
#endif
    return h_val;
}

float CudaVolumeGrid::sample_trilinear(const glm::vec3& world_pos) const {
    if (!is_valid()) return 1000.0f;
    float h_val = 1000.0f;
#ifdef __CUDACC__
    if (CudaRasterizer::is_cuda_available()) {
        float* d_sdf = nullptr;
        float* d_out = nullptr;
        size_t bytes = data.size() * sizeof(float);
        if (cudaMalloc(&d_sdf, bytes) == cudaSuccess &&
            cudaMalloc(&d_out, sizeof(float)) == cudaSuccess) {
            cudaMemcpy(d_sdf, data.data(), bytes, cudaMemcpyHostToDevice);
            gpu_sample_volume_point_kernel<<<1, 1>>>(d_sdf, min_coord, dims, voxel_size, world_pos, 1, d_out);
            cudaMemcpy(&h_val, d_out, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_sdf);
            cudaFree(d_out);
        } else {
            if (d_sdf) cudaFree(d_sdf);
            if (d_out) cudaFree(d_out);
        }
    }
#endif
    return h_val;
}

#ifdef __CUDACC__
// --- 28. GPU Kernels for Root Graph Node-to-Capsule Construction ---
__global__ void gpu_mark_node_children_kernel(
    const GpuColonizationNodeLayout* __restrict__ nodes,
    int num_nodes,
    int* __restrict__ out_has_child
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    int p = nodes[i].parent_index;
    if (p >= 0 && p < num_nodes) {
        out_has_child[p] = 1;
    }
}

__device__ inline float dev_eval_node_radius(
    const GpuColonizationNodeLayout& n,
    bool use_caliber_taper,
    float trunk_caliber,
    float tendril_caliber,
    float taper_curve
) {
    if (!use_caliber_taper) {
        return n.radius;
    }
    float t = fminf(fmaxf(n.energy, 0.0f), 1.0f);
    return tendril_caliber + (trunk_caliber - tendril_caliber) * powf(t, taper_curve);
}

__global__ void gpu_build_node_capsules_kernel(
    const GpuColonizationNodeLayout* __restrict__ nodes,
    const int* __restrict__ has_child,
    int num_nodes,
    float pad,
    bool use_caliber_taper,
    float trunk_caliber,
    float tendril_caliber,
    float taper_curve,
    CudaCapsule* __restrict__ out_capsules,
    int* __restrict__ out_cap_count,
    float* __restrict__ out_total_min_xyz,
    float* __restrict__ out_total_max_xyz,
    bool colorize_by_root
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    int p = nodes[i].parent_index;
    float rb = dev_eval_node_radius(nodes[i], use_caliber_taper, trunk_caliber, tendril_caliber, taper_curve);

    bool emit = false;
    glm::vec3 a(0.0f), b(0.0f);
    float ra = rb;

    if (p >= 0 && p < num_nodes) {
        a = nodes[p].position;
        b = nodes[i].position;
        ra = dev_eval_node_radius(nodes[p], use_caliber_taper, trunk_caliber, tendril_caliber, taper_curve);
        if (nodes[p].strand_id != nodes[i].strand_id) {
            ra = fminf(ra, rb * 1.15f);
        }
        emit = true;
    } else if (has_child[i] == 0) {
        a = nodes[i].position;
        b = nodes[i].position;
        ra = rb;
        emit = true;
    }

    if (emit) {
        float max_r = fmaxf(ra, rb) + pad;
        glm::vec3 min_b = glm::min(a, b) - glm::vec3(max_r);
        glm::vec3 max_b = glm::max(a, b) + glm::vec3(max_r);

        int slot = atomicAdd(out_cap_count, 1);
        if (slot < num_nodes) {
            CudaCapsule c;
            c.ax = a.x; c.ay = a.y; c.az = a.z;
            c.bx = b.x; c.by = b.y; c.bz = b.z;
            c.ra = ra;  c.rb = rb;
            c.min_x = min_b.x; c.min_y = min_b.y; c.min_z = min_b.z;
            c.max_x = max_b.x; c.max_y = max_b.y; c.max_z = max_b.z;
            c.strand_id = colorize_by_root
                ? dev_trace_node_root_seed_index(nodes, num_nodes, i)
                : (nodes[i].strand_id & 0xFFFF);
            out_capsules[slot] = c;
        }

        dev_atomicMin_float(&out_total_min_xyz[0], min_b.x);
        dev_atomicMin_float(&out_total_min_xyz[1], min_b.y);
        dev_atomicMin_float(&out_total_min_xyz[2], min_b.z);
        dev_atomicMax_float(&out_total_max_xyz[0], max_b.x);
        dev_atomicMax_float(&out_total_max_xyz[1], max_b.y);
        dev_atomicMax_float(&out_total_max_xyz[2], max_b.z);
    }
}
#endif

bool CudaRasterizer::rasterize_root_nodes_gpu(
    const void* nodes_data,
    size_t num_nodes,
    float voxel_size,
    float half_width_voxels,
    bool use_caliber_taper,
    float trunk_caliber,
    float tendril_caliber,
    float taper_curve,
    CudaVolumeGrid& out_volume,
    float& out_time_ms,
    bool preserve_beaded,
    float branch_smooth_k
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_volume.voxel_size = voxel_size;
    out_volume.data.clear();

    if (!nodes_data || num_nodes == 0) {
        out_time_ms = 0.0f;
        return true;
    }

    const float smooth_k = std::max(0.0f, preserve_beaded ? std::max(branch_smooth_k, voxel_size * 1.25f) : branch_smooth_k);
    const float background = half_width_voxels * voxel_size;
    const float pad = background + smooth_k + 2.0f * voxel_size;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n = static_cast<int>(num_nodes);
        GpuColonizationNodeLayout* d_nodes = nullptr;
        int* d_has_child = nullptr;
        CudaCapsule* d_capsules = nullptr;
        int* d_cap_count = nullptr;
        float* d_min_xyz = nullptr;
        float* d_max_xyz = nullptr;

        if (cudaMalloc(&d_nodes, num_nodes * sizeof(GpuColonizationNodeLayout)) == cudaSuccess &&
            cudaMalloc(&d_has_child, num_nodes * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_capsules, num_nodes * sizeof(CudaCapsule)) == cudaSuccess &&
            cudaMalloc(&d_cap_count, sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_min_xyz, 3 * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_max_xyz, 3 * sizeof(float)) == cudaSuccess) {

            cudaMemcpy(d_nodes, nodes_data, num_nodes * sizeof(GpuColonizationNodeLayout), cudaMemcpyHostToDevice);
            cudaMemset(d_has_child, 0, num_nodes * sizeof(int));
            cudaMemset(d_cap_count, 0, sizeof(int));

            glm::vec3 init_min(1e30f), init_max(-1e30f);
            cudaMemcpy(d_min_xyz, &init_min, 3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_max_xyz, &init_max, 3 * sizeof(float), cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (n + threads - 1) / threads;

            gpu_mark_node_children_kernel<<<blocks, threads>>>(d_nodes, n, d_has_child);
            gpu_build_node_capsules_kernel<<<blocks, threads>>>(
                d_nodes, d_has_child, n, pad,
                use_caliber_taper, trunk_caliber, tendril_caliber, taper_curve,
                d_capsules, d_cap_count, d_min_xyz, d_max_xyz,
                false
            );

            int h_cap_count = 0;
            glm::vec3 total_min(0.0f), total_max(0.0f);
            cudaMemcpy(&h_cap_count, d_cap_count, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&total_min, d_min_xyz, 3 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&total_max, d_max_xyz, 3 * sizeof(float), cudaMemcpyDeviceToHost);

            std::vector<CudaCapsule> cuda_capsules;
            if (h_cap_count > 0) {
                cuda_capsules.resize(h_cap_count);
                cudaMemcpy(cuda_capsules.data(), d_capsules, static_cast<size_t>(h_cap_count) * sizeof(CudaCapsule), cudaMemcpyDeviceToHost);
            }

            cudaFree(d_nodes);
            cudaFree(d_has_child);
            cudaFree(d_capsules);
            cudaFree(d_cap_count);
            cudaFree(d_min_xyz);
            cudaFree(d_max_xyz);

            if (h_cap_count > 0) {
                out_volume.min_coord = glm::ivec3(
                    static_cast<int>(std::floor(total_min.x / voxel_size)),
                    static_cast<int>(std::floor(total_min.y / voxel_size)),
                    static_cast<int>(std::floor(total_min.z / voxel_size))
                );
                out_volume.max_coord = glm::ivec3(
                    static_cast<int>(std::ceil(total_max.x / voxel_size)),
                    static_cast<int>(std::ceil(total_max.y / voxel_size)),
                    static_cast<int>(std::ceil(total_max.z / voxel_size))
                );
                out_volume.dims = out_volume.max_coord - out_volume.min_coord + glm::ivec3(1);

                float rast_ms = 0.0f;
                rasterize(
                    cuda_capsules, voxel_size, background,
                    out_volume.min_coord, out_volume.max_coord,
                    out_volume.data, rast_ms, smooth_k
                );
                out_volume.version++;
            }

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_nodes) cudaFree(d_nodes);
        if (d_has_child) cudaFree(d_has_child);
        if (d_capsules) cudaFree(d_capsules);
        if (d_cap_count) cudaFree(d_cap_count);
        if (d_min_xyz) cudaFree(d_min_xyz);
        if (d_max_xyz) cudaFree(d_max_xyz);
    }
#endif
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_build_vol_section_capsules_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    float pad,
    CudaCapsule* __restrict__ out_capsules,
    float* __restrict__ out_total_min_xyz,
    float* __restrict__ out_total_max_xyz,
    bool colorize_by_root
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;

    CudaVolSection sec = sections[i];
    int p = sec.parent_sec;
    glm::vec3 a = (p >= 0 && p < num_sections) ? sections[p].pos : sec.pos;
    glm::vec3 b = sec.pos;
    float rb = sec.radius;
    float ra = (p >= 0 && p < num_sections)
        ? ((sections[p].root_id == sec.root_id) ? sections[p].radius : fminf(sections[p].radius, rb * 1.15f))
        : rb;
    float max_r = fmaxf(ra, rb) + pad;
    glm::vec3 bmin = glm::min(a, b) - glm::vec3(max_r);
    glm::vec3 bmax = glm::max(a, b) + glm::vec3(max_r);

    CudaCapsule cap;
    cap.ax = a.x; cap.ay = a.y; cap.az = a.z;
    cap.bx = b.x; cap.by = b.y; cap.bz = b.z;
    cap.ra = ra;  cap.rb = rb;
    cap.min_x = bmin.x; cap.min_y = bmin.y; cap.min_z = bmin.z;
    cap.max_x = bmax.x; cap.max_y = bmax.y; cap.max_z = bmax.z;
    cap.strand_id = colorize_by_root
        ? dev_trace_vol_section_root_seed_index(sections, num_sections, i)
        : sec.root_id;
    out_capsules[i] = cap;

    dev_atomicMin_float(&out_total_min_xyz[0], bmin.x);
    dev_atomicMin_float(&out_total_min_xyz[1], bmin.y);
    dev_atomicMin_float(&out_total_min_xyz[2], bmin.z);
    dev_atomicMax_float(&out_total_max_xyz[0], bmax.x);
    dev_atomicMax_float(&out_total_max_xyz[1], bmax.y);
    dev_atomicMax_float(&out_total_max_xyz[2], bmax.z);
}
#endif

bool CudaRasterizer::rasterize_volumetric_sections_gpu(
    const CudaVolSection* sections_data,
    size_t num_sections,
    float voxel_size,
    CudaVolumeGrid& out_volume,
    float& out_time_ms,
    bool preserve_beaded,
    float branch_smooth_k
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_volume.voxel_size = voxel_size;
    out_volume.data.clear();
    if (!sections_data || num_sections == 0) {
        out_time_ms = 0.0f;
        return true;
    }

    const float smooth_k = std::max(0.0f, preserve_beaded ? std::max(branch_smooth_k, voxel_size * 1.25f) : branch_smooth_k);
    const float background = 4.0f * voxel_size;
    const float pad = background + smooth_k + 2.0f * voxel_size;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n = static_cast<int>(num_sections);
        CudaVolSection* d_secs = nullptr;
        CudaCapsule* d_caps = nullptr;
        float* d_min_xyz = nullptr;
        float* d_max_xyz = nullptr;
        std::cout << "[DEBUG] Rasterizing " << n << " sections." << std::endl;

        // Prepare for batched processing
        std::vector<CudaCapsule> all_capsules;
        all_capsules.reserve(num_sections);
        if (cudaMalloc(&d_secs, n * sizeof(CudaVolSection)) == cudaSuccess &&
            cudaMalloc(&d_caps, n * sizeof(CudaCapsule)) == cudaSuccess &&
            cudaMalloc(&d_min_xyz, 3 * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_max_xyz, 3 * sizeof(float)) == cudaSuccess) {

            cudaMemcpy(d_secs, sections_data, n * sizeof(CudaVolSection), cudaMemcpyHostToDevice);
            glm::vec3 init_min(1e30f), init_max(-1e30f);
            cudaMemcpy(d_min_xyz, &init_min, 3 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_max_xyz, &init_max, 3 * sizeof(float), cudaMemcpyHostToDevice);

            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            gpu_build_vol_section_capsules_kernel<<<blocks, threads>>>(
                d_secs, n, pad, d_caps, d_min_xyz, d_max_xyz,
                false
            );

            glm::vec3 total_min(0.0f), total_max(0.0f);
            bool bounds_ok =
                cudaMemcpy(&total_min, d_min_xyz, 3 * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(&total_max, d_max_xyz, 3 * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
                std::isfinite(total_min.x) && std::isfinite(total_min.y) && std::isfinite(total_min.z) &&
                std::isfinite(total_max.x) && std::isfinite(total_max.y) && std::isfinite(total_max.z);

            // v24.11 Track E R3 (bake fail-fast guard): validate the GPU-side
            // section bounds BEFORE the int coordinate casts below. A
            // wild-but-finite section position on a continued (second Start)
            // growth session can inflate an axis span into the multi-km
            // range: the static_cast<int> of the out-of-range quotient is UB,
            // and any dims that survive make rasterize()'s
            // out_distances.assign() throw std::bad_alloc / length_error.
            // The growth path has no exception handler, so the throw escalates
            // std::terminate -> abort() -> UCRT __fastfail -> process exit
            // code 0xC0000409 (the reproducible quick-second-Start crash the
            // watchdog captured, 3/3 runs; run 3 staged-release timeline
            // confirms the v24.10 re-bake release path itself works). The
            // same guard also catches a failed/no-op capsule kernel: the
            // 1e30/-1e30 init bounds come back finite but with a negative
            // span, which the span check rejects. Failing here degrades to
            // the SAME graceful path as any other CUDA failure in this file:
            // the bake returns false, the controller keeps a null roots grid
            // until the next successful bake. Caps sit far above any
            // buildable grid (mug at 0.10 mm is ~1021x787x871 = 0.70 GVox);
            // healthy data never trips them.
            const double bake_guard_axis_cap = 1073741824.0;   // 2^30 voxels/axis (~107 km at 0.10 mm)
            const double bake_guard_total_cap = 8589934592.0;  // 2^33 voxels total (32 GiB float grid)
            const double bg_vmin_x = static_cast<double>(total_min.x) / static_cast<double>(voxel_size);
            const double bg_vmin_y = static_cast<double>(total_min.y) / static_cast<double>(voxel_size);
            const double bg_vmin_z = static_cast<double>(total_min.z) / static_cast<double>(voxel_size);
            const double bg_vmax_x = static_cast<double>(total_max.x) / static_cast<double>(voxel_size);
            const double bg_vmax_y = static_cast<double>(total_max.y) / static_cast<double>(voxel_size);
            const double bg_vmax_z = static_cast<double>(total_max.z) / static_cast<double>(voxel_size);
            if (bounds_ok) {
                const double bg_sx = bg_vmax_x - bg_vmin_x;
                const double bg_sy = bg_vmax_y - bg_vmin_y;
                const double bg_sz = bg_vmax_z - bg_vmin_z;
                bounds_ok =
                    std::fabs(bg_vmin_x) <= bake_guard_axis_cap && std::fabs(bg_vmin_y) <= bake_guard_axis_cap &&
                    std::fabs(bg_vmin_z) <= bake_guard_axis_cap && std::fabs(bg_vmax_x) <= bake_guard_axis_cap &&
                    std::fabs(bg_vmax_y) <= bake_guard_axis_cap && std::fabs(bg_vmax_z) <= bake_guard_axis_cap &&
                    bg_sx >= 1.0 && bg_sx <= bake_guard_axis_cap &&
                    bg_sy >= 1.0 && bg_sy <= bake_guard_axis_cap &&
                    bg_sz >= 1.0 && bg_sz <= bake_guard_axis_cap &&
                    (bg_sx * bg_sy * bg_sz) <= bake_guard_total_cap;
            }
            if (!bounds_ok) {
                char bake_guard_msg[512];
                snprintf(bake_guard_msg, sizeof(bake_guard_msg),
                    "volumetric bake rejected: section bounds failed validation "
                    "(sections=%d voxel=%.4g min=[%g %g %g] max=[%g %g %g]) - "
                    "roots grid not rebuilt, app continues",
                    n, voxel_size,
                    total_min.x, total_min.y, total_min.z,
                    total_max.x, total_max.y, total_max.z);
                std::cerr << "[Bake Guard] " << bake_guard_msg << std::endl;
                RhizomeLog("CUDA", bake_guard_msg);
                cudaFree(d_secs);
                cudaFree(d_caps);
                cudaFree(d_min_xyz);
                cudaFree(d_max_xyz);
                return false;
            }

            std::vector<CudaCapsule> capsules(n);
            cudaMemcpy(capsules.data(), d_caps, n * sizeof(CudaCapsule), cudaMemcpyDeviceToHost);

            cudaFree(d_secs);
            cudaFree(d_caps);
            cudaFree(d_min_xyz);
            cudaFree(d_max_xyz);

            out_volume.min_coord = glm::ivec3(
                static_cast<int>(std::floor(total_min.x / voxel_size)),
                static_cast<int>(std::floor(total_min.y / voxel_size)),
                static_cast<int>(std::floor(total_min.z / voxel_size))
            );
            out_volume.max_coord = glm::ivec3(
                static_cast<int>(std::ceil(total_max.x / voxel_size)),
                static_cast<int>(std::ceil(total_max.y / voxel_size)),
                static_cast<int>(std::ceil(total_max.z / voxel_size))
            );
            out_volume.dims = out_volume.max_coord - out_volume.min_coord + glm::ivec3(1);

            float rast_ms = 0.0f;
            rasterize(
                capsules, voxel_size, background,
                out_volume.min_coord, out_volume.max_coord,
                out_volume.data, rast_ms, smooth_k
            );
            out_volume.version++;

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_secs) cudaFree(d_secs);
        if (d_caps) cudaFree(d_caps);
        if (d_min_xyz) cudaFree(d_min_xyz);
        if (d_max_xyz) cudaFree(d_max_xyz);
    }
#endif
    return false;
}

} // namespace rhizome
