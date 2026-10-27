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
// 31. Direct GPU Raymarching & Implicit Volume Raycasting (Zero Mesh Extraction)
// ============================================================================
#ifdef __CUDACC__

__device__ inline bool dev_ray_aabb_intersect(
    const glm::vec3& ro,
    const glm::vec3& rd,
    const glm::vec3& box_min,
    const glm::vec3& box_max,
    float& out_tnear,
    float& out_tfar
) {
    glm::vec3 inv_d(
        (fabsf(rd.x) > 1e-8f) ? (1.0f / rd.x) : (rd.x >= 0.0f ? 1e8f : -1e8f),
        (fabsf(rd.y) > 1e-8f) ? (1.0f / rd.y) : (rd.y >= 0.0f ? 1e8f : -1e8f),
        (fabsf(rd.z) > 1e-8f) ? (1.0f / rd.z) : (rd.z >= 0.0f ? 1e8f : -1e8f)
    );
    glm::vec3 t0s = (box_min - ro) * inv_d;
    glm::vec3 t1s = (box_max - ro) * inv_d;
    glm::vec3 tsmaller = glm::min(t0s, t1s);
    glm::vec3 tbigger  = glm::max(t0s, t1s);
    out_tnear = fmaxf(0.0f, fmaxf(tsmaller.x, fmaxf(tsmaller.y, tsmaller.z)));
    out_tfar  = fminf(tbigger.x, fminf(tbigger.y, tbigger.z));
    return (out_tnear <= out_tfar && out_tfar > 0.0f);
}

__global__ void gpu_raycast_sdf_volume_kernel(
    const float* __restrict__ sdf_grid,
    glm::ivec3 min_coord,
    glm::ivec3 dims,
    float voxel_size,
    glm::vec3 ro,
    glm::vec3 rd,
    glm::vec3* __restrict__ out_hit_pt,
    glm::vec3* __restrict__ out_hit_norm,
    float* __restrict__ out_hit_t
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *out_hit_t = -1.0f;

    glm::vec3 box_min = glm::vec3(min_coord) * voxel_size;
    glm::vec3 box_max = glm::vec3(min_coord + dims - glm::ivec3(1)) * voxel_size;
    float t_near = 0.0f, t_far = 0.0f;
    if (!dev_ray_aabb_intersect(ro, rd, box_min, box_max, t_near, t_far)) return;

    float t = t_near + 0.01f * voxel_size;
    float prev_d = dev_sample_sdf_trilinear(sdf_grid, min_coord, dims, voxel_size, ro + rd * t);
    const float min_step = 0.35f * voxel_size;
    const float eps = 0.02f * voxel_size;

    for (int step = 0; step < 512 && t <= t_far; ++step) {
        glm::vec3 p = ro + rd * t;
        float d = dev_sample_sdf_trilinear(sdf_grid, min_coord, dims, voxel_size, p);
        if (fabsf(d) <= eps || (prev_d > 0.0f && d <= 0.0f)) {
            // Secant refinement across zero crossing
            float t_hit = t;
            if (prev_d > 0.0f && d < 0.0f) {
                float frac = prev_d / (prev_d - d);
                t_hit = t - fmaxf(min_step, fabsf(prev_d) * 0.85f) * (1.0f - frac);
            }
            glm::vec3 hp = ro + rd * t_hit;
            *out_hit_pt = hp;
            *out_hit_norm = dev_sample_sdf_grad(sdf_grid, min_coord, dims, voxel_size, hp);
            *out_hit_t = t_hit;
            return;
        }
        prev_d = d;
        t += fmaxf(min_step, fabsf(d) * 0.85f);
    }
}

__global__ void gpu_count_interior_voxels_kernel(
    const float* __restrict__ sdf_grid,
    size_t num_voxels,
    unsigned long long* __restrict__ out_inside_count
) {
    __shared__ unsigned int s_cnt[256];
    int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    unsigned int local_c = 0;
    for (size_t i = idx; i < num_voxels; i += stride) {
        if (sdf_grid[i] <= 0.0f) {
            local_c++;
        }
    }
    s_cnt[tid] = local_c;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) s_cnt[tid] += s_cnt[tid + s];
        __syncthreads();
    }
    if (tid == 0 && s_cnt[0] > 0) {
        atomicAdd(out_inside_count, static_cast<unsigned long long>(s_cnt[0]));
    }
}

__device__ inline float dev_eval_capsules_bvh_grid(
    const CudaCapsule* __restrict__ capsules,
    const int* __restrict__ cell_head,
    const int* __restrict__ cell_next,
    uint32_t hash_mask,
    float inv_cell_size,
    const glm::vec3& p,
    float smooth_k,
    int* out_closest_strand_id = nullptr
) {
    int cx = static_cast<int>(floorf(p.x * inv_cell_size));
    int cy = static_cast<int>(floorf(p.y * inv_cell_size));
    int cz = static_cast<int>(floorf(p.z * inv_cell_size));

    float cell_size = 1.0f / inv_cell_size;
    float d_min = cell_size * 0.70f;
    float best_raw_dist = 1e20f;
    int best_strand = 0;
    bool found_any = false;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                uint32_t slot = dev_hash_coords(cx + dx, cy + dy, cz + dz, hash_mask);
                for (int ci = cell_head[slot]; ci >= 0; ci = cell_next[ci]) {
                    CudaCapsule c = capsules[ci];
                    float bax = c.bx - c.ax, bay = c.by - c.ay, baz = c.bz - c.az;
                    float pax = p.x - c.ax, pay = p.y - c.ay, paz = p.z - c.az;
                    float baba = bax * bax + bay * bay + baz * baz;
                    float paba = pax * bax + pay * bay + paz * baz;
                    float h = (baba > 1e-8f) ? fminf(fmaxf(paba / baba, 0.0f), 1.0f) : 0.0f;
                    float dx_c = pax - bax * h;
                    float dy_c = pay - bay * h;
                    float dz_c = paz - baz * h;
                    float r = c.ra + (c.rb - c.ra) * h;
                    float dist = sqrtf(dx_c * dx_c + dy_c * dy_c + dz_c * dz_c) - r;

                    if (dist < best_raw_dist) {
                        best_raw_dist = dist;
                        best_strand = c.strand_id;
                    }

                    if (!found_any) {
                        d_min = fminf(d_min, dist);
                        found_any = true;
                    } else if (smooth_k > 1e-5f) {
                        float hk = fmaxf(smooth_k - fabsf(d_min - dist), 0.0f) / smooth_k;
                        d_min = fminf(d_min, dist) - hk * hk * smooth_k * 0.25f;
                    } else {
                        d_min = fminf(d_min, dist);
                    }
                }
            }
        }
    }
    if (out_closest_strand_id) {
        *out_closest_strand_id = best_strand;
    }
    return d_min;
}

__global__ void gpu_insert_capsules_spatial_hash_kernel(
    const CudaCapsule* __restrict__ capsules,
    int num_capsules,
    float inv_cell_size,
    uint32_t hash_mask,
    int* __restrict__ cell_head,
    int* __restrict__ cell_next
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_capsules) return;

    CudaCapsule c = capsules[i];
    float mx = 0.5f * (c.ax + c.bx);
    float my = 0.5f * (c.ay + c.by);
    float mz = 0.5f * (c.az + c.bz);
    int cx = static_cast<int>(floorf(mx * inv_cell_size));
    int cy = static_cast<int>(floorf(my * inv_cell_size));
    int cz = static_cast<int>(floorf(mz * inv_cell_size));
    uint32_t slot = dev_hash_coords(cx, cy, cz, hash_mask);
    cell_next[i] = atomicExch(&cell_head[slot], i);
}

__device__ inline glm::vec3 dev_shade_clay_surface(
    const glm::vec3& world_n,
    const glm::mat4& view_mat,
    const glm::vec3& base_rgb,
    float diffuse_strength,
    float edge_darkening,
    float crevice_ao_strength,
    float specular_strength,
    float ao_factor,
    float sss_factor = 0.0f,
    const glm::vec3& hit_p = glm::vec3(0.0f),
    float bark_strength = 0.0f,
    float bark_freq = 0.40f,
    // v24.01 Phase 1 (lighting expansion): promoted v23.29 constants. The
    // key/fill directions arrive pre-derived on the host
    // (core/lighting_rig.hpp) so both render paths share one rig; defaults
    // reproduce the previous compiled-in look exactly.
    const glm::vec3& key_dir = glm::vec3(-0.22f, 0.34f, 0.91f),
    const glm::vec3& fill_dir = glm::vec3(0.38f, -0.28f, 0.88f),
    float light_ambient = 0.32f,
    float light_fill = 0.16f,
    float light_glossiness = 24.0f,
    float light_highlight_tint = 0.55f,
    float sss_strength = 0.45f,
    float sss_warmth = 0.35f
) {
    glm::vec3 mod_rgb = base_rgb;
    glm::vec3 mod_n = world_n;

    // 100% CUDA GPU Procedural Bark & Organic Texture via 3D Simplex Noise
    if (bark_strength > 1e-4f) {
        float noise_val = dev_bark_texture_3d(hit_p, bark_freq); // ~[-1.0, 1.0]
        float dark_mult = 1.0f - (noise_val * 0.5f + 0.5f) * (bark_strength * 0.55f);
        mod_rgb = glm::clamp(base_rgb * dark_mult, glm::vec3(0.0f), glm::vec3(1.0f));

        // Perturb surface normal slightly according to noise gradient for tactile grooves
        float eps = 0.05f;
        float nx = dev_bark_texture_3d(hit_p + glm::vec3(eps, 0, 0), bark_freq) - dev_bark_texture_3d(hit_p - glm::vec3(eps, 0, 0), bark_freq);
        float ny = dev_bark_texture_3d(hit_p + glm::vec3(0, eps, 0), bark_freq) - dev_bark_texture_3d(hit_p - glm::vec3(0, eps, 0), bark_freq);
        float nz_g = dev_bark_texture_3d(hit_p + glm::vec3(0, 0, eps), bark_freq) - dev_bark_texture_3d(hit_p - glm::vec3(0, 0, eps), bark_freq);
        glm::vec3 bump_n = dev_safe_norm(world_n - glm::vec3(nx, ny, nz_g) * (bark_strength * 0.35f), world_n);
        mod_n = bump_n;
    }

    glm::vec3 nV = dev_safe_norm(glm::mat3(view_mat) * mod_n, glm::vec3(0.0f, 0.0f, 1.0f));
    float nz = fminf(fmaxf(nV.z, 0.0f), 1.0f);

    glm::vec3 keyDirV = glm::normalize(key_dir);
    float diffKey = fmaxf(glm::dot(nV, keyDirV), 0.0f);
    glm::vec3 fillDirV = glm::normalize(fill_dir);
    float diffFill = fmaxf(glm::dot(nV, fillDirV), 0.0f);

    // Forward translucent subsurface transmission through thin root walls
    float backKey = fmaxf(glm::dot(-nV, keyDirV), 0.0f);
    float sssIllum = backKey * sss_factor * sss_strength;
    glm::vec3 sssTint = glm::mix(mod_rgb, glm::vec3(0.95f, 0.45f, 0.25f), sss_warmth);

    float baseIllum = fminf(fmaxf(light_ambient + 0.62f * diffKey + light_fill * diffFill, 0.0f), 1.15f) * diffuse_strength;

    float t_rim = fminf(fmaxf(nz / 0.72f, 0.0f), 1.0f);
    float rimProfile = t_rim * t_rim * (3.0f - 2.0f * t_rim);
    float edgeContour = 1.0f + (0.08f + 0.92f * powf(rimProfile, 0.65f) - 1.0f) * fminf(fmaxf(edge_darkening, 0.0f), 1.0f);

    float creviceAO = 1.0f - (1.0f - ao_factor) * fminf(fmaxf(crevice_ao_strength, 0.0f), 1.0f);

    glm::vec3 halfV = glm::normalize(keyDirV + glm::vec3(0.0f, 0.0f, 1.0f));
    float spec = powf(fmaxf(glm::dot(nV, halfV), 0.0f), light_glossiness) * specular_strength * edgeContour * creviceAO;
    glm::vec3 specColor = (mod_rgb * (1.0f - light_highlight_tint) + glm::vec3(light_highlight_tint)) * spec;

    glm::vec3 finalRgb = mod_rgb * (baseIllum * edgeContour * creviceAO) + specColor + sssTint * sssIllum;
    return glm::clamp(finalRgb, glm::vec3(0.0f), glm::vec3(1.0f));
}

__device__ inline glm::vec4 dev_shade_matte_glass(
    const glm::vec3& N,
    const glm::vec3& V,
    const glm::vec3& glass_rgb,
    float glass_alpha,
    float frosting,
    // v24.01 Phase 1: promoted v23.29 fresnel rim constant (0.30).
    float fresnel_rim = 0.30f
) {
    float ndotv = fminf(fmaxf(fabsf(glm::dot(N, V)), 0.0f), 1.0f);
    float fresnel = (1.0f - ndotv) * (1.0f - ndotv);
    glm::vec3 keyDir = glm::normalize(glm::vec3(0.5f, 0.4f, 0.8f));
    float wrapKey = fmaxf(glm::dot(N, keyDir) * 0.6f + 0.4f, 0.0f);
    glm::vec3 diffuse = glass_rgb * (wrapKey * 0.65f);
    glm::vec3 hKey = glm::normalize(keyDir + V);
    float exp_p = 28.0f + (8.0f - 28.0f) * frosting;
    float specRough = powf(fmaxf(glm::dot(N, hKey), 0.0f), exp_p);
    glm::vec3 specular = glm::vec3(1.0f, 0.98f, 0.95f) * (specRough * 0.35f);
    float base_a = 0.12f + (0.35f - 0.12f) * frosting;
    float alpha = fminf(fmaxf(base_a + fresnel * 0.55f, 0.08f), 0.92f) * glass_alpha;
    glm::vec3 rgb = glm::clamp(diffuse + specular + glass_rgb * (fresnel * fresnel_rim), glm::vec3(0.0f), glm::vec3(1.0f));
    return glm::vec4(rgb, alpha);
}

__global__ void gpu_raymarch_implicit_viewport_kernel(
    int width,
    int height,
    glm::mat4 view_mat,
    glm::mat4 inv_vp_mat,
    glm::vec3 cam_pos,
    bool is_ortho,
    const float* __restrict__ spec_sdf,
    glm::ivec3 spec_min,
    glm::ivec3 spec_dims,
    float spec_vox,
    bool show_spec,
    bool spec_glass,
    float glass_frosting,
    float spec_opacity,
    glm::vec4 color_spec,
    const float* __restrict__ roots_sdf,
    glm::ivec3 roots_min,
    glm::ivec3 roots_dims,
    float roots_vox,
    const CudaCapsule* __restrict__ capsules,
    int num_capsules,
    const int* __restrict__ cap_head,
    const int* __restrict__ cap_next,
    uint32_t cap_hash_mask,
    float inv_cap_cell,
    glm::vec3 cap_bbox_min,
    glm::vec3 cap_bbox_max,
    float smooth_k,
    float wall_cut_smooth,
    bool show_roots,
    bool colorize_strands,
    glm::vec4 color_roots,
    float shader_diffuse,
    float shader_edge_darkening,
    float shader_crevice_ao,
    float shader_specular,
    uint32_t* __restrict__ out_rgba,
    float shader_bark_strength = 0.0f,
    float shader_bark_frequency = 0.40f,
    glm::mat4 vp_mat = glm::mat4(1.0f),
    float* __restrict__ out_depth = nullptr,
    // v24.01 Phase 1 (lighting expansion): the host-derived rig
    // (core/lighting_rig.hpp) + promoted scalars. Defaults reproduce the
    // v23.29 compiled-in constants for any future launch site.
    glm::vec3 light_dir_key = glm::vec3(-0.22f, 0.34f, 0.91f),
    glm::vec3 light_dir_fill = glm::vec3(0.38f, -0.28f, 0.88f),
    float light_ambient = 0.32f,
    float light_fill = 0.16f,
    float light_glossiness = 24.0f,
    float light_highlight_tint = 0.55f,
    float light_fresnel_rim = 0.30f,
    float volumetric_ao = 0.85f,
    float sss_strength = 0.45f,
    float sss_warmth = 0.35f
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    // v24.01 Phase 1: global volumetric-AO gain. The three AO sites carry
    // stylized per-site strengths (roots SDF 0.85 / capsules 0.80 /
    // specimen 0.70 with floors 0.15/0.20/0.25); the slider scales the
    // occlusion proportionally — default 0.85 -> gain 1.0 -> byte-identical
    // math at every site.
    const float ao_gain = volumetric_ao / 0.85f;

    int pixel_idx = py * width + px;
    float ndc_x = ((static_cast<float>(px) + 0.5f) / static_cast<float>(width)) * 2.0f - 1.0f;
    float ndc_y = ((static_cast<float>(py) + 0.5f) / static_cast<float>(height)) * 2.0f - 1.0f;

    glm::vec4 near4 = inv_vp_mat * glm::vec4(ndc_x, ndc_y, -1.0f, 1.0f);
    glm::vec4 far4  = inv_vp_mat * glm::vec4(ndc_x, ndc_y,  1.0f, 1.0f);
    glm::vec3 r_near = glm::vec3(near4) / near4.w;
    glm::vec3 r_far  = glm::vec3(far4)  / far4.w;
    glm::vec3 rd = glm::normalize(r_far - r_near);
    glm::vec3 ro = is_ortho ? r_near : cam_pos;

    bool hit_roots = false;
    float t_roots = 1e20f;
    glm::vec3 n_roots(0.0f, 0.0f, 1.0f);
    float ao_roots = 1.0f;
    float sss_roots = 0.0f;
    int hit_strand_id = 0;

    // 1. Raymarch Implicit Roots (either via confined SDF volume OR live smooth-blended GPU capsules)
    if (show_roots) {
        if (roots_sdf != nullptr) {
            glm::vec3 bmin = glm::vec3(roots_min) * roots_vox;
            glm::vec3 bmax = glm::vec3(roots_min + roots_dims - glm::ivec3(1)) * roots_vox;
            float tn = 0.0f, tf = 0.0f;
            if (dev_ray_aabb_intersect(ro, rd, bmin, bmax, tn, tf)) {
                float t = tn + 0.01f * roots_vox;
                const float min_step = 0.35f * roots_vox;
                const float eps = 0.025f * roots_vox;
                // v24.08 Track D R5: roots-march travel budget 280 -> 2048.
                // The 280-step cap dated from the 0.25-0.30 mm detail era:
                // at those voxels the banded step (0.85 * 4 * vox) crossed
                // any then-possible roots grid well within 280 steps. Since
                // v24.04 the default Detail Level is 0.10 mm, and the baked
                // roots grid (vessel 100x76x85 mm + lateral rootlets to
                // 8 mm + rasterizer pad) has a ~150-185 mm diagonal while
                // the budget reach shrank to 280 * 0.34 mm = ~95 mm: rays
                // entering the far half of the grid exhausted the loop
                // mid-volume and registered no hit, so everything beyond a
                // camera-dependent sheet vanished — the "dynamic planar
                // trimming" the user reported on v24.07, affecting ONLY the
                // roots (the specimen march had already received its budget
                // fix in R4). Unlike the R4 specimen defect there is no
                // punch-through component here: roots grids are direct
                // capsule rasterizations whose values are clamped to the
                // 4-voxel background (never overestimated — conservative
                // lower bounds outside the band), and the hit test
                // d <= eps also catches interior (negative) samples, so
                // the march can only ever fail by running out of budget.
                // 2048 (not the specimen's 1024): the roots background band
                // is 4 voxels vs the specimen's 8, so every step covers half
                // the distance — the same per-mm headroom R4 gave the
                // specimen needs 2x its budget. Reach = 2048 * 0.34 = ~696
                // mm = ~3.8x the worst 0.10 mm diagonal; a plain background
                // crossing ends via t <= tf at ~539 steps, so the budget is
                // a ceiling for pathological band-skimming rays, not a cost
                // every ray pays.
                for (int s = 0; s < 2048 && t <= tf; ++s) {
                    glm::vec3 p = ro + rd * t;
                    float d = dev_sample_sdf_trilinear(roots_sdf, roots_min, roots_dims, roots_vox, p);
                    if (d <= eps) {
                        hit_roots = true;
                        t_roots = t;
                        n_roots = dev_sample_sdf_grad(roots_sdf, roots_min, roots_dims, roots_vox, p);
                        if (colorize_strands && capsules != nullptr && num_capsules > 0) {
                            dev_eval_capsules_bvh_grid(capsules, cap_head, cap_next, cap_hash_mask, inv_cap_cell, p, smooth_k, &hit_strand_id);
                        }
                        // Volumetric SDF Ambient Occlusion along surface normal
                        float d_ao1 = dev_sample_sdf_trilinear(roots_sdf, roots_min, roots_dims, roots_vox, p + n_roots * (roots_vox * 1.5f));
                        float d_ao2 = dev_sample_sdf_trilinear(roots_sdf, roots_min, roots_dims, roots_vox, p + n_roots * (roots_vox * 3.0f));
                        float occ = fmaxf(0.0f, (roots_vox * 1.5f - d_ao1) / (roots_vox * 1.5f)) * 0.5f +
                                    fmaxf(0.0f, (roots_vox * 3.0f - d_ao2) / (roots_vox * 3.0f)) * 0.5f;
                        ao_roots = fminf(fmaxf(1.0f - occ * 0.85f * ao_gain, 0.15f), 1.0f);

                        // Subsurface scattering: sample interior thickness along inward normal
                        float d_internal = dev_sample_sdf_trilinear(roots_sdf, roots_min, roots_dims, roots_vox, p - n_roots * (roots_vox * 1.8f));
                        float thickness = fmaxf(0.0f, -d_internal);
                        sss_roots = expf(-thickness * 0.65f);
                        break;
                    }
                    // TRACK C FIX 2: cap the advance at the narrow-band
                    // background (4 voxels, matching the rasterizer's
                    // background = 4 * voxel_size) so a stale out-of-bounds
                    // sentinel (1000.0f) sampled from any grid can never
                    // launch a ray past the whole root volume in one step.
                    // With properly banded grids this cap is a no-op
                    // (d <= 4 * roots_vox => d * 0.85f < cap).
                    t += fminf(4.0f * roots_vox, fmaxf(min_step, d * 0.85f));
                }
            }
        } else if (capsules != nullptr && num_capsules > 0) {
            float tn = 0.0f, tf = 0.0f;
            if (dev_ray_aabb_intersect(ro, rd, cap_bbox_min, cap_bbox_max, tn, tf)) {
                float step_vox = fmaxf(spec_vox, 0.20f);
                float t = tn;
                const float min_step = 0.25f * step_vox;
                const float max_step = (0.65f / inv_cap_cell);
                const float eps = 0.03f * step_vox;
                for (int s = 0; s < 384 && t <= tf; ++s) {
                    glm::vec3 p = ro + rd * t;
                    int cur_strand = 0;
                    float d = dev_eval_capsules_bvh_grid(capsules, cap_head, cap_next, cap_hash_mask, inv_cap_cell, p, smooth_k, &cur_strand);
                    if (spec_sdf != nullptr) {
                        float vd = dev_sample_sdf_trilinear(spec_sdf, spec_min, spec_dims, spec_vox, p);
                        float d_int = fmaxf(d, vd);
                        if (wall_cut_smooth > 1e-5f) {
                            float hk = fmaxf(wall_cut_smooth - fabsf(d - vd), 0.0f) / wall_cut_smooth;
                            d_int += hk * hk * wall_cut_smooth * 0.25f;
                        }
                        d = d_int;
                    }
                    if (d <= eps) {
                        hit_roots = true;
                        t_roots = t;
                        hit_strand_id = cur_strand;
                        float ge = 0.25f * step_vox;
                        auto eval_combined = [&](const glm::vec3& qp) -> float {
                            float cd = dev_eval_capsules_bvh_grid(capsules, cap_head, cap_next, cap_hash_mask, inv_cap_cell, qp, smooth_k);
                            if (spec_sdf != nullptr) {
                                float cvd = dev_sample_sdf_trilinear(spec_sdf, spec_min, spec_dims, spec_vox, qp);
                                float cint = fmaxf(cd, cvd);
                                if (wall_cut_smooth > 1e-5f) {
                                    float chk = fmaxf(wall_cut_smooth - fabsf(cd - cvd), 0.0f) / wall_cut_smooth;
                                    cint += chk * chk * wall_cut_smooth * 0.25f;
                                }
                                cd = cint;
                            }
                            return cd;
                        };
                        float gx = eval_combined(p + glm::vec3(ge, 0, 0)) - eval_combined(p - glm::vec3(ge, 0, 0));
                        float gy = eval_combined(p + glm::vec3(0, ge, 0)) - eval_combined(p - glm::vec3(0, ge, 0));
                        float gz = eval_combined(p + glm::vec3(0, 0, ge)) - eval_combined(p - glm::vec3(0, 0, ge));
                        n_roots = dev_safe_norm(glm::vec3(gx, gy, gz), glm::vec3(0.0f, 0.0f, 1.0f));
                        float d_ao = dev_eval_capsules_bvh_grid(capsules, cap_head, cap_next, cap_hash_mask, inv_cap_cell, p + n_roots * (step_vox * 2.0f), smooth_k);
                        float occ = fmaxf(0.0f, (step_vox * 2.0f - d_ao) / (step_vox * 2.0f));
                        ao_roots = fminf(fmaxf(1.0f - occ * 0.80f * ao_gain, 0.20f), 1.0f);

                        float d_internal = eval_combined(p - n_roots * (step_vox * 1.5f));
                        float thickness = fmaxf(0.0f, -d_internal);
                        sss_roots = expf(-thickness * 0.65f);
                        break;
                    }
                    t += fminf(max_step, fmaxf(min_step, d * 0.80f));
                }
            }
        }
    }

    // 2. Raymarch Implicit Specimen SDF Volume
    bool hit_spec = false;
    float t_spec = 1e20f;
    glm::vec3 n_spec(0.0f, 0.0f, 1.0f);
    float ao_spec = 1.0f;

    if (show_spec && spec_sdf != nullptr) {
        glm::vec3 bmin = glm::vec3(spec_min) * spec_vox;
        glm::vec3 bmax = glm::vec3(spec_min + spec_dims - glm::ivec3(1)) * spec_vox;
        float tn = 0.0f, tf = 0.0f;
        if (dev_ray_aabb_intersect(ro, rd, bmin, bmax, tn, tf)) {
            float t = tn + 0.01f * spec_vox;
            const float min_step = 0.35f * spec_vox;
            const float eps = 0.025f * spec_vox;
            // v24.07 Track D R4: band-limited specimen marching, 1024-step
            // budget. The fast SDF path (v24.04+) re-distances the far field
            // with a sweep-capped eikonal; the cap leaves REAL but
            // OVERESTIMATED distances in tortuous regions (measured up to
            // ~13 mm at mug scale after the 8-round cap - and v24.06's
            // far-sentinel audit cannot see them, because no sentinels
            // remain: the user's v24.06 log shows "far residual 0, extra
            // rounds 0, safe-clamped 0" while the trim persisted). Sphere
            // tracing steps 0.85 * (true + overestimate), which can clear a
            // thin wall plus both band edges in a single step - rays punch
            // through the rim/walls and hit the interior or exit the box,
            // producing plane-like trims that move while rotating (user
            // reports on v24.05/v24.06, mug.obj at the 0.10 mm default;
            // reproduced and quantified in the R4 truncation study:
            // scripts/r4_phase2*.py). The exact band (|d| <= 8 voxels,
            // RHI_SDF_BAND_VOXELS) is always trustworthy - the eikonal is
            // monotone and never undercuts the [A] splat seeds (verified:
            // identical band minima truncated-vs-converged). Outside it,
            // clamp the marched value to the band edge: a conservative
            // LOWER bound of the true distance, so a step (0.85 * 8 vox =
            // 6.8 vox) can never clear surface + band on both sides. This
            // subsumes the v24.06 R3 guard (d > 1e5). The cap doubles
            // (512 -> 1024): banded far marching at the 0.10 mm default
            // crosses the 1555-voxel diagonal in <= 229 steps, but grazing
            // rays near the silhouette need the per-mm headroom the
            // 0.25 mm-era 512 budget had.
            const float spec_band = 8.0f * spec_vox;
            for (int s = 0; s < 1024 && t <= tf; ++s) {
                glm::vec3 p = ro + rd * t;
                float d = dev_sample_sdf_trilinear(spec_sdf, spec_min, spec_dims, spec_vox, p);
                if (d > spec_band) d = spec_band;
                if (d <= eps) {
                    hit_spec = true;
                    t_spec = t;
                    n_spec = dev_sample_sdf_grad(spec_sdf, spec_min, spec_dims, spec_vox, p);
                    float d_ao = dev_sample_sdf_trilinear(spec_sdf, spec_min, spec_dims, spec_vox, p + n_spec * (spec_vox * 2.0f));
                    float occ = fmaxf(0.0f, (spec_vox * 2.0f - d_ao) / (spec_vox * 2.0f));
                    ao_spec = fminf(fmaxf(1.0f - occ * 0.70f * ao_gain, 0.25f), 1.0f);
                    break;
                }
                t += fmaxf(min_step, d * 0.85f);
            }
        }
    }

    if (!hit_roots && !hit_spec) {
        out_rgba[pixel_idx] = 0u;
        if (out_depth != nullptr) {
            out_depth[pixel_idx] = 1.0f;
        }
        return;
    }

    glm::vec4 out_col(0.0f);
    glm::vec3 active_root_rgb = colorize_strands ? dev_get_strand_color(hit_strand_id) : glm::vec3(color_roots);
    if (hit_roots && (!hit_spec || t_roots <= t_spec)) {
        glm::vec3 hit_pt = ro + rd * t_roots;
        glm::vec3 rgb = dev_shade_clay_surface(
            n_roots, view_mat, active_root_rgb,
            shader_diffuse, shader_edge_darkening, shader_crevice_ao, shader_specular, ao_roots, sss_roots,
            hit_pt, shader_bark_strength, shader_bark_frequency,
            light_dir_key, light_dir_fill,
            light_ambient, light_fill, light_glossiness, light_highlight_tint,
            sss_strength, sss_warmth
        );
        out_col = glm::vec4(rgb, 1.0f);
    } else if (hit_spec) {
        if (!spec_glass) {
            glm::vec3 rgb = dev_shade_clay_surface(
                n_spec, view_mat, glm::vec3(color_spec),
                shader_diffuse, shader_edge_darkening, shader_crevice_ao, shader_specular, ao_spec,
                0.0f, glm::vec3(0.0f), 0.0f, 0.40f,
                light_dir_key, light_dir_fill,
                light_ambient, light_fill, light_glossiness, light_highlight_tint,
                sss_strength, sss_warmth
            );
            out_col = glm::vec4(rgb, 1.0f);
        } else {
            glm::vec4 glass_front = dev_shade_matte_glass(
                n_spec, -rd, glm::vec3(color_spec), fminf(fmaxf(spec_opacity, 0.05f), 1.0f), glass_frosting,
                light_fresnel_rim
            );
            if (hit_roots) {
                glm::vec3 root_rgb = dev_shade_clay_surface(
                    n_roots, view_mat, active_root_rgb,
                    shader_diffuse, shader_edge_darkening, shader_crevice_ao, shader_specular, ao_roots,
                    0.0f, glm::vec3(0.0f), 0.0f, 0.40f,
                    light_dir_key, light_dir_fill,
                    light_ambient, light_fill, light_glossiness, light_highlight_tint,
                    sss_strength, sss_warmth
                );
                glm::vec3 blended = glm::vec3(glass_front) * glass_front.a + root_rgb * (1.0f - glass_front.a);
                out_col = glm::vec4(glm::clamp(blended, glm::vec3(0.0f), glm::vec3(1.0f)), 1.0f);
            } else {
                out_col = glass_front;
            }
        }
    }

    uint32_t r8 = static_cast<uint32_t>(fminf(fmaxf(out_col.r * 255.0f + 0.5f, 0.0f), 255.0f));
    uint32_t g8 = static_cast<uint32_t>(fminf(fmaxf(out_col.g * 255.0f + 0.5f, 0.0f), 255.0f));
    uint32_t b8 = static_cast<uint32_t>(fminf(fmaxf(out_col.b * 255.0f + 0.5f, 0.0f), 255.0f));
    uint32_t a8 = static_cast<uint32_t>(fminf(fmaxf(out_col.a * 255.0f + 0.5f, 0.0f), 255.0f));
    out_rgba[pixel_idx] = (a8 << 24) | (b8 << 16) | (g8 << 8) | r8;

    if (out_depth != nullptr) {
        float hit_depth = 1.0f;
        if (hit_roots || (hit_spec && !spec_glass)) {
            float t_closest = 1e20f;
            if (hit_roots && (!hit_spec || spec_glass || t_roots <= t_spec)) {
                t_closest = t_roots;
            } else if (hit_spec && !spec_glass) {
                t_closest = t_spec;
            }
            if (t_closest < 1e19f) {
                glm::vec3 hit_world = ro + rd * t_closest;
                glm::vec4 clip_p = vp_mat * glm::vec4(hit_world, 1.0f);
                if (clip_p.w > 1e-6f) {
                    float ndc_z = clip_p.z / clip_p.w;
                    hit_depth = dev_clamp(ndc_z * 0.5f + 0.5f, 0.0f, 1.0f);
                }
            }
        }
        out_depth[pixel_idx] = hit_depth;
    }
}

#endif // __CUDACC__

bool CudaRasterizer::raycast_sdf_volume_gpu(
    const CudaVolumeGrid& grid,
    const glm::vec3& ray_origin,
    const glm::vec3& ray_dir,
    glm::vec3& out_hit_point,
    glm::vec3& out_hit_normal,
    float& out_hit_t,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_hit_t = -1.0f;
    if (!grid.is_valid()) {
        out_time_ms = 0.0f;
        return false;
    }
#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_sdf = nullptr;
        glm::vec3* d_pt = nullptr;
        glm::vec3* d_norm = nullptr;
        float* d_t = nullptr;
        size_t sdf_bytes = grid.data.size() * sizeof(float);
        if (cudaMalloc(&d_sdf, sdf_bytes) == cudaSuccess &&
            cudaMalloc(&d_pt, sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_norm, sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_t, sizeof(float)) == cudaSuccess) {

            cudaMemcpy(d_sdf, grid.data.data(), sdf_bytes, cudaMemcpyHostToDevice);
            gpu_raycast_sdf_volume_kernel<<<1, 1>>>(
                d_sdf, grid.min_coord, grid.dims, grid.voxel_size,
                ray_origin, ray_dir, d_pt, d_norm, d_t
            );
            cudaMemcpy(&out_hit_point, d_pt, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
            cudaMemcpy(&out_hit_normal, d_norm, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
            cudaMemcpy(&out_hit_t, d_t, sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_sdf);
            cudaFree(d_pt);
            cudaFree(d_norm);
            cudaFree(d_t);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return (out_hit_t >= 0.0f);
        }
        if (d_sdf) cudaFree(d_sdf);
        if (d_pt) cudaFree(d_pt);
        if (d_norm) cudaFree(d_norm);
        if (d_t) cudaFree(d_t);
    }
#endif
    return false;
}

bool CudaRasterizer::compute_sdf_interior_volume_gpu(
    const CudaVolumeGrid& grid,
    float& out_volume_mm3,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_volume_mm3 = 0.0f;
    if (!grid.is_valid()) {
        out_time_ms = 0.0f;
        return false;
    }
#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_sdf = nullptr;
        unsigned long long* d_count = nullptr;
        size_t n = grid.data.size();
        if (cudaMalloc(&d_sdf, n * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_count, sizeof(unsigned long long)) == cudaSuccess) {
            cudaMemcpy(d_sdf, grid.data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_count, 0, sizeof(unsigned long long));
            int threads = 256;
            int blocks = std::min(1024, static_cast<int>((n + threads - 1) / threads));
            gpu_count_interior_voxels_kernel<<<blocks, threads>>>(d_sdf, n, d_count);
            unsigned long long h_count = 0;
            cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            cudaFree(d_sdf);
            cudaFree(d_count);
            float vox_vol = grid.voxel_size * grid.voxel_size * grid.voxel_size;
            out_volume_mm3 = static_cast<float>(h_count) * vox_vol;
            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_sdf) cudaFree(d_sdf);
        if (d_count) cudaFree(d_count);
    }
#endif
    return false;
}

// v24.10 Track E R2: roots-grid device cache, hoisted from function-local
// statics in raymarch_implicit_viewport_gpu to namespace scope so
// release_roots_viewport_cache_gpu() can drop it from the growth controller
// BEFORE a bake rebuilds the roots grid. Without this, a continuing Start
// carried the stale device copy of the previous bake (freed only when a NEW
// grid appeared on a later frame) through the entire rasterize + intersect
// rebuild - the measured second-Start VRAM spike and OOM crash.
static float* s_d_roots_sdf = nullptr;
static const float* s_cached_roots_ptr = nullptr;
static size_t s_cached_roots_size = 0;
static uint64_t s_cached_roots_ver = 0;

void CudaRasterizer::release_roots_viewport_cache_gpu() {
#ifdef __CUDACC__
    if (s_d_roots_sdf) {
        if (is_cuda_available()) {
            cudaFree(s_d_roots_sdf);
        }
        s_d_roots_sdf = nullptr;
    }
#endif
    s_cached_roots_ptr = nullptr;
    s_cached_roots_size = 0;
    s_cached_roots_ver = 0;
}

bool CudaRasterizer::raymarch_implicit_viewport_gpu(
    int width,
    int height,
    const glm::mat4& view_mat,
    const glm::mat4& inv_vp_mat,
    const glm::vec3& cam_pos,
    bool is_orthographic,
    const CudaVolumeGrid* specimen_grid,
    bool show_specimen,
    bool specimen_matte_glass,
    float glass_frosting,
    float specimen_opacity,
    const glm::vec4& color_specimen,
    const CudaVolumeGrid* roots_grid,
    const std::vector<CudaVolSection>* vol_sections,
    const void* classic_nodes_data,
    size_t num_classic_nodes,
    bool show_roots,
    bool colorize_strands,
    const glm::vec4& color_roots,
    float shader_diffuse,
    float shader_edge_darkening,
    float shader_crevice_ao,
    float shader_specular,
    std::vector<uint32_t>& out_rgba_pixels,
    float& out_time_ms,
    float branch_smooth_k,
    float wall_cut_smooth,
    bool preserve_beaded,
    bool colorize_roots,
    float shader_bark_strength,
    float shader_bark_frequency,
    const glm::mat4* view_proj_mat,
    std::vector<float>* out_depth_pixels,
    // v24.01 Phase 1 (lighting expansion): host-derived rig (see the
    // declaration in cuda_rasterizer.hpp; fed from core/lighting_rig.hpp).
    glm::vec3 light_dir_key,
    glm::vec3 light_dir_fill,
    float light_ambient,
    float light_fill,
    float light_glossiness,
    float light_highlight_tint,
    float light_fresnel_rim,
    float volumetric_ao,
    float sss_strength,
    float sss_warmth
) {
    auto start = std::chrono::high_resolution_clock::now();
    if (width <= 0 || height <= 0) return false;
    out_rgba_pixels.assign(static_cast<size_t>(width) * height, 0u);

#ifdef __CUDACC__
    if (is_cuda_available()) {
        // Persistent cached device buffers for interactive 60+ FPS viewport raymarching
        static float* s_d_spec_sdf = nullptr;
        static const float* s_cached_spec_ptr = nullptr;
        static size_t s_cached_spec_size = 0;
        static uint64_t s_cached_spec_ver = 0;

        // v24.10 Track E R2: the roots-grid cache statics moved to namespace
        // scope (s_d_roots_sdf and friends, above) so the growth controller
        // can release the stale device copy before a bake rebuild.

        static uint32_t* s_d_rgba = nullptr;
        static size_t s_cap_rgba = 0;
        static float* s_d_depth = nullptr;
        static size_t s_cap_depth = 0;

        static CudaCapsule* s_d_caps = nullptr;
        static int* s_d_cap_next = nullptr;
        static size_t s_cap_capsules = 0;
        static int* s_d_cap_head = nullptr;
        const uint32_t cap_hash_size = 65536u;
        const uint32_t cap_hash_mask = cap_hash_size - 1u;

        if (!s_d_cap_head) {
            cudaMalloc(&s_d_cap_head, cap_hash_size * sizeof(int));
        }

        // 1. Sync Specimen SDF Volume to GPU if changed
        bool valid_spec = (specimen_grid != nullptr && specimen_grid->is_valid() && show_specimen);
        if (valid_spec) {
            if (s_cached_spec_ptr != specimen_grid->data.data() ||
                s_cached_spec_size != specimen_grid->data.size() ||
                s_cached_spec_ver != specimen_grid->version) {
                if (s_d_spec_sdf) cudaFree(s_d_spec_sdf);
                s_d_spec_sdf = nullptr;
                size_t bytes = specimen_grid->data.size() * sizeof(float);
                if (cudaMalloc(&s_d_spec_sdf, bytes) == cudaSuccess) {
                    cudaMemcpy(s_d_spec_sdf, specimen_grid->data.data(), bytes, cudaMemcpyHostToDevice);
                    s_cached_spec_ptr = specimen_grid->data.data();
                    s_cached_spec_size = specimen_grid->data.size();
                    s_cached_spec_ver = specimen_grid->version;
                }
            }
        }

        // 2. Sync Roots SDF Volume OR Live Capsules to GPU
        bool valid_roots_sdf = (roots_grid != nullptr && roots_grid->is_valid() && show_roots);
        if (valid_roots_sdf) {
            if (s_cached_roots_ptr != roots_grid->data.data() ||
                s_cached_roots_size != roots_grid->data.size() ||
                s_cached_roots_ver != roots_grid->version) {
                if (s_d_roots_sdf) cudaFree(s_d_roots_sdf);
                s_d_roots_sdf = nullptr;
                size_t bytes = roots_grid->data.size() * sizeof(float);
                if (cudaMalloc(&s_d_roots_sdf, bytes) == cudaSuccess) {
                    cudaMemcpy(s_d_roots_sdf, roots_grid->data.data(), bytes, cudaMemcpyHostToDevice);
                    s_cached_roots_ptr = roots_grid->data.data();
                    s_cached_roots_size = roots_grid->data.size();
                    s_cached_roots_ver = roots_grid->version;
                }
            }
        } else if (s_d_roots_sdf) {
            // v24.10 Track E R2: no roots grid this frame (Clear / model
            // reload / roots hidden / live growth rendering capsules) - free
            // the stale device copy now instead of carrying it until the
            // next bake appears. Previously this copy lingered indefinitely.
            cudaFree(s_d_roots_sdf);
            s_d_roots_sdf = nullptr;
            s_cached_roots_ptr = nullptr;
            s_cached_roots_size = 0;
            s_cached_roots_ver = 0;
        }

        int num_capsules = 0;
        float inv_cap_cell = 1.0f / 4.0f;
        float vox = (specimen_grid && specimen_grid->is_valid()) ? specimen_grid->voxel_size : 0.25f;
        float smooth_k = std::max(0.0f, preserve_beaded ? std::max(branch_smooth_k, vox * 1.25f) : branch_smooth_k);
        glm::vec3 cap_bbox_min(-1e5f), cap_bbox_max(1e5f);

        bool colorize_any = (colorize_strands || colorize_roots);
        if (show_roots && (!valid_roots_sdf || colorize_any)) {
            smooth_k = std::max(0.0f, preserve_beaded ? std::max(branch_smooth_k, vox * 1.25f) : branch_smooth_k);
            if (vol_sections != nullptr && vol_sections->size() > 1) {
                int n_sec = static_cast<int>(vol_sections->size());
                if (static_cast<size_t>(n_sec) > s_cap_capsules) {
                    if (s_d_caps) cudaFree(s_d_caps);
                    if (s_d_cap_next) cudaFree(s_d_cap_next);
                    s_cap_capsules = static_cast<size_t>(n_sec) * 2;
                    cudaMalloc(&s_d_caps, s_cap_capsules * sizeof(CudaCapsule));
                    cudaMalloc(&s_d_cap_next, s_cap_capsules * sizeof(int));
                }

                CudaVolSection* d_sec = nullptr;
                float* d_bmin = nullptr;
                float* d_bmax = nullptr;
                if (cudaMalloc(&d_sec, n_sec * sizeof(CudaVolSection)) == cudaSuccess &&
                    cudaMalloc(&d_bmin, 3 * sizeof(float)) == cudaSuccess &&
                    cudaMalloc(&d_bmax, 3 * sizeof(float)) == cudaSuccess) {
                    cudaMemcpy(d_sec, vol_sections->data(), n_sec * sizeof(CudaVolSection), cudaMemcpyHostToDevice);
                    float init_min[3] = { 1e20f, 1e20f, 1e20f };
                    float init_max[3] = { -1e20f, -1e20f, -1e20f };
                    cudaMemcpy(d_bmin, init_min, 3 * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_bmax, init_max, 3 * sizeof(float), cudaMemcpyHostToDevice);

                    int threads = 256;
                    int blocks = (n_sec + threads - 1) / threads;
                    gpu_build_vol_section_capsules_kernel<<<blocks, threads>>>(
                        d_sec, n_sec, smooth_k + vox * 2.0f, s_d_caps, d_bmin, d_bmax, colorize_roots
                    );
                    num_capsules = n_sec;
                    cudaMemcpy(&cap_bbox_min.x, d_bmin, 3 * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&cap_bbox_max.x, d_bmax, 3 * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaFree(d_sec);
                    cudaFree(d_bmin);
                    cudaFree(d_bmax);
                }
            } else if (classic_nodes_data != nullptr && num_classic_nodes > 1) {
                int n_nodes = static_cast<int>(num_classic_nodes);
                if (static_cast<size_t>(n_nodes) > s_cap_capsules) {
                    if (s_d_caps) cudaFree(s_d_caps);
                    if (s_d_cap_next) cudaFree(s_d_cap_next);
                    s_cap_capsules = static_cast<size_t>(n_nodes) * 2;
                    cudaMalloc(&s_d_caps, s_cap_capsules * sizeof(CudaCapsule));
                    cudaMalloc(&s_d_cap_next, s_cap_capsules * sizeof(int));
                }

                GpuColonizationNodeLayout* d_nodes = nullptr;
                int* d_has_child = nullptr;
                int* d_cnt = nullptr;
                float* d_bmin = nullptr;
                float* d_bmax = nullptr;
                if (cudaMalloc(&d_nodes, n_nodes * sizeof(GpuColonizationNodeLayout)) == cudaSuccess &&
                    cudaMalloc(&d_has_child, n_nodes * sizeof(int)) == cudaSuccess &&
                    cudaMalloc(&d_cnt, sizeof(int)) == cudaSuccess &&
                    cudaMalloc(&d_bmin, 3 * sizeof(float)) == cudaSuccess &&
                    cudaMalloc(&d_bmax, 3 * sizeof(float)) == cudaSuccess) {
                    cudaMemcpy(d_nodes, classic_nodes_data, n_nodes * sizeof(GpuColonizationNodeLayout), cudaMemcpyHostToDevice);
                    cudaMemset(d_has_child, 0, n_nodes * sizeof(int));
                    cudaMemset(d_cnt, 0, sizeof(int));
                    float init_min[3] = { 1e20f, 1e20f, 1e20f };
                    float init_max[3] = { -1e20f, -1e20f, -1e20f };
                    cudaMemcpy(d_bmin, init_min, 3 * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_bmax, init_max, 3 * sizeof(float), cudaMemcpyHostToDevice);

                    int threads = 256;
                    int blocks = (n_nodes + threads - 1) / threads;
                    gpu_mark_node_children_kernel<<<blocks, threads>>>(d_nodes, n_nodes, d_has_child);
                    gpu_build_node_capsules_kernel<<<blocks, threads>>>(
                        d_nodes, d_has_child, n_nodes, smooth_k + vox * 2.0f,
                        false, 0.0f, 0.0f, 1.0f,
                        s_d_caps, d_cnt, d_bmin, d_bmax, colorize_roots
                    );
                    cudaMemcpy(&num_capsules, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&cap_bbox_min.x, d_bmin, 3 * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&cap_bbox_max.x, d_bmax, 3 * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaFree(d_nodes);
                    cudaFree(d_has_child);
                    cudaFree(d_cnt);
                    cudaFree(d_bmin);
                    cudaFree(d_bmax);
                }
            }

            if (num_capsules > 0) {
                inv_cap_cell = 1.0f / 4.5f;
                cudaMemset(s_d_cap_head, 0xFF, cap_hash_size * sizeof(int));
                int threads = 256;
                int blocks = (num_capsules + threads - 1) / threads;
                gpu_insert_capsules_spatial_hash_kernel<<<blocks, threads>>>(
                    s_d_caps, num_capsules, inv_cap_cell, cap_hash_mask, s_d_cap_head, s_d_cap_next
                );
            }
        }

        size_t total_px = static_cast<size_t>(width) * height;
        if (total_px > s_cap_rgba) {
            if (s_d_rgba) cudaFree(s_d_rgba);
            s_cap_rgba = total_px;
            cudaMalloc(&s_d_rgba, s_cap_rgba * sizeof(uint32_t));
        }
        if (out_depth_pixels != nullptr) {
            out_depth_pixels->assign(total_px, 1.0f);
            if (total_px > s_cap_depth) {
                if (s_d_depth) cudaFree(s_d_depth);
                s_cap_depth = total_px;
                cudaMalloc(&s_d_depth, s_cap_depth * sizeof(float));
            }
        }

        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);

        glm::mat4 actual_vp = (view_proj_mat != nullptr) ? (*view_proj_mat) : (view_mat);

        gpu_raymarch_implicit_viewport_kernel<<<grid, block>>>(
            width, height, view_mat, inv_vp_mat, cam_pos, is_orthographic,
            valid_spec ? s_d_spec_sdf : nullptr,
            valid_spec ? specimen_grid->min_coord : glm::ivec3(0),
            valid_spec ? specimen_grid->dims : glm::ivec3(0),
            valid_spec ? specimen_grid->voxel_size : 0.25f,
            valid_spec, specimen_matte_glass, glass_frosting, specimen_opacity, color_specimen,
            valid_roots_sdf ? s_d_roots_sdf : nullptr,
            valid_roots_sdf ? roots_grid->min_coord : glm::ivec3(0),
            valid_roots_sdf ? roots_grid->dims : glm::ivec3(0),
            valid_roots_sdf ? roots_grid->voxel_size : 0.25f,
            (show_roots && num_capsules > 0) ? s_d_caps : nullptr,
            num_capsules,
            s_d_cap_head, s_d_cap_next, cap_hash_mask, inv_cap_cell,
            cap_bbox_min, cap_bbox_max, smooth_k, std::max(0.0f, wall_cut_smooth),
            show_roots, colorize_any, color_roots,
            shader_diffuse, shader_edge_darkening, shader_crevice_ao, shader_specular,
            s_d_rgba,
            shader_bark_strength, shader_bark_frequency,
            actual_vp,
            (out_depth_pixels != nullptr) ? s_d_depth : nullptr,
            // v24.01 Phase 1: lighting rig — the same derived vectors the
            // GL uniforms carry (single host-side derivation).
            light_dir_key, light_dir_fill,
            light_ambient, light_fill, light_glossiness, light_highlight_tint,
            light_fresnel_rim, volumetric_ao, sss_strength, sss_warmth
        );

        cudaMemcpy(out_rgba_pixels.data(), s_d_rgba, total_px * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (out_depth_pixels != nullptr) {
            cudaMemcpy(out_depth_pixels->data(), s_d_depth, total_px * sizeof(float), cudaMemcpyDeviceToHost);
        }

        auto end = std::chrono::high_resolution_clock::now();
        out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
        return true;
    }
#endif
    return false;
}

} // namespace rhizome
