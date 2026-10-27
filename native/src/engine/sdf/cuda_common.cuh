#pragma once

// =========================================================================
// RHIZOME CUDA COMMON — engine/sdf module (v23.29 Phase 3)
// =========================================================================
// Memory-management & launch-safety layer of the 100% CUDA GPU pipeline.
//
// Per AGENTS.md Sec A.2, shared CUDA utilities are split across two homes:
//   * core/math_utils.cuh    — pure host/device math helpers, atomic CAS
//                              guards, tiling constants (RASTER_TILE_SIZE,
//                              RHIZOME_DIV_UP). No runtime API usage.
//   * engine/sdf/cuda_common.cuh (THIS FILE) — everything that touches the
//                              CUDA runtime API:
//       1. RHIZOME_CUDA_CHECK / RHIZOME_CUDA_CHECK_VOID error macros
//       2. rhizome_cuda_alloc / to_device / to_host / free wrappers
//       3. DeviceBuffer<T> — RAII guard for engine-owned VRAM
//       4. Shared cross-TU kernel DECLARATIONS (Phase 2 finding F10 +
//          Phase 3 finding F13): kernels that are DEFINED in exactly one
//          .cu but LAUNCHED from others are declared here so no kernel
//          definition is ever duplicated across translation units:
//              gpu_reduce_min_float_kernel, gpu_mark_node_children_kernel,
//              gpu_build_node_capsules_kernel,
//              gpu_build_vol_section_capsules_kernel   -> defined in
//                                              engine/sdf/cuda_rasterizer.cu
//              gpu_accumulate_face_normals_kernel,
//              gpu_normalize_normals_kernel,
//              gpu_reduce_mesh_bbox_kernel,
//              gpu_compact_non_degenerate_tris_kernel  -> defined in
//                                              engine/mesh/cuda_mesh_ops.cu
//          (the F13 duplicate `static` body formerly duplicated in
//          engine/mesh/cuda_mesh_topology.cu was removed in Phase 3 —
//          verified byte-identical to the survivor before deletion)
//
// CALL-SITE POLICY (deliberate, documented deviation): the existing
// `cudaMalloc(...) == cudaSuccess` chains in the engine keep their v23.28
// error-handling shape in this phase. The Phase 3 migration is structural
// (file tree + include paths); every kernel and host function body moved
// verbatim per the approved Phase 1 proposal ("kernel bodies are not
// altered"). Adopting these helpers at the 282 allocation call sites is
// queued for Phase 5, AFTER the VS2022 + CUDA 12 build validates the new
// tree, so error-path behavior changes can never hide inside move-diff
// noise. AGENTS.md Sec 1 (functional contract 100% intact) governs.
//
// This header is consumed exclusively by .cu translation units (NVCC).
// =========================================================================

#include "core/math_utils.cuh"
#include "core/unified_logger.hpp"   // RhizomeLog (2-arg compat API)

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstddef>

namespace rhizome {

// --- 1. CUDA error checking -----------------------------------------------
// Two forms mirroring the engine's two error-handling shapes:
//   * RHIZOME_CUDA_CHECK(call)      — log via UnifiedLogger, return false.
//     For host driver functions returning bool (the engine's dominant
//     `if (cudaMalloc(...) == cudaSuccess)` pattern).
//   * RHIZOME_CUDA_CHECK_VOID(call) — log via UnifiedLogger, continue.
//     For void teardown / fire-and-report contexts.

#define RHIZOME_CUDA_CHECK(call)                                             \
    do {                                                                     \
        cudaError_t rhizome_cuda_err_ = (call);                              \
        if (rhizome_cuda_err_ != cudaSuccess) {                              \
            char rhizome_cuda_msg_[256];                                     \
            snprintf(rhizome_cuda_msg_, sizeof(rhizome_cuda_msg_),           \
                     "%s failed at %s:%d - %s", #call, __FILE__, __LINE__,   \
                     cudaGetErrorString(rhizome_cuda_err_));                 \
            RhizomeLog("CUDA", rhizome_cuda_msg_);                           \
            return false;                                                    \
        }                                                                    \
    } while (0)

#define RHIZOME_CUDA_CHECK_VOID(call)                                        \
    do {                                                                     \
        cudaError_t rhizome_cuda_err_ = (call);                              \
        if (rhizome_cuda_err_ != cudaSuccess) {                              \
            char rhizome_cuda_msg_[256];                                     \
            snprintf(rhizome_cuda_msg_, sizeof(rhizome_cuda_msg_),           \
                     "%s failed at %s:%d - %s", #call, __FILE__, __LINE__,   \
                     cudaGetErrorString(rhizome_cuda_err_));                 \
            RhizomeLog("CUDA", rhizome_cuda_msg_);                           \
        }                                                                    \
    } while (0)

// --- 2. VRAM alloc / copy / free wrappers ----------------------------------
// Semantically identical to the raw calls they wrap; zero overhead (inline),
// single point of truth for future instrumentation (allocation accounting,
// RhizomeLogMemory hooks) and error-context enrichment.

template <typename T>
inline bool rhizome_cuda_alloc(T** dev_ptr, size_t count) {
    return cudaMalloc(reinterpret_cast<void**>(dev_ptr),
                      count * sizeof(T)) == cudaSuccess;
}

inline bool rhizome_cuda_alloc_bytes(void** dev_ptr, size_t bytes) {
    return cudaMalloc(dev_ptr, bytes) == cudaSuccess;
}

template <typename T>
inline bool rhizome_cuda_to_device(T* dev_dst, const T* host_src, size_t count) {
    return cudaMemcpy(dev_dst, host_src, count * sizeof(T),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

template <typename T>
inline bool rhizome_cuda_to_host(T* host_dst, const T* dev_src, size_t count) {
    return cudaMemcpy(host_dst, dev_src, count * sizeof(T),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

inline void rhizome_cuda_free(void* dev_ptr) {
    if (dev_ptr) cudaFree(dev_ptr);
}

// --- 3. DeviceBuffer<T> — RAII guard for engine-owned VRAM -----------------
// Non-copyable, movable. Frees on scope exit; explicit release() for
// teardown paths that need deterministic ordering.

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(size_t count) { reset(count); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    bool reset(size_t count) {
        release();
        if (count == 0) return true;
        count_ = count;
        return cudaMalloc(reinterpret_cast<void**>(&ptr_),
                          count * sizeof(T)) == cudaSuccess;
    }

    void release() noexcept {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    bool   valid() const noexcept { return ptr_ != nullptr; }
    size_t size()  const noexcept { return count_; }
    T*     get()         noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }

    bool upload(const T* host_src, size_t count) {
        return count <= count_ && ptr_ != nullptr &&
               cudaMemcpy(ptr_, host_src, count * sizeof(T),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    }
    bool download(T* host_dst, size_t count) const {
        return count <= count_ && ptr_ != nullptr &&
               cudaMemcpy(host_dst, ptr_, count * sizeof(T),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    }

private:
    T*     ptr_   = nullptr;
    size_t count_ = 0;
};

// --- 4. Shared cross-TU kernel declarations (F10/F13 relocation) -----------
// Relocated verbatim from core/math_utils.cuh (Phase 2, F10) and
// consolidated from a removed TU-local duplicate (Phase 3, F13). Each
// kernel below is defined in exactly one translation unit (see the map in
// the header comment above) and launched from one or more others; these
// declarations make the host-side launch stubs visible to every consumer.

__global__ void gpu_accumulate_face_normals_kernel(
    const float* __restrict__ vertices,
    int num_vertices,
    const uint32_t* __restrict__ indices,
    int num_triangles,
    float* __restrict__ out_normals
);

__global__ void gpu_normalize_normals_kernel(
    float* __restrict__ normals,
    int num_vertices
);

__global__ void gpu_reduce_mesh_bbox_kernel(
    const float* __restrict__ vertices,
    int num_vertices,
    float* __restrict__ out_min_xyz,
    float* __restrict__ out_max_xyz
);

__global__ void gpu_reduce_min_float_kernel(
    const float* __restrict__ data,
    size_t n,
    float* __restrict__ out_min
);

__global__ void gpu_mark_node_children_kernel(
    const GpuColonizationNodeLayout* __restrict__ nodes,
    int num_nodes,
    int* __restrict__ out_has_child
);

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
);

__global__ void gpu_build_vol_section_capsules_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    float pad,
    CudaCapsule* __restrict__ out_capsules,
    float* __restrict__ out_total_min_xyz,
    float* __restrict__ out_total_max_xyz,
    bool colorize_by_root
);

// F13 (Phase 3): was a TU-local `static` duplicate in
// engine/mesh/cuda_mesh_topology.cu, byte-identical to the external
// definition in engine/mesh/cuda_mesh_ops.cu. Duplicate body deleted;
// both TUs now share this declaration + the single mesh_ops definition.
__global__ void gpu_compact_non_degenerate_tris_kernel(
    const uint32_t* __restrict__ in_indices,
    int num_tris,
    const uint8_t* __restrict__ remove_mask,
    uint32_t* __restrict__ out_indices,
    int* __restrict__ out_tri_count
);

} // namespace rhizome

#endif // __CUDACC__
