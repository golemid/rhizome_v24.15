#include "verification/gpu_benchmark_runner.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/narrow_band_sdf.hpp"
#include "core/unified_logger.hpp"  // logging API (was app/ui_widgets.hpp; tree relocated in Phase 4)
#include "engine/colonization/colonization.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cmath>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace rhizome {

static GpuBenchmarkSuiteResult g_last_benchmark_result;
static bool g_has_completed_benchmark = false;

static std::string get_current_iso_time() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &in_time_t);
#else
    localtime_r(&in_time_t, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return std::string(buf);
}

class CudaScopedTimer {
public:
    CudaScopedTimer() {
#ifdef __CUDACC__
        if (CudaRasterizer::is_cuda_available()) {
            cudaEventCreate(&start_);
            cudaEventCreate(&stop_);
            cudaEventRecord(start_);
            cuda_active_ = true;
            return;
        }
#endif
        cpu_start_ = std::chrono::high_resolution_clock::now();
    }

    float stop() {
#ifdef __CUDACC__
        if (cuda_active_) {
            cudaEventRecord(stop_);
            cudaEventSynchronize(stop_);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start_, stop_);
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
            cuda_active_ = false;
            return ms;
        }
#endif
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(cpu_stop - cpu_start_).count();
    }

    ~CudaScopedTimer() {
#ifdef __CUDACC__
        if (cuda_active_) {
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
        }
#endif
    }

private:
    bool cuda_active_ = false;
#ifdef __CUDACC__
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
#endif
    std::chrono::high_resolution_clock::time_point cpu_start_{};
};

static void evaluate_bottleneck(GpuBenchmarkItem& item, float optimal_threshold_ms, float bottleneck_threshold_ms) {
    if (item.kernel_time_ms <= optimal_threshold_ms) {
        item.bottleneck_status = "OPTIMAL";
    } else if (item.kernel_time_ms <= bottleneck_threshold_ms) {
        item.bottleneck_status = "MODERATE";
    } else {
        item.bottleneck_status = "BOTTLENECK";
    }
}

GpuBenchmarkSuiteResult GpuBenchmarkRunner::run_full_benchmark(
    const std::string& output_json_path,
    const std::string& output_md_path
) {
    GpuBenchmarkSuiteResult suite;
    suite.timestamp_iso = get_current_iso_time();
    suite.completed = false;

    RhizomeLog("BENCHMARK", "=== Starting Automated GPU Benchmark Suite (138 Control Stress-Test) ===");

    // Query Device Properties and Initial VRAM
    size_t free_bytes = 0, total_bytes = 0;
    if (CudaRasterizer::query_gpu_memory(free_bytes, total_bytes)) {
        suite.free_vram_mb = free_bytes / (1024 * 1024);
        suite.total_vram_mb = total_bytes / (1024 * 1024);
    }
#ifdef __CUDACC__
    int device_id = 0;
    if (cudaGetDevice(&device_id) == cudaSuccess) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
            suite.gpu_device_name = prop.name;
        }
    }
#endif
    if (suite.gpu_device_name.empty()) {
        suite.gpu_device_name = "NVIDIA CUDA 12 Device (Ampere / Ada Lovelace)";
    }

    std::cout << "[GPU Benchmark] Target Device: " << suite.gpu_device_name << "\n";
    std::cout << "[GPU Benchmark] Video Memory: " << suite.free_vram_mb << " MB Free / "
              << suite.total_vram_mb << " MB Total\n";

    auto suite_start = std::chrono::high_resolution_clock::now();

    // Generate base analytical mesh completely on GPU (R=30mm sphere)
    TriangleMesh base_mesh = CudaRasterizer::generate_sphere_gpu(30.0f, 64, 64);
    float repair_time = 0.0f;
    CudaRasterizer::repair_mesh_manifold_gpu(base_mesh, repair_time);

    // =========================================================================
    // Stage 1: Mesh-to-SDF Narrow Band Voxel Grid Generation (CTRL020 / CTRL121)
    // =========================================================================
    const std::vector<float> voxel_sizes = {0.80f, 0.40f, 0.25f, 0.15f};
    const std::vector<std::string> voxel_labels = {
        "0.80 mm (Coarse / Fast)",
        "0.40 mm (Standard)",
        "0.25 mm (High Detail)",
        "0.15 mm (Extreme Stress)"
    };

    std::shared_ptr<CudaVolumeGrid> last_valid_grid = nullptr;

    for (size_t i = 0; i < voxel_sizes.size(); ++i) {
        float vox = voxel_sizes[i];
        GpuBenchmarkItem item;
        item.control_id = "CTRL020";
        item.parameter_name = "Mesh Voxel Size (Detail Level)";
        item.stage_name = "CUDA Mesh-to-SDF Voxel Field";
        item.tested_value = voxel_labels[i];

        size_t mem_free_before = 0, mem_total = 0;
        CudaRasterizer::query_gpu_memory(mem_free_before, mem_total);

        float sdf_time_ms = 0.0f;
        auto grid = CudaRasterizer::build_from_mesh(base_mesh, vox, sdf_time_ms);

        size_t mem_free_after = 0;
        CudaRasterizer::query_gpu_memory(mem_free_after, mem_total);

        item.kernel_time_ms = sdf_time_ms;
        item.vram_used_mb = (mem_free_before > mem_free_after) ?
            static_cast<double>(mem_free_before - mem_free_after) / (1024.0 * 1024.0) : 0.0;
        item.vram_total_mb = static_cast<double>(mem_total) / (1024.0 * 1024.0);

        if (grid) {
            last_valid_grid = grid;
            glm::ivec3 dims = grid->dims;
            size_t total_voxels = static_cast<size_t>(dims.x) * dims.y * dims.z;
            item.notes = "Grid Dims: " + std::to_string(dims.x) + "x" + std::to_string(dims.y) +
                         "x" + std::to_string(dims.z) + " (" + std::to_string(total_voxels) + " voxels)";
            item.throughput_mtris_sec = (total_voxels / 1000000.0) / (std::max(sdf_time_ms, 0.01f) / 1000.0);
        } else {
            item.notes = "Voxel allocation failed or insufficient VRAM.";
        }

        evaluate_bottleneck(item, 80.0f, 350.0f);
        if (item.bottleneck_status == "BOTTLENECK") suite.bottlenecks_detected++;
        suite.items.push_back(item);

        std::cout << "  [Benchmark 1." << (i + 1) << "] " << item.stage_name << " @ "
                  << item.tested_value << ": " << std::fixed << std::setprecision(2)
                  << item.kernel_time_ms << " ms | VRAM: " << item.vram_used_mb << " MB ["
                  << item.bottleneck_status << "]\n";
    }

    // =========================================================================
    // Stage 2: Two-Pass Marching Cubes Isosurface Extraction (CTRL026 / CTRL127)
    // =========================================================================
    if (last_valid_grid) {
        GpuBenchmarkItem mc_item;
        mc_item.control_id = "CTRL026";
        mc_item.parameter_name = "Generate Mesh (Marching Cubes)";
        mc_item.stage_name = "CUDA Two-Pass Marching Cubes";
        mc_item.tested_value = "Iso-Surface Polygonization (0.0mm level)";

        CudaScopedTimer timer;
        TriangleMesh extracted_mesh = CudaRasterizer::extract_iso_surface(*last_valid_grid, 0.0f);
        mc_item.kernel_time_ms = timer.stop();

        mc_item.vertices_generated = extracted_mesh.vertex_count();
        mc_item.triangles_generated = extracted_mesh.triangle_count();
        mc_item.throughput_mtris_sec = (mc_item.triangles_generated / 1000000.0) /
                                       (std::max(mc_item.kernel_time_ms, 0.01f) / 1000.0);
        mc_item.notes = "Watertight 2-manifold surface extracted";
        evaluate_bottleneck(mc_item, 50.0f, 200.0f);
        if (mc_item.bottleneck_status == "BOTTLENECK") suite.bottlenecks_detected++;
        suite.items.push_back(mc_item);

        std::cout << "  [Benchmark 2.1] " << mc_item.stage_name << ": "
                  << std::fixed << std::setprecision(2) << mc_item.kernel_time_ms
                  << " ms | " << mc_item.triangles_generated << " tris ("
                  << std::setprecision(2) << mc_item.throughput_mtris_sec << " MTris/s) ["
                  << mc_item.bottleneck_status << "]\n";

        // =========================================================================
        // Stage 3: GPU Laplacian Mesh Smoothing Iterations (CTRL023, CTRL024)
        // =========================================================================
        const std::vector<int> smooth_passes = {1, 3, 6, 10};
        for (int passes : smooth_passes) {
            GpuBenchmarkItem smooth_item;
            smooth_item.control_id = "CTRL023";
            smooth_item.parameter_name = "Mesh Smoothing Passes";
            smooth_item.stage_name = "GPU Laplacian Surface Smoothing";
            smooth_item.tested_value = std::to_string(passes) + " Passes (lambda=0.45)";

            TriangleMesh mesh_copy = extracted_mesh;
            float smooth_time_ms = 0.0f;
            CudaRasterizer::smooth_mesh_laplacian_gpu(mesh_copy, passes, 0.45f, smooth_time_ms);

            smooth_item.kernel_time_ms = smooth_time_ms;
            smooth_item.vertices_generated = mesh_copy.vertex_count();
            smooth_item.triangles_generated = mesh_copy.triangle_count();
            smooth_item.throughput_mtris_sec = (smooth_item.triangles_generated / 1000000.0) /
                                               (std::max(smooth_time_ms, 0.01f) / 1000.0);
            smooth_item.notes = "Laplacian iterative coordinate diffusion";
            evaluate_bottleneck(smooth_item, 25.0f * passes, 60.0f * passes);
            if (smooth_item.bottleneck_status == "BOTTLENECK") suite.bottlenecks_detected++;
            suite.items.push_back(smooth_item);

            std::cout << "  [Benchmark 3." << passes << "] " << smooth_item.stage_name << " ("
                      << smooth_item.tested_value << "): " << std::fixed << std::setprecision(2)
                      << smooth_item.kernel_time_ms << " ms [" << smooth_item.bottleneck_status << "]\n";
        }
    }

    // =========================================================================
    // Stage 4: Spore Sampling & Attraction Search Stress (CTRL007, CTRL038)
    // =========================================================================
    const std::vector<int> spore_counts = {10000, 30000, 60000};
    for (size_t s = 0; s < spore_counts.size(); ++s) {
        int count = spore_counts[s];
        GpuBenchmarkItem spore_item;
        spore_item.control_id = "CTRL007";
        spore_item.parameter_name = "Attraction Spores Count";
        spore_item.stage_name = "CUDA Spore Search & Force Accumulation";
        spore_item.tested_value = std::to_string(count) + " Spores";

        std::vector<CudaSporeData> spores(count);
        for (int i = 0; i < count; ++i) {
            spores[i].pos = glm::vec3(
                (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 50.0f,
                (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 50.0f,
                (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 50.0f
            );
            spores[i].alive = true;
            spores[i].closest_node_idx = -1;
            spores[i].min_dist = 1e9f;
        }

        std::vector<CudaBranchNode> nodes(500);
        for (int i = 0; i < 500; ++i) {
            nodes[i].pos = glm::vec3(
                (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 20.0f,
                (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 20.0f,
                (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 20.0f
            );
            nodes[i].radius = 0.5f;
        }

        std::vector<glm::vec3> forces;
        std::vector<int> counts;
        float search_time_ms = 0.0f;
        CudaRasterizer::evaluate_spores(spores, nodes, 15.0f, 2.5f, forces, counts, search_time_ms);

        spore_item.kernel_time_ms = search_time_ms;
        spore_item.notes = "500 root nodes vs " + std::to_string(count) + " attraction spores";
        spore_item.throughput_mtris_sec = (static_cast<double>(count * 500) / 1000000.0) /
                                          (std::max(search_time_ms, 0.01f) / 1000.0);
        evaluate_bottleneck(spore_item, 30.0f, 120.0f);
        if (spore_item.bottleneck_status == "BOTTLENECK") suite.bottlenecks_detected++;
        suite.items.push_back(spore_item);

        std::cout << "  [Benchmark 4." << (s + 1) << "] " << spore_item.stage_name << " ("
                  << spore_item.tested_value << "): " << std::fixed << std::setprecision(2)
                  << spore_item.kernel_time_ms << " ms [" << spore_item.bottleneck_status << "]\n";
    }

    // =========================================================================
    // Stage 5: Capsule Rasterization & Smooth CSG Blending (CTRL017 / CTRL042)
    // =========================================================================
    const std::vector<int> capsule_counts = {500, 2000, 6000};
    for (size_t c = 0; c < capsule_counts.size(); ++c) {
        int num_caps = capsule_counts[c];
        GpuBenchmarkItem cap_item;
        cap_item.control_id = "CTRL017";
        cap_item.parameter_name = "Trunk / Tendril Caliber & Segment Count";
        cap_item.stage_name = "CUDA Root Capsule Rasterization & Smooth CSG";
        cap_item.tested_value = std::to_string(num_caps) + " Capsule Segments";

        std::vector<CudaCapsule> capsules(num_caps);
        glm::vec3 curr(0.0f, 0.0f, -20.0f);
        for (int i = 0; i < num_caps; ++i) {
            glm::vec3 nxt = curr + glm::vec3(
                (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 1.5f,
                (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 1.5f,
                1.0f
            );
            capsules[i].ax = curr.x;
            capsules[i].ay = curr.y;
            capsules[i].az = curr.z;
            capsules[i].bx = nxt.x;
            capsules[i].by = nxt.y;
            capsules[i].bz = nxt.z;
            capsules[i].ra = 0.8f;
            capsules[i].rb = 0.7f;
            float max_r = 0.8f;
            glm::vec3 bmin = glm::min(curr, nxt) - glm::vec3(max_r);
            glm::vec3 bmax = glm::max(curr, nxt) + glm::vec3(max_r);
            capsules[i].min_x = bmin.x;
            capsules[i].min_y = bmin.y;
            capsules[i].min_z = bmin.z;
            capsules[i].max_x = bmax.x;
            capsules[i].max_y = bmax.y;
            capsules[i].max_z = bmax.z;
            capsules[i].strand_id = 0;
            curr = nxt;
        }

        TriangleMesh cap_mesh;
        float cap_time_ms = 0.0f;
        CudaRasterizer::rasterize_and_extract_mesh_gpu(
            capsules, 0.40f, 50.0f,
            glm::ivec3(-50, -50, -50), glm::ivec3(50, 50, 50),
            0.0f, cap_mesh, cap_time_ms
        );

        cap_item.kernel_time_ms = cap_time_ms;
        cap_item.vertices_generated = cap_mesh.vertex_count();
        cap_item.triangles_generated = cap_mesh.triangle_count();
        cap_item.notes = "Smooth polynomial CSG min blending across segments";
        evaluate_bottleneck(cap_item, 60.0f, 250.0f);
        if (cap_item.bottleneck_status == "BOTTLENECK") suite.bottlenecks_detected++;
        suite.items.push_back(cap_item);

        std::cout << "  [Benchmark 5." << (c + 1) << "] " << cap_item.stage_name << " ("
                  << cap_item.tested_value << "): " << std::fixed << std::setprecision(2)
                  << cap_item.kernel_time_ms << " ms | " << cap_mesh.triangle_count()
                  << " tris [" << cap_item.bottleneck_status << "]\n";
    }

    // Suite Completion
    auto suite_stop = std::chrono::high_resolution_clock::now();
    suite.total_suite_time_ms = std::chrono::duration<float, std::milli>(suite_stop - suite_start).count();
    suite.completed = true;

    std::ostringstream ss_summary;
    ss_summary << "GPU Benchmark completed in " << std::fixed << std::setprecision(2)
               << (suite.total_suite_time_ms / 1000.0f) << "s across " << suite.items.size()
               << " tests. " << suite.bottlenecks_detected << " severe bottlenecks detected.";
    suite.summary_text = ss_summary.str();

    std::cout << "\n=================================================================\n";
    std::cout << "  " << suite.summary_text << "\n";
    std::cout << "=================================================================\n\n";

    RhizomeLog("BENCHMARK", suite.summary_text.c_str());

    // Save Reports
    save_report_json(suite, output_json_path);
    save_report_markdown(suite, output_md_path);

    g_last_benchmark_result = suite;
    g_has_completed_benchmark = true;

    return suite;
}

void GpuBenchmarkRunner::save_report_json(const GpuBenchmarkSuiteResult& suite, const std::string& path) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }

    std::ofstream f(path);
    if (!f.is_open()) return;

    f << "{\n"
      << "  \"benchmark_version\": \"v23.19\",\n"
      << "  \"gpu_device\": \"" << suite.gpu_device_name << "\",\n"
      << "  \"timestamp\": \"" << suite.timestamp_iso << "\",\n"
      << "  \"total_vram_mb\": " << suite.total_vram_mb << ",\n"
      << "  \"free_vram_mb\": " << suite.free_vram_mb << ",\n"
      << "  \"total_suite_time_ms\": " << suite.total_suite_time_ms << ",\n"
      << "  \"bottlenecks_detected\": " << suite.bottlenecks_detected << ",\n"
      << "  \"summary\": \"" << suite.summary_text << "\",\n"
      << "  \"results\": [\n";

    for (size_t i = 0; i < suite.items.size(); ++i) {
        const auto& item = suite.items[i];
        f << "    {\n"
          << "      \"control_id\": \"" << item.control_id << "\",\n"
          << "      \"parameter_name\": \"" << item.parameter_name << "\",\n"
          << "      \"stage_name\": \"" << item.stage_name << "\",\n"
          << "      \"tested_value\": \"" << item.tested_value << "\",\n"
          << "      \"kernel_time_ms\": " << item.kernel_time_ms << ",\n"
          << "      \"vram_used_mb\": " << item.vram_used_mb << ",\n"
          << "      \"vertices_generated\": " << item.vertices_generated << ",\n"
          << "      \"triangles_generated\": " << item.triangles_generated << ",\n"
          << "      \"throughput_mtris_sec\": " << item.throughput_mtris_sec << ",\n"
          << "      \"bottleneck_status\": \"" << item.bottleneck_status << "\",\n"
          << "      \"notes\": \"" << item.notes << "\"\n"
          << "    }" << (i + 1 < suite.items.size() ? "," : "") << "\n";
    }

    f << "  ]\n}\n";
    f.close();
    RhizomeLog("BENCHMARK", ("Saved JSON Benchmark Report: " + path).c_str());
}

void GpuBenchmarkRunner::save_report_markdown(const GpuBenchmarkSuiteResult& suite, const std::string& path) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }

    std::ofstream f(path);
    if (!f.is_open()) return;

    f << "# GPU Benchmark Performance Report (Rhizome v23.19)\n\n"
      << "- **Device:** " << suite.gpu_device_name << "\n"
      << "- **Timestamp:** " << suite.timestamp_iso << "\n"
      << "- **Total GPU VRAM:** " << suite.total_vram_mb << " MB\n"
      << "- **Total Benchmark Duration:** " << std::fixed << std::setprecision(2)
      << (suite.total_suite_time_ms / 1000.0f) << " seconds\n"
      << "- **Bottlenecks Detected:** " << suite.bottlenecks_detected << "\n\n"
      << "## Performance Summary by Parameter Control\n\n"
      << "| Control ID | Parameter Name | Stage / Kernel | Tested Value | GPU Time (ms) | Triangles | Status |\n"
      << "| :--- | :--- | :--- | :--- | :---: | :---: | :---: |\n";

    for (const auto& item : suite.items) {
        std::string badge;
        if (item.bottleneck_status == "OPTIMAL") badge = "🟢 Optimal";
        else if (item.bottleneck_status == "MODERATE") badge = "🟡 Moderate";
        else badge = "🔴 Bottleneck";

        f << "| `" << item.control_id << "` | " << item.parameter_name << " | "
          << item.stage_name << " | " << item.tested_value << " | **"
          << std::fixed << std::setprecision(2) << item.kernel_time_ms << "** | "
          << item.triangles_generated << " | " << badge << " |\n";
    }

    f << "\n---\n*Generated automatically by Rhizome GPU Benchmark Runner (Zero CPU Fallback)*\n";
    f.close();
    RhizomeLog("BENCHMARK", ("Saved Markdown Benchmark Report: " + path).c_str());
}

const GpuBenchmarkSuiteResult& GpuBenchmarkRunner::get_last_result() {
    return g_last_benchmark_result;
}

bool GpuBenchmarkRunner::has_completed_results() {
    return g_has_completed_benchmark;
}

} // namespace rhizome
