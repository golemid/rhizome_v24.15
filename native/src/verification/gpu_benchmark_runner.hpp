#pragma once

#include <string>
#include <vector>
#include <cstddef>

namespace rhizome {

struct GpuBenchmarkItem {
    std::string control_id;         // Corresponding control from param_controls_complete_list.csv (e.g. CTRL020)
    std::string parameter_name;     // Parameter / Slider name
    std::string stage_name;         // GPU kernel / Pipeline stage
    std::string tested_value;       // Tested value label (e.g. "0.20 mm (High Detail)")
    float kernel_time_ms = 0.0f;    // Accurate GPU hardware execution time in milliseconds
    double vram_used_mb = 0.0;      // VRAM allocated during kernel execution
    double vram_total_mb = 0.0;     // Total GPU VRAM
    size_t vertices_generated = 0;  // Extracted vertex count
    size_t triangles_generated = 0; // Extracted triangle count
    double throughput_mtris_sec = 0.0; // Million triangles/sec or voxels/sec throughput
    std::string bottleneck_status;  // "OPTIMAL", "MODERATE", "BOTTLENECK"
    std::string notes;              // Architectural / performance insight
};

struct GpuBenchmarkSuiteResult {
    std::string gpu_device_name;
    size_t total_vram_mb = 0;
    size_t free_vram_mb = 0;
    float total_suite_time_ms = 0.0f;
    std::vector<GpuBenchmarkItem> items;
    size_t bottlenecks_detected = 0;
    std::string timestamp_iso;
    std::string summary_text;
    bool completed = false;
};

class GpuBenchmarkRunner {
public:
    // Execute comprehensive 100% GPU benchmark across all demanding parameters in the 138-control catalog
    static GpuBenchmarkSuiteResult run_full_benchmark(
        const std::string& output_json_path = "logs/gpu_benchmark_report.json",
        const std::string& output_md_path = "logs/gpu_benchmark_report.md"
    );

    static void save_report_json(const GpuBenchmarkSuiteResult& suite, const std::string& path);
    static void save_report_markdown(const GpuBenchmarkSuiteResult& suite, const std::string& path);

    static const GpuBenchmarkSuiteResult& get_last_result();
    static bool has_completed_results();
};

} // namespace rhizome
