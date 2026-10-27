# UI Interaction Monitoring System for Rhizome Application
## Complete Stream Capture & Backend Reaction Logger for All 241 UI Controls

> **As-built status (v23.29):** this specification is implemented. The event stream / JSON-Lines monitor lives in `native/src/core/unified_logger.{hpp,cpp}` (`UnifiedLogger`, session API `init_ui_interaction_monitor` / `log_ui_interaction` / `log_ui_reaction`), and the control registry is the data-driven `ControlDefinition` table in `native/src/ui/widgets/control_registry.{hpp,cpp}` (`log_control_interaction()` resolves exact CTRL-IDs; the 241-row catalog is audited against `param_controls_complete_list.csv` at build time and by `--verify` at runtime). The `monitoring/` paths and `EventStreamManager` / `ControlMetadata` names below are the original design vocabulary; the shipped classes are `UnifiedLogger` and `ControlDefinition`.
>
> **Catalog boundary (v24.01 Phase 2, EXECUTED):** the Styles panel's "ADVANCED - INTERFACE STYLE EDITOR" pills (SIZES / COLORS / UI RENDERING (IMGUI), 93 widgets) are registered as `CTRL149-241`: sliders and toggles stream through `render_control()` exactly like domain controls; the COLORS pickers (63 census rows at the 8d reference, ImGui 1.92.8+; the live count follows the ImGui build) log commit-only on `IsItemDeactivatedAfterEdit` (drag-start -> committed rgba values), the policy that keeps color drags from flooding the JSON-Lines stream.
>
> **Program closure (2026-10-22):** the full-telemetry goal is shipped and verified — the 241-control catalog (including the 93 style-editor widgets) is final, audited at build time and by `--verify`, and the v24.01 program is closed by user sign-off. The policies above reflect the shipped behavior; optional future work is scoped in `IMPROVEMENT_PLAN.md`.

---

## 🎯 OBJECTIVE

Create a comprehensive monitoring and telemetry system that captures:
- **Every UI Interaction:** All 241 interactive controls (`CTRL001` through `CTRL241`) across buttons, toggle switches, compact sliders, dropdown combos, color pickers, and text inputs.
- **Detailed Backend Reaction:** Correlating every UI adjustment with backend GPU performance (mesh rebuild duration, CUDA kernel execution time, VRAM usage delta, and geometry changes).
- **Real-Time Event Stream:** Emitting structured JSON-Lines events (`logs/ui_events.jsonl`) for profiling, automated regression testing, and interactive benchmark dashboards.

---

## 📋 CURRENT LOGGING INFRASTRUCTURE

### Existing Implementation (`core/unified_logger.cpp`, homed there in the v23.29 restructuring; the console-compatible free functions `RhizomeLog` / `RhizomeLogMemory` / `RhizomeLogVramSnapshot` remain the call API)

```cpp
void RhizomeLog(const char* category, const char* message) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t timer = std::chrono::system_clock::to_time_t(now);
    std::tm bt;
#if defined(_WIN32)
    localtime_s(&bt, &timer);
#else
    localtime_r(&timer, &bt);
#endif
    char time_str[32];
    std::strftime(time_str, sizeof(time_str), "%H:%M:%S", &bt);

    std::cout << "[" << time_str << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ["
              << (category ? category : "INFO") << "] "
              << (message ? message : "") << std::endl;
}
```

### Active Categories:
- **`ACTION`** – User-triggered UI interactions (slider drags, button clicks, toggle changes).
- **`REACTION`** – Backend completion metrics (e.g., `Mesh rebuilt in 23ms, triangles: 48,291`).
- **`MEM:BUFFER`** – GPU buffer allocation and vertex upload events.
- **`MEM:VRAM`** – Real-time snapshots of NVIDIA GPU VRAM occupancy and headroom.

---

## 🆕 ENHANCED MONITORING ARCHITECTURE

```
┌────────────────────────────────────────────────────────┐
│               Rhizome Dear ImGui UI Layer              │
│       241 Controls (Sliders, Toggles, Buttons, Combos) │
└───────────────────────────┬────────────────────────────┘
                            │
               Triggers User Interaction Event
                            ▼
┌────────────────────────────────────────────────────────┐
│             Control Registry (`CTRL001`-`CTRL241`)     │
│   Maps ID -> Section, Friendly Name, Unit, & Category  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               Event Stream Manager                     │
│  - Captures Timestamp (Nanosecond precision)           │
│  - Captures Old Value -> New Value                     │
│  - Launches High-Resolution Reaction Timer             │
└───────────────────────────┬────────────────────────────┘
                            │
               Executes CUDA GPU Pipeline
                            ▼
┌────────────────────────────────────────────────────────┐
│          Native CUDA GPU Backend & Solvers             │
│   SDF Rasterization | Colonization | Marching Cubes    │
└───────────────────────────┬────────────────────────────┘
                            │
               Reports Execution Reaction
                            ▼
┌────────────────────────────────────────────────────────┐
│                 Correlated Event Stream                │
│  - Processing Time (ms)     - Geometry Delta (Tris)   │
│  - GPU Kernel Time (ms)     - VRAM Occupancy (%)      │
│  - Structured JSON Stream (`logs/ui_events.jsonl`)     │
└────────────────────────────────────────────────────────┘
```

---

## 🛠️ CORE MONITORING COMPONENTS

### 1. Event Stream Manager — as built: `UnifiedLogger` (`core/unified_logger.hpp`)

```cpp
#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <fstream>
#include <cstdint>

namespace rhizome {

struct UiEvent {
    uint64_t timestamp_ns;           // Nanosecond timestamp
    std::string event_type;          // "ACTION", "REACTION", "ERROR"
    std::string control_id;          // e.g., "CTRL099"
    std::string control_name;        // e.g., "Detail Level Override"
    std::string category_id;         // e.g., "SPECIMEN_01"
    std::string panel_section;       // e.g., "Implicit Specimen"
    float old_value;                 // Value before interaction
    float new_value;                 // Value after interaction
    std::string unit;                // "mm", "Degrees", "count", etc.
    int64_t backend_reaction_ms;     // Total turnaround time
    float gpu_kernel_time_ms;        // Direct CUDA kernel time
    size_t triangles_before;         // Geometry state before
    size_t triangles_after;          // Geometry state after
    double vram_used_gb;             // Active GPU memory in GB
    double vram_total_gb;            // Total GPU memory in GB
    std::string reaction_message;    // Summary message
    bool success;                    // Execution status
};

class EventStreamManager {
public:
    static EventStreamManager& instance();

    void init(const char* log_file_path = "logs/ui_events.jsonl");
    void shutdown();

    void capture_interaction(
        const std::string& control_id,
        float old_val,
        float new_val
    );

    void capture_reaction(
        const std::string& control_id,
        int64_t reaction_time_ms,
        float gpu_kernel_ms,
        size_t tris_before,
        size_t tris_after,
        const std::string& reaction_message,
        bool success = true
    );

    void stream_event(const UiEvent& event);
    void flush_events();
    uint64_t get_event_count() const;

private:
    EventStreamManager();
    ~EventStreamManager();

    std::mutex mutex_;
    std::ofstream log_file_;
    std::atomic<uint64_t> event_count_{0};
    std::string active_control_id_;
    float active_old_val_{0.0f};
    float active_new_val_{0.0f};
    uint64_t interaction_start_ns_{0};
};

} // namespace rhizome
```

---

### 2. Control Registry — as built: `ControlDefinition` table (`ui/widgets/control_registry.hpp`)

Covers **100% of the 241 controls** defined in `param_controls_complete_list.csv` (25 CSV sections — Specimen Appearance CTRL104-106 since the v24.01 Phase 0 renumber; ten lighting sliders at CTRL093-102 since v24.01 Phase 1 with ex-CTRL093..138 slid to CTRL103-148; the 93 style-editor widgets at CTRL149-241 since v24.01 Phase 2 (63-color census since the 8d re-pin); the 21 groupings below merge the sections for readability — ranges renumbered to the live catalog):

```cpp
#pragma once

#include <string>
#include <unordered_map>

namespace rhizome {

struct ControlMetadata {
    std::string control_id;      // CTRL001 - CTRL241
    std::string source_file;     // panel_growth.cpp / panel_styles.cpp / panel_scene.cpp / panel_verification.cpp
    std::string panel_section;   // One of 22 CSV sections (19 functional groupings)
    std::string control_type;    // Slider Float, Slider Int, Toggle Switch, Button, Combo, etc.
    std::string label;           // User-visible label
    float min_value;
    float max_value;
    std::string unit;
    std::string category_id;
    bool logging_triggered;
};

class ControlRegistry {
public:
    static ControlRegistry& instance();

    const ControlMetadata* lookup(const std::string& control_id) const;
    const ControlMetadata* lookup_by_label(const std::string& label) const;
    size_t total_controls() const { return controls_.size(); }

private:
    ControlRegistry();
    void register_all_138_controls();

    std::unordered_map<std::string, ControlMetadata> controls_;
    std::unordered_map<std::string, std::string> label_to_id_;
};

} // namespace rhizome
```

#### Registry Section Ranges:
1. **`CTRL001` – `CTRL004`**: Growth Command Bar (Import, Start, Stop, Clear)
2. **`CTRL005` – `CTRL016`**: Algorithm Selector & Physics Parameters
3. **`CTRL017` – `CTRL020`**: Colonization Seeds (Spores, Primary Roots, Search/Kill Radii)
4. **`CTRL021` – `CTRL027`**: Branching & Growth (Steps, Rates, Steering Cone)
5. **`CTRL028` – `CTRL034`**: Vessel & Caliber (Radii, Taper, Smoothness, Clearance)
6. **`CTRL035` – `CTRL040`**: Lateral Roots (Spacing, Diameters, Length, Angles)
7. **`CTRL041` – `CTRL049`**: Seeding Management & Per-Seed Toggles
8. **`CTRL050` – `CTRL052`**: Growth Execution (Live Mode, Cadence, Step Multiplier)
9. **`CTRL053` – `CTRL059`**: Growth Presets Library & Serialization
10. **`CTRL060` – `CTRL065`**: Color Themes & Layout INI Serialization
11. **`CTRL066` – `CTRL070`**: Display Font & Typography Scaling
12. **`CTRL071` – `CTRL086`**: 15 Viewport Element Colors & Reset Button
13. **`CTRL087` – `CTRL106`**: 3D Lighting & Shading (16 sliders + reset) and Specimen Appearance (matte-glass trio)
14. **`CTRL107` – `CTRL111`**: 20-Test GPU Verification Suite (suite bar + window)
15. **`CTRL112` – `CTRL115`**: Implicit Specimen & Analytical GPU Primitives
16. **`CTRL116` – `CTRL126`**: 11 Layers Visibility & Turntable Orbit Toggles
17. **`CTRL127` – `CTRL138`**: Mesh Extraction, Adaptive Resolution & 3D Exporters
18. **`CTRL139` – `CTRL148`**: Procedural Bark Texture Synthesis & Surface Displacement (texture + relief)
19. **`CTRL149` – `CTRL171`**: Style Editor: Sizes (23 ImGuiStyle layout sliders)
20. **`CTRL172` – `CTRL234`**: Style Editor: Colors (census row per `ImGuiCol_` slot — 63 at the 8d reference baseline (ImGui 1.92.8+), live count follows the build; commit-only telemetry)
21. **`CTRL235` – `CTRL241`**: Style Editor: UI Rendering (alpha, anti-aliasing, curve/circle tessellation)

---

### 3. High-Resolution Timing Macros

```cpp
#pragma once

#include "core/unified_logger.hpp"

#define RHIZOME_LOG_ACTION(ctrl_id, old_v, new_v) \
    rhizome::EventStreamManager::instance().capture_interaction((ctrl_id), (old_v), (new_v))

#define RHIZOME_LOG_REACTION(ctrl_id, ms, gpu_ms, tris_before, tris_after, msg) \
    rhizome::EventStreamManager::instance().capture_reaction( \
        (ctrl_id), (ms), (gpu_ms), (tris_before), (tris_after), (msg), true)
```

---

## 🔄 CODEBASE INTEGRATION POINTS

### 1. In `native/src/ui/panels/panel_growth.cpp` (Growth & Caliber Example)

```cpp
// Vessel & Caliber Section: Trunk Caliber Slider (CTRL028)
float old_trunk = ctx.app_state.growth.trunk_caliber;
if (render_compact_slider_float("Trunk Radius", &ctx.app_state.growth.trunk_caliber, 0.10f, 3.00f, "%.2f mm")) {
    RHIZOME_LOG_ACTION("CTRL028", old_trunk, ctx.app_state.growth.trunk_caliber);

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t tris_before = ctx.roots_mesh.triangle_count();

    sync_caliber_live();

    auto end_time = std::chrono::high_resolution_clock::now();
    int64_t total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    RHIZOME_LOG_REACTION("CTRL028", total_ms, 0.0f, tris_before, ctx.roots_mesh.triangle_count(),
                         "Updated live root caliber and regenerated visual buffers");
}
```

### 2. In `native/src/ui/panels/panel_scene.cpp` (Specimen Detail Example)

```cpp
// Implicit Specimen Section: Detail Level Override (CTRL099)
float old_detail = ctx.app_state.growth.detail_level;
bool changed = render_compact_slider_float("Detail Level", &ctx.app_state.growth.detail_level, 0.05f, 0.80f, "%.2f mm");
if (changed) {
    RHIZOME_LOG_ACTION("CTRL099", old_detail, ctx.app_state.growth.detail_level);
}
if (ImGui::IsItemDeactivatedAfterEdit() || (changed && !ImGui::IsItemActive())) {
    auto start_rebuild = std::chrono::high_resolution_clock::now();
    size_t tris_before = ctx.specimen_triangle_count;

    ctx.convert_and_update_model(ctx.original_mesh, ctx.loaded_model_name);

    int64_t rebuild_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_rebuild
    ).count();

    RHIZOME_LOG_REACTION("CTRL099", rebuild_ms, ctx.specimen_build_time_ms,
                         tris_before, ctx.specimen_triangle_count,
                         "SDF volume rebuilt and meshed via CUDA marching cubes");
}
```

---

## 📊 STRUCTURED JSON EVENT FORMAT

Every event recorded into `logs/ui_events.jsonl` follows this validated schema:

```json
{
  "timestamp_ns": 1705689234567890123,
  "iso_time": "2026-10-18T10:14:32.567Z",
  "event_type": "INTERACTION_COMPLETE",
  "control": {
    "id": "CTRL099",
    "name": "Detail Level Override",
    "category": "SPECIMEN_01",
    "section": "Implicit Specimen",
    "source_file": "panel_scene.cpp",
    "widget_type": "Slider Float",
    "unit": "mm"
  },
  "interaction": {
    "old_value": 0.40,
    "new_value": 0.25,
    "delta": -0.15
  },
  "backend_reaction": {
    "total_latency_ms": 38,
    "cuda_kernel_time_ms": 28.4,
    "triangle_count_before": 24190,
    "triangle_count_after": 61840,
    "geometry_delta_triangles": 37650,
    "message": "SDF volume rebuilt and meshed via CUDA marching cubes",
    "success": true
  },
  "gpu_telemetry": {
    "vram_used_gb": 3.8,
    "vram_total_gb": 12.0,
    "vram_occupancy_percent": 31.7,
    "device_name": "NVIDIA GeForce RTX 4070"
  }
}
```

---

## 📈 BENEFIT & USE CASES

1. **Deterministic Performance Profiling:** Pinpoints exactly which slider adjustments cause heavy GPU latency or VRAM spikes.
2. **Automated Regression Testing:** Replays recorded user session files to test for visual differences or memory leaks across software releases.
3. **Hardware Scaling Verification:** Compares millisecond reactions across different NVIDIA GPUs (RTX 3060 vs 4090 vs A100).
