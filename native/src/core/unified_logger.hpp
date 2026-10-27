#pragma once
// =========================================================================
// RHIZOME UNIFIED DEBUG & MEMORY LOGGER — core module (v23.29 Phase 2)
// =========================================================================
// Single Source of Truth for host-side logging, GPU VRAM telemetry, UI
// interaction monitoring (JSON-Lines), and step performance tracing.
//
// Architecture (CODE_MAP.md Sec 3.9 / AGENTS.md Sec 2 -> core/):
//   * UnifiedLogger  — process-wide singleton. Zero-allocation fixed-size
//                      ring buffer; ONE background flusher thread performs
//                      all asynchronous console + file emission (with
//                      shutdown drain guards).
//   * Free-function compatibility API (RhizomeLog, RhizomeLogMemory,
//     RhizomeLogVramSnapshot, log_ui_interaction, log_ui_reaction,
//     init/shutdown_ui_interaction_monitor, query_gpu_memory_stats) —
//     preserved 1:1 from v23.28 so every existing call site (panels,
//     growth controller, GPU benchmark runner, verification harness)
//     compiles and emits byte-identical formats without modification.
//
// Migrated in Phase 2 from the bottom section of src/app/ui_widgets.cpp,
// fixing layering violations F1/F2 (Phase 1 findings): rhizome_core code
// may now log without depending on the ImGui application layer.
// =========================================================================

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

namespace rhizome {

// ---- Real-time GPU Memory Tracking (moved from app/ui_widgets.hpp) ----
struct GpuMemoryStats {
    size_t total_bytes = 0;
    size_t free_bytes = 0;
    size_t used_bytes = 0;
    float used_percent = 0.0f;
    bool valid = false;
};

GpuMemoryStats query_gpu_memory_stats();

class UnifiedLogger {
public:
    // Fixed-capacity ring storage: zero heap allocation at steady state.
    static constexpr size_t kRingCapacity  = 1024;
    static constexpr size_t kCategoryChars = 24;
    static constexpr size_t kConsoleChars  = 1024;
    static constexpr size_t kJsonlChars    = 1024;

    enum class Sink : uint8_t {
        ConsoleOnly,      // "[HH:MM:SS.mmm] [CAT] msg" to stdout
        JsonlOnly,        // raw JSON line to the monitor stream
        ConsoleAndJsonl   // both
    };

    struct Record {
        std::chrono::system_clock::time_point timestamp; // captured at enqueue
        Sink sink = Sink::ConsoleOnly;
        char category[kCategoryChars] = {};
        char console_line[kConsoleChars] = {};
        char jsonl_line[kJsonlChars] = {};
    };

    static UnifiedLogger& instance();

    // ---- CODE_MAP.md Sec 3.9 canonical API ----
    void log_info(const char* message);
    void log_gpu_memory(const char* context = nullptr);   // MEM:VRAM snapshot
    void log_step(const char* stage, double elapsed_ms);  // step execution perf
    void log_interaction(const char* control_id_or_label, float old_val,
                         float new_val, const char* details = nullptr);

    // ---- Ring buffer management ----
    void enqueue(Record&& rec);   // thread-safe; drops newest on overflow
    void flush_now();             // synchronous drain (shutdown guard)
    size_t dropped_records() const {
        return m_dropped.load(std::memory_order_relaxed);
    }
    bool jsonl_active() const {
        return m_jsonl_active.load(std::memory_order_relaxed);
    }

    // ---- JSON-Lines monitor session (logs/ui_events.jsonl) ----
    void open_jsonl_session(const char* file_path);  // idempotent
    void close_jsonl_session();                      // SESSION_END + drain

    // Synchronous oversize bypass (records exceeding the fixed ring payload,
    // e.g. long benchmark summaries) — preserves full text fidelity.
    void write_jsonl_direct(const char* json_line);

private:
    UnifiedLogger();
    ~UnifiedLogger();
    UnifiedLogger(const UnifiedLogger&) = delete;
    UnifiedLogger& operator=(const UnifiedLogger&) = delete;

    void flusher_loop();
    void drain_locked();  // requires m_mutex

    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    std::thread m_flusher;
    std::array<Record, kRingCapacity> m_ring;
    size_t m_head = 0;    // next write slot
    size_t m_count = 0;   // pending records
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_jsonl_active{false};
    std::atomic<size_t> m_dropped{0};
    std::string m_jsonl_path = "logs/ui_events.jsonl";
    std::ofstream m_jsonl_file; // written only under m_mutex (flusher thread)
};

// ---- Free-function compatibility API (v23.28 call sites unchanged) ----
void RhizomeLog(const char* category, const char* message);
void RhizomeLogMemory(const char* buffer_name, size_t count, size_t element_size_bytes);
void RhizomeLogVramSnapshot(const char* context = nullptr);

void init_ui_interaction_monitor(const char* log_file_path = "logs/ui_events.jsonl");
void shutdown_ui_interaction_monitor();
void log_ui_interaction(const char* control_id_or_label, float old_val, float new_val,
                        const char* details = nullptr);
void log_ui_reaction(const char* control_id_or_label, int64_t elapsed_ms,
                     float gpu_ms = 0.0f, size_t tris_before = 0, size_t tris_after = 0,
                     const char* message = nullptr);

} // namespace rhizome
