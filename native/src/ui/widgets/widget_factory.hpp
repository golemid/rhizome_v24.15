#pragma once

// =========================================================================
// RHIZOME WIDGET FACTORY — ui/widgets module (v23.29 Phase 4)
// =========================================================================
// The custom ImGui widget primitives: floating pill accordion headers,
// pill combos & buttons, iOS-style toggle switches, labeled and compact
// sleek-track sliders. Every primitive implements the exact rhizome::UiLayout
// pixel constants (AGENTS.md Sec A.1: ROW_HEIGHT_SLIDER = 16.0f,
// SLIDER_LABEL_WIDTH = 104.0f, ...). Drawing code only — migrated verbatim
// from include/app/ui_widgets.hpp (the logging half moved to core/ in
// Phase 2; the 138-control catalog now lives in ui/widgets/
// control_registry.hpp).
// =========================================================================

#include <imgui.h>
#include <algorithm>
#include <cstddef>

namespace rhizome {

namespace UiLayout {
    // Sizing & Heights
    constexpr float ROW_HEIGHT_SLIDER   = 16.0f; // Standard height of compact sliders & toggle switches
    constexpr float ROW_HEIGHT_BUTTON   = 22.0f; // Standard height of buttons, dropdowns, inputs
    constexpr float ROW_HEIGHT_HEADER   = 22.0f; // Height of section header cards
    constexpr float STATUS_BAR_HEIGHT   = 20.0f; // Height of status bar

    // Spacing & Padding
    constexpr float ITEM_SPACING_X      = 4.0f;  // Gap between inline buttons / items
    constexpr float ITEM_SPACING_Y      = 4.0f;  // Gap between consecutive rows (center-to-center = 20px)
    constexpr float WINDOW_PADDING_X    = 10.0f; // Horizontal panel padding
    constexpr float WINDOW_PADDING_Y    = 6.0f;  // Vertical panel padding
    constexpr float FRAME_PADDING_X     = 8.0f;  // Button/input horizontal padding
    constexpr float FRAME_PADDING_Y     = 3.0f;  // Button/input vertical padding

    // Slider & Toggle Internal Columns
    constexpr float SLIDER_LABEL_WIDTH  = 104.0f;// Width allocated to parameter name
    constexpr float SLIDER_VALUE_WIDTH  = 56.0f; // Width allocated to numerical value
    constexpr float SLIDER_TRACK_HEIGHT = 4.0f;  // Height of slider rail
    constexpr float SLIDER_KNOB_RADIUS  = 5.0f;  // Resting knob radius (10px diameter)
    constexpr float TOGGLE_TRACK_WIDTH  = 34.0f; // Width of toggle switch track
    constexpr float TOGGLE_TRACK_HEIGHT = 4.0f;  // Height of toggle switch track
    constexpr float TOGGLE_KNOB_RADIUS  = 5.0f;  // Radius of toggle switch knob

    // Rounding & Borders
    constexpr float FRAME_ROUNDING      = 4.0f;  // Button & input corners
    constexpr float WINDOW_ROUNDING     = 6.0f;  // Panel & popup corners
    constexpr float BORDER_THICKNESS    = 1.0f;  // Standard 1px border
}

enum class PillButtonAccent {
    Normal = 0,
    TealPrimary,
    OrangeAction
};

// "No-Scroll" Floating Pill Accordion Header
bool render_floating_pill_header(const char* label, bool default_open = true);

// Custom Pill Dropdown Combo with right-aligned arrow
bool begin_pill_combo(const char* id, const char* preview_value, float width_arg = 0.0f, float height = 22.0f);
void end_pill_combo();
bool render_pill_combo(const char* id, int* current_item, const char* const items[], int items_count, float width_arg = 0.0f, float height = 22.0f);

// Custom Pill Button
bool render_pill_button(const char* label, const ImVec2& size_arg = ImVec2(0.0f, 22.0f), PillButtonAccent accent = PillButtonAccent::Normal);

// iOS-style Right-Aligned Toggle Switch
bool render_toggle_switch(const char* label, bool* v);

// Labeled Sliders
bool render_labeled_slider_float(const char* label, float* val, float v_min, float v_max, const char* format);
bool render_labeled_slider_int(const char* label, int* val, int v_min, int v_max, const char* format = "%d");

// Compact Sleek Track Sliders
bool render_compact_slider_float(const char* label, float* val, float v_min, float v_max, const char* format = "%.2f");
bool render_compact_slider_int(const char* label, int* val, int v_min, int v_max, const char* format = "%d");

} // namespace rhizome
