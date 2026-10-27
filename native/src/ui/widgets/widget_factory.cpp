// =========================================================================
// RHIZOME WIDGET FACTORY — drawing implementation (v23.29 Phase 4)
// Migrated verbatim from src/app/ui_widgets.cpp (widget-drawing half; the
// logging system moved to core/unified_logger.cpp in Phase 2). The dead
// upward include of the CUDA rasterizer header (zero symbols used — same
// class as Phase 3 finding F12) has been dropped.
// =========================================================================

#include "ui/widgets/widget_factory.hpp"
#include "core/unified_logger.hpp"

#include <iostream>
#include <fstream>
#include <mutex>
#include <filesystem>
#include <sstream>
#include <unordered_map>
#include <iomanip>
#include <string>
#include <chrono>
#include <ctime>
#include <imgui_internal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace rhizome {

// Custom UI Widgets: "No-Scroll" Floating Pill Accordion Header (Dear ImGui CollapsingHeader semantics bound to live style parameters), iOS-style Right-Aligned Toggle Switch & Clean Sliders
bool render_floating_pill_header(const char* label, bool default_open) {
    ImGuiStorage* storage = ImGui::GetStateStorage();
    ImGuiID id = ImGui::GetID(label);
    bool is_open = storage->GetBool(id, default_open);

    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    float avail_w = ImGui::GetContentRegionAvail().x;
    const ImGuiStyle& style = ImGui::GetStyle();
    float height = 22.0f;
    float rounding = style.FrameRounding;

    bool clicked = ImGui::InvisibleButton(label, ImVec2(avail_w, height));
    if (clicked) {
        is_open = !is_open;
        storage->SetBool(id, is_open);
    }

    bool hovered = ImGui::IsItemHovered();
    bool active = ImGui::IsItemActive();

    // 1. Outer Drop Shadow (soft ambient depth)
    draw_list->AddRectFilled(
        ImVec2(p.x, p.y + 2.0f),
        ImVec2(p.x + avail_w, p.y + height + 3.0f),
        IM_COL32(0, 0, 0, 45),
        rounding
    );
    draw_list->AddRectFilled(
        ImVec2(p.x + 1.0f, p.y + 1.0f),
        ImVec2(p.x + avail_w - 1.0f, p.y + height + 1.5f),
        IM_COL32(0, 0, 0, 30),
        rounding
    );

    // 2. Custom Container Box (Header / Card background with border)
    ImU32 bg_col = active ? ImGui::GetColorU32(ImGuiCol_HeaderActive) : (hovered ? ImGui::GetColorU32(ImGuiCol_HeaderHovered) : ImGui::GetColorU32(ImGuiCol_Header));
    draw_list->AddRectFilled(p, ImVec2(p.x + avail_w, p.y + height), bg_col, rounding);
    draw_list->AddRect(p, ImVec2(p.x + avail_w, p.y + height), ImGui::GetColorU32(ImGuiCol_Border), rounding, 0, style.FrameBorderSize > 0.0f ? style.FrameBorderSize : 1.0f);

    // 3. Subtle Top Border Highlight
    draw_list->AddLine(
        ImVec2(p.x + rounding, p.y + 1.0f),
        ImVec2(p.x + avail_w - rounding, p.y + 1.0f),
        IM_COL32(255, 255, 255, 28),
        1.0f
    );

    // 4. Triangle Icon (Primary Teal CheckMark/SliderGrab color)
    float tri_center_x = p.x + 14.0f;
    float tri_center_y = p.y + height * 0.5f;
    float tri_sz = 4.5f;
    ImU32 tri_col = hovered ? ImGui::GetColorU32(ImGuiCol_SliderGrabActive) : ImGui::GetColorU32(ImGuiCol_CheckMark);

    if (is_open) {
        draw_list->AddTriangleFilled(
            ImVec2(tri_center_x - tri_sz, tri_center_y - tri_sz * 0.5f),
            ImVec2(tri_center_x + tri_sz, tri_center_y - tri_sz * 0.5f),
            ImVec2(tri_center_x, tri_center_y + tri_sz * 0.7f),
            tri_col
        );
    } else {
        draw_list->AddTriangleFilled(
            ImVec2(tri_center_x - tri_sz * 0.5f, tri_center_y - tri_sz),
            ImVec2(tri_center_x + tri_sz * 0.7f, tri_center_y),
            ImVec2(tri_center_x - tri_sz * 0.5f, tri_center_y + tri_sz),
            tri_col
        );
    }

    // 5. Section Title (#E9F0F4 main text)
    const char* text_end = ImGui::FindRenderedTextEnd(label);
    ImVec2 text_pos(p.x + 28.0f, p.y + (height - ImGui::GetTextLineHeight()) * 0.5f);
    draw_list->AddText(text_pos, hovered ? IM_COL32(255, 255, 255, 255) : IM_COL32(233, 240, 244, 255), label, text_end);

    return is_open;
}

// Custom Pill Dropdown Combo with right-aligned arrow (distinct from collapsible section headers)
bool begin_pill_combo(const char* id, const char* preview_value, float width_arg, float height) {
    ImGui::PushID(id);
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    float avail_w = width_arg > 0.0f ? width_arg : ImGui::GetContentRegionAvail().x;
    const ImGuiStyle& style = ImGui::GetStyle();
    float rounding = style.FrameRounding;

    ImGuiID popup_id = ImGui::GetID("##pill_combo_popup");
    bool is_open = ImGui::IsPopupOpen(popup_id, ImGuiPopupFlags_None);

    bool clicked = ImGui::InvisibleButton("##pill_combo_btn", ImVec2(avail_w, height));
    if (clicked) {
        ImGui::OpenPopup(popup_id);
        is_open = true;
    }

    bool hovered = ImGui::IsItemHovered();
    bool active = ImGui::IsItemActive() || is_open;

    // 1. Outer Drop Shadow (soft ambient depth)
    draw_list->AddRectFilled(
        ImVec2(p.x, p.y + 2.0f),
        ImVec2(p.x + avail_w, p.y + height + 3.0f),
        IM_COL32(0, 0, 0, 45),
        rounding
    );
    draw_list->AddRectFilled(
        ImVec2(p.x + 1.0f, p.y + 1.0f),
        ImVec2(p.x + avail_w - 1.0f, p.y + height + 1.5f),
        IM_COL32(0, 0, 0, 30),
        rounding
    );

    // 2. Custom Container Box (Input/Frame background with border)
    ImU32 bg_col = active ? ImGui::GetColorU32(ImGuiCol_FrameBgActive) : (hovered ? ImGui::GetColorU32(ImGuiCol_FrameBgHovered) : ImGui::GetColorU32(ImGuiCol_FrameBg));
    draw_list->AddRectFilled(p, ImVec2(p.x + avail_w, p.y + height), bg_col, rounding);
    draw_list->AddRect(p, ImVec2(p.x + avail_w, p.y + height), ImGui::GetColorU32(ImGuiCol_Border), rounding, 0, style.FrameBorderSize > 0.0f ? style.FrameBorderSize : 1.0f);

    // 3. Subtle Top Border Highlight
    draw_list->AddLine(
        ImVec2(p.x + rounding, p.y + 1.0f),
        ImVec2(p.x + avail_w - rounding, p.y + 1.0f),
        IM_COL32(255, 255, 255, 28),
        1.0f
    );

    // 4. Selected Preview Label on the left (#E9F0F4 main text)
    if (preview_value) {
        const char* preview_end = ImGui::FindRenderedTextEnd(preview_value);
        ImVec2 text_pos(p.x + 10.0f, p.y + (height - ImGui::GetTextLineHeight()) * 0.5f);
        draw_list->PushClipRect(ImVec2(p.x + 6.0f, p.y), ImVec2(p.x + avail_w - 26.0f, p.y + height), true);
        draw_list->AddText(text_pos, (hovered || is_open) ? IM_COL32(255, 255, 255, 255) : IM_COL32(233, 240, 244, 255), preview_value, preview_end);
        draw_list->PopClipRect();
    }

    // 5. Triangle Icon on the right (Primary Teal CheckMark/SliderGrab color, no divider line)
    float tri_center_x = p.x + avail_w - 14.0f;
    float tri_center_y = p.y + height * 0.5f;
    float tri_sz = 4.5f;
    ImU32 tri_col = (hovered || is_open) ? ImGui::GetColorU32(ImGuiCol_SliderGrabActive) : ImGui::GetColorU32(ImGuiCol_CheckMark);

    draw_list->AddTriangleFilled(
        ImVec2(tri_center_x - tri_sz, tri_center_y - tri_sz * 0.5f),
        ImVec2(tri_center_x + tri_sz, tri_center_y - tri_sz * 0.5f),
        ImVec2(tri_center_x, tri_center_y + tri_sz * 0.7f),
        tri_col
    );

    ImGui::SetNextWindowPos(ImVec2(p.x, p.y + height + 2.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSizeConstraints(ImVec2(avail_w, 0.0f), ImVec2(std::max(avail_w, 260.0f), 360.0f));
    if (ImGui::BeginPopupEx(popup_id, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove)) {
        return true;
    }

    ImGui::PopID();
    return false;
}

void end_pill_combo() {
    ImGui::EndPopup();
    ImGui::PopID();
}

bool render_pill_combo(const char* id, int* current_item, const char* const items[], int items_count, float width_arg, float height) {
    const char* preview = (*current_item >= 0 && *current_item < items_count) ? items[*current_item] : "";
    bool changed = false;
    if (begin_pill_combo(id, preview, width_arg, height)) {
        for (int i = 0; i < items_count; ++i) {
            bool is_selected = (*current_item == i);
            if (ImGui::Selectable(items[i], is_selected)) {
                if (*current_item != i) {
                    *current_item = i;
                    changed = true;
                }
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        end_pill_combo();
    }
    return changed;
}

bool render_pill_button(const char* label, const ImVec2& size_arg, PillButtonAccent accent) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImGuiStyle& style = ImGui::GetStyle();

    float height = size_arg.y > 0.0f ? size_arg.y : 22.0f;
    float text_width = ImGui::CalcTextSize(label).x;
    float width = size_arg.x > 0.0f ? size_arg.x : (size_arg.x < 0.0f ? ImGui::GetContentRegionAvail().x : (text_width + 24.0f));
    float rounding = style.FrameRounding;

    bool clicked = ImGui::InvisibleButton(label, ImVec2(width, height));
    bool hovered = ImGui::IsItemHovered();
    bool active = ImGui::IsItemActive();

    // 1. Outer Drop Shadow (soft ambient depth matching Colonization Seeds)
    draw_list->AddRectFilled(
        ImVec2(p.x, p.y + 2.0f),
        ImVec2(p.x + width, p.y + height + 3.0f),
        IM_COL32(0, 0, 0, 45),
        rounding
    );
    draw_list->AddRectFilled(
        ImVec2(p.x + 1.0f, p.y + 1.0f),
        ImVec2(p.x + width - 1.0f, p.y + height + 1.5f),
        IM_COL32(0, 0, 0, 30),
        rounding
    );

    // 2. Custom Container Box (Header / Card background with border)
    ImU32 bg_col;
    ImU32 text_col;
    ImU32 border_col = ImGui::GetColorU32(ImGuiCol_Border);

    if (accent == PillButtonAccent::TealPrimary) {
        if (active) bg_col = IM_COL32(20, 178, 135, 255);
        else if (hovered) bg_col = IM_COL32(40, 229, 180, 255);
        else bg_col = IM_COL32(25, 211, 162, 255);
        text_col = IM_COL32(7, 17, 28, 255);
    } else if (accent == PillButtonAccent::OrangeAction) {
        if (active) bg_col = IM_COL32(230, 115, 20, 255);
        else if (hovered) bg_col = IM_COL32(255, 158, 64, 255);
        else bg_col = IM_COL32(255, 138, 31, 255);
        text_col = IM_COL32(255, 255, 255, 255);
    } else {
        if (active) bg_col = ImGui::GetColorU32(ImGuiCol_HeaderActive);
        else if (hovered) bg_col = ImGui::GetColorU32(ImGuiCol_HeaderHovered);
        else bg_col = ImGui::GetColorU32(ImGuiCol_Header);
        text_col = hovered ? IM_COL32(255, 255, 255, 255) : IM_COL32(233, 240, 244, 255);
    }

    draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), bg_col, rounding);
    draw_list->AddRect(p, ImVec2(p.x + width, p.y + height), border_col, rounding, 0, style.FrameBorderSize > 0.0f ? style.FrameBorderSize : 1.0f);

    // 3. Subtle Top Border Highlight
    draw_list->AddLine(
        ImVec2(p.x + rounding, p.y + 1.0f),
        ImVec2(p.x + width - rounding, p.y + 1.0f),
        accent != PillButtonAccent::Normal ? IM_COL32(255, 255, 255, 60) : IM_COL32(255, 255, 255, 28),
        1.0f
    );

    // 4. Centered Text
    ImVec2 text_pos(
        p.x + (width - text_width) * 0.5f,
        p.y + (height - ImGui::GetTextLineHeight()) * 0.5f
    );
    draw_list->AddText(text_pos, text_col, label);

    if (clicked) RhizomeLog("ACTION", ("Button: " + std::string(label)).c_str());
    return clicked;
}

bool render_toggle_switch(const char* label, bool* v) {
    ImGui::PushID(label);
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float row_h = UiLayout::ROW_HEIGHT_SLIDER;
    float avail_width = ImGui::GetContentRegionAvail().x;

    // Mini slider rail dimensions matching slider styling
    float track_w = UiLayout::TOGGLE_TRACK_WIDTH;
    float track_h = UiLayout::TOGGLE_TRACK_HEIGHT;
    float knob_r = UiLayout::TOGGLE_KNOB_RADIUS;

    // Full row clickable button matching compact slider row height
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
    bool clicked = ImGui::InvisibleButton("##toggle_row", ImVec2(avail_width, row_h));
    ImGui::PopStyleVar();
    if (clicked) {
        *v = !*v;
        RhizomeLog("ACTION", ("Toggle: " + std::string(label) + " -> " + std::string(*v ? "ON" : "OFF")).c_str());
    }

    bool hovered = ImGui::IsItemHovered();
    // Label on the left
    float text_y = p.y + (row_h - ImGui::GetTextLineHeight()) * 0.5f;
    draw_list->AddText(ImVec2(p.x, text_y),
                       hovered ? IM_COL32(233, 240, 244, 255) : IM_COL32(137, 152, 167, 255),
                       label);

    // Mini slider track on the right
    float switch_x = p.x + avail_width - track_w - 4.0f;
    float cy = p.y + row_h * 0.5f;

    // Unfilled background track (#182738)
    ImU32 col_bg_track = hovered ? IM_COL32(32, 49, 68, 255) : IM_COL32(24, 39, 56, 255);
    draw_list->AddRectFilled(
        ImVec2(switch_x, cy - track_h * 0.5f),
        ImVec2(switch_x + track_w, cy + track_h * 0.5f),
        col_bg_track,
        track_h * 0.5f
    );

    float t = *v ? 1.0f : 0.0f;
    float usable_w = track_w - 2.0f * knob_r;
    float knob_cx = (switch_x + knob_r) + t * usable_w;

    // Active track fill when toggled on (#19D3A2 / #28E5B4)
    if (*v) {
        ImU32 col_fill = hovered ? IM_COL32(40, 229, 180, 255) : IM_COL32(25, 211, 162, 255);
        draw_list->AddRectFilled(
            ImVec2(switch_x, cy - track_h * 0.5f),
            ImVec2(knob_cx, cy + track_h * 0.5f),
            col_fill,
            track_h * 0.5f
        );
    }

    // Circular Knob / Thumb matching slider knob styling
    float current_r = hovered ? (knob_r + 0.5f) : knob_r;
    if (*v) {
        ImU32 knob_col = hovered ? IM_COL32(40, 229, 180, 255) : IM_COL32(25, 211, 162, 255);
        draw_list->AddCircleFilled(ImVec2(knob_cx, cy), current_r, knob_col);
        draw_list->AddCircle(ImVec2(knob_cx, cy), current_r, IM_COL32(38, 56, 74, 220), 16, 1.0f);
    } else {
        ImU32 knob_col = hovered ? IM_COL32(137, 152, 167, 255) : IM_COL32(88, 106, 126, 255);
        draw_list->AddCircleFilled(ImVec2(knob_cx, cy), current_r, knob_col);
        draw_list->AddCircle(ImVec2(knob_cx, cy), current_r, IM_COL32(24, 39, 56, 220), 16, 1.0f);
    }

    ImGui::PopID();
    return clicked;
}

static ImGuiID g_active_slider_edit_id = 0;
static char g_slider_edit_buf[64] = "";
static bool g_slider_edit_just_opened = false;

bool render_labeled_slider_float(const char* label, float* val, float v_min, float v_max, const char* format) {
    ImGui::PushID(label);
    ImGuiID slider_id = ImGui::GetID("##slider_root");
    bool is_editing = (g_active_slider_edit_id == slider_id);
    bool value_changed = false;

    char val_buf[64];
    snprintf(val_buf, sizeof(val_buf), format, *val);

    // Format range reference string [min - max]
    char range_buf[64];
    if (strstr(format, "%.0f") != nullptr) {
        snprintf(range_buf, sizeof(range_buf), "[%.0f - %.0f]", v_min, v_max);
    } else if (strstr(format, "%.1f") != nullptr) {
        snprintf(range_buf, sizeof(range_buf), "[%.1f - %.1f]", v_min, v_max);
    } else {
        snprintf(range_buf, sizeof(range_buf), "[%.2f - %.2f]", v_min, v_max);
    }

    // Label on the left (#8998A7)
    ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "%s", label);

    if (!is_editing) {
        // Formatted value on the right (#19D3A2) - Clickable to open inline edit
        float val_w = ImGui::CalcTextSize(val_buf).x;
        float avail_w = ImGui::GetContentRegionAvail().x;
        if (avail_w > val_w) {
            ImGui::SameLine(ImGui::GetCursorPosX() + avail_w - val_w - 4.0f);
            ImVec2 click_pos = ImGui::GetCursorScreenPos();
            if (ImGui::InvisibleButton("##num_btn", ImVec2(val_w + 4.0f, ImGui::GetTextLineHeight()))) {
                g_active_slider_edit_id = slider_id;
                g_slider_edit_just_opened = true;
                snprintf(g_slider_edit_buf, sizeof(g_slider_edit_buf), format, *val);
                char* p = g_slider_edit_buf;
                while (*p) {
                    if (*p != '-' && *p != '.' && (*p < '0' || *p > '9')) {
                        *p = '\0';
                        break;
                    }
                    p++;
                }
            }
            bool hovered = ImGui::IsItemHovered();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            ImU32 num_col = hovered ? IM_COL32(40, 229, 180, 255) : IM_COL32(25, 211, 162, 255);
            dl->AddText(click_pos, num_col, val_buf);
            if (hovered) {
                dl->AddLine(ImVec2(click_pos.x, click_pos.y + ImGui::GetTextLineHeight() - 1.0f),
                            ImVec2(click_pos.x + val_w, click_pos.y + ImGui::GetTextLineHeight() - 1.0f),
                            IM_COL32(40, 229, 180, 180), 1.0f);
                ImGui::SetTooltip("Click to enter value manually");
            }
        }
    } else {
        // Inline Edit Box with Range Reference popup on its left
        float range_w = ImGui::CalcTextSize(range_buf).x;
        float input_box_w = 68.0f;
        float total_cluster_w = range_w + 8.0f + input_box_w;
        float avail_w = ImGui::GetContentRegionAvail().x;
        if (avail_w > total_cluster_w) {
            ImGui::SameLine(ImGui::GetCursorPosX() + avail_w - total_cluster_w);
        } else {
            ImGui::SameLine();
        }

        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(ImVec4(1.0f, 0.702f, 0.400f, 0.85f), "%s", range_buf);
        ImGui::SameLine(0.0f, 6.0f);

        ImGui::SetNextItemWidth(input_box_w);
        if (g_slider_edit_just_opened) {
            ImGui::SetKeyboardFocusHere();
            g_slider_edit_just_opened = false;
        }
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.086f, 0.129f, 0.169f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.098f, 0.827f, 0.635f, 0.8f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);

        bool enter_pressed = ImGui::InputText("##input", g_slider_edit_buf, sizeof(g_slider_edit_buf), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);

        if (enter_pressed) {
            char* end_ptr = nullptr;
            float parsed = strtof(g_slider_edit_buf, &end_ptr);
            if (end_ptr != g_slider_edit_buf) {
                if (parsed < v_min) parsed = v_min;
                if (parsed > v_max) parsed = v_max;
                if (*val != parsed) {
                    *val = parsed;
                    value_changed = true;
                    RhizomeLog("ACTION", ("Labeled Slider Adjusted: " + std::string(label)).c_str());
                }
            }
            g_active_slider_edit_id = 0;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            g_active_slider_edit_id = 0;
        } else if (ImGui::IsItemDeactivated() && !enter_pressed) {
            g_active_slider_edit_id = 0;
        }

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);
    }

    // Custom sleek track slider matching reference image
    ImVec2 cursor = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float avail_w = ImGui::GetContentRegionAvail().x;
    float row_h = 16.0f;
    float track_h = 4.0f;
    float knob_r = 6.0f;

    ImGui::InvisibleButton("##track_btn", ImVec2(avail_w, row_h));
    bool is_hovered = ImGui::IsItemHovered();
    bool is_active = ImGui::IsItemActive();

    float usable_w = avail_w - 2.0f * knob_r;
    if (usable_w < 10.0f) usable_w = 10.0f;

    // Mouse drag input
    if (is_active && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        float mouse_x = ImGui::GetIO().MousePos.x;
        float t = (mouse_x - (cursor.x + knob_r)) / usable_w;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        float new_val = v_min + t * (v_max - v_min);
        if (*val != new_val) {
            *val = new_val;
            value_changed = true;
            RhizomeLog("ACTION", ("Labeled Slider Adjusted: " + std::string(label)).c_str());
        }
    }

    if (is_hovered || is_active) {
        float range = (v_max - v_min);
        float step = range * 0.01f;
        if (ImGui::GetIO().KeyShift) {
            step *= 0.1f;
        }

        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, true)) {
            float next_val = std::max(v_min, *val - step);
            if (*val != next_val) {
                *val = next_val;
                value_changed = true;
                std::string msg = std::string(label) + " set to " + std::to_string(*val);
                RhizomeLog("ACTION", msg.c_str());
            }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, true)) {
            float next_val = std::min(v_max, *val + step);
            if (*val != next_val) {
                *val = next_val;
                value_changed = true;
                std::string msg = std::string(label) + " set to " + std::to_string(*val);
                RhizomeLog("ACTION", msg.c_str());
            }
        }
    }

    float t = (*val - v_min) / (v_max - v_min);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    float cy = cursor.y + row_h * 0.5f;
    float knob_cx = cursor.x + knob_r + t * usable_w;

    // Unfilled background track (#182738)
    ImU32 col_bg_track = IM_COL32(24, 39, 56, 255);
    draw_list->AddRectFilled(
        ImVec2(cursor.x, cy - track_h * 0.5f),
        ImVec2(cursor.x + avail_w, cy + track_h * 0.5f),
        col_bg_track,
        track_h * 0.5f
    );

    // Filled active track (#19D3A2 / #28E5B4)
    ImU32 col_fill = is_hovered || is_active ? IM_COL32(40, 229, 180, 255) : IM_COL32(25, 211, 162, 255);
    if (knob_cx > cursor.x) {
        draw_list->AddRectFilled(
            ImVec2(cursor.x, cy - track_h * 0.5f),
            ImVec2(knob_cx, cy + track_h * 0.5f),
            col_fill,
            track_h * 0.5f
        );
        if (is_active) RhizomeLog("ACTION", ("Slider Adjusted: " + std::string(label)).c_str());
    }

    // Circular Knob / Thumb centered on track
    float current_r = is_active ? (knob_r + 1.2f) : (is_hovered ? (knob_r + 0.6f) : knob_r);
    draw_list->AddCircleFilled(ImVec2(knob_cx, cy), current_r, col_fill);
    draw_list->AddCircle(ImVec2(knob_cx, cy), current_r, IM_COL32(38, 56, 74, 220), 16, 1.0f);

    ImGui::PopID();
    return value_changed;
}

bool render_labeled_slider_int(const char* label, int* val, int v_min, int v_max, const char* format) {
    ImGui::PushID(label);
    ImGuiID slider_id = ImGui::GetID("##slider_root");
    bool is_editing = (g_active_slider_edit_id == slider_id);
    bool value_changed = false;

    char val_buf[64];
    snprintf(val_buf, sizeof(val_buf), format, *val);

    char range_buf[64];
    snprintf(range_buf, sizeof(range_buf), "[%d - %d]", v_min, v_max);

    // Label on the left (#8998A7)
    ImGui::TextColored(ImVec4(0.537f, 0.596f, 0.655f, 1.0f), "%s", label);

    if (!is_editing) {
        // Formatted value on the right (#19D3A2) - Clickable to open inline edit
        float val_w = ImGui::CalcTextSize(val_buf).x;
        float avail_w = ImGui::GetContentRegionAvail().x;
        if (avail_w > val_w) {
            ImGui::SameLine(ImGui::GetCursorPosX() + avail_w - val_w - 4.0f);
            ImVec2 click_pos = ImGui::GetCursorScreenPos();
            if (ImGui::InvisibleButton("##num_btn", ImVec2(val_w + 4.0f, ImGui::GetTextLineHeight()))) {
                g_active_slider_edit_id = slider_id;
                g_slider_edit_just_opened = true;
                snprintf(g_slider_edit_buf, sizeof(g_slider_edit_buf), "%d", *val);
            }
            bool hovered = ImGui::IsItemHovered();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            ImU32 num_col = hovered ? IM_COL32(40, 229, 180, 255) : IM_COL32(25, 211, 162, 255);
            dl->AddText(click_pos, num_col, val_buf);
            if (hovered) {
                dl->AddLine(ImVec2(click_pos.x, click_pos.y + ImGui::GetTextLineHeight() - 1.0f),
                            ImVec2(click_pos.x + val_w, click_pos.y + ImGui::GetTextLineHeight() - 1.0f),
                            IM_COL32(40, 229, 180, 180), 1.0f);
                ImGui::SetTooltip("Click to enter value manually");
            }
        }
    } else {
        // Inline Edit Box with Range Reference popup on its left
        float range_w = ImGui::CalcTextSize(range_buf).x;
        float input_box_w = 68.0f;
        float total_cluster_w = range_w + 8.0f + input_box_w;
        float avail_w = ImGui::GetContentRegionAvail().x;
        if (avail_w > total_cluster_w) {
            ImGui::SameLine(ImGui::GetCursorPosX() + avail_w - total_cluster_w);
        } else {
            ImGui::SameLine();
        }

        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(ImVec4(1.0f, 0.702f, 0.400f, 0.85f), "%s", range_buf);
        ImGui::SameLine(0.0f, 6.0f);

        ImGui::SetNextItemWidth(input_box_w);
        if (g_slider_edit_just_opened) {
            ImGui::SetKeyboardFocusHere();
            g_slider_edit_just_opened = false;
        }
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.086f, 0.129f, 0.169f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.098f, 0.827f, 0.635f, 0.8f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);

        bool enter_pressed = ImGui::InputText("##input", g_slider_edit_buf, sizeof(g_slider_edit_buf), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);

        if (enter_pressed) {
            char* end_ptr = nullptr;
            long parsed = strtol(g_slider_edit_buf, &end_ptr, 10);
            if (end_ptr != g_slider_edit_buf) {
                int clamped = static_cast<int>(std::clamp<long>(parsed, v_min, v_max));
                if (*val != clamped) {
                    *val = clamped;
                    value_changed = true;
                    std::string msg = std::string(label) + " set to " + std::to_string(*val);
                    RhizomeLog("ACTION", msg.c_str());
                }
            }
            g_active_slider_edit_id = 0;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            g_active_slider_edit_id = 0;
        } else if (ImGui::IsItemDeactivated() && !enter_pressed) {
            g_active_slider_edit_id = 0;
        }

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);
    }

    // Custom sleek track slider
    ImVec2 cursor = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float avail_w = ImGui::GetContentRegionAvail().x;
    float row_h = 16.0f;
    float track_h = 4.0f;
    float knob_r = 6.0f;

    ImGui::InvisibleButton("##track_btn", ImVec2(avail_w, row_h));
    bool is_hovered = ImGui::IsItemHovered();
    bool is_active = ImGui::IsItemActive();

    float usable_w = avail_w - 2.0f * knob_r;
    if (usable_w < 10.0f) usable_w = 10.0f;

    // Mouse drag input
    if (is_active && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        float mouse_x = ImGui::GetIO().MousePos.x;
        float t = (mouse_x - (cursor.x + knob_r)) / usable_w;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        int new_val = v_min + static_cast<int>(std::round(t * static_cast<float>(v_max - v_min)));
        if (*val != new_val) {
            *val = new_val;
            value_changed = true;
            std::string msg = std::string(label) + " set to " + std::to_string(*val);
            RhizomeLog("ACTION", msg.c_str());
        }
    }

    // Keyboard 'Left' and 'Right' arrow keystrokes
    if (is_hovered || is_active) {
        int range = (v_max - v_min);
        int step = std::max(1, static_cast<int>(range * 0.01f));
        if (ImGui::GetIO().KeyShift) {
            step = 1;
        }

        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, true)) {
            int next_val = std::max(v_min, *val - step);
            if (*val != next_val) {
                *val = next_val;
                value_changed = true;
                std::string msg = std::string(label) + " set to " + std::to_string(*val);
                RhizomeLog("ACTION", msg.c_str());
            }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, true)) {
            int next_val = std::min(v_max, *val + step);
            if (*val != next_val) {
                *val = next_val;
                value_changed = true;
                std::string msg = std::string(label) + " set to " + std::to_string(*val);
                RhizomeLog("ACTION", msg.c_str());
            }
        }
    }

    float t = static_cast<float>(*val - v_min) / static_cast<float>(v_max - v_min);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    float cy = cursor.y + row_h * 0.5f;
    float knob_cx = cursor.x + knob_r + t * usable_w;

    // Unfilled background track
    ImU32 col_bg_track = IM_COL32(24, 39, 56, 255);
    draw_list->AddRectFilled(
        ImVec2(cursor.x, cy - track_h * 0.5f),
        ImVec2(cursor.x + avail_w, cy + track_h * 0.5f),
        col_bg_track,
        track_h * 0.5f
    );

    // Filled active track
    ImU32 col_fill = is_hovered || is_active ? IM_COL32(40, 229, 180, 255) : IM_COL32(25, 211, 162, 255);
    if (knob_cx > cursor.x) {
        draw_list->AddRectFilled(
            ImVec2(cursor.x, cy - track_h * 0.5f),
            ImVec2(knob_cx, cy + track_h * 0.5f),
            col_fill,
            track_h * 0.5f
        );
        if (is_active) {
            std::string msg = std::string(label) + " set to " + std::to_string(*val);
            RhizomeLog("ACTION", msg.c_str());
        }
    }

    // Circular Knob / Thumb centered on track
    float current_r = is_active ? (knob_r + 1.2f) : (is_hovered ? (knob_r + 0.6f) : knob_r);
    draw_list->AddCircleFilled(ImVec2(knob_cx, cy), current_r, col_fill);
    draw_list->AddCircle(ImVec2(knob_cx, cy), current_r, IM_COL32(38, 56, 74, 220), 16, 1.0f);

    ImGui::PopID();
    return value_changed;
}

bool render_compact_slider_float(const char* label, float* val, float v_min, float v_max, const char* format) {
    ImGui::PushID(label);
    ImGuiID slider_id = ImGui::GetID("##c_slider_f");
    bool is_editing = (g_active_slider_edit_id == slider_id);
    bool value_changed = false;

    char val_buf[64];
    snprintf(val_buf, sizeof(val_buf), format, *val);

    ImVec2 row_start = ImGui::GetCursorScreenPos();
    float avail_w = ImGui::GetContentRegionAvail().x;
    float row_h = UiLayout::ROW_HEIGHT_SLIDER;
    float label_w = UiLayout::SLIDER_LABEL_WIDTH;
    float val_col_w = UiLayout::SLIDER_VALUE_WIDTH;
    float track_w = std::max(36.0f, avail_w - label_w - val_col_w - 10.0f);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    float text_y = row_start.y + (row_h - ImGui::GetTextLineHeight()) * 0.5f;
    dl->AddText(ImVec2(row_start.x, text_y), IM_COL32(137, 152, 167, 255), label);

    ImGui::SetCursorScreenPos(ImVec2(row_start.x + label_w, row_start.y));
    ImVec2 track_pos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##c_track_f", ImVec2(track_w, row_h));
    bool is_hovered = ImGui::IsItemHovered();
    bool is_active = ImGui::IsItemActive();

    const float track_h = UiLayout::SLIDER_TRACK_HEIGHT;
    const float knob_r = UiLayout::SLIDER_KNOB_RADIUS;
    float usable_w = std::max(8.0f, track_w - 2.0f * knob_r);

    if (is_active && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        float mouse_x = ImGui::GetIO().MousePos.x;
        float t = std::clamp((mouse_x - (track_pos.x + knob_r)) / usable_w, 0.0f, 1.0f);
        float new_val = v_min + t * (v_max - v_min);
        if (*val != new_val) {
            *val = new_val;
            value_changed = true;
        }
    }

    if (is_hovered || is_active) {
        float step = (v_max - v_min) * (ImGui::GetIO().KeyShift ? 0.001f : 0.01f);
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, true)) {
            float nv = std::max(v_min, *val - step);
            if (*val != nv) { *val = nv; value_changed = true; }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, true)) {
            float nv = std::min(v_max, *val + step);
            if (*val != nv) { *val = nv; value_changed = true; }
        }
    }

    float t = std::clamp((*val - v_min) / (v_max - v_min), 0.0f, 1.0f);
    float cy = track_pos.y + row_h * 0.5f;
    float knob_cx = track_pos.x + knob_r + t * usable_w;

    dl->AddRectFilled(
        ImVec2(track_pos.x, cy - track_h * 0.5f),
        ImVec2(track_pos.x + track_w, cy + track_h * 0.5f),
        IM_COL32(24, 39, 56, 255),
        track_h * 0.5f
    );
    ImU32 col_fill = (is_hovered || is_active) ? IM_COL32(40, 229, 180, 255) : IM_COL32(25, 211, 162, 255);
    if (knob_cx > track_pos.x) {
        dl->AddRectFilled(
            ImVec2(track_pos.x, cy - track_h * 0.5f),
            ImVec2(knob_cx, cy + track_h * 0.5f),
            col_fill,
            track_h * 0.5f
        );
    }
    float cur_r = is_active ? (knob_r + 1.0f) : (is_hovered ? (knob_r + 0.5f) : knob_r);
    dl->AddCircleFilled(ImVec2(knob_cx, cy), cur_r, col_fill);
    dl->AddCircle(ImVec2(knob_cx, cy), cur_r, IM_COL32(38, 56, 74, 220), 16, 1.0f);

    ImGui::SetCursorScreenPos(ImVec2(row_start.x + label_w + track_w + 6.0f, row_start.y));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
    if (!is_editing) {
        float val_w = ImGui::CalcTextSize(val_buf).x;
        ImVec2 click_pos(row_start.x + avail_w - val_w, text_y);
        if (ImGui::InvisibleButton("##c_val_f", ImVec2(val_col_w, row_h))) {
            g_active_slider_edit_id = slider_id;
            g_slider_edit_just_opened = true;
            snprintf(g_slider_edit_buf, sizeof(g_slider_edit_buf), "%.3f", *val);
        }
        bool v_hov = ImGui::IsItemHovered();
        dl->AddText(click_pos, v_hov ? IM_COL32(40, 229, 180, 255) : IM_COL32(25, 211, 162, 255), val_buf);
        if (v_hov) {
            dl->AddLine(
                ImVec2(click_pos.x, click_pos.y + ImGui::GetTextLineHeight() - 1.0f),
                ImVec2(click_pos.x + val_w, click_pos.y + ImGui::GetTextLineHeight() - 1.0f),
                IM_COL32(40, 229, 180, 180),
                1.0f
            );
        }
    } else {
        ImGui::SetNextItemWidth(val_col_w);
        if (g_slider_edit_just_opened) {
            ImGui::SetKeyboardFocusHere();
            g_slider_edit_just_opened = false;
        }
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3.0f, 1.0f));
        bool enter_pressed = ImGui::InputText("##c_in_f", g_slider_edit_buf, sizeof(g_slider_edit_buf), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
        if (enter_pressed) {
            char* end_ptr = nullptr;
            float parsed = strtof(g_slider_edit_buf, &end_ptr);
            if (end_ptr != g_slider_edit_buf) {
                parsed = std::clamp(parsed, v_min, v_max);
                if (*val != parsed) { *val = parsed; value_changed = true; }
            }
            g_active_slider_edit_id = 0;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Escape) || (ImGui::IsItemDeactivated() && !enter_pressed)) {
            g_active_slider_edit_id = 0;
        }
        ImGui::PopStyleVar();
    }
    ImGui::PopStyleVar();

    if (value_changed) {
        std::string msg = std::string(label) + " set to " + std::to_string(*val);
        RhizomeLog("ACTION", msg.c_str());
    }

    ImGui::PopID();
    return value_changed;
}

bool render_compact_slider_int(const char* label, int* val, int v_min, int v_max, const char* format) {
    ImGui::PushID(label);
    ImGuiID slider_id = ImGui::GetID("##c_slider_i");
    bool is_editing = (g_active_slider_edit_id == slider_id);
    bool value_changed = false;

    char val_buf[64];
    snprintf(val_buf, sizeof(val_buf), format, *val);

    ImVec2 row_start = ImGui::GetCursorScreenPos();
    float avail_w = ImGui::GetContentRegionAvail().x;
    float row_h = UiLayout::ROW_HEIGHT_SLIDER;
    float label_w = UiLayout::SLIDER_LABEL_WIDTH;
    float val_col_w = UiLayout::SLIDER_VALUE_WIDTH;
    float track_w = std::max(36.0f, avail_w - label_w - val_col_w - 10.0f);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    float text_y = row_start.y + (row_h - ImGui::GetTextLineHeight()) * 0.5f;
    dl->AddText(ImVec2(row_start.x, text_y), IM_COL32(137, 152, 167, 255), label);

    ImGui::SetCursorScreenPos(ImVec2(row_start.x + label_w, row_start.y));
    ImVec2 track_pos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##c_track_i", ImVec2(track_w, row_h));
    bool is_hovered = ImGui::IsItemHovered();
    bool is_active = ImGui::IsItemActive();

    const float track_h = UiLayout::SLIDER_TRACK_HEIGHT;
    const float knob_r = UiLayout::SLIDER_KNOB_RADIUS;
    float usable_w = std::max(8.0f, track_w - 2.0f * knob_r);

    if (is_active && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        float mouse_x = ImGui::GetIO().MousePos.x;
        float t = std::clamp((mouse_x - (track_pos.x + knob_r)) / usable_w, 0.0f, 1.0f);
        int new_val = static_cast<int>(std::round(v_min + t * (v_max - v_min)));
        if (*val != new_val) {
            *val = new_val;
            value_changed = true;
        }
    }

    if (is_hovered || is_active) {
        int step = std::max(1, static_cast<int>((v_max - v_min) * (ImGui::GetIO().KeyShift ? 0.002f : 0.01f)));
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, true)) {
            int nv = std::max(v_min, *val - step);
            if (*val != nv) { *val = nv; value_changed = true; }
        }
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, true)) {
            int nv = std::min(v_max, *val + step);
            if (*val != nv) { *val = nv; value_changed = true; }
        }
    }

    float t = std::clamp(static_cast<float>(*val - v_min) / static_cast<float>(v_max - v_min), 0.0f, 1.0f);
    float cy = track_pos.y + row_h * 0.5f;
    float knob_cx = track_pos.x + knob_r + t * usable_w;

    dl->AddRectFilled(
        ImVec2(track_pos.x, cy - track_h * 0.5f),
        ImVec2(track_pos.x + track_w, cy + track_h * 0.5f),
        IM_COL32(24, 39, 56, 255),
        track_h * 0.5f
    );
    ImU32 col_fill = (is_hovered || is_active) ? IM_COL32(40, 229, 180, 255) : IM_COL32(25, 211, 162, 255);
    if (knob_cx > track_pos.x) {
        dl->AddRectFilled(
            ImVec2(track_pos.x, cy - track_h * 0.5f),
            ImVec2(knob_cx, cy + track_h * 0.5f),
            col_fill,
            track_h * 0.5f
        );
    }
    float cur_r = is_active ? (knob_r + 1.0f) : (is_hovered ? (knob_r + 0.5f) : knob_r);
    dl->AddCircleFilled(ImVec2(knob_cx, cy), cur_r, col_fill);
    dl->AddCircle(ImVec2(knob_cx, cy), cur_r, IM_COL32(38, 56, 74, 220), 16, 1.0f);

    ImGui::SetCursorScreenPos(ImVec2(row_start.x + label_w + track_w + 6.0f, row_start.y));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
    if (!is_editing) {
        float val_w = ImGui::CalcTextSize(val_buf).x;
        ImVec2 click_pos(row_start.x + avail_w - val_w, text_y);
        if (ImGui::InvisibleButton("##c_val_i", ImVec2(val_col_w, row_h))) {
            g_active_slider_edit_id = slider_id;
            g_slider_edit_just_opened = true;
            snprintf(g_slider_edit_buf, sizeof(g_slider_edit_buf), "%d", *val);
        }
        bool v_hov = ImGui::IsItemHovered();
        dl->AddText(click_pos, v_hov ? IM_COL32(40, 229, 180, 255) : IM_COL32(25, 211, 162, 255), val_buf);
        if (v_hov) {
            dl->AddLine(
                ImVec2(click_pos.x, click_pos.y + ImGui::GetTextLineHeight() - 1.0f),
                ImVec2(click_pos.x + val_w, click_pos.y + ImGui::GetTextLineHeight() - 1.0f),
                IM_COL32(40, 229, 180, 180),
                1.0f
            );
        }
    } else {
        ImGui::SetNextItemWidth(val_col_w);
        if (g_slider_edit_just_opened) {
            ImGui::SetKeyboardFocusHere();
            g_slider_edit_just_opened = false;
        }
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3.0f, 1.0f));
        bool enter_pressed = ImGui::InputText("##c_in_i", g_slider_edit_buf, sizeof(g_slider_edit_buf), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
        if (enter_pressed) {
            char* end_ptr = nullptr;
            long parsed = strtol(g_slider_edit_buf, &end_ptr, 10);
            if (end_ptr != g_slider_edit_buf) {
                int clamped = std::clamp(static_cast<int>(parsed), v_min, v_max);
                if (*val != clamped) { *val = clamped; value_changed = true; }
            }
            g_active_slider_edit_id = 0;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Escape) || (ImGui::IsItemDeactivated() && !enter_pressed)) {
            g_active_slider_edit_id = 0;
        }
        ImGui::PopStyleVar();
    }
    ImGui::PopStyleVar();

    if (value_changed) {
        std::string msg = std::string(label) + " set to " + std::to_string(*val);
        RhizomeLog("ACTION", msg.c_str());
    }

    ImGui::PopID();
    return value_changed;
}



} // namespace rhizome
