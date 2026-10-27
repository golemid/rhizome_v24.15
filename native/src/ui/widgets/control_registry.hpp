#pragma once
// =========================================================================
// RHIZOME CONTROL REGISTRY — ui/widgets module (v24.01 Phase 2)
// =========================================================================
// Data-driven declaration of the 241 UI controls (AGENTS.md Sec A.1 /
// contract Sec 1.3). The single authority is
// param_controls_complete_list.csv: every row CTRL001..CTRL241 appears here
// exactly once, in CSV order, with catalog fields (id, section, label,
// category, unit, logging flag) transcribed 1:1 from the CSV.
//
// v24.01 Phase 0 (Option A renumber): the shading trio (Matte Glass
// Specimen / Glass Frosting / Opacity) moved from "Layers Visibility" to the
// new "Specimen Appearance" section (now CTRL104-106). v24.01 Phase 1
// (lighting expansion): ten lighting sliders inserted at CTRL093-102
// ("3D Lighting & Shading" 6 -> 16 sliders); ex-CTRL093..138 slid +10 to
// CTRL103-148. v24.01 Phase 2 (style-editor registration, append-only):
// the 93 Dear ImGui style-editor widgets joined as CTRL149-241 in three
// terminal sections — "Style Editor: Sizes" (23), "Style Editor: Colors"
// (63 == ImGuiCol_COUNT at the reference census, ImGui 1.92.8-docking),
// "Style Editor: UI Rendering" (7). 25 CSV sections; the catalog count
// is 241.
//
// Design (approved Phase 1 proposal Sec 4):
//   * VALUE controls (sliders / toggles / color pickers / text input —
//     104 rows) carry a typed thunk binding into AppState. Thunks are
//     captureless lambdas decayed to plain function pointers: zero heap,
//     zero std::function, resolved against the AppState singleton each
//     frame.
//   * ACTION controls (buttons / combos / selectables — 44 rows) keep a
//     declarative row for catalog completeness, uniform logging and
//     benchmark enumeration; their handlers live in the owning panel .cpp
//     exactly as in v23.28 (layout-sensitive SameLine chains, width
//     calculations, tooltips and accent colors are orchestrated there).
//
// `draw_label` preserves the exact ImGui label string drawn by v23.28
// (e.g. CTRL009 draws "Roughness" while the CSV catalog label is
// "Root Roughness"), so the rendered UI stays pixel-identical while the
// catalog identity matches the CSV / g_all_controls monitor resolution
// byte-for-byte. Where the CSV label and the drawn label coincide,
// draw_label simply repeats the catalog label.
//
// Automatic logging (proposal Sec 4.3/4.4 — "one code path"): when a value
// control changes and def.logs is true, render_control() emits
// log_ui_interaction(def.id, old, new, def.unit) — exact CTRL-ID
// resolution into the JSON-Lines monitor (logs/ui_events.jsonl), field
// schema unchanged from v23.28.
// =========================================================================

#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>

#include "core/app_state.hpp"
#include "ui/widgets/widget_factory.hpp"

namespace rhizome {

// ---- Control kinds (one per CSV Control_Function family) ----------------
enum class ControlKind : uint8_t {
    SliderFloat,   // render_compact_slider_float   (49 rows)
    SliderInt,     // render_compact_slider_int     ( 7 rows)
    Toggle,        // render_toggle_switch          (22 rows)
    Button,        // ImGui::Button                 (34 rows)
    PillButton,    // render_pill_button            ( 4 rows: CTRL001-004)
    Combo,         // begin_pill_combo / render_pill_combo (5 rows)
    ColorEdit4,    // ImGui::ColorEdit4             (15 rows)
    InputText,     // ImGui::InputText              ( 1 row: CTRL057)
    Selectable     // ImGui::Selectable             ( 1 row: CTRL101)
};

// ---- The declarative control row -----------------------------------------
struct ControlDefinition {
    // ---- catalog identity (param_controls_complete_list.csv, 1:1) ----
    const char* id;         // "CTRL009"
    const char* section;    // "Algorithm"
    const char* label;      // CSV catalog label: "Root Roughness"
    const char* category;   // "GROWTH_01"
    ControlKind kind;
    bool        logs;       // CSV Logging_Triggered == YES

    // ---- presentation (v23.28-identical) ----
    const char* draw_label; // ImGui label as drawn by v23.28 ("Roughness")
    const char* format;     // printf format shown in the value column

    // ---- ranges (exact CSV Min_Value / Max_Value) ----
    float f_min = 0.0f;
    float f_max = 1.0f;
    int   i_min = 0;
    int   i_max = 1;

    // ---- CSV Unit column ----
    const char* unit = "";

    // ---- value binding: captureless-lambda thunks -> plain fn pointers ----
    // Exactly one resolver is non-null for value controls; all are null for
    // action controls (handlers are registered by the owning panel).
    float*     (*resolve_float)(AppState&) = nullptr;
    int*       (*resolve_int  )(AppState&) = nullptr;
    bool*      (*resolve_bool )(AppState&) = nullptr;
    glm::vec4* (*resolve_color)(AppState&) = nullptr;
    char*      (*resolve_text )(AppState&) = nullptr;
};

// ---- Binding sugar (macro-free, C++20 captureless lambdas) ---------------
// Each AppState sub-state is a distinct member with a distinct type, so
// member access stays type-safe through tiny per-domain wrappers. The first
// argument is the AppState SUB-STATE MEMBER name (growth, seed, display,
// mesh_export, bark_texture, verification, ui), not the type name:
//   RHZ_BIND_F(growth, root_roughness)  ->  float* (*)(AppState&)
#define RHZ_BIND_F(sub, member) \
    +[](AppState& s) -> float*     { return &s.sub.member; }
#define RHZ_BIND_I(sub, member) \
    +[](AppState& s) -> int*       { return &s.sub.member; }
#define RHZ_BIND_B(sub, member) \
    +[](AppState& s) -> bool*      { return &s.sub.member; }
#define RHZ_BIND_C(sub, member) \
    +[](AppState& s) -> glm::vec4* { return &s.sub.member; }
#define RHZ_BIND_T(sub, member) \
    +[](AppState& s) -> char*      { return s.sub.member; }

// ImGui-global bindings (documented shims): CTRL068 "Font Scale" and
// CTRL069 "Item Spacing Y" bound ImGui::GetIO().FontGlobalScale and
// ImGui::GetStyle().ItemSpacing.y directly in v23.28 — live ImGui state,
// not AppState members. The thunks below preserve that exact behavior
// (they are only ever invoked inside an ImGui frame).
#define RHZ_BIND_IMGUI_IO_F(member) \
    +[](AppState&) -> float* { return &ImGui::GetIO().member; }
#define RHZ_BIND_IMGUI_STYLE_F(member) \
    +[](AppState&) -> float* { return &ImGui::GetStyle().member; }

// v24.01 Phase 2 (proposal 2.1): style-editor bindings. Same thunk shape
// as the CTRL068/069 shims above — resolved against the live ImGui context
// style, never against AppState (whose bindings are untouched). The ImVec2
// component form (F2) addresses style members like WindowPadding.x; the
// color form (COL) reinterprets style.Colors[index] (ImVec4, standard-
// layout 4xfloat) as glm::vec4 — every downstream access goes through a
// float* exactly like the v23.28 `(float*)&style.Colors[i]` cast it
// replaces. Thunks are only invoked inside an ImGui frame, with ONE
// documented exception: the startup loader (load_imgui_style_from_ini,
// main.cpp, after ImGui::CreateContext).
#define RHZ_BIND_IMGUI_STYLE_B(member) \
    +[](AppState&) -> bool* { return &ImGui::GetStyle().member; }
#define RHZ_BIND_IMGUI_STYLE_F2(member, axis) \
    +[](AppState&) -> float* { return &ImGui::GetStyle().member.axis; }
#define RHZ_BIND_IMGUI_STYLE_COL(index) \
    +[](AppState&) -> glm::vec4* { \
        return reinterpret_cast<glm::vec4*>(&ImGui::GetStyle().Colors[index]); }

// ---- The registry: 251 rows in CSV order (CTRL001..CTRL251) --------------
// 148 domain rows (v23.29 catalog after the v24.01 Phase 0/1 renumbers) +
// 93 style-editor rows appended by Phase 2 + 10 developmental rows appended
// by v24.12 Track F R1 (terminal section "Developmental (Mode 4)"). The
// array length is tied to the CSV section anchors below. The Colors rows
// are INDEX-bound (Colors[0..62],
// no ImGuiCol_ enumerator names) and version-adaptive (8c): a build against
// an older Dear ImGui with a smaller ImGuiCol_COUNT simply leaves the tail
// rows dormant - every runtime enumeration is bounded by the built count.
extern const ControlDefinition k_all_controls[251];

// ---- v24.01 Phase 2: style-editor section anchors (append-only end) ------
// The three style-editor sections tile the catalog tail. k_style_colors_count
// is the CSV-SIDE census, fixed at the reference Dear ImGui build the catalog
// was generated against (63 colors at ImGui 1.92.8-docking). 8c version-
// adaptive semantics: the census COVERS every build with ImGuiCol_COUNT <=
// 63 - the live colors of a build are the first ImGuiCol_COUNT rows (labels
// resolve at runtime through GetStyleColorName), and tail rows beyond the
// built count are dormant: carried for catalog completeness, never rendered,
// resolved or logged. A build with ImGuiCol_COUNT > 63 (newer ImGui than the
// census) is stopped by the coverage gate below. Census refresh:
// V2401_PHASE2_REPORT.md Sec 8 + scripts/v2401_phase2_repin_colors.py.
inline constexpr size_t k_style_sizes_first = 148;    // CTRL149..CTRL171
inline constexpr size_t k_style_sizes_count = 23;
inline constexpr size_t k_style_colors_first = 171;   // CTRL172..CTRL234
inline constexpr size_t k_style_colors_count = 63;    // CSV census at the reference baseline (ImGui 1.92.8)
inline constexpr size_t k_style_render_first = 234;   // CTRL235..CTRL241 (0-based index of CTRL235)
inline constexpr size_t k_style_render_count = 7;
static_assert(k_style_sizes_first + k_style_sizes_count == k_style_colors_first,
              "style-editor sections must tile the catalog tail contiguously");
static_assert(k_style_colors_first + k_style_colors_count == k_style_render_first,
              "Style Editor: Colors section must span exactly k_style_colors_count rows");
static_assert(k_style_render_first + k_style_render_count == 241,
              "style-editor tail must end the 241-row Phase 2 catalog");

// v24.12 Track F R1: developmental section (append-only terminal; the ten
// dev_* user-surface sliders, panel_growth.cpp ALGORITHM pill).
inline constexpr size_t k_dev_first = 241;    // CTRL242..CTRL251
inline constexpr size_t k_dev_count = 10;
static_assert(k_style_render_first + k_style_render_count == k_dev_first,
              "developmental section must tile the catalog tail contiguously");
static_assert(k_dev_first + k_dev_count == 251,
              "developmental tail must end the 251-row catalog");

// v24.01 Phase 2 8c: ImGuiCol coverage gate (version-adaptive). The census
// must COVER this build: ImGuiCol_COUNT may be SMALLER (older Dear ImGui -
// tail color rows go dormant; validated at runtime by the --verify Test 20
// [Style Editor Registry] gate), never LARGER (newer Dear ImGui - the live
// COLORS loop and the commit-log lookup would index past the Colors section
// into Style Editor: UI Rendering). When the count exceeds the census the
// gate trips and the ImguiColCoverageCheck false-branch below makes the
// compiler PRINT the actual count as a template argument (MSVC: "use of
// undefined type 'rhizome::ImguiColCountMismatch<NN>'"; GCC: "invalid
// application of 'sizeof' to incomplete type ... <NN>'"). Re-pin per
// V2401_PHASE2_REPORT.md Sec 8 (scripts/v2401_phase2_repin_colors.py).
// While the count fits, only the silent empty true-branch is ever
// instantiated, so a covered build compiles exactly as before.
template <int N> struct ImguiColCountMismatch;   // never defined (printer)

template <size_t Cover, int Actual,
          bool Fits = (static_cast<size_t>(Actual) <= Cover)>
struct ImguiColCoverageCheck;                    // primary: never defined

template <size_t Cover, int Actual>              // fits: silent, empty
struct ImguiColCoverageCheck<Cover, Actual, true> {};

template <size_t Cover, int Actual>              // over census: print NN
struct ImguiColCoverageCheck<Cover, Actual, false> {
    static_assert(sizeof(ImguiColCountMismatch<Actual>) > 0,
                  "Style Editor: this ImGui build has MORE style colors than "
                  "the 241-row catalog covers - the actual count is printed "
                  "in the template argument of the error above; re-pin per "
                  "V2401_PHASE2_REPORT.md Sec 8");
};

static_assert(sizeof(ImguiColCoverageCheck<k_style_colors_count, ImGuiCol_COUNT>) > 0,
              "Style Editor: Colors census (63 rows at the reference "
              "baseline, ImGui 1.92.8) must cover this build's "
              "ImGuiCol_COUNT - on overflow the actual count is printed in "
              "the ImguiColCountMismatch error above");

// Supported-window floor: the style-editor surface (Sizes members, AA
// toggles, tessellation tolerances) expects Dear ImGui 1.90.x or newer.
static_assert(ImGuiCol_COUNT >= 50,
              "Style Editor: ImGuiCol_COUNT below 50 - Dear ImGui older than "
              "the supported window (the style-editor bindings expect "
              "1.90.x-era ImGuiStyle members)");

// Named per-row references for panel call sites:
//   render_control(CTRL028, app_state)   // "Trunk Radius"
namespace ctrldefs {
inline const ControlDefinition& CTRL001 = k_all_controls[0];
inline const ControlDefinition& CTRL002 = k_all_controls[1];
inline const ControlDefinition& CTRL003 = k_all_controls[2];
inline const ControlDefinition& CTRL004 = k_all_controls[3];
inline const ControlDefinition& CTRL005 = k_all_controls[4];
inline const ControlDefinition& CTRL006 = k_all_controls[5];
inline const ControlDefinition& CTRL007 = k_all_controls[6];
inline const ControlDefinition& CTRL008 = k_all_controls[7];
inline const ControlDefinition& CTRL009 = k_all_controls[8];
inline const ControlDefinition& CTRL010 = k_all_controls[9];
inline const ControlDefinition& CTRL011 = k_all_controls[10];
inline const ControlDefinition& CTRL012 = k_all_controls[11];
inline const ControlDefinition& CTRL013 = k_all_controls[12];
inline const ControlDefinition& CTRL014 = k_all_controls[13];
inline const ControlDefinition& CTRL015 = k_all_controls[14];
inline const ControlDefinition& CTRL016 = k_all_controls[15];
inline const ControlDefinition& CTRL017 = k_all_controls[16];
inline const ControlDefinition& CTRL018 = k_all_controls[17];
inline const ControlDefinition& CTRL019 = k_all_controls[18];
inline const ControlDefinition& CTRL020 = k_all_controls[19];
inline const ControlDefinition& CTRL021 = k_all_controls[20];
inline const ControlDefinition& CTRL022 = k_all_controls[21];
inline const ControlDefinition& CTRL023 = k_all_controls[22];
inline const ControlDefinition& CTRL024 = k_all_controls[23];
inline const ControlDefinition& CTRL025 = k_all_controls[24];
inline const ControlDefinition& CTRL026 = k_all_controls[25];
inline const ControlDefinition& CTRL027 = k_all_controls[26];
inline const ControlDefinition& CTRL028 = k_all_controls[27];
inline const ControlDefinition& CTRL029 = k_all_controls[28];
inline const ControlDefinition& CTRL030 = k_all_controls[29];
inline const ControlDefinition& CTRL031 = k_all_controls[30];
inline const ControlDefinition& CTRL032 = k_all_controls[31];
inline const ControlDefinition& CTRL033 = k_all_controls[32];
inline const ControlDefinition& CTRL034 = k_all_controls[33];
inline const ControlDefinition& CTRL035 = k_all_controls[34];
inline const ControlDefinition& CTRL036 = k_all_controls[35];
inline const ControlDefinition& CTRL037 = k_all_controls[36];
inline const ControlDefinition& CTRL038 = k_all_controls[37];
inline const ControlDefinition& CTRL039 = k_all_controls[38];
inline const ControlDefinition& CTRL040 = k_all_controls[39];
inline const ControlDefinition& CTRL041 = k_all_controls[40];
inline const ControlDefinition& CTRL042 = k_all_controls[41];
inline const ControlDefinition& CTRL043 = k_all_controls[42];
inline const ControlDefinition& CTRL044 = k_all_controls[43];
inline const ControlDefinition& CTRL045 = k_all_controls[44];
inline const ControlDefinition& CTRL046 = k_all_controls[45];
inline const ControlDefinition& CTRL047 = k_all_controls[46];
inline const ControlDefinition& CTRL048 = k_all_controls[47];
inline const ControlDefinition& CTRL049 = k_all_controls[48];
inline const ControlDefinition& CTRL050 = k_all_controls[49];
inline const ControlDefinition& CTRL051 = k_all_controls[50];
inline const ControlDefinition& CTRL052 = k_all_controls[51];
inline const ControlDefinition& CTRL053 = k_all_controls[52];
inline const ControlDefinition& CTRL054 = k_all_controls[53];
inline const ControlDefinition& CTRL055 = k_all_controls[54];
inline const ControlDefinition& CTRL056 = k_all_controls[55];
inline const ControlDefinition& CTRL057 = k_all_controls[56];
inline const ControlDefinition& CTRL058 = k_all_controls[57];
inline const ControlDefinition& CTRL059 = k_all_controls[58];
inline const ControlDefinition& CTRL060 = k_all_controls[59];
inline const ControlDefinition& CTRL061 = k_all_controls[60];
inline const ControlDefinition& CTRL062 = k_all_controls[61];
inline const ControlDefinition& CTRL063 = k_all_controls[62];
inline const ControlDefinition& CTRL064 = k_all_controls[63];
inline const ControlDefinition& CTRL065 = k_all_controls[64];
inline const ControlDefinition& CTRL066 = k_all_controls[65];
inline const ControlDefinition& CTRL067 = k_all_controls[66];
inline const ControlDefinition& CTRL068 = k_all_controls[67];
inline const ControlDefinition& CTRL069 = k_all_controls[68];
inline const ControlDefinition& CTRL070 = k_all_controls[69];
inline const ControlDefinition& CTRL071 = k_all_controls[70];
inline const ControlDefinition& CTRL072 = k_all_controls[71];
inline const ControlDefinition& CTRL073 = k_all_controls[72];
inline const ControlDefinition& CTRL074 = k_all_controls[73];
inline const ControlDefinition& CTRL075 = k_all_controls[74];
inline const ControlDefinition& CTRL076 = k_all_controls[75];
inline const ControlDefinition& CTRL077 = k_all_controls[76];
inline const ControlDefinition& CTRL078 = k_all_controls[77];
inline const ControlDefinition& CTRL079 = k_all_controls[78];
inline const ControlDefinition& CTRL080 = k_all_controls[79];
inline const ControlDefinition& CTRL081 = k_all_controls[80];
inline const ControlDefinition& CTRL082 = k_all_controls[81];
inline const ControlDefinition& CTRL083 = k_all_controls[82];
inline const ControlDefinition& CTRL084 = k_all_controls[83];
inline const ControlDefinition& CTRL085 = k_all_controls[84];
inline const ControlDefinition& CTRL086 = k_all_controls[85];
inline const ControlDefinition& CTRL087 = k_all_controls[86];
inline const ControlDefinition& CTRL088 = k_all_controls[87];
inline const ControlDefinition& CTRL089 = k_all_controls[88];
inline const ControlDefinition& CTRL090 = k_all_controls[89];
inline const ControlDefinition& CTRL091 = k_all_controls[90];
inline const ControlDefinition& CTRL092 = k_all_controls[91];
inline const ControlDefinition& CTRL093 = k_all_controls[92];
inline const ControlDefinition& CTRL094 = k_all_controls[93];
inline const ControlDefinition& CTRL095 = k_all_controls[94];
inline const ControlDefinition& CTRL096 = k_all_controls[95];
inline const ControlDefinition& CTRL097 = k_all_controls[96];
inline const ControlDefinition& CTRL098 = k_all_controls[97];
inline const ControlDefinition& CTRL099 = k_all_controls[98];
inline const ControlDefinition& CTRL100 = k_all_controls[99];
inline const ControlDefinition& CTRL101 = k_all_controls[100];
inline const ControlDefinition& CTRL102 = k_all_controls[101];
inline const ControlDefinition& CTRL103 = k_all_controls[102];
inline const ControlDefinition& CTRL104 = k_all_controls[103];
inline const ControlDefinition& CTRL105 = k_all_controls[104];
inline const ControlDefinition& CTRL106 = k_all_controls[105];
inline const ControlDefinition& CTRL107 = k_all_controls[106];
inline const ControlDefinition& CTRL108 = k_all_controls[107];
inline const ControlDefinition& CTRL109 = k_all_controls[108];
inline const ControlDefinition& CTRL110 = k_all_controls[109];
inline const ControlDefinition& CTRL111 = k_all_controls[110];
inline const ControlDefinition& CTRL112 = k_all_controls[111];
inline const ControlDefinition& CTRL113 = k_all_controls[112];
inline const ControlDefinition& CTRL114 = k_all_controls[113];
inline const ControlDefinition& CTRL115 = k_all_controls[114];
inline const ControlDefinition& CTRL116 = k_all_controls[115];
inline const ControlDefinition& CTRL117 = k_all_controls[116];
inline const ControlDefinition& CTRL118 = k_all_controls[117];
inline const ControlDefinition& CTRL119 = k_all_controls[118];
inline const ControlDefinition& CTRL120 = k_all_controls[119];
inline const ControlDefinition& CTRL121 = k_all_controls[120];
inline const ControlDefinition& CTRL122 = k_all_controls[121];
inline const ControlDefinition& CTRL123 = k_all_controls[122];
inline const ControlDefinition& CTRL124 = k_all_controls[123];
inline const ControlDefinition& CTRL125 = k_all_controls[124];
inline const ControlDefinition& CTRL126 = k_all_controls[125];
inline const ControlDefinition& CTRL127 = k_all_controls[126];
inline const ControlDefinition& CTRL128 = k_all_controls[127];
inline const ControlDefinition& CTRL129 = k_all_controls[128];
inline const ControlDefinition& CTRL130 = k_all_controls[129];
inline const ControlDefinition& CTRL131 = k_all_controls[130];
inline const ControlDefinition& CTRL132 = k_all_controls[131];
inline const ControlDefinition& CTRL133 = k_all_controls[132];
inline const ControlDefinition& CTRL134 = k_all_controls[133];
inline const ControlDefinition& CTRL135 = k_all_controls[134];
inline const ControlDefinition& CTRL136 = k_all_controls[135];
inline const ControlDefinition& CTRL137 = k_all_controls[136];
inline const ControlDefinition& CTRL138 = k_all_controls[137];
inline const ControlDefinition& CTRL139 = k_all_controls[138];
inline const ControlDefinition& CTRL140 = k_all_controls[139];
inline const ControlDefinition& CTRL141 = k_all_controls[140];
inline const ControlDefinition& CTRL142 = k_all_controls[141];
inline const ControlDefinition& CTRL143 = k_all_controls[142];
inline const ControlDefinition& CTRL144 = k_all_controls[143];
inline const ControlDefinition& CTRL145 = k_all_controls[144];
inline const ControlDefinition& CTRL146 = k_all_controls[145];
inline const ControlDefinition& CTRL147 = k_all_controls[146];
inline const ControlDefinition& CTRL148 = k_all_controls[147];
// v24.01 Phase 2: style-editor rows (append-only tail, CTRL149-241)
inline const ControlDefinition& CTRL149 = k_all_controls[148];
inline const ControlDefinition& CTRL150 = k_all_controls[149];
inline const ControlDefinition& CTRL151 = k_all_controls[150];
inline const ControlDefinition& CTRL152 = k_all_controls[151];
inline const ControlDefinition& CTRL153 = k_all_controls[152];
inline const ControlDefinition& CTRL154 = k_all_controls[153];
inline const ControlDefinition& CTRL155 = k_all_controls[154];
inline const ControlDefinition& CTRL156 = k_all_controls[155];
inline const ControlDefinition& CTRL157 = k_all_controls[156];
inline const ControlDefinition& CTRL158 = k_all_controls[157];
inline const ControlDefinition& CTRL159 = k_all_controls[158];
inline const ControlDefinition& CTRL160 = k_all_controls[159];
inline const ControlDefinition& CTRL161 = k_all_controls[160];
inline const ControlDefinition& CTRL162 = k_all_controls[161];
inline const ControlDefinition& CTRL163 = k_all_controls[162];
inline const ControlDefinition& CTRL164 = k_all_controls[163];
inline const ControlDefinition& CTRL165 = k_all_controls[164];
inline const ControlDefinition& CTRL166 = k_all_controls[165];
inline const ControlDefinition& CTRL167 = k_all_controls[166];
inline const ControlDefinition& CTRL168 = k_all_controls[167];
inline const ControlDefinition& CTRL169 = k_all_controls[168];
inline const ControlDefinition& CTRL170 = k_all_controls[169];
inline const ControlDefinition& CTRL171 = k_all_controls[170];
inline const ControlDefinition& CTRL172 = k_all_controls[171];
inline const ControlDefinition& CTRL173 = k_all_controls[172];
inline const ControlDefinition& CTRL174 = k_all_controls[173];
inline const ControlDefinition& CTRL175 = k_all_controls[174];
inline const ControlDefinition& CTRL176 = k_all_controls[175];
inline const ControlDefinition& CTRL177 = k_all_controls[176];
inline const ControlDefinition& CTRL178 = k_all_controls[177];
inline const ControlDefinition& CTRL179 = k_all_controls[178];
inline const ControlDefinition& CTRL180 = k_all_controls[179];
inline const ControlDefinition& CTRL181 = k_all_controls[180];
inline const ControlDefinition& CTRL182 = k_all_controls[181];
inline const ControlDefinition& CTRL183 = k_all_controls[182];
inline const ControlDefinition& CTRL184 = k_all_controls[183];
inline const ControlDefinition& CTRL185 = k_all_controls[184];
inline const ControlDefinition& CTRL186 = k_all_controls[185];
inline const ControlDefinition& CTRL187 = k_all_controls[186];
inline const ControlDefinition& CTRL188 = k_all_controls[187];
inline const ControlDefinition& CTRL189 = k_all_controls[188];
inline const ControlDefinition& CTRL190 = k_all_controls[189];
inline const ControlDefinition& CTRL191 = k_all_controls[190];
inline const ControlDefinition& CTRL192 = k_all_controls[191];
inline const ControlDefinition& CTRL193 = k_all_controls[192];
inline const ControlDefinition& CTRL194 = k_all_controls[193];
inline const ControlDefinition& CTRL195 = k_all_controls[194];
inline const ControlDefinition& CTRL196 = k_all_controls[195];
inline const ControlDefinition& CTRL197 = k_all_controls[196];
inline const ControlDefinition& CTRL198 = k_all_controls[197];
inline const ControlDefinition& CTRL199 = k_all_controls[198];
inline const ControlDefinition& CTRL200 = k_all_controls[199];
inline const ControlDefinition& CTRL201 = k_all_controls[200];
inline const ControlDefinition& CTRL202 = k_all_controls[201];
inline const ControlDefinition& CTRL203 = k_all_controls[202];
inline const ControlDefinition& CTRL204 = k_all_controls[203];
inline const ControlDefinition& CTRL205 = k_all_controls[204];
inline const ControlDefinition& CTRL206 = k_all_controls[205];
inline const ControlDefinition& CTRL207 = k_all_controls[206];
inline const ControlDefinition& CTRL208 = k_all_controls[207];
inline const ControlDefinition& CTRL209 = k_all_controls[208];
inline const ControlDefinition& CTRL210 = k_all_controls[209];
inline const ControlDefinition& CTRL211 = k_all_controls[210];
inline const ControlDefinition& CTRL212 = k_all_controls[211];
inline const ControlDefinition& CTRL213 = k_all_controls[212];
inline const ControlDefinition& CTRL214 = k_all_controls[213];
inline const ControlDefinition& CTRL215 = k_all_controls[214];
inline const ControlDefinition& CTRL216 = k_all_controls[215];
inline const ControlDefinition& CTRL217 = k_all_controls[216];
inline const ControlDefinition& CTRL218 = k_all_controls[217];
inline const ControlDefinition& CTRL219 = k_all_controls[218];
inline const ControlDefinition& CTRL220 = k_all_controls[219];
inline const ControlDefinition& CTRL221 = k_all_controls[220];
inline const ControlDefinition& CTRL222 = k_all_controls[221];
inline const ControlDefinition& CTRL223 = k_all_controls[222];
inline const ControlDefinition& CTRL224 = k_all_controls[223];
inline const ControlDefinition& CTRL225 = k_all_controls[224];
inline const ControlDefinition& CTRL226 = k_all_controls[225];
inline const ControlDefinition& CTRL227 = k_all_controls[226];
inline const ControlDefinition& CTRL228 = k_all_controls[227];
inline const ControlDefinition& CTRL229 = k_all_controls[228];
inline const ControlDefinition& CTRL230 = k_all_controls[229];
inline const ControlDefinition& CTRL231 = k_all_controls[230];
inline const ControlDefinition& CTRL232 = k_all_controls[231];
inline const ControlDefinition& CTRL233 = k_all_controls[232];
inline const ControlDefinition& CTRL234 = k_all_controls[233];
inline const ControlDefinition& CTRL235 = k_all_controls[234];
inline const ControlDefinition& CTRL236 = k_all_controls[235];
inline const ControlDefinition& CTRL237 = k_all_controls[236];
inline const ControlDefinition& CTRL238 = k_all_controls[237];
inline const ControlDefinition& CTRL239 = k_all_controls[238];
inline const ControlDefinition& CTRL240 = k_all_controls[239];
inline const ControlDefinition& CTRL241 = k_all_controls[240];
inline const ControlDefinition& CTRL242 = k_all_controls[241];
inline const ControlDefinition& CTRL243 = k_all_controls[242];
inline const ControlDefinition& CTRL244 = k_all_controls[243];
inline const ControlDefinition& CTRL245 = k_all_controls[244];
inline const ControlDefinition& CTRL246 = k_all_controls[245];
inline const ControlDefinition& CTRL247 = k_all_controls[246];
inline const ControlDefinition& CTRL248 = k_all_controls[247];
inline const ControlDefinition& CTRL249 = k_all_controls[248];
inline const ControlDefinition& CTRL250 = k_all_controls[249];
inline const ControlDefinition& CTRL251 = k_all_controls[250];
} // namespace ctrldefs

// ---- CSV section views (contiguous ranges inside k_all_controls) --------
struct ControlSectionView {
    const char*           section; // CSV Panel_Section name
    const ControlDefinition* defs;
    size_t                count;
};

// 26 CSV sections, in CSV order (22 domain sections + the 3 style-editor
// sections appended by v24.01 Phase 2 + the developmental section appended
// by v24.12 Track F R1).
extern const ControlSectionView k_all_sections[26];

// ---- Registry services ----------------------------------------------------

// Find a row by CTRL id ("CTRL028"); nullptr when not found.
const ControlDefinition* find_control(const char* ctrl_id);

// Registry integrity check (Phase 5 audit hook): verifies 251 rows,
// contiguous CTRL001..CTRL251 ids, CSV-order sections and that every value
// row carries exactly one non-null resolver. Returns true when healthy.
bool control_registry_integrity_check();

// ---- Generic value-control renderer ---------------------------------------
//
// Draws the widget with the exact v23.28 primitive, draw_label, static CSV
// range and format, then auto-logs through the JSON-Lines monitor when the
// value changed and def.logs is set. Returns true when the value changed
// (the owning panel then runs its side effects).
//
// f_min_override / f_max_override (optional): a handful of v23.28 sliders
// compute their range at runtime (e.g. Min/Max Growth Step derive their
// floor from trunk_caliber). Pass pointers to the live values; pass nullptr
// to use the static CSV range.
bool render_control(const ControlDefinition& def, AppState& state,
                    const float* f_min_override = nullptr,
                    const float* f_max_override = nullptr);

// Log a value interaction for a control that could not bind a plain
// resolver (e.g. the per-selected-seed toggles CTRL047-049 write indexed
// seed vectors). Same JSON-Lines emission path as render_control.
void log_control_interaction(const ControlDefinition& def,
                             float old_val, float new_val);

} // namespace rhizome
