# Rhizome UI Parameter Controls Complete Inventory
## 100% Comprehensive Codebase Audit & Specification Report

> **Program closure (2026-10-22):** the v24.01 catalog program is COMPLETE and user-signed-off at **241 controls / 25 sections** (`CTRL001`–`CTRL241`). *(v24.12 Track F R1, append-only: the ten developmental sliders joined as `CTRL242`–`CTRL251` in one new terminal section "Developmental (Mode 4)" — catalog now **251 controls / 26 sections**; the summary table below reflects the live catalog.)* Final target-hardware `--verify`: 19/20 + the documented Test 8 known-issue; `[Style Editor Registry]` census 63 == live 63 on Dear ImGui 1.92.9b (exact-fit). This inventory is the frozen as-built record of the closed program; future modifications should begin from `IMPROVEMENT_PLAN.md`.

---

## 📊 SUMMARY COUNTS

| Control Type | Rendering Function / API | Instance Count | Source Files (v24.01) |
|:---|:---|:---:|:---|
| **Compact Sliders (Float)** | `render_compact_slider_float` | **95** | `panel_growth.cpp`, `panel_styles.cpp`, `panel_scene.cpp` |
| **Compact Sliders (Int)** | `render_compact_slider_int` | **8** | `panel_growth.cpp`, `panel_scene.cpp` |
| **Toggle Switches** | `render_toggle_switch` | **25** | `panel_growth.cpp`, `panel_styles.cpp`, `panel_scene.cpp` |
| **Action Buttons** | `ImGui::Button` | **34** | all four panel `.cpp` files |
| **Pill Buttons** | `render_pill_button` | **4** | `panel_growth.cpp` |
| **Dropdown Combo Boxes** | `begin_pill_combo`, `render_pill_combo` | **5** | `panel_growth.cpp`, `panel_styles.cpp`, `panel_scene.cpp` |
| **Color Pickers** | `ImGui::ColorEdit4` | **73** | `panel_styles.cpp` |
| **Text Inputs** | `ImGui::InputText` | **1** | `panel_styles.cpp` |
| **Selectable Items** | `ImGui::Selectable` | **1** | `panel_verification.cpp` |
| **TOTAL INTERACTIVE CONTROLS** | | **251** | Complete Codebase Coverage |

*Note: All 251 controls are sequentially indexed from `CTRL001` through `CTRL251` in `param_controls_complete_list.csv`.*

*v23.29 correction (F5): the v23.28 type table summed to 125 (38 float / 8 int sliders, 34 buttons, 6 combos) and under-counted against the CSV. The counts above are the machine-audited CSV census, enforced at build time by the `rhizome_control_audit` CMake target and at runtime by `--verify`. The CSV defines 22 panel sections; the functional groupings in the detailed inventory below merge them into 19 subsections for readability. *(v24.01 Phase 0: the shading trio — Matte Glass Specimen / Glass Frosting / Opacity — moved from "Layers Visibility" to the new "Specimen Appearance" section (CTRL104-106, `panel_styles.cpp`), and every row's Source_File now names its real panel home and is verified by the build-time audit.)*
*v24.01 Phase 1 (lighting expansion): ten promoted lighting constants joined as `CTRL093-102` (3D Lighting & Shading 6 -> 16 sliders); ex-`CTRL093..138` slid +10 to `CTRL103-148`; the catalog became 148 controls / 22 sections. v24.01 Phase 2 (style-editor registration, append-only): the interface style-editor widgets joined as `CTRL149-241` in three terminal sections — Style Editor: Sizes (23), Style Editor: Colors (63 == ImGuiCol_COUNT at the 8d census, ImGui 1.92.8+; 8c version-adaptive — covered builds with fewer colors leave tail rows dormant; 8d re-pinned from the 58-color 1.91.3 census when the target machine reported 63), Style Editor: UI Rendering (7); the catalog is now **241 controls / 25 sections**.*
*v24.12 Track F R1 (developmental growth mode 4, append-only): the ten developmental user-surface sliders joined as `CTRL242`–`CTRL251` in one new terminal section "Developmental (Mode 4)" (`panel_growth.cpp`, ALGORITHM pill, `algorithm_mode == 4` branch; bindings resolve into the ten `GrowthState::dev_*` members — app_state.hpp); SliderFloat 86 -> 95, SliderInt 7 -> 8; the catalog is now **251 controls / 26 sections**.*

---

## 📁 SOURCE FILE BREAKDOWN (v23.29 panel files)

### 1. `native/src/ui/panels/panel_growth.cpp` (62 Controls Total — 52 + CTRL242-251, v24.12 Track F R1)
Contains the growth command bar, algorithm selector, colonization seeds, branching / vessel / lateral-root physics, seeding management, and growth execution controls.

| CSV Section | Control ID Range | Sliders | Toggles | Buttons | Combos / Pickers / Text | Section Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Growth Command Bar | `CTRL001` – `CTRL004` | 0 | 0 | 4 | 0 | **4** |
| Algorithm | `CTRL005` – `CTRL016` | 8 | 0 | 3 | 1 | **12** |
| Colonization Seeds | `CTRL017` – `CTRL020` | 4 | 0 | 0 | 0 | **4** |
| Branching & Growth | `CTRL021` – `CTRL027` | 6 | 1 | 0 | 0 | **7** |
| Vessel & Caliber | `CTRL028` – `CTRL034` | 6 | 1 | 0 | 0 | **7** |
| Lateral Roots | `CTRL035` – `CTRL040` | 6 | 0 | 0 | 0 | **6** |
| Seeding | `CTRL041` – `CTRL049` | 1 | 3 | 5 | 0 | **9** |
| Execution | `CTRL050` – `CTRL052` | 2 | 1 | 0 | 0 | **3** |
| **Total `panel_growth.cpp`** | | **33** | **6** | **12** | **1** | **52** |

### 2. `native/src/ui/panels/panel_styles.cpp` (54 Controls Total)
Contains the growth presets library, record-new-preset toolchain, color themes, display typography, viewport element colors, 3D lighting / shading controls, and the specimen appearance trio (re-homed v24.01).

| CSV Section | Control ID Range | Sliders | Toggles | Buttons | Combos / Pickers / Text | Section Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Growth Presets Library | `CTRL053` – `CTRL056` | 0 | 0 | 3 | 1 | **4** |
| Record New Preset | `CTRL057` – `CTRL059` | 0 | 0 | 2 | 1 | **3** |
| Color Themes & Presets | `CTRL060` – `CTRL065` | 0 | 0 | 5 | 1 | **6** |
| Display Font / Typography | `CTRL066` – `CTRL070` | 2 | 1 | 1 | 1 | **5** |
| Viewport Element Colors | `CTRL071` – `CTRL086` | 0 | 0 | 1 | 15 | **16** |
| 3D Lighting & Shading *(v24.01 Phase 1: 6 -> 16 sliders)* | `CTRL087` – `CTRL103` | 16 | 0 | 1 | 0 | **17** |
| Specimen Appearance *(v24.01)* | `CTRL104` – `CTRL106` | 2 | 1 | 0 | 0 | **3** |
| **Total `panel_styles.cpp`** | | **20** | **2** | **13** | **19** | **54** |

### 3. `native/src/ui/panels/panel_verification.cpp` (3 Controls Total)
Contains the verification log window controls.

| CSV Section | Control ID Range | Sliders | Toggles | Buttons | Combos / Selectables | Section Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Verification Window | `CTRL109` – `CTRL111` | 0 | 0 | 2 | 1 | **3** |

### 4. `native/src/ui/panels/panel_scene.cpp` (39 Controls Total)
Contains the verification suite runner, implicit specimen parameters, GPU shape generators, layers visibility switches (shading trio re-homed to `panel_styles.cpp` in v24.01), mesh generation / export pipeline, and bark texture synthesis.

| CSV Section | Control ID Range | Sliders | Toggles | Buttons | Combos / Pickers / Text | Section Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Verification Suite | `CTRL107` – `CTRL108` | 0 | 0 | 2 | 0 | **2** |
| Implicit Specimen | `CTRL112` – `CTRL115` | 1 | 0 | 3 | 0 | **4** |
| Layers Visibility | `CTRL116` – `CTRL126` | 0 | 11 | 0 | 0 | **11** |
| Export Options | `CTRL127` – `CTRL138` | 5 | 2 | 4 | 1 | **12** |
| Bark Texture | `CTRL139` – `CTRL143` | 4 | 0 | 1 | 0 | **5** |
| Bark Relief | `CTRL144` – `CTRL148` | 3 | 1 | 1 | 0 | **5** |
| **Total `panel_scene.cpp`** | | **13** | **14** | **11** | **1** | **39** |

---

## 🔍 CONTROL INVENTORY BY FUNCTIONAL SECTION

### Section 1: Growth Command Bar (`panel_growth.cpp`)
- `CTRL001`: **Import** (`render_pill_button`) – Launches native file dialog to import 3D OBJ/STL specimen mesh.
- `CTRL002`: **Start** (`render_pill_button`) – Executes space-colonization root growth algorithm.
- `CTRL003`: **Stop** (`render_pill_button`) – Halts or pauses ongoing live root growth.
- `CTRL004`: **Clear** (`render_pill_button`) – Clears generated roots, resets growth simulation, and resets test objects.

### Section 2: Algorithm Selector & Parameters (`panel_growth.cpp`)
- `CTRL005`: **Algorithm Selector** (`begin_pill_combo`) – Switches between presets (Colonization, Volumetric, Vascular Murray, Root Bound, Creeping).
- `CTRL006`: **Export Algorithm** (`ImGui::Button`) – Exports active algorithm definition to `.json` or `.cu`.
- `CTRL007`: **Import Algorithm** (`ImGui::Button`) – Imports custom algorithm definition file.
- `CTRL008`: **Delete Algorithm** (`ImGui::Button`) – Deletes active algorithm profile.
- `CTRL009`: **Root Roughness** (`render_compact_slider_float`, `0.00` - `1.00`) – Meandering and turbulence in root trajectory.
- `CTRL010`: **Push Strength** (`render_compact_slider_float`, `0.00` - `1.00`) – Forward driving force for root propagation.
- `CTRL011`: **Space Fill Drive** (`render_compact_slider_float`, `0.10` - `1.00`) – Force pulling roots into unvisited interior volume.
- `CTRL012`: **Backward Pull** (`render_compact_slider_float`, `0.00` - `1.00`) – Curvature bias back toward seed origin.
- `CTRL013`: **Tensile Straightening** (`render_compact_slider_float`, `0.00` - `1.00`) – Structural tension straightening branches.
- `CTRL014`: **Surface Adherence** (`render_compact_slider_float`, `0.00` - `1.00`) – Attraction clamping roots to specimen shell surface.
- `CTRL015`: **Murray Exponent** (`render_compact_slider_float`, `2.00` - `4.00`, unit: $\gamma$) – Murray law branching power law exponent.
- `CTRL016`: **Murray Branch Angle** (`render_compact_slider_float`, `15.0` - `65.0`, unit: Degrees) – Bifurcation angle between daughter branches.

### Section 3: Colonization Seeds (`panel_growth.cpp`)
- `CTRL017`: **Spore Count** (`render_compact_slider_int`, `2000` - `50000`, unit: count) – Number of nutrient attractor spores distributed inside vessel.
- `CTRL018`: **Primary Roots** (`render_compact_slider_int`, `1` - `8`, unit: count) – Number of primary root trunks sprouted from seed.
- `CTRL019`: **Search Radius** (`render_compact_slider_float`, `4.0` - `30.0`, unit: mm) – Maximum distance nodes can detect spores.
- `CTRL020`: **Kill Radius** (`render_compact_slider_float`, `0.5` - `10.0`, unit: mm) – Distance threshold at which spores are consumed.

### Section 4: Branching & Growth (`panel_growth.cpp`)
- `CTRL021`: **Variable Step** (`render_toggle_switch`) – Toggles adaptive step sizing along root length.
- `CTRL022`: **Min Growth Step** (`render_compact_slider_float`, `0.25` - `4.00`, unit: mm) – Lower bound for adaptive growth step.
- `CTRL023`: **Max Growth Step** (`render_compact_slider_float`, `0.25` - `6.00`, unit: mm) – Upper bound for adaptive growth step.
- `CTRL024`: **Growth Step Size** (`render_compact_slider_float`, `0.25` - `6.00`, unit: mm) – Fixed node segment step length.
- `CTRL025`: **Branching Rate** (`render_compact_slider_float`, `0.10` - `1.00`) – Frequency of lateral daughter branch splitting.
- `CTRL026`: **Steering Cone** (`render_compact_slider_float`, `5.0` - `90.0`, unit: Degrees) – Angular cone limit for growth direction.
- `CTRL027`: **Max Steps** (`render_compact_slider_int`, `5` - `150`, unit: count) – Maximum growth iterations before auto-termination.

### Section 5: Vessel & Caliber (`panel_growth.cpp`)
- `CTRL028`: **Trunk Radius** (`render_compact_slider_float`, `0.10` - `3.00`, unit: mm) – Basal trunk radius at seed point.
- `CTRL029`: **Tip Radius** (`render_compact_slider_float`, `0.05` - `2.00`, unit: mm) – Terminal tendril radius at root tips.
- `CTRL030`: **Variable Taper** (`render_toggle_switch`) – Toggles continuous non-linear tapering.
- `CTRL031`: **Taper Curve** (`render_compact_slider_float`, `0.20` - `5.00`) – Exponent controlling taper profile curve.
- `CTRL032`: **Branch Smoothness (K)** (`render_compact_slider_float`, `0.00` - `2.00`, unit: mm) – SDF blend radius between branch junctions.
- `CTRL033`: **Wall Clearance** (`render_compact_slider_float`, `0.50` - `1.50`, unit: x) – Safe standoff distance multiplier from container boundary.
- `CTRL034`: **Wall Cut Smooth** (`render_compact_slider_float`, `0.00` - `2.00`, unit: mm) – Smoothness parameter for boolean boundary clipping.

### Section 6: Lateral Roots (`panel_growth.cpp`)
- `CTRL035`: **Sprout Spacing** (`render_compact_slider_float`, `0.50` - `6.00`, unit: mm) – Spatial interval between secondary lateral roots.
- `CTRL036`: **Lateral Base** (`render_compact_slider_float`, `0.05` - `0.80`, unit: mm) – Micro-radius at base of lateral rootlets.
- `CTRL037`: **Lateral Tip** (`render_compact_slider_float`, `0.02` - `0.40`, unit: mm) – Micro-radius at tip of lateral rootlets.
- `CTRL038`: **Max Reach** (`render_compact_slider_float`, `1.00` - `25.00`, unit: mm) – Maximum allowable length of secondary rootlets.
- `CTRL039`: **Surface Wander** (`render_compact_slider_float`, `0.00` - `1.00`) – Degree of lateral root curling across host surfaces.
- `CTRL040`: **Branch Angle** (`render_compact_slider_float`, `15.0` - `85.0`, unit: Degrees) – Outward takeoff angle for lateral rootlets.

### Section 7: Seeding Management (`panel_growth.cpp`)
- `CTRL041`: **Seed Offset** (`render_compact_slider_float`, `0.00` - `5.00`, unit: mm) – Clearance offset from container interior surface.
- `CTRL042`: **Add Seed (+)** (`ImGui::Button`) – Dispatches GPU raycast to place a new seed inside specimen.
- `CTRL043`: **Remove Seed (-)** (`ImGui::Button`) – Deletes active seed and prunes all branches grown from it on GPU.
- `CTRL044`: **Freeze All Seeds** (`ImGui::Button`) – Locks current caliber on all seeds simultaneously.
- `CTRL045`: **Unfreeze All Seeds** (`ImGui::Button`) – Unlocks all seeds to respond to global caliber changes.
- `CTRL046`: **Select Seed (S1-Sn)** (`ImGui::Button`) – Selects active seed slot for interactive inspection and positioning.
- `CTRL047`: **Freeze Seed** (`render_toggle_switch`) – Locks individual seed radius.
- `CTRL048`: **Snap to Root** (`render_toggle_switch`) – Forces lateral roots to hug parent root surface.
- `CTRL049`: **Lateral Roots** (`render_toggle_switch`) – Enables secondary rootlet sprouting on active seed.

### Section 8: Growth Execution (`panel_growth.cpp`)
- `CTRL050`: **Live Animated Growth** (`render_toggle_switch`) – Toggles continuous frame-by-frame growth simulation.
- `CTRL051`: **Step Rate (Live FPS)** (`render_compact_slider_int`, `1` - `30`, unit: fps) – Growth animation cadence limiter.
- `CTRL052`: **Steps Per Frame** (`render_compact_slider_int`, `1` - `10`, unit: count) – Number of growth algorithm iterations dispatched per frame.

### Section 9: Growth Presets Library (`panel_styles.cpp`)
- `CTRL053`: **Growth Preset Selector** (`begin_pill_combo`) – Selects active saved growth preset.
- `CTRL054`: **Apply Preset** (`ImGui::Button`) – Loads preset parameters into growth state.
- `CTRL055`: **Update Preset** (`ImGui::Button`) – Overwrites selected preset with active parameters.
- `CTRL056`: **Delete Preset** (`ImGui::Button`) – Removes selected preset from `imgui.ini`.
- `CTRL057`: **New Preset Name** (`ImGui::InputText`) – Input buffer for recording a custom preset.
- `CTRL058`: **Save New Preset** (`ImGui::Button`) – Commits new preset to persistent storage.
- `CTRL059`: **Export to Clipboard** (`ImGui::Button`) – Copies all preset configurations to Windows clipboard.

### Section 10: Color Themes & Layout Presets (`panel_styles.cpp`)
- `CTRL060`: **Theme Preset Selector** (`render_pill_combo`) – Switches theme (Modern Teal, Dark, Classic, Light).
- `CTRL061`: **Reset Theme** (`ImGui::Button`) – Restores Modern Dark Engineering teal palette.
- `CTRL062`: **Save Style Reference** (`ImGui::Button`) – Caches active ImGui style metrics.
- `CTRL063`: **Revert Style Reference** (`ImGui::Button`) – Restores cached style metrics.
- `CTRL064`: **Copy Style C++** (`ImGui::Button`) – Generates C++ code for active style colors to clipboard.
- `CTRL065`: **Copy Layout INI** (`ImGui::Button`) – Saves docking layout and styles to clipboard.

### Section 11: Display Font & Typography (`panel_styles.cpp`)
- `CTRL066`: **Display Font Selector** (`begin_pill_combo`) – Selects font family (Inter, Segoe UI, Roboto, etc.).
- `CTRL067`: **Bold Weight** (`render_toggle_switch`) – Toggles between Regular and Bold font weights.
- `CTRL068`: **Font Scale** (`render_compact_slider_float`, `0.50` - `2.00`, unit: x) – Global ImGui font scale multiplier.
- `CTRL069`: **Item Spacing Y** (`render_compact_slider_float`, `2.0` - `24.0`, unit: px) – Vertical UI element padding.
- `CTRL070`: **Reset Typography Settings** (`ImGui::Button`) – Restores default 1.0x scale and 4px padding.

### Section 12: Viewport Element Colors (`panel_styles.cpp`)
- `CTRL071` – `CTRL085`: **15 RGBA Color Pickers** (`ImGui::ColorEdit4`):
  - Viewport Background, Floor Grid, Floor X-Axis, Floor Y-Axis, Original Mesh, Body / Glass, Root Network, Branch Skeleton, Spores Cloud, Active Seed, Inactive Seeds, Bounding Box, Wireframe, Test Pass, Test Fail.
- `CTRL086`: **Reset Viewport Colors** (`ImGui::Button`) – Resets all 15 element colors to engineering defaults.

### Section 13: 3D Lighting & Shading (`panel_styles.cpp`)
v24.01 Phase 1 expanded this section from 6 to 16 sliders: every constant that was hardcoded in the two render paths (OpenGL mesh shader + CUDA raymarcher) is now a first-class, logged, persisted parameter. Defaults reproduce the v23.29 compiled-in look exactly (key/fill directions derived from Azimuth/Elevation in `core/lighting_rig.hpp`; `--verify` Test 20 asserts it). The "(raymarch)" trio affects only the CUDA raymarch path.
- `CTRL087`: **Diffuse Light** (`render_compact_slider_float`, `0.20` - `1.50`) – Viewport directional light intensity.
- `CTRL088`: **Edge Darkening** (`render_compact_slider_float`, `0.00` - `1.00`) – Silhouette edge darkening intensity.
- `CTRL089`: **Crevice Occlusion** (`render_compact_slider_float`, `0.00` - `1.00`) – Curvature-based crevice darkening.
- `CTRL090`: **Specular** (`render_compact_slider_float`, `0.00` - `0.60`) – Blinn-Phong specular highlight strength.
- `CTRL091`: **Bark Relief Shading** (`render_compact_slider_float`, `0.00` - `2.00`) – Shading normal enhancement for bark displacement.
- `CTRL092`: **Bark Relief Scale** (`render_compact_slider_float`, `0.10` - `10.00`) – Procedural frequency for relief shading.
- `CTRL093`: **Light Azimuth** (`render_compact_slider_float`, `-180.0` - `180.0`, unit: Degrees) – View-space key-light azimuth; the derived direction feeds both the GL mesh shader and the CUDA raymarcher (v24.01 Phase 1).
- `CTRL094`: **Light Elevation** (`render_compact_slider_float`, `-89.0` - `89.0`, unit: Degrees) – View-space key-light elevation; the fill light follows at a fixed angular offset (`core/lighting_rig.hpp`).
- `CTRL095`: **Fill Light** (`render_compact_slider_float`, `0.00` - `1.00`) – Counter-fill weight in the base illumination expression (was hardcoded 0.16).
- `CTRL096`: **Ambient** (`render_compact_slider_float`, `0.00` - `1.00`) – Ambient floor in the base illumination expression (was hardcoded 0.32).
- `CTRL097`: **Glossiness** (`render_compact_slider_float`, `4.0` - `64.0`) – Blinn-Phong specular exponent (was hardcoded 24).
- `CTRL098`: **Highlight Tint** (`render_compact_slider_float`, `0.00` - `1.00`) – Specular highlight mix toward white (was hardcoded 0.55).
- `CTRL099`: **Fresnel Rim** (`render_compact_slider_float`, `0.00` - `1.00`) – Matte-glass fresnel rim gain (was hardcoded 0.30).
- `CTRL100`: **Volumetric AO (raymarch)** (`render_compact_slider_float`, `0.00` - `1.00`) – CUDA raymarch ambient-occlusion gain across the three stylized AO sites (was hardcoded 0.85).
- `CTRL101`: **SSS Strength (raymarch)** (`render_compact_slider_float`, `0.00` - `1.00`) – Subsurface transmittance strength through thin root walls (was hardcoded 0.45).
- `CTRL102`: **SSS Warmth (raymarch)** (`render_compact_slider_float`, `0.00` - `1.00`) – Warm tint mix of the subsurface scattering glow (was hardcoded 0.35).
- `CTRL103`: **Reset 3D Lighting** (`ImGui::Button`) – Resets all lighting coefficients to default values.

### Section 14: Specimen Appearance (`panel_styles.cpp`) — v24.01 re-home
The shading trio, formerly in Layers Visibility (v23.28 ids CTRL105-107). All three belong with the lighting family: `DisplayState` declares them in its "3D Shading & Lighting" block, the Reset 3D Lighting button covers Glass Frosting, and the `[LightingAndShading]` INI section serializes them (v24.01 also added GlassFrosting / SpecimenOpacity keys and a startup loader).
- `CTRL104`: **Matte Glass Specimen** (`render_toggle_switch`) – Enables semi-transparent glass shader on body.
- `CTRL105`: **Glass Frosting** (`render_compact_slider_float`, `0.10` - `1.00`) – Roughness/blur factor of matte glass.
- `CTRL106`: **Opacity** (`render_compact_slider_float`, `0.10` - `1.00`) – Alpha transparency of specimen body.

Nesting conditions preserved from v23.29: the toggle is drawn only when the converted-body layer is visible, and the two sliders only in matte-glass mode.

### Section 15: Verification Suite (`panel_scene.cpp` + `panel_verification.cpp`)
- `CTRL107`: **Run All Tests (Scene Panel)** (`ImGui::Button`) – Dispatches full 20-test geometry verification harness on GPU.
- `CTRL108`: **Hide/View Log** (`ImGui::Button`) – Toggles visibility of floating verification results window.
- `CTRL109`: **Run Tests (Window)** (`ImGui::Button`) – Dispatches verification tests from within the tests window.
- `CTRL110`: **Clear 3D Tests** (`ImGui::Button`) – Removes test visual meshes from viewport and resets result list.
- `CTRL111`: **Load Test Model to Viewport** (`ImGui::Selectable`) – Loads a specific test result mesh as active specimen.

### Section 16: Implicit Specimen (`panel_scene.cpp`)
- `CTRL112`: **Detail Level Override** (`render_compact_slider_float`, `0.05` - `0.80`, unit: mm) – Voxel resolution for SDF rasterization.
- `CTRL113`: **Sphere (GPU Shape)** (`ImGui::Button`) – Synthesizes analytical 3D sphere mesh on GPU ($R = 20\text{ mm}$).
- `CTRL114`: **Torus (GPU Shape)** (`ImGui::Button`) – Synthesizes analytical 3D torus mesh on GPU ($R = 25\text{ mm}, r = 10\text{ mm}$).
- `CTRL115`: **Box (GPU Shape)** (`ImGui::Button`) – Synthesizes filleted box mesh on GPU ($40\text{ mm}$ with $4\text{ mm}$ fillet).

### Section 17: Layers Visibility (`panel_scene.cpp`)
- `CTRL116`: **Original Mesh** (`render_toggle_switch`) – Displays imported polygonal surface wireframe.
- `CTRL117`: **Converted Body** (`render_toggle_switch`) – Displays SDF reconstructed specimen body.
- `CTRL118`: **Roots** (`render_toggle_switch`) – Displays solid 3D root network mesh.
- `CTRL119`: **Colorize Roots** (`render_toggle_switch`) – Renders branches with distinct per-seed colors.
- `CTRL120`: **Colorize Strands** (`render_toggle_switch`) – Renders individual tendrils with gradient color ramps.
- `CTRL121`: **Branch Skeleton** (`render_toggle_switch`) – Renders 3D centerline graph line segments.
- `CTRL122`: **Seeds** (`render_toggle_switch`) – Renders 3D spheres at root initiation seed positions.
- `CTRL123`: **Spores** (`render_toggle_switch`) – Renders 3D nutrient point cloud inside container.
- `CTRL124`: **Bounding Box** (`render_toggle_switch`) – Renders axis-aligned bounding box cage.
- `CTRL125`: **Generated Mesh (Wireframe)** (`render_toggle_switch`) – Renders polygon wireframe overlay on generated mesh.
- `CTRL126`: **Auto Orbit** (`render_toggle_switch`) – Activates turntable camera rotation.

### Section 18: Mesh Generation & Export (`panel_scene.cpp`)
- `CTRL127`: **Export Target** (`render_pill_combo`) – Merged Union, Roots Only, or Converted Body Only.
- `CTRL128`: **Adaptive Voxel Size** (`render_toggle_switch`) – Enables dual-resolution adaptive marching cubes.
- `CTRL129`: **Mesh Voxel Size** (`render_compact_slider_float`, `0.05` - `0.60`, unit: mm) – Uniform marching cubes cell size.
- `CTRL130`: **Min Voxel Size** (`render_compact_slider_float`, `0.05` - `0.60`, unit: mm) – Finest detail cell size in adaptive mode.
- `CTRL131`: **Max Voxel Size** (`render_compact_slider_float`, `0.05` - `0.60`, unit: mm) – Coarsest cell size in adaptive mode.
- `CTRL132`: **Smoothing Passes** (`render_compact_slider_int`, `0` - `15`, unit: count) – Number of GPU Laplacian smoothing iterations.
- `CTRL133`: **Smooth Strength** (`render_compact_slider_float`, `0.05` - `0.80`) – Weight multiplier per smoothing pass.
- `CTRL134`: **Manifold Repair** (`render_toggle_switch`) – Dispatches GPU topological repair to seal non-manifold edges.
- `CTRL135`: **Generate Mesh** (`ImGui::Button`) – Dispatches CUDA isosurface extraction and post-processing pipeline.
- `CTRL136`: **Export Mesh (OBJ)** (`ImGui::Button`) – Saves final mesh to Wavefront OBJ format.
- `CTRL137`: **Export Binary STL** (`ImGui::Button`) – Saves final mesh to binary 3D-printable STL format.
- `CTRL138`: **Export Binary PLY** (`ImGui::Button`) – Saves final mesh to binary Polygon File Format (PLY).

### Section 19: Bark Texture & Relief (`panel_scene.cpp`)
- `CTRL139`: **Grain Scale** (`render_compact_slider_float`, `1.0` - `20.0`) – Cellular noise frequency scale.
- `CTRL140`: **Vertical Stretch** (`render_compact_slider_float`, `1.0` - `10.0`) – Anisotropic stretching along root axis.
- `CTRL141`: **Furrow Depth** (`render_compact_slider_float`, `0.0` - `1.0`) – Deep groove non-linear contrast exponent.
- `CTRL142`: **Noise Octaves** (`render_compact_slider_int`, `1` - `6`, unit: count) – Fractal noise detail octaves.
- `CTRL143`: **Generate 2D Bark Texture (GPU)** (`ImGui::Button`) – Computes 2D grayscale heightmap and OpenGL texture preview.
- `CTRL144`: **Displacement Depth** (`render_compact_slider_float`, `-5.0` - `5.0`, unit: mm) – Physical geometric vertex displacement depth.
- `CTRL145`: **Tile Repeat U** (`render_compact_slider_float`, `1.0` - `16.0`) – Circumferential texture tiling count.
- `CTRL146`: **Tile Repeat V** (`render_compact_slider_float`, `1.0` - `32.0`) – Longitudinal texture tiling count.
- `CTRL147`: **Recompute Normals** (`render_toggle_switch`) – Recalculates smooth analytical surface normals on GPU after vertex displacement.
- `CTRL148`: **Apply Bark Relief to Mesh (GPU)** (`ImGui::Button`) – Dispatches CUDA vertex displacement kernel directly onto mesh buffers.

---

## 🧩 STYLE EDITOR: REGISTERED (v24.01 Phase 2, EXECUTED)

**Status:** the 93 widgets below are **inside** the 241-control contract as `CTRL149-241` (three terminal CSV sections: Style Editor: Sizes / Colors / UI Rendering). Registry rows, telemetry, audit coverage and `[Style]`/`[StyleColors]` INI persistence all shipped in v24.01 Phase 2.

The Styles panel renders one block of widgets that is deliberately **outside** the 148-control contract: the **ADVANCED - INTERFACE STYLE EDITOR** parent pill (collapsed by default), containing three nested pills — **SIZES** (23 sliders: paddings, spacings, border sizes, rounding radii), **COLORS** (one `ImGui::ColorEdit4` per `ImGuiCol_` enumerator, the exact count determined by the ImGui version resolved through vcpkg), and **UI RENDERING (IMGUI)** (7 widgets: Global Alpha, Disabled Alpha, three anti-aliasing toggles, curve and circle tessellation tolerances).

These ~82 widgets are a Rhizome-skinned re-imagining of Dear ImGui's built-in `ShowStyleEditor()`, implemented as `render_style_editor_with_modern_preset()` in `native/src/ui/panels/panel_styles.cpp`. They bind **directly to `ImGuiStyle`** — the Dear ImGui toolkit's own interface state — rather than to `AppState`, through the `RHZ_BIND_IMGUI_STYLE_{F,F2,B,COL}` thunk family (the documented CTRL068/069 ImGui-global precedent, extended). Since v24.01 Phase 2 they are first-class catalog rows with three concrete enforcement consequences, all now in force: every widget has a `ControlDefinition` registry row, they are covered by the build-time and `--verify` CSV↔registry audits (plus the runtime census-coverage gate: the live Colors rows must equal the build's ImGuiCol_COUNT within the 63-row reference census (8d)), and they emit `ui_events.jsonl` telemetry — sliders/toggles through `render_control()`, the 63 census color pickers commit-only on `IsItemDeactivatedAfterEdit` (color drags fire per-frame; the event carries the drag-start and committed rgba values).

As executed (v24.01 Phase 2): the three pills sit under the single ADVANCED parent pill (one stable section home) and the RENDERING pill keeps the **UI RENDERING (IMGUI)** name. Numbering is append-only — Sizes `CTRL149-171`, Colors `CTRL172-234`, UI Rendering `CTRL235-241` — so no domain id moved. The dynamic `ImGuiCol_COUNT` handling landed as a version-adaptive census (8c, re-pinned 8d): the 63 CTRL172-234 rows are INDEX-bound (no enumerator names), so any Dear ImGui with ImGuiCol_COUNT <= 63 builds cleanly — rows beyond the built count stay dormant (8d note: the census layout follows the 1.92.8+ enum order; where upstream inserted colors mid-enum, an older covered build's census labels past the insertion point follow the census layout, not that older enum's) — while newer builds are stopped by the `ImguiColCoverageCheck` compile gate (actual count printed; refresh via `scripts/v2401_phase2_repin_colors.py`), with the `--verify` Test 20 `[Style Editor Registry]` gate reporting the census vs live counts and the `IMGUI_VERSION` string. Style changes now survive restarts: the Copy Layout INI exporter writes the full `[Style]` + `[StyleColors]` block and `load_imgui_style_from_ini` (panel_styles.cpp, called from main.cpp after `ImGui::CreateContext`) restores it.

---

## ⚙️ LOGGING SYSTEM INTEGRATION

Every UI interaction triggers a standardized call to `log_control_interaction()` (`native/src/ui/widgets/control_registry.hpp`), which resolves the exact CTRL-ID from the registry row and forwards to `RhizomeLog("ACTION", message)` (declared in `native/src/core/unified_logger.hpp`; implemented by the `UnifiedLogger` zero-alloc ring buffer with async flushing — console format unchanged):

```cpp
void RhizomeLog(const char* category, const char* message) {
    auto now = std::chrono::system_clock::now();
    // [HH:MM:SS.mmm] [CATEGORY] Message
    std::cout << "[" << time_str << "] [" << category << "] " 
              << (message ? message : "") << std::endl;
}
```

### Log Categories Mapped:
- **`ACTION`**: Slider adjustments, toggle flips, button clicks, and dropdown selections.
- **`REACTION`**: GPU execution metrics (e.g. `Mesh rebuilt in 23ms, triangles: 48,291`).
- **`MEM:BUFFER`**: Dynamic GPU memory allocations and buffer uploads.
- **`MEM:VRAM`**: Real-time snapshot of NVIDIA GPU VRAM occupancy.
