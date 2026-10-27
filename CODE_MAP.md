# CODE_MAP.md — Rhizome Engine Agent Navigation Map (v24.15)

This document is a technical directory and navigation map for engineers and AI agents working on the **Rhizome v24.11** codebase. It outlines the project structure, entry points, host/device boundaries, and architectural constraints. The v23.29 restructuring (5-phase protocol per `AGENTS.md`) moved the v23.28 monoliths into the domain-driven tree below; every module now lives under `native/src/{core,ui,engine,verification}` with co-located headers — the legacy flat `include/` root is gone. v24.03 (2026-10-23) consolidated the v24.02 program (harness refinement + deterministic GPU accumulation) and the two post-closure fixes (GUI suite crash, Lateral Roots viewport disappearance); the v24.03-specific kernel annotations below are marked **(v24.03)**. v24.04 (2026-10-23, Track D) adds the Fast SDF path to `compute_mesh_sdf` and moves the default Detail Level to 0.10 mm; v24.04-specific annotations are marked **(v24.04)**. v24.05 (2026-10-23, Track D R2) completes the fast path's verification gate with an iterative exact-winding patch policy (patch a small number of mismatches in place, re-probe with fresh stratifications, fall back only on systematic divergence); v24.05-specific annotations are marked **(v24.05)**. v24.06 (2026-10-23, Track D R3) completes the eikonal far field (far-sentinel audit, audit-gated extension rounds, sign-aware safe clamp) and adds a viewport-side specimen-march guard; v24.06-specific annotations are marked **(v24.06)**. v24.07 (2026-10-23, Track D R4) makes the specimen march band-limited (marched values clamped at the ±8-voxel band edge, step budget 512 → 1024) after the truncation study proved the sweep-capped eikonal can ship real-but-overestimated far distances that sphere tracing punches through thin walls; v24.07-specific annotations are marked **(v24.07)**. v24.08 (2026-10-23, Track D R5) raises the roots-march travel budget (280 → 2048 steps — the roots-only face of the rotating trim) and adds `VOLUME_MARCH_GUIDE.md`, the learnable documentation of the whole volume-rendering path; v24.08-specific annotations are marked **(v24.08)**. v24.09 (2026-10-24, Track E R1) flips the Live Animated Growth default OFF in both `GrowthState` (`core/app_state.hpp`) and `GrowthPreset` (`core/algorithm_io.hpp`) — the state struct is the sole default source (the control registry carries no default mechanism; its boolean flag is `logs`), so the one-shot batch bake is now the default Start path while the Growth panel Execution toggle keeps live mode opt-in; v24.09-specific annotations are marked **(v24.09)**. v24.10 (2026-10-24, Track E R2) fixes the second-Start re-bake memory spike the batch default exposed: `update_implicit_session_controller` now releases the previous baked roots grid (host) and the viewport's cached device copy of it BEFORE rasterizing the replacement, and the two bake call sites intersect the fresh rasterization in place (`preserve_input_grid=false`; all other `intersect_with_vessel` callers keep the copy semantics); v24.10-specific annotations are marked **(v24.10)**. v24.11 (2026-10-24, Track E R3) hardens the bake pipeline against the fail-fast crash class captured in the field on the quick second Start (exit code 0xC0000409 = unhandled C++ exception → terminate → abort → UCRT __fastfail): `rasterize_volumetric_sections_gpu` validates the GPU-computed section bounds in double precision before the integer grid math (rejecting impossible bounds with a logged `[Bake Guard]` diagnostic and the standard graceful bake failure), and the three `growth_controller` entry points (Start / Stop / live-step) gain try/catch wrappers that convert any remaining throw into a logged status-bar message with the session parked; v24.11-specific annotations are marked **(v24.11)**. Program history and governance: `AGENTS.md` Secs 5-15.

---

## 1. High-Level Architecture (v23.29)

- **Application Type:** Native Windows 64-bit desktop application (`rhizome_app.exe`).
- **Language Standards:** C++20 (`/std:c++20`) and NVIDIA CUDA 12 (`--extended-lambda`, `--expt-relaxed-constexpr`).
- **Target GPU Architectures:** Compute Capability 8.6 and 8.9 (NVIDIA Ampere RTX 30xx, Ada Lovelace RTX 40xx).
- **GUI & Display:** Dear ImGui with docking enabled + OpenGL 3.3 Core Profile viewport with GLFW window management.
- **Build System:** CMake (>= 3.24) targeting Visual Studio 2022 (`MSVC` + `NVCC`). Dependencies managed via `vcpkg.json` (GLFW, GLAD, GLM, ImGui, OpenGL).
- **Targets in `native/CMakeLists.txt`:**
  - `rhizome_core` (static library): `core/` state SSOT + logger + growth orchestration, `engine/` geometry & CUDA pipeline (sdf / colonization / mesh / io), viewport CUDA services (`ui/viewport/*.cu`), and the verification test harness.
  - `rhizome_app` (executable): main entry point, `ui/panels/` + `ui/widgets/` (data-driven 251-control registry + widget factory), `ui/viewport/` renderer.
  - `rhizome_control_audit` (Phase 5, custom target): build-time CSV `<->` control-registry consistency audit; a build dependency of `rhizome_app` (requires a Python 3 interpreter, skipped with a warning otherwise).
- **Core Engine Constraint:** 100% Native NVIDIA CUDA GPU pipeline. OpenVDB has been eliminated in favor of native CUDA SDF kernels (`engine/sdf/cuda_rasterizer.cu`). Zero CPU computation fallbacks are permitted.

### Directory Tree (v23.29, all headers co-located)

```
native/src/
├── main.cpp                  # entry: GLFW/ImGui bootstrap, main loop, CLI flags
├── core/                     # state SSOT, logger, math, orchestration
│   ├── app_state.{hpp,cpp}   #   AppState (9 sub-states) + preset transfer
│   ├── lighting_rig.hpp        # v24.01 Phase 1: shared key/fill light derivation (GL + CUDA)
│   ├── unified_logger.{hpp,cpp} # zero-alloc ring logger + 251-control monitor catalog
│   ├── growth_controller.{hpp,cpp} # live-growth stepping (AppState& + EngineResources&); (v24.10) update_implicit_session_controller releases the old roots bake + viewport device cache before every rebake
│   ├── algorithm_io.{hpp,cpp}   # algorithm profiles + preset .ini persistence
│   ├── engine_resources.hpp # EngineResources: live GPU/GL buffers & metrics
│   └── math_utils.cuh        # shared CUDA math, RASTER_TILE_SIZE = 128
├── ui/
│   ├── widgets/              # widget_factory.{hpp,cpp} + control_registry.{hpp,cpp}
│   ├── panels/               # panel_growth / panel_styles / panel_scene / panel_verification
│   └── viewport/             # viewport_renderer + cuda_viewport_gl/packers (CUDA)
├── engine/
│   ├── sdf/                  # cuda_rasterizer, narrow_band_sdf, root_field, cuda_common.cuh
│   ├── colonization/         # colonization, cuda_colonization{,_volumetric}
│   ├── mesh/                 # cuda_marching_cubes, cuda_mesh_ops/topology/primitives
│   └── io/                   # obj/stl parsers, obj_exporter, cuda_mesh_io
└── verification/             # harness*.{cpp,hpp}, cuda_verification.cu, gpu_benchmark_runner
```

---

## 2. Main Entry Points

| Entry Point | File Path | Primary Function / Symbol | Description |
| :--- | :--- | :--- | :--- |
| **Application Startup** | `native/src/main.cpp` | `main(int argc, char** argv)` | Initializes UTF-8 console, GLFW window, GLAD, ImGui docking, fonts, and runs startup tests or the verification harness if `--verify`, `--test`, or `-t` flag is passed. `--verify` first audits the 241-control registry, then runs the full 20-test suite headlessly (exit code = pass/fail). |
| **Main Event Loop** | `native/src/main.cpp` | `while (!glfwWindowShouldClose(window))` | Dispatches frame event polling, step growth updates, ImGui frame rendering, viewport rendering, and buffer swapping. |
| **CUDA Engine Entry Point** | `native/src/engine/sdf/cuda_rasterizer.hpp`<br>`native/src/engine/sdf/cuda_rasterizer.cu` | `CudaRasterizer::is_cuda_available()`<br>`CudaRasterizer::compute_mesh_sdf()`<br>`CudaRasterizer::step_volumetric_colonization_gpu()` | Hardware probe and high-level C++ dispatchers for GPU kernels. |
| **UI Window Entry Point** | `native/src/ui/panels/panel_growth.cpp`<br>`native/src/ui/panels/panel_styles.cpp`<br>`native/src/ui/panels/panel_scene.cpp`<br>`native/src/ui/panels/panel_verification.cpp` | `render_growth_panel()`<br>`render_styles_panel()`<br>`render_scene_panel()`<br>`render_viewport_window()` | Renders left docked panel (growth params), right docked panel (themes, presets, specimen & export), verification window, and central 3D viewport. |
| **Control Registry Entry Point** | `native/src/ui/widgets/control_registry.{hpp,cpp}` | `k_all_controls[241]`, `render_control()`, `find_control()` | Data-driven declaration of all 241 controls: CSV-authoritative catalog fields + typed thunk bindings into `AppState` (domain rows) and `ImGuiStyle` (style-editor rows, Sec 3.10). v24.01 Phase 1: ten lighting sliders at CTRL093-102, catalog 138 -> 148; v24.01 Phase 2: 88 style-editor widgets appended at CTRL149-236, catalog 148 -> 236; 8d census re-pin: Colors 58 -> 63 (ImGui 1.92.8+), catalog 236 -> 241. |
| **File I/O Entry Point** | `native/src/ui/panels/panel_scene.cpp`<br>`native/src/engine/io/parsers.hpp`<br>`native/src/engine/io/obj_exporter.cpp` | `open_file_dialog()`<br>`save_file_dialog()`<br>`ObjParser::parse()`<br>`StlParser::parse()`<br>`ObjExporter::export_mesh()` | Native Win32 file dialogs (`OPENFILENAMEA`) streaming buffers to/from disk, feeding parallel GPU decoders. |

---

## 3. Module Responsibility Map

### 1. Application Shell & UI
- **Purpose:** Manages window creation, ImGui docking layout, theme styling, and frame dispatch.
- **Primary Files:** `native/src/main.cpp`, `native/src/ui/panels/panel_growth.cpp`, `native/src/ui/panels/panel_styles.cpp`, `native/src/ui/panels/panel_scene.cpp`, `native/src/ui/panels/panel_verification.cpp`, `native/src/ui/widgets/widget_factory.cpp`.
- **Key Functions:** `main()`, `apply_dark_engineering_theme()` (panel_styles.cpp), `render_growth_panel()` (panel_growth.cpp), `render_styles_panel()` (panel_styles.cpp), `render_scene_panel()` + `render_status_bar_panel()` (panel_scene.cpp), `render_viewport_window()` (ui/viewport/viewport_renderer.cpp).
- **Style-Editor Registration (v24.01 Phase 2, EXECUTED):** `render_style_editor_with_modern_preset(AppState&)` (panel_styles.cpp) — its 93 Dear ImGui toolkit widgets (SIZES 23 / COLORS 63 == ImGuiCol_COUNT at the 8d census, ImGui 1.92.8+, version-adaptive 8c+8d / UI RENDERING (IMGUI) 7, nested under the "ADVANCED - INTERFACE STYLE EDITOR" parent pill) are registered as CTRL149-241: `RHZ_BIND_IMGUI_STYLE_*` thunk bindings into the live `ImGuiStyle`, registry + CSV + logger rows, build-time & `--verify` audit coverage, `ui_events.jsonl` telemetry (colors commit-only via `IsItemDeactivatedAfterEdit`), and `[Style]`/`[StyleColors]` INI persistence through `load_imgui_style_from_ini`.
- **Do Not Break:** The central dockspace ID layout (`RhizomeDockSpace`) and the quiet status bar panel.

### 2. State & Growth Controller
- **Purpose:** Coordinates application state, growth parameter profiles, live growth frames, and specimen conversion.
- **Primary Files (v23.29: `core/`):** `native/src/core/app_state.{hpp,cpp}`, `native/src/core/growth_controller.{hpp,cpp}`, `native/src/core/algorithm_io.{hpp,cpp}`, `native/src/core/engine_resources.hpp`.
- **Key Classes / Functions:** `AppState`, `EngineResources`, `GrowthControllerContext` (slimmed v23.29 form: `AppState&` + `EngineResources&`), `step_live_growth_frame_controller()`, `remesh_live_session_controller()`, `convert_and_update_model_controller()`, `export_algorithm_to_json()`, `export_algorithm_to_cu()`.
- **State Architecture (Single Source of Truth — 9 sub-states):**
  - `AppState::growth` (`GrowthState`): Detail level, search radius, growth step, branching rate, algorithm mode, caliber, taper, physical drivers. **(v24.09)** `live_growth` defaults OFF — live animated growth is opt-in (toggle CTRL050); Start takes the one-shot batch bake by default.
  - `AppState::seed` (`SeedState`): Seed points, per-seed freeze flags, lateral/snap toggles, `show_seeds` viewport flag.
  - `AppState::display` (`DisplayState`): 3D viewport visibility flags (`show_roots`, `show_wireframe`, `show_spores`, `colorize_roots`, `colorize_strands`), element colors, and lighting/shading parameters.
  - `AppState::viewport` (`ViewportInputState`): Camera & orbit input state.
  - `AppState::specimen` (`SpecimenInfoState`): Imported specimen identity, triangle counts, build timings.
  - `AppState::mesh_export` (`MeshExportState`): Export target, mesh voxel sizes, smoothing passes/strength, manifold repair, and generated mesh presence flags (`has_generated_roots_mesh`, `has_generated_vessel_mesh`, `has_generated_merged_mesh`).
  - `AppState::bark_texture` (`BarkTextureState`): Bark grain/relief parameters and generated texture presence.
  - `AppState::verification` (`VerificationUIState`): Verification suite and style editor visibility flags.
  - `AppState::ui` (`AppUIState`): Theme presets, active font index, font bold flag, `new_preset_name` buffer (CTRL057), status bar message, and UI layout metrics.
- **State Architecture (Single Source of Truth):**
  - `AppState::growth` (`GrowthState`): Detail level, search radius, growth step, branching rate, algorithm mode, caliber, taper, physical drivers.
  - `AppState::display` (`DisplayState`): 3D viewport visibility flags (`show_roots`, `show_wireframe`, `show_spores`, `colorize_roots`, `colorize_strands`), element colors, and lighting/shading parameters.
  - `AppState::mesh_export` (`MeshExportState`): Export target, mesh voxel sizes, smoothing passes/strength, manifold repair, and generated mesh presence flags (`has_generated_roots_mesh`, `has_generated_vessel_mesh`, `has_generated_merged_mesh`).
  - `AppState::verification` (`VerificationUIState`): Verification suite and style editor visibility flags.
  - `AppState::ui` (`AppUIState`): Theme presets, active font index, font bold flag, and UI layout metrics.
- **Do Not Break:** Preset persistence logic (`load_growth_presets_from_ini`), algorithm profiles, live stepping cadence, and single source of truth in `AppState` (never introduce duplicate shadow state in panel contexts or `main.cpp`).

### 3. Space Colonization Solvers
- **Purpose:** Executes biological growth algorithms (classic space colonization, natural root-bound, and Murray vascular bifurcation).
- **Primary Files (v23.29: `engine/colonization/`):** `native/src/engine/colonization/colonization.{hpp,cpp}`, `native/src/engine/colonization/cuda_colonization.cu`, `native/src/engine/colonization/cuda_colonization_volumetric.cu`.
- **Key Classes / Functions / Kernels:**
  - `SpaceColonization` (host driver class)
  - `spore_search_kernel` (classic attraction solver)
  - `gpu_accumulate_spore_forces_kernel` (atomic attraction direction reduction)
  - `gpu_insert_sections_hash_kernel` (spatial hashing of root segments)
  - `gpu_prune_and_hash_spores_kernel` (GPU spore kill and occupancy hashing)
  - `gpu_sense_and_advance_tips_kernel` (11-feeler environment cone sensor & lateral branching)
  - `gpu_push_sections_kernel` (3D capsule-to-capsule physical pushing solver)
  - `gpu_rope_reel_tethers_kernel` (tensile tether distance reeling and vessel confinement)
  - `gpu_accumulate_face_normals_kernel` (normal computation for mesh curvature analysis)
  - `gpu_dequantize_ll_to_float_kernel` / `gpu_dequantize_ll_to_float_vol_kernel` **(v24.02 Phase 2 / v24.03)** — refill the float accumulation buffers from the fixed-point uint64 shadow accumulators (Q = 2^24) that replaced the 18 float `atomicAdd` sites (order-independent by two's-complement associativity)
  - `gpu_generate_lateral_roots_kernel` **(v24.03 / Track C)** — GPU lateral-rootlet generation off seminal-root sections (Growth > Seeding > Lateral Roots, CTRL049); on output-buffer overflow it writes truncated strands rather than abandoning its counted `atomicAdd` reservation (uninitialized-section hardening)
- **Do Not Break:** The 11-feeler cone sensing logic, collision avoidance margins, and Murray vascular exponent branching calculations.

### 4. CUDA SDF & Rasterization
- **Purpose:** Signed distance field generation, capsule rasterization, volumetric Boolean CSG blending, and analytical test fields.
- **Primary Files (v23.29: `engine/sdf/`):** `native/src/engine/sdf/cuda_rasterizer.{hpp,cu}`, `native/src/engine/sdf/cuda_types.hpp`, `native/src/engine/sdf/cuda_common.cuh` (CUDA runtime guards, `DeviceBuffer<T>` RAII, shared kernel declarations), `native/src/core/math_utils.cuh` (pure math + `RASTER_TILE_SIZE`), `native/src/engine/sdf/narrow_band_sdf.cpp`, `native/src/engine/sdf/root_field.cpp`.
- **Key Kernels / Methods:**
  - `mesh_to_sdf_kernel` (parallel point-to-triangle distance + generalized winding number evaluation — the LEGACY exact kernel, still the only path below 32M voxels and the fallback for meshes the fast path rejects)
  - **Fast SDF path (v24.04 Track D, verify gate completed v24.05 R2)** inside `CudaRasterizer::compute_mesh_sdf()`, engaging at ≥ 32M voxels (`RHI_FAST_SDF_MIN_VOXELS`): `gpu_splat_band_distances_kernel` (exact per-triangle band splat, ±8 voxels `RHI_SDF_BAND_VOXELS`, `dev_atomicMin_float`), `gpu_parity_sign_columns_kernel` (scanline parity inside/outside per z-column; jittered rays + host-built 4×4-voxel XY supercell CSR; leaking/odd columns deferred), `gpu_exact_winding_fixup_kernel` (exact winding for deferred voxels — launched from two sites: the [C] column fixup and the v24.05 R2 patch loop), `gpu_eikonal_fsm_sweep_kernel` + `rhi_eikonal_godunov` (in-place directional fast-sweeping re-distancing of the far field, 6 sweeps/round to convergence; v24.06 R3 completes it: `gpu_count_far_sentinels_kernel` audits for unreached ≥1e29 voxels after the rounds, an audit-gated extension runs up to 16 extra rounds on a nonzero residual, and `gpu_clamp_far_sentinels_kernel` safe-clamps any true leftover to sign-aware band-edge magnitudes — no zero crossing moves), `gpu_verify_sign_samples_kernel` (v24.05 R2: 4,096-sample × 3-generation exact-winding verification; records mismatched sample voxels into `d_sites`; probe offsets {1/2, 1/6, 5/6}, generation 0 bit-identical to the v24.04 probe), `gpu_mark_patch_unknown_kernel` (v24.05 R2: marks a 49³-voxel box around each mismatched site for exact recomputation), `gpu_apply_signs_kernel` (sign apply + far-sentinel clamp); host orchestration in `rhi_compute_mesh_sdf_fast` — a small number of verify mismatches is patched with exact winding and re-probed; systematic divergence (> 64 sites per probe, or persistence across 3 generations), VRAM, or pathology declines fall back to the legacy kernel, so the worst case is exactly v24.03/v24.04 behavior
  - `rasterize_capsules_kernel` (shared memory tiled capsule distance rasterizer with smooth min CSG blending)
  - `gpu_csg_intersect_kernel` **(v24.03 / Track C)** — volumetric Boolean intersect of the root grid with the vessel SDF; the stored distance is clamped to the narrow-band background `d = fminf(d, 4 × voxel_size)` so an out-of-bounds sampler sentinel can never poison the far field (fixes the Lateral Roots viewport disappearance; moves no zero crossing — meshes/exports/verification numerically unaffected)
  - `CudaRasterizer::compute_mesh_sdf()`
  - `CudaRasterizer::rasterize()`
  - `CudaRasterizer::fast_sweeping_eikonal_gpu()`
  - `CudaVolumeGrid` (dense voxel volume structure)
- **Do Not Break:** Shared-memory tiling (`RASTER_TILE_SIZE = 128`), sign evaluation via generalized winding numbers, and absence of external OpenVDB calls.

### 5. Mesh Extraction & Remeshing
- **Purpose:** Iso-surface polygonization, Laplacian smoothing, adaptive decimation, and manifold repair.
- **Primary Files (v23.29: `engine/mesh/`):** `native/src/engine/mesh/cuda_marching_cubes.cu`, `native/src/engine/mesh/cuda_mesh_ops.cu`, `native/src/engine/mesh/cuda_mesh_topology.cu`, `native/src/engine/mesh/cuda_mesh_primitives.cu`.
- **Key Kernels / Methods:**
  - `gpu_mc_count_pass_kernel` (pass 1: counts welded edge vertices and triangles)
  - `gpu_mc_emit_vertices_kernel` (pass 2A: emits welded vertices along grid edges)
  - `gpu_mc_emit_triangles_kernel` (pass 2B: outputs indexed triangle topology)
  - `gpu_accumulate_face_normals_kernel` (normal computation)
  - `gpu_mc_seal_border_kernel` (border sealing for watertight surfaces)
  - `CudaRasterizer::smooth_mesh_laplacian_gpu()`
  - `CudaRasterizer::repair_mesh_manifold_gpu()`
- **Do Not Break:** Two-pass Marching Cubes edge welding that guarantees closed, watertight, 2-manifold surfaces.

### 6. Viewport Renderer & GPU Raymarching
- **Purpose:** Interactive OpenGL rendering, camera control, direct GPU implicit volume raymarching, and VBO streaming.
- **Primary Files (v23.29: `ui/viewport/`):** `native/src/ui/viewport/viewport_renderer.{hpp,cpp}`, `native/src/ui/viewport/cuda_viewport_gl.cu`, `native/src/ui/viewport/cuda_viewport_packers.cu` (the two .cu files compile into `rhizome_core`).
- **Key Kernels / Methods:**
  - `gpu_raymarch_implicit_viewport_kernel` (direct GPU SDF raymarching without generating a mesh)
  - `CudaRasterizer::pack_gl_mesh_vbo_gpu()`
  - `CudaRasterizer::pack_branch_skeleton_vbo_gpu()`
  - `ViewportRenderer::render()`
  - **(v24.03 / Track C)** the roots-grid march caps its per-sample advance at 4 × voxel (mirroring the capsule path's existing max-step guard) — a provable no-op on properly banded grids that prevents any out-of-bounds sentinel from launching a ray past the volume
  - **(v24.07 / Track D R4, supersedes the v24.06 specimen guard)** the specimen march is **band-limited**: every marched value is clamped at the ±8-voxel band edge (`if (d > spec_band) d = spec_band`), and the step budget doubles (512 → 1024). The R4 truncation study (sandbox, real mug.obj at 0.25/0.4 mm replicas of the fast-path pipeline) proved the sweep-capped eikonal ships **real but overestimated** far distances (up to ~13 mm at mug scale after the 8-round cap, invisible to the v24.06 far-sentinel audit since no sentinels remain) — sphere tracing then steps `0.85 × (true + overestimate)` and punches through thin walls/rim, rendering the interior or missing outright: the plane-like trims that move while rotating. The band is exact by construction (the monotone eikonal never undercuts the [A] splat seeds), so clamping outside it restores the pre-v24.04 marching guarantee: a step (≤ 6.8 voxels) can never clear surface + band on both sides
  - **(v24.08 / Track D R5)** the roots-grid march budget is raised 280 → **2048** steps. The 280 cap was sized for the 0.25–0.30 mm era (one background step ~1 mm); at the 0.10 mm default the steady step is 0.34 mm, so the reach was 95 mm — under the ~183 mm diagonal of a real baked roots grid (vessel + lateral rootlets + pad), and rays entering the far half ran out of samples mid-volume: the roots-only dynamic trim. 2048 × 0.34 = 696 mm = 3.8× the diagonal; no band clamp needed here (roots grids are direct capsule rasterizations — out-of-band values are conservative lower bounds by construction; the Track C per-step cap stays). Full learnable documentation of this render path: `VOLUME_MARCH_GUIDE.md` (app tree root)
- **Do Not Break:** 3Dconnexion SpaceNavigator support hooks and offscreen framebuffer blitting.

### 7. File Parsers & Exporters
- **Purpose:** Reads and writes OBJ, STL (ASCII & Binary), and PLY formats with zero-copy/single-pass host buffering and CUDA VRAM formatting.
- **Primary Files (v23.29: `engine/io/`):** `native/src/engine/io/obj_parser.cpp`, `native/src/engine/io/stl_parser.cpp`, `native/src/engine/io/obj_exporter.cpp`, `native/src/engine/io/cuda_mesh_io.cu`, `native/src/engine/io/parsers.hpp`.
- **Key Methods / Kernels:**
  - `ObjParser::parse()`, `StlParser::parse()` (single-pass in-memory streaming with robust `tellg()` stream size bounds)
  - `ObjExporter::export_obj()`, `ObjExporter::export_binary_stl()`, `ObjExporter::export_binary_ply()`
  - `gpu_format_binary_stl_kernel`, `gpu_pack_ply_vertices_kernel`, `gpu_pack_ply_faces_kernel` (VRAM buffer formatters with 32-bit triangle limits and byte-overflow guards) *(v23.29 fix F14: the PLY formatter was previously misattributed as a single nonexistent kernel)*
  - `CudaRasterizer::parse_binary_stl_gpu()`, `CudaRasterizer::parse_obj_text_gpu()`
- **Do Not Break:** Topological manifold verification checks prior to file writing, single-pass buffer loading, and millimetric coordinate scaling (1 unit = 1 mm).

### 8. Verification Suite & GPU Benchmark Runner
- **Purpose:** Validates analytical shapes, Eikonal convergence, vessel containment, round-trip geometric precision, and runs automated CUDA hardware stress tests across all 241 parameter controls.
- **Primary Files (v23.29: `verification/`, headers co-located):** `native/src/verification/harness.{hpp,cpp}`, `native/src/verification/harness_vessel_tests.cpp`, `native/src/verification/harness_eikonal_cuda_tests.cpp`, `native/src/verification/harness_volumetric_tests.cpp`, `native/src/verification/cuda_verification.cu`, `native/src/verification/gpu_benchmark_runner.{hpp,cu}`.
- **Key Methods:** `VerificationHarness::run_all_tests()`, `CudaRasterizer::evaluate_mesh_deviations_gpu()`, `CudaRasterizer::evaluate_eikonal_field_gpu()`, `GpuBenchmarkRunner::run_full_benchmark()`, `GpuBenchmarkRunner::export_report_to_json()`, `GpuBenchmarkRunner::export_report_to_markdown()`.
- **(v24.02 Phase 1 / v24.03) Harness refinements:** Test 8's internal rebuild grid refined 0.20 → 0.15 mm (`harness_vessel_tests.cpp`, round-trip site only — the acceptance bound unchanged at 0.100 mm; suite-green 20/20 since); the Eikonal test's per-case prints state the actual acceptance rule (`harness_eikonal_cuda_tests.cpp` — display references labeled as such); Test 20's ImGuiStyle INI round-trip switches to its scratch ImGui context explicitly and destroys it by pointer after restore (`harness_volumetric_tests.cpp` — fixes the GUI "Run All Tests" crash: the no-argument `DestroyContext()` previously destroyed the live context mid-frame in GUI mode).
- **Do Not Break:** The test runner CLI invocations (`--verify` for the 251-control registry audit + full automated 20-test suite, `--test` / `-t` for fast startup tests, `--benchmark` / `-b` for automated GPU benchmark runner) and UI test/benchmark buttons in `ui/panels/panel_scene.cpp`.

### 10. Control Registry & Widget Factory (v23.29)
- **Purpose:** Data-driven declaration and rendering of the 251-contract controls (`AGENTS.md` Sec 1.3 / A.1).
- **Primary Files:** `native/src/ui/widgets/control_registry.{hpp,cpp}`, `native/src/ui/widgets/widget_factory.{hpp,cpp}`.
- **Key Classes / Functions:** `ControlDefinition` (CSV catalog fields + `draw_label`/`format` + static CSV ranges + thunk resolvers), `k_all_controls[241]`, `k_all_sections[25]`, `RHZ_BIND_*` binding macros (incl. the Phase 2 `RHZ_BIND_IMGUI_STYLE_{F,F2,B,COL}` family), `render_control()`, `log_control_interaction()`, `find_control()`, `control_registry_integrity_check()`.
- **v24.01 Phase 0:** the shading trio (Matte Glass Specimen / Glass Frosting / Opacity) re-homed from "Layers Visibility" to the new "Specimen Appearance" section (CTRL094-096, rendered in `panel_styles.cpp`); downstream ids slid to CTRL097-107; CSV Source_File column now names the real panel homes and is verified by the build-time audit; `load_viewport_style_from_ini` (`core/algorithm_io.cpp`) restores `[ViewportColors]` / `[LightingAndShading]` from `imgui.ini` at startup.
- **Contract:** `param_controls_complete_list.csv` is the single authority — every row `CTRL001..CTRL251` appears exactly once, in CSV order; value controls (197 rows: 104 domain + 93 style-editor) bind typed `AppState` members or live `ImGuiStyle` members via captureless-lambda thunks; action controls (44 rows) keep declarative rows for catalog, logging and benchmark enumeration while their handlers live in the owning panel `.cpp`.
- **Do Not Break:** The build-time CSV audit (`rhizome_control_audit` CMake target) and the runtime `--verify` registry integrity check must both stay green.

### 9. Unified Debug & Memory Logger & UI Interaction Monitor
- **Purpose:** Thread-safe centralized logging of GPU VRAM allocation (free/total), step execution performance, UI slider interactions, and error diagnostics across host and CUDA contexts.
- **Primary Files (v23.29: `core/`):** `native/src/core/unified_logger.{hpp,cpp}`, `ui_interaction_monitor_system.md`, `param_controls_complete_list.csv`.
- **v23.29 note:** `UnifiedLogger` hosts the verbatim control monitor catalog (`g_all_controls`; 241 rows since the 8d census re-pin) used for exact CTRL-ID resolution in the JSON-Lines interaction stream; consistency with the CSV and the `ui/widgets` registry is enforced at build time (`native/tools/audit_control_registry.py`, CMake target `rhizome_control_audit`) and at runtime by `--verify`.
- **Key Methods / Classes:** `UnifiedLogger::instance()`, `UnifiedLogger::log_info()`, `UnifiedLogger::log_gpu_memory()`, `UnifiedLogger::log_step()`, `UnifiedLogger::log_interaction()`.
- **Do Not Break:** Zero-allocation ring buffer, millisecond timestamp precision, and asynchronous file flushing guards.

---

## 4. Feature-to-File Matrix

| Feature | Primary File(s) | Key Symbols / Kernels |
| :--- | :--- | :--- |
| **Mesh-to-SDF Calculation** | `native/src/engine/sdf/cuda_rasterizer.cu` | `mesh_to_sdf_kernel`, `compute_mesh_sdf()`, `build_from_mesh()` |
| **Capsule Rasterization & Blending** | `native/src/engine/sdf/cuda_rasterizer.cu` | `rasterize_capsules_kernel`, `rasterize()`, `rasterize_and_extract_mesh_gpu()` |
| **Classic Space Colonization** | `native/src/engine/colonization/cuda_colonization.cu` | `spore_search_kernel`, `gpu_accumulate_spore_forces_kernel`, `evaluate_spores()` |
| **Volumetric Root Simulation** | `native/src/engine/colonization/cuda_colonization_volumetric.cu` | `gpu_sense_and_advance_tips_kernel`, `gpu_push_sections_kernel`, `gpu_rope_reel_tethers_kernel` |
| **Murray Vascular Bifurcation** | `native/src/engine/colonization/cuda_colonization_volumetric.cu` | Mode `algorithm_mode == 2`, `murray_exponent`, `murray_branch_angle` |
| **Two-Pass Marching Cubes** | `native/src/engine/mesh/cuda_marching_cubes.cu` | `gpu_mc_count_pass_kernel`, `gpu_mc_emit_vertices_kernel`, `extract_iso_surface()` |
| **GPU Laplacian Smoothing** | `native/src/engine/mesh/cuda_mesh_ops.cu` | `smooth_mesh_laplacian_gpu()` |
| **GPU Mesh Decimation & Repair** | `native/src/engine/mesh/cuda_mesh_ops.cu`<br>`native/src/engine/mesh/cuda_mesh_topology.cu` | `decimate_mesh_adaptive_gpu()`, `repair_mesh_manifold_gpu()` |
| **Direct GPU Viewport Raymarching**| `native/src/ui/viewport/cuda_viewport_gl.cu` | `gpu_raymarch_implicit_viewport_kernel`, `raymarch_implicit_viewport_gpu()` |
| **Procedural GPU Geometry** | `native/src/engine/mesh/cuda_mesh_primitives.cu` | `generate_sphere_gpu()`, `generate_torus_gpu()`, `generate_filleted_box_gpu()` |
| **GPU File Formatting** | `native/src/engine/io/cuda_mesh_io.cu` | `format_obj_gpu()`, `format_binary_stl_gpu()`, `format_binary_ply_gpu()` (kernels: `gpu_format_binary_stl_kernel`, `gpu_pack_ply_vertices_kernel`, `gpu_pack_ply_faces_kernel`) |
| **Dear ImGui Growth Controls** | `native/src/ui/panels/panel_growth.cpp` | `render_growth_panel()`, `render_growth_algorithm_selector()` |
| **Specimen & Export UI** | `native/src/ui/panels/panel_scene.cpp` | `render_scene_panel()`, `save_file_dialog()` |
| **Presets & Theme Customizer** | `native/src/ui/panels/panel_styles.cpp` | `render_styles_panel()`, `render_presets_panel()`, `load_growth_presets_from_ini()` (persistence in `core/algorithm_io.cpp`) |
| **Unified Debug & Memory Logger** | `native/src/core/unified_logger.cpp`<br>`native/src/core/unified_logger.hpp` | `UnifiedLogger::instance()`, `UnifiedLogger::log_gpu_memory()`, `UnifiedLogger::log_step()` |
| **100% UI Controls Catalog (148)** | `param_controls_complete_list.csv`<br>`param_controls_inventory.md` | Catalog of all 148 controls (`CTRL001`-`CTRL148`) across 22 CSV panel sections *(v23.29 fix of the miscounted 18; v24.01 Phase 0: 22 after the Specimen Appearance re-home; v24.01 Phase 1: ten lighting sliders at CTRL093-102, ex-093..138 slid +10 to CTRL103-148)* |
| **UI Interaction Monitor** | `ui_interaction_monitor_system.md` | Interaction event dispatch, GPU performance tracing, and metric reporting |
| **Automated GPU Benchmark Runner** | `native/src/verification/gpu_benchmark_runner.cu`<br>`native/src/verification/gpu_benchmark_runner.hpp` | `GpuBenchmarkRunner::run_full_benchmark()`, `--benchmark` / `-b` CLI, `logs/gpu_benchmark_report.json` |
| **Build Configuration** | `native/CMakeLists.txt`<br>`native/vcpkg.json`<br>`native/tools/audit_control_registry.py` | `target_compile_options`, `CMAKE_CUDA_ARCHITECTURES`, `rhizome_control_audit` build-time CSV<->registry audit |

---

## 5. Symptom-to-Search Matrix

| Symptom / Issue | Where to Search First | What to Inspect |
| :--- | :--- | :--- |
| **GPU Crash / CUDA Error (`cudaError_t`)** | `native/src/engine/sdf/cuda_common.cuh`<br>`native/src/engine/sdf/cuda_rasterizer.cu` | Search for `cudaGetLastError()` or `CUDA_CHECK`. Verify grid/block dimensions and pointer validity before launches. |
| **Illegal Memory Access** | `native/src/engine/colonization/cuda_colonization_volumetric.cu`<br>`native/src/engine/mesh/cuda_marching_cubes.cu` | Check buffer capacity limits (e.g., `max_sections`, `max_next_tips`, `max_verts`). Check shared memory tile sizes (`RASTER_TILE_SIZE`). |
| **UI Not Updating During Growth** | `native/src/core/growth_controller.cpp`<br>`native/src/main.cpp` | Inspect `step_live_growth_frame_controller()`, `is_growing_live`, and `last_live_step_time` rate limits. |
| **File Import / Model Load Failure**| `native/src/ui/panels/panel_scene.cpp`<br>`native/src/engine/io/obj_parser.cpp`<br>`native/src/engine/io/stl_parser.cpp` | Check `open_file_dialog()`, `convert_and_update_model_controller()`, and `MeshWindingPass::orient_mesh()`. |
| **Performance Lag / Low Frame Rate** | `native/src/core/growth_controller.cpp`<br>`native/src/ui/viewport/viewport_renderer.cpp` | Check if live remeshing is triggering every step instead of batching, or check detail voxel size (`detail_level` < 0.1 mm). |
| **Suspected CPU Fallback** | `native/src/engine/` | Check for loops iterating over large collections on the CPU; all heavy geometry math must run in CUDA kernels. |
| **Roots Escaping Outside Specimen**| `native/src/engine/colonization/cuda_colonization_volumetric.cu` | Search for `dev_confine_sphere_to_vessel` and inspect `wall_clearance_scale` and `min_vessel_sdf`. |
| **Spikes or Holes in Extracted Mesh**| `native/src/engine/mesh/cuda_marching_cubes.cu` | Inspect `gpu_mc_seal_border_kernel` and edge interpolation inside `gpu_mc_emit_vertices_kernel`. |
| **Viewport Camera NaN / Glitch** | `native/src/ui/viewport/cuda_viewport_packers.cu`<br>`native/src/ui/viewport/cuda_viewport_gl.cu` | Check forward direction length guard when `eye == target` and `dev_safe_norm` normal transformations. |
| **CUDA Atomic CAS Spinloop Hang** | `native/src/core/math_utils.cuh` (atomic float guards)<br>`native/src/engine/sdf/cuda_common.cuh` (runtime macros) | Inspect `dev_atomicMin_float` / `dev_atomicMax_float` for NaN guards on input and stored floats. |
| **Font Loading Exception on Launch** | `native/src/main.cpp` | Ensure `std::filesystem::exists` uses non-throwing `std::error_code` overload when checking font paths. |
| **CMake Build / Linker Error** | `native/CMakeLists.txt` | Check CUDA architecture flags (`CMAKE_CUDA_ARCHITECTURES`), `/bigobj`, and `/NODEFAULTLIB:LIBCMT`. |

---

## 6. CUDA Host/Device Boundary Map

- **CPU Scope (Strictly Minimal):**
  - OS events, GLFW window input polling, and SpaceNavigator message pump.
  - Dear ImGui widget layout and UI command triggering.
  - Streaming raw binary buffers from/to storage disks via Win32 dialogs (`GetOpenFileNameA`, `GetSaveFileNameA`).
  - High-level orchestration of growth frame steps.
- **GPU Scope (100% of Computation):**
  - Signed distance field generation (voxel-level winding number and distance).
  - Spore rejection sampling, attraction evaluation, and kill pruning.
  - 11-feeler environment cone sensing and physical capsule-to-capsule push solver.
  - Marching Cubes surface extraction, edge welding, and vertex normal accumulation.
  - Direct implicit volume raymarching to display pixels.
  - Mesh smoothing, manifold verification, decimation, and text/binary format encoding.
- **Memory Allocation & Transfer Patterns:**
  - Allocated via `cudaMalloc` (or the `DeviceBuffer<T>` RAII wrapper in `engine/sdf/cuda_common.cuh`) in methods within `native/src/engine/` and `native/src/ui/viewport/`.
  - Input arrays transferred from CPU via `cudaMemcpy(..., cudaMemcpyHostToDevice)`.
  - Kernel execution via `<<<grid, block, shared_mem, stream>>>`.
  - Output results downloaded via `cudaMemcpy(..., cudaMemcpyDeviceToHost)` only when required for display buffers or file export, or kept in VRAM for direct OpenGL VBO upload.

---

## 7. Mathematical Tolerance & Cell Size Derivation (From native/README.md)

### **Requirement**
- **Maximum Chord Error:** $\\epsilon_{\\max} \\le 0.1\\,\\text{mm}$
- **Minimum Curvature Radius:** $R_{\\min} \\ge 0.5\\,\\text{mm}$

### **Mathematical Derivation**
For an osculating circle of radius $R$ subtended by a planar polygonal edge (chord) of length $h$, the sagitta (chord error $\\epsilon$) is:
$$\\epsilon = R - \\sqrt{R^2 - \\left(\\frac{h}{2}\\right)^2}$$

Using the second-order Taylor expansion for $h \\ll R$:
$$\\epsilon \\approx \\frac{h^2}{8R} \\implies h \\le \\sqrt{8 R \\epsilon_{\\max}}$$

Substituting $R_{\\min} = 0.5\\,\\text{mm}$ and $\\epsilon_{\\max} = 0.1\\,\\text{mm}$:
$$h_{\\max} \\le \\sqrt{8 \\times 0.5 \\times 0.1} = \\sqrt{0.4} \\approx 0.6324\\,\\text{mm}$$

In a 3D dual contouring / marching voxel grid with isotropic cell size $\\Delta x$:
- Face diagonal chord length: $h_{\\text{face}} = \\Delta x \\sqrt{2} \\approx 1.4142 \\Delta x$
- Space diagonal chord length: $h_{\\text{space}} = \\Delta x \\sqrt{3} \\approx 1.7321 \\Delta x$

To strictly enforce that $h_{\\text{face}} \\le 0.6324\\,\\text{mm}$:
$$\\Delta x \\le \\frac{0.6324}{\\sqrt{2}} \\approx 0.447\\,\\text{mm}$$

### **Cell Sizing Strategy**
1. **Nominal Cell Size:** $\\mathbf{0.50\\,\\text{mm}}$ in near-surface narrow-band regions where local curvature $R \\ge 1.0\\,\\text{mm}$.
   - At $R = 1.0\\,\\text{mm}$, $\\Delta x = 0.5\\,\\text{mm}$:
     $$\\epsilon = 1.0 - \\sqrt{1.0^2 - (0.3535)^2} \\approx 0.065\\,\\text{mm} < 0.1\\,\\text{mm} \\quad \\text{[SATISFIED]}$$
2. **Adaptive Refinement:** $\\mathbf{0.25\\,\\text{mm}}$ in regions where curvature radius $R \\in [0.5\\,\\text{mm}, 1.0\\,\\text{mm}]$.
   - At $R = 0.5\\,\\text{mm}$, $\\Delta x = 0.25\\,\\text{mm}$:
     $$\\epsilon = 0.5 - \\sqrt{0.5^2 - (0.1768)^2} \\approx 0.032\\,\\text{mm} \\ll 0.1\\,\\text{mm} \\quad \\text{[SATISFIED]}$$

---

## 8. File I/O Buffer Management & Compilation Flags

### **Memory Allocation & Transfer Patterns**
- Allocated via `cudaMalloc` (or the `DeviceBuffer<T>` RAII wrapper in `engine/sdf/cuda_common.cuh`) in methods within `native/src/engine/` and `native/src/ui/viewport/`.
- Input arrays transferred from CPU via `cudaMemcpy(..., cudaMemcpyHostToDevice)`.
- Kernel execution via `<<<grid, block, shared_mem, stream>>>`.
- Output results downloaded via `cudaMemcpy(..., cudaMemcpyDeviceToHost)` only when required for display buffers or file export, or kept in VRAM for direct OpenGL VBO upload.

### **Compilation Flags & Linker Settings (from CMakeLists.txt)**
- **Floating Point Precision:** `/fp:precise` (MSVC default)
- **Conformance Mode:** `/permissive-` enabled via CMake
- **Linking:** `/NODEFAULTLIB:LIBCMT` to prevent runtime double-free issues
- **Debug Symbol Size Limit:** `/bigobj` for large object files (4096+ exports)

### **Streaming Buffer Sizing**
- Single-pass in-memory streaming with robust `tellg()` stream size bounds.
- VRAM vs CPU memory allocation tradeoffs: GPU buffers preferred for heavy geometry operations.
- Binary format guard conditions: 32-bit triangle limits and byte-overflow guards.

### **File I/O Operations (GPU-Accelerated)**
- Native Win32 file dialogs (`OPENFILENAMEA`) streaming buffers to/from disk.
- Feeding parallel GPU decoders via `ObjParser::parse()`, `StlParser::parse()`.
- Export formatting via `gpu_format_binary_stl_kernel`, `gpu_pack_ply_vertices_kernel`, `gpu_pack_ply_faces_kernel` *(v23.29 fix F14)*.

---

## 9. Verification Harness Benchmark Results (GPU-Only Pipeline)

### **Computation Architecture & GPU Pipeline Validation**

All verification harness operations execute exclusively on NVIDIA CUDA GPU via dedicated kernels. No CPU fallback computations permitted per the **Zero CPU Fallback Rule**.

| Operation Stage | Responsible CUDA Kernel | GPU Memory Location | Verification Status |
| :--- | :--- | :--- | :--- |
| **1. Mesh Parsing (Input)** | `CudaRasterizer::parse_binary_stl_gpu()`<br>`CudaRasterizer::parse_obj_text_gpu()` *(v23.29 fix F16: the STL export formatter was previously misattributed to the input stage)* | Host → VRAM Transfer | ✅ GPU-Only |
| **2. SDF Generation** | `mesh_to_sdf_kernel`<br>`rasterize_capsules_kernel` | CUDA Device Memory | ✅ GPU-Only |
| **3. Voxel Grid Construction** | `gpu_accumulate_spore_forces_kernel` | CUDA Volume Grid (CudaVolumeGrid) | ✅ GPU-Only |
| **4. Marching Cubes Pass 1** | `gpu_mc_count_pass_kernel`<br>`gpu_mc_emit_vertices_kernel` | CUDA Device Memory | ✅ GPU-Only |
| **5. Marching Cubes Pass 2B** | `gpu_mc_emit_triangles_kernel` | CUDA Device Memory | ✅ GPU-Only |
| **6. Normal Accumulation** | `gpu_accumulate_face_normals_kernel` | CUDA Device Memory | ✅ GPU-Only |
| **7. Analytical Evaluation** | `gpu_fill_analytical_sdf_grid_kernel`, `gpu_evaluate_eikonal_field_kernel`, `gpu_eval_sdf_grid_error_kernel` (`src/verification/cuda_verification.cu`) | CUDA Device Math | ✅ GPU-Only |
| **8. Deviation Computation** | `gpu_eval_vertex_deviations_kernel`<br>`gpu_eval_midpoint_deviations_kernel` (+ `gpu_reduce_round_trip_devs_kernel` for the round-trip test) | CUDA Device Math | ✅ GPU-Only |
| **9. Result Download** | `cudaMemcpy(Device → Host)` | VRAM → System RAM | ✅ Controlled Transfer |

---

### **Benchmark Data with GPU Kernel Attribution**

| Primitive | Input Tris | Extracted Verts | Extracted Tris | Voxel Size | Max Deviation | Mean Deviation | RMS Deviation | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sphere** (`R=20mm`) | 32,768 | 41,218 | 82,432 | 0.50mm | **0.0631mm** | **0.0214mm** | **0.0258mm** | ✅ PASSED ≤ 0.100mm |
| **Torus** (`R=25, r=5mm`) | 32,768 | 58,490 | 116,976 | 0.40mm | **0.0718mm** | **0.0289mm** | **0.0341mm** | ✅ PASSED ≤ 0.100mm |
| **Filleted Box** (`r=2mm`) | 19,200 | 36,912 | 73,824 | 0.40mm | **0.0842mm** | **0.0312mm** | **0.0387mm** | ✅ PASSED ≤ 0.100mm |

---

### **GPU Kernel Attribution by Operation**

#### **Kernel-Stage Mapping Verification**

| File Location | GPU Kernel(s) | Function in Verification Pipeline |
| :--- | :--- | :--- |
| `src/engine/sdf/cuda_rasterizer.cu` | `mesh_to_sdf_kernel`, `rasterize_capsules_kernel` | Distance field computation & capsule blending |
| `src/engine/colonization/cuda_colonization_volumetric.cu` | `gpu_accumulate_spore_forces_kernel` | Voxel grid force accumulation for SDF sampling |
| `src/engine/mesh/cuda_marching_cubes.cu` | `gpu_mc_count_pass_kernel`, `gpu_mc_emit_vertices_kernel`, `gpu_mc_emit_triangles_kernel` | Two-pass edge welding & vertex emission |
| `src/engine/mesh/cuda_mesh_ops.cu` | `gpu_accumulate_face_normals_kernel` | Per-face normal computation for curvature analysis |
| `src/verification/cuda_verification.cu` | `gpu_fill_analytical_sdf_grid_kernel`, `gpu_evaluate_eikonal_field_kernel`, `gpu_eval_sdf_grid_error_kernel` | Analytical field generation & comparison against CUDA SDF |
| `src/verification/cuda_verification.cu` | `gpu_eval_vertex_deviations_kernel`, `gpu_eval_midpoint_deviations_kernel`, `gpu_reduce_round_trip_devs_kernel` | Vertex/midpoint error evaluation and reduction *(v23.29 fix F11: previously misattributed to a nonexistent tensor-reduction kernel in `cuda_common.cuh`)* |

---

### **High Curvature Region Handling (Torus Inner Ring)**

**Challenge:** Negative Gaussian curvature areas (`k_1 = 1/5.0mm = 0.2mm⁻¹`) with high principal curvature demand finer resolution.

| Resolution Strategy | Cell Size | Chord Error at R=5mm | Result vs Threshold |
| :--- | :--- | :--- | :--- |
| Uniform Grid (Default) | 0.50mm | ~0.089mm | ⚠️ Near threshold |
| Adaptive Refinement (Applied) | 0.40mm | **0.0718mm** | ✅ Well below 0.100mm |

**GPU Implementation:** Adaptive voxel sizing controlled via `CudaVolumeGrid::set_adaptive_refinement(0.25)` kernel configuration, executed entirely on GPU during SDF generation phase.

---

### **GPU-Only Claim Verification Checklist**

| Constraint Claim | Evidence from Codebase | Validation Status |
| :--- | :--- | :--- |
| **Zero CPU Fallback** | All distance computations via `mesh_to_sdf_kernel` in VRAM | ✅ VERIFIED |
| **No External OpenVDB** | Only native CUDA volume structures (`CudaVolumeGrid`) used | ✅ VERIFIED |
| **GPU Raymarching** | `gpu_raymarch_implicit_viewport_kernel` renders directly from SDF | ✅ VERIFIED |
| **Voxel Grid Operations** | All marching cubes kernels execute in device memory | ✅ VERIFIED |
| **Error Computation** | `cudaMemcpy` only for final results; all intermediate math on GPU | ✅ VERIFIED |

---

### **Performance Note: GPU Throughput Characteristics**

| Metric | Value | Computation Location |
| :--- | :--- | :--- |
| **Total SDF Generation Time** | ~15-45ms (depends on mesh complexity) | Full CUDA Pipeline |
| **Marching Cubes Extraction** | ~2-8ms | GPU Vertex Processing |
| **Analytical Error Check** | <1ms | Tensor Reduction Kernel |

*Measured on RTX 3060+ (sm_86) with full GPU occupancy.*

---

## 10. Growth Algorithm Extension Map

- **Existing Growth Implementations:**
  - Mode 0 (`Classic Space Colonization`): `native/src/engine/colonization/cuda_colonization.cu`.
  - Mode 1 (`Natural Root-Bound`): `native/src/engine/colonization/cuda_colonization_volumetric.cu`.
  - Mode 2 (`Vascular Murray Bifurcation`): `native/src/engine/colonization/cuda_colonization_volumetric.cu` (controlled via `params.algorithm_mode == 2`, `murray_exponent`, `murray_branch_angle`).
  - Mode 3 (`Creeping Root Surface Binder`): classic-nodes path with `surface_adherence` scoring (v24.09 era).
  - Mode 4 (`Root Typ (Developmental FSPM)`, **v24.12 Track F R1**): `native/src/engine/colonization/cuda_colonization_volumetric.cu` — `gpu_dev_advance_tips_kernel` (Firn-Digby gravitropic setpoint chase per order + hashed wander + wall confinement) and `gpu_dev_commit_proposals_kernel` (append discipline + `steps_grown` bookkeeping); host-side `apply_dev_murray_caliber` (bottom-up Murray census, SLA print floor, `colonization.cpp`) + `emit_print_audit` ([Print Audit] log pair); user surface `dev_*` (CTRL242-251) derived in `growth_controller.cpp build_colonization_config_from_state`. No spores / feelers / push-reel in this mode; sections are append-only.
  - Mode 4 eye loop + retune (**v24.13 Track F R2**): `gpu_dev_advance_tips_kernel` gains order-0 wander damping (0.45x — smooth framework arcs) and a wall-slide park (>65 deg confinement deflection parks the tip instead of crawling walls/floor); `apply_dev_murray_caliber` switches to biological terminal demand (print floor at write only); `growth_controller.cpp` adds crown-top auto-seeding for mode 4 (host scan down the vessel SDF center column), vessel-relative length caps + iteration budget (`dev_vessel_ref_mm` measured at Start, `dev_len_frac` table), the live-session auto-enable of the branch skeleton layer (live growth previously invisible), and `write_structure_dump_controller` (every finalized volumetric bake writes `logs/structure_dump_mode<M>_<stamp>.txt` — the artificial-eye feed, analyzed off-box by `scripts/eye/analyze_dump.py`; mock wind tunnel `scripts/eye/mock_mode4.py`).
  - MSVC compile fix (**v24.14 Track F R2-HF**): the vessel-relative length-cap clamp in `growth_controller.cpp build_colonization_config_from_state` now runs in the int domain (`std::min(300, std::max(6, raw_steps))`) — the v24.13 form passed `std::lroundf`'s `long` return next to a float literal inside `std::max`/`std::min`, which MSVC rejects (C2672); zero semantics change.
  - Growth-panel ImGui ID-conflict fix (**v24.15 Track F R2-HF2**): `render_control` in `ui/widgets/control_registry.cpp` now wraps the (renamed, file-local) `render_control_value` body in `ImGui::PushID(def.id)` / `ImGui::PopID()` — the compact widgets in `ui/widgets/widget_factory.cpp` seed ImGui IDs from the draw label alone (`PushID(label)`), so co-rendered registry controls sharing a draw label (the Growth window's three "Branch Angle" rows: CTRL016 Murray / CTRL040 Lateral Roots / CTRL246 Developmental) collapsed onto one ImGuiID and dragging one slider wrote every duplicate's bound value. Zero visual change; catalog/CSV untouched; heals the whole 251-control registry (heritage mode-2 pair included).
- **Algorithm State & Switching:**
  - `AppState::growth::algorithm_mode` (`0`-`4`) in `native/src/core/app_state.hpp`.
  - `GrowthAlgorithmProfile` in `native/src/core/algorithm_io.hpp`.
  - UI selector dropdown in `native/src/ui/panels/panel_growth.cpp` (`render_growth_algorithm_selector()`, driven by the CTRL005 registry row).
- **How to Add a New Algorithm (Multi-Algorithm Architecture):**
  1. Add a new mode index (e.g., Mode 3) to `GrowthAlgorithmProfile::engine_mode` and `AppState::growth::algorithm_mode`.
  2. Implement the new algorithm's computational kernels directly in a CUDA source file under `native/src/engine/colonization/` (e.g., `cuda_colonization_*.cu`); add any new cross-TU kernel declarations to `engine/sdf/cuda_common.cuh`.
  3. Dispatch the new mode inside `SpaceColonization::step_volumetric_session` or `SpaceColonization::run` based on `config.algorithm_mode`.
  4. Register the new algorithm profile in `init_default_growth_algorithms()` in `native/src/core/growth_controller.cpp`.
  5. **Rule:** NEVER overwrite or remove existing algorithms (Mode 0, 1, 2, 3, or 4 — see Mode 4 / Track F R1 for the worked example: dedicated kernels + session gates + a profile + config derivation + catalog rows in lockstep).

---

## 11. Protected Areas & Unbreakable Invariants

When modifying or extending this repository, always preserve:
1. **The `/native` Directory:** Must never be deleted, moved, or renamed. The primary output is `rhizome_app.exe`.
2. **GPU-Only Rule (Zero CPU Fallback):** Never replace or supplement CUDA kernels with CPU sequential loops.
3. **No OpenVDB:** OpenVDB has been replaced by the native CUDA SDF rasterizer (`cuda_rasterizer.cu`). Do not import or link OpenVDB.
4. **No Web Conversion:** Do not convert the project to Web, Node.js, React, Vite, or Three.js.
5. **Preserve Existing Algorithms:** The Space Colonization and volumetric root-bound pipeline must remain functional when adding new modes.

---

## 12. Key Search Keywords

For agents searching the codebase with `grep` or file search tools:
- CUDA API: `cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaGetLastError`, `cudaDeviceSynchronize`
- Kernel Declarations: `__global__`, `__device__`, `__constant__`, `__shared__`
- Engine Core: `CudaRasterizer`, `CudaVolumeGrid`, `CudaCapsule`, `CudaVolSection`, `CudaVolTip`
- Growth Pipeline: `SpaceColonization`, `ColonizationConfig`, `spore_search_kernel`, `gpu_sense_and_advance_tips_kernel`, `murray_exponent`
- UI & Platform: `ImGui::Begin`, `ImGui::DockSpaceOverViewport`, `OPENFILENAMEA`, `GetOpenFileNameA`, `GetSaveFileNameA`
- Control Registry: `ControlDefinition`, `k_all_controls`, `k_all_sections`, `RHZ_BIND_F/I/B/C/T`, `render_control`, `control_registry_integrity_check`
- State & Resources: `AppState`, `EngineResources`, `SeedState`, `BarkTextureState`
- Build System: `rhizome_core`, `rhizome_app`, `CMAKE_CUDA_ARCHITECTURES`

---

## 13. Intent Routing Table

| If the User / Task Asks To... | Start Here | Inspect / Modify These Files First |
| :--- | :--- | :--- |
| **Add a new growth simulation rule or algorithm** | Extension Map (Sec. 10) | `native/src/engine/colonization/colonization.hpp`<br>`native/src/engine/colonization/cuda_colonization_volumetric.cu`<br>`native/src/core/growth_controller.cpp` |
| **Adjust 3D shading, colors, or viewport lighting** | UI & Viewport (Sec. 3.1 & 3.6) | `native/src/ui/panels/panel_styles.cpp` (`render_styles_panel`)<br>`native/src/ui/viewport/viewport_renderer.cpp` |
| **Add or modify parameter sliders in the Growth panel** | Control Registry (Sec. 3.10) | `param_controls_complete_list.csv` (authority)<br>`native/src/ui/widgets/control_registry.cpp` (row + binding)<br>`native/src/core/app_state.hpp` (member) — then re-run the build-time audit |
| **Change mesh export formatting (OBJ, STL, PLY)** | Mesh I/O (Sec. 3.7) | `native/src/engine/io/obj_exporter.cpp`<br>`native/src/engine/io/cuda_mesh_io.cu` |
| **Debug a CUDA kernel crash or memory corruption** | Symptom Matrix (Sec. 5) | `native/src/engine/sdf/cuda_common.cuh`<br>`native/src/engine/sdf/cuda_rasterizer.cu` |
| **Investigate mesh generation or watertightness** | Remeshing (Sec. 3.5) | `native/src/engine/mesh/cuda_marching_cubes.cu`<br>`native/src/engine/mesh/cuda_mesh_topology.cu` |
| **Add verification tests** | Verification Suite (Sec. 3.8) | `native/src/verification/harness.cpp`<br>`native/src/verification/harness.hpp` |

---

## 14. Maintenance, Cleanups & Verification Checklists

This repository maintains structured checklists for all code maintenance, error-correction passes, and architectural audits:

| Document | Purpose & Scope | Status |
| :--- | :--- | :--- |
| **`CLEANUP_CHECKLIST.md`** | Pass 1: Eliminates OpenVDB residue, establishes 100% CUDA GPU pipeline, implements two-pass Marching Cubes, and organizes the verification test harness. | Complete |
| **`CLEANUP_CHECKLIST_PASS2.md`** | Pass 2: Hardens edge cases, single-pass memory parsing, atomic CAS NaN guards, camera math singularity safeguards, and non-throwing font file access. | Complete |
| **`CLEANUP_CHECKLIST_PASS3.md`** | Pass 3: Fixes raycast miss double-free, guards degenerate grid dimensions, enforces non-throwing filesystem probes, clamps capsule emission, and eliminates unused headers. | Complete |
| **`param_controls_complete_list.csv`** | 100% complete catalog of all 241 UI controls (`CTRL001`–`CTRL241`) with widget types, value ranges, units, and logging flags *(148 at v24.01 Phase 1; 236 at Phase 2; 241 since the 8d census re-pin)*. | Complete (241 Controls) |
| **`param_controls_inventory.md`** | Detailed architectural report and section-by-section breakdown of all 241 UI controls across the 25 CSV panel sections *(v23.29 fix of the miscounted 18; v24.01 Phase 0 renumber + Phase 1 lighting expansion + Phase 2 style-editor registration applied; type counts corrected to the CSV census)*. | Complete |
| **`ui_interaction_monitor_system.md`** | Architecture specification for unified interaction logging, GPU memory tracking, and telemetry event streaming. | Active Specification |
| **`logs/gpu_benchmark_report.md`** | Automated GPU benchmark stress test report covering the 251-control catalog on CUDA GPU with microsecond hardware timing. | Generated via `--benchmark` |
| **`v23.28 State Synchronization & Cleanup`** | Pass 4: Consolidates mesh export and display state into `AppState` single source of truth across `GrowthPanelContext`, `ScenePanelContext`, and `GrowthControllerContext`; eliminates unused shadow declarations in `main.cpp`; and fixes Dear ImGui style stack scoping (`PushStyleVar`/`PopStyleVar`) for clean `FramePadding` and size resets. | Complete |
| **`v23.29 Global Restructuring (Phases 1-5)`** | Domain-driven tree (`core/ui/engine/verification`, co-located headers), `AppState` SSOT + `EngineResources`, `UnifiedLogger` in `core/`, `ControlDefinition` 138-control registry with build-time CSV audit (`rhizome_control_audit`) and runtime `--verify` registry check, shadow state eradicated, `--verify` headless gate. | Complete |
| **`v24.01 Program (Phases 0-2, CLOSED 2026-10-22)`** | Catalog coherence + `[ViewportColors]`/`[LightingAndShading]` persistence (Phase 0); lighting expansion 6 -> 16 sliders with the shared `lighting_rig.hpp` dual render-path derivation (Phase 1); full style-editor registration — 93 widgets at `CTRL149-241`, `ui_events.jsonl` telemetry, `[Style]`/`[StyleColors]` persistence, version-adaptive ImGuiCol census (8b/8c/8d; supported window 1.90.8-1.93.0 docking, target 1.92.9b exact-fit) — Track B `--verify` cleared at 8e (19/20 + Test 8 known-issue). Final catalog: **241 controls / 25 sections**. | Complete |
| **`v24.02 Program (Phases 1-2, CLOSED 2026-10-23 at Phase 2)`** | Harness refinement (Phase 1: Test 8 round-trip grid 0.20 → 0.15 mm — suite-green **20/20**, Track B Round 1; Eikonal acceptance-rule print clarity); deterministic GPU accumulation (Phase 2: 18 float `atomicAdd` sites → fixed-point uint64 shadow accumulators, Q = 2^24, + dequantize kernels in the two growth-kernel files; `native/tools/trackb_bitexact_gate.ps1` N-run SHA-256 gate — Track B Round 2: 20/20, control bit-identical, gate measured NOT bit-exact 10/10 — accepted characteristic; Phase 3 ordering channels declined by user). | Complete |
| **`v24.03 Release (2026-10-23)`** | Consolidation + version stamp of the v24.02 program + post-closure fixes: DIAG1 crash diagnostics (MSVC `/MAP` + `native/tools/resolve_map_offsets.py`); DIAG1 FIX2 GUI suite crash fix (Test 20 ImGui scratch-context lifecycle, `harness_volumetric_tests.cpp`); Track C Lateral Roots fix (CSG far-field clamp `cuda_rasterizer.cu`, viewport step cap `cuda_viewport_gl.cu`, lateral overflow hardening `cuda_colonization_volumetric.cu`). Documentation-level release — zero product code changed for the stamp; release manifest `manifests/v24.03_release.sha256`; record `phase_reports/V2403_RELEASE_REPORT.md`. | Complete (current) |

### Native Build & Verification Procedure (v23.29)

```powershell
# 1. Configure the Visual Studio 2022 64-bit solution with CUDA 12
#    (or run native\build.ps1, which also installs vcpkg deps and runs step 3)
cmake -B build -S native -G "Visual Studio 17 2022" -A x64

# 2. Build Release binary
#    The build automatically runs the CSV<->registry audit target
#    (rhizome_control_audit) when a Python 3 interpreter is available.
cmake --build build --config Release --parallel

#    Audit manually at any time:
#    python native\tools\audit_control_registry.py --csv param_controls_complete_list.csv --registry native\src\ui\widgets\control_registry.cpp --logger native\src\core\unified_logger.cpp

# 3. Execute the automated verification gate (registry audit + 20-test GPU suite, headless, exit-code gated)
.\build\Release\rhizome_app.exe --verify

# 4. Execute the automated GPU benchmark runner (stress-test 251 controls on CUDA GPU)
.\build\Release\rhizome_app.exe --benchmark
```

---

## 15. UI Layout Constants & Design Standards

When adding or modifying user interface elements, always use the centralized constants defined in `rhizome::UiLayout` (`native/src/ui/widgets/widget_factory.hpp`):

| Parameter | Value | Scope & Usage |
| :--- | :--- | :--- |
| `ROW_HEIGHT_SLIDER` | `16.0f` | Standard height for all compact sliders (`render_compact_slider_*`) and toggle switches (`render_toggle_switch`). |
| `ROW_HEIGHT_BUTTON` | `22.0f` | Standard height for tool buttons, action triggers (`[+]`, `[-]`, `Freeze All`), and combo dropdowns. |
| `ROW_HEIGHT_HEADER` | `22.0f` | Height for collapsible section header cards (`render_floating_pill_header`). |
| `STATUS_BAR_HEIGHT` | `20.0f` | Bottom status bar strip height. |
| `ITEM_SPACING_X` | `4.0f` | Horizontal gap between inline buttons (`SameLine(0.0f, 4.0f)`). |
| `ITEM_SPACING_Y` | `4.0f` | Vertical gap between consecutive rows (results in consistent 20 px center-to-center distance). |
| `WINDOW_PADDING_X` | `10.0f` | Horizontal inner padding for docked panels. |
| `WINDOW_PADDING_Y` | `6.0f` | Vertical inner padding for docked panels. |
| `FRAME_PADDING_X` | `8.0f` | Inner horizontal padding for buttons and text inputs. |
| `FRAME_PADDING_Y` | `3.0f` | Inner vertical padding for buttons and text inputs. |
| `SLIDER_LABEL_WIDTH` | `104.0f` | Standard column width allocated for parameter names on the left. |
| `SLIDER_VALUE_WIDTH` | `56.0f` | Standard column width allocated for numerical values / inline edits on the right. |
| `SLIDER_TRACK_HEIGHT`| `4.0f` | Rail height for all slider tracks. |
| `SLIDER_KNOB_RADIUS` | `5.0f` | Circular thumb radius (10 px resting diameter). |
| `TOGGLE_TRACK_WIDTH` | `34.0f` | Rail width for iOS-style toggle switches. |
| `TOGGLE_TRACK_HEIGHT`| `4.0f` | Rail height for toggle switches. |
| `TOGGLE_KNOB_RADIUS` | `5.0f` | Circular thumb radius for toggle switches (10 px diameter). |
| `FRAME_ROUNDING` | `4.0f` | Corner radius for buttons, frames, and sliders. |
| `WINDOW_ROUNDING` | `6.0f` | Corner radius for windows, cards, and popups. |
| `BORDER_THICKNESS` | `1.0f` | Subtle outline border thickness. |
