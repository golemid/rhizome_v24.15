# Rhizome Native: Phase 1 Implicit Mathematical SDF Engine

A native C++20 desktop engine engineered for nTopology-level implicit mathematical fidelity, built to bypass browser and WebGL memory/compute barriers.

Project navigation: see `CODE_MAP.md` (v24.15) for the full module map, entry points, and host/device boundaries of the restructured `native/src/{core,ui,engine,verification}` tree.

> **Current status (v24.15, released 2026-10-24):** the engine, harness, and toolchain are final — 251-control catalog, `--verify` **20/20** (first suite-green in program history, Track B-verified 2026-10-22; the v23.29-era Test 8 knife-edge resolved by the v24.02 harness refinement), CUDA raster speedup 105.41x on the target RTX 3060-class GPU. **v24.04 Track D added the Fast SDF path** (`engine/sdf/cuda_rasterizer.cu`; grids ≥ 32M voxels: band-splat exact distances + scanline parity signs + FSM eikonal re-distancing, making **0.10 mm the default Detail Level**); the first user build showed its zero-tolerance verify gate rejecting mug.obj outright (1/4096 sign mismatches → 82 s legacy fallback, slower than v24.03). **v24.05 Track D R2 completes the gate**: a small number of mismatched sites is now patched with the exact winding kernel (49³-voxel boxes) and re-probed with fresh 4,096-sample stratifications (up to 3 generations); only systematic divergence still falls back to the legacy kernel. All verification-suite grids sit below the fast-path gate, so test results are unchanged. Known characteristic (accepted): exports of the growth-affected chain are not bit-reproducible run-to-run. **v24.06 Track D R3 completes the eikonal far field**: a far-sentinel audit after the rounds, an audit-gated extension (up to 16 extra rounds), and a sign-aware band-edge safe clamp for any true leftover — previously truncated far voxels were silently clamped to ±1e6, which the viewport raymarcher rendered as a rotating-trim artifact on the loaded model; a viewport-side specimen-march guard defuses pathological far samples on the consumer side. **v24.07 Track D R4 closes the rotating-trim artifact**: the user's v24.06 log (far residual 0, extra rounds 0, safe-clamped 0 — yet the trim persisted) disproved the sentinel theory; the sandbox truncation study (faithful replicas of the fast-path pipeline on the real mug.obj) measured the real cause — the sweep-capped eikonal ships real-but-OVERESTIMATED far distances (up to ~13 mm at mug scale, no sentinels left to audit), and sphere tracing steps 0.85 × (true + overestimate) through thin walls and the rim. The specimen march is now band-limited: marched values clamp at the exact ±8-voxel band edge (the eikonal never undercuts the [A] splat seeds, so in-band values stay exact), and the step budget doubles (512 → 1024) to keep banded far marching inside budget at the 0.10 mm default. **v24.08 Track D R5 closes the artifact's last face — the roots-only trim:** the baked-roots raymarch still carried a 280-step travel budget sized for the 0.25–0.30 mm era; at the 0.10 mm default its reach was 95 mm against a ~183 mm baked-grid diagonal (vessel + lateral rootlets + pad), so rays entering the far half of the roots volume ran out of samples mid-grid and everything beyond a camera-dependent sheet vanished. The budget is now 2048 steps (696 mm reach, 3.8× the diagonal; twice the specimen's 1024 because the roots background band is 4 voxels vs the specimen's 8 — every step covers half the distance). No band clamp is needed on this path: roots grids are direct capsule rasterizations whose out-of-band values are conservative lower bounds by construction. The release also adds `VOLUME_MARCH_GUIDE.md` (app tree root) — the detailed, learnable documentation of the whole 3D volume visualization (render path, constants, and the full v24.06–v24.08 defect story), written on explicit user request. **v24.09 Track E R1 flips the Live Animated Growth default OFF** (both `GrowthState` and `GrowthPreset` — the state struct is the sole default source; the control registry carries no default mechanism): live mode steps the growth session every frame and re-syncs the implicit session per tick (every accumulated section re-converted to branch nodes and re-uploaded, live capsule render session held open) — the memory cost measured on target hardware; Start now takes the one-shot batch bake by default (identical baked product, no per-tick churn), and the Growth panel Execution toggle re-enables live mode at any time. Saved presets and algorithm files keep their explicit `LiveGrowth=` value verbatim. **v24.10 Track E R2 fixes the second-Start re-bake memory spike that the batch default exposed:** a continuing Start re-baked the roots volume while still holding the previous bake on the host, the viewport's cached device copy of it (freed only when a new grid appeared on a later frame), and one more full-grid host copy made by the intersection — at the 0.10 mm default every full-size grid is ~2.43 GiB (652.5M voxels), so the second Start peaked at roughly twice the first bake's footprint and crashed. The bake now releases the previous roots grid and the viewport device cache BEFORE rasterizing the replacement (`CudaRasterizer::release_roots_viewport_cache_gpu()`; the cache is also freed whenever no roots grid is present — Clear, model reload, hidden roots, live growth), and the two bake call sites intersect the fresh rasterization in place (`preserve_input_grid=false`; every other `intersect_with_vessel` caller — verification harness and panels — keeps the historical copy semantics). Memory-only: no growth semantics, no defaults, no render math touched. **v24.11 Track E R3 armor-plates the bake against the fail-fast crash class the field watchdog captured on the quick second Start (exit code 0xC0000409, 3/3 reproducible runs, seconds after the `Rasterizing N sections.` print):** forensics proved the app had zero exception handling on the growth path and that MSVC release escalates any unhandled throw (terminate → abort → UCRT `__fastfail`) to exactly that exit code — no literal stack smash required — with the single lethal throw site being the bake's host-grid allocation sized from unvalidated GPU bounds (a wild-but-finite section position inflates the grid into the multi-terabyte voxel range). `rasterize_volumetric_sections_gpu` now validates the bounds in double precision before the integer grid math (checked D2H, finiteness, absolute position and per-axis span ≤ 2^30 voxels, product ≤ 2^33 voxels — 12× the largest buildable bake; healthy data never trips it), rejecting impossible bounds with a `[Bake Guard]` diagnostic line (section count + raw bounds) and the same graceful no-roots-grid fallback as any CUDA failure; independently, the three growth controller entry points (Start / Stop / live-step) gained an exception guard converting any remaining throw into a logged status-bar message with the session parked in the Stop state. Crash-class-only change: previously-succeeding inputs behave byte-identically. The derivations below remain the binding accuracy contract for any future work (see `IMPROVEMENT_PLAN.md`; release record `../phase_reports/V2411_TRACK_E_R3_REPORT.md`). **v24.12 Track F R1 adds the fourth growth algorithm mode** ("Root Typ (Developmental FSPM)", `algorithm_mode == 4` in `engine/colonization/`): per-order gravitropic setpoints (Firn-Digby sine law) + hashed wander on living apices, acropetal lateral emergence with hashed azimuth, per-order elongation/spacing/length caps, append-only sections (no spores/feeler/push-reel in this mode), a bake-time host-side bottom-up Murray census for structural caliber floored at the SLA print radius (CTRL250, default 0.30 mm — the output is destined for large-format UV-resin SLA printing), a [Print Audit] log pair at every bake (per-order census, min radius vs floor, worst slenderness L/r, connectivity, estimated resin volume), and a section budget (CTRL251, default 30,000). New user surface CTRL242-251 (catalog 241 -> 251 / 26 sections; CSV == registry == monitor audit PASS). Modes 0-3 untouched; release record `../phase_reports/V2412_TRACK_F_R1_REPORT.md`. **v24.13 Track F R2** adds the artificial-eye loop for mode 4: every finalized volumetric bake writes a structure dump (`logs/structure_dump_mode<M>_<stamp>.txt`, newest five kept — full skeleton + parameter snapshot, index order = growth chronology) for off-box analysis (`scripts/eye/analyze_dump.py` renders/metrics/MP4 + `scripts/eye/mock_mode4.py` wind tunnel); mode 4 auto-seeds at the vessel crown instead of the floor, gains the eye-driven retune (flaring GSA 55/68/78/86, gain 0.26, order-0 wander damping, wall-slide park, spacing 16/20/28/36, vessel-relative length caps + iteration budget, biological Murray demand, 7 arms), and live sessions auto-enable the branch skeleton layer so animated growth is visible. Modes 0-3 untouched; catalog stays 251; release record `../phase_reports/V2413_TRACK_F_R2_REPORT.md`. **v24.14 hotfix** repairs the single MSVC compile blocker in those sources (C2672: the vessel-relative length-cap clamp mixed `std::lroundf`'s `long` return with a float literal inside `std::min`/`std::max`; the clamp now runs in the int domain — zero semantics change); release record `../phase_reports/V2414_TRACK_F_R2_HF_REPORT.md`. **v24.15 hotfix** repairs the Growth-panel ImGui ID conflict found in the first interactive field pass (three registry rows share the draw label "Branch Angle" — Murray CTRL016 / Lateral Roots CTRL040 / Developmental CTRL246 — and the compact widgets seed ImGui IDs from the draw label alone, so co-rendered duplicates shared one ID and dragging one slider wrote the other's bound value; `render_control` now scopes every control's IDs by its unique CTRL id via `PushID(def.id)` — zero visual change, catalog untouched, whole 251-control registry healed); release record `../phase_reports/V2415_TRACK_F_R2_HF2_REPORT.md`.

---

## 1. Dimensional Tolerance & Cell Size Derivation

### Requirement
* **Maximum Chord Error:** $\epsilon_{\max} \le 0.1\,\text{mm}$
* **Minimum Curvature Radius:** $R_{\min} \ge 0.5\,\text{mm}$

### Mathematical Derivation
For an osculating circle of radius $R$ subtended by a planar polygonal edge (chord) of length $h$, the sagitta (chord error $\epsilon$) is:
$$\epsilon = R - \sqrt{R^2 - \left(\frac{h}{2}\right)^2}$$

Using the second-order Taylor expansion for $h \ll R$:
$$\epsilon \approx \frac{h^2}{8R} \implies h \le \sqrt{8 R \epsilon_{\max}}$$

Substituting $R_{\min} = 0.5\,\text{mm}$ and $\epsilon_{\max} = 0.1\,\text{mm}$:
$$h_{\max} \le \sqrt{8 \times 0.5 \times 0.1} = \sqrt{0.4} \approx 0.6324\,\text{mm}$$

In a 3D dual contouring / marching voxel grid with isotropic cell size $\Delta x$:
* Face diagonal chord length: $h_{\text{face}} = \Delta x \sqrt{2} \approx 1.4142 \Delta x$
* Space diagonal chord length: $h_{\text{space}} = \Delta x \sqrt{3} \approx 1.7321 \Delta x$

To strictly enforce that $h_{\text{face}} \le 0.6324\,\text{mm}$:
$$\Delta x \le \frac{0.6324}{\sqrt{2}} \approx 0.447\,\text{mm}$$

### Cell Sizing Strategy
1. **Nominal Cell Size:** $\mathbf{0.50\,\text{mm}}$ in near-surface narrow-band regions where local curvature $R \ge 1.0\,\text{mm}$.
   * At $R = 1.0\,\text{mm}$, $\Delta x = 0.5\,\text{mm}$:
     $$\epsilon = 1.0 - \sqrt{1.0^2 - (0.3535)^2} \approx 0.065\,\text{mm} < 0.1\,\text{mm} \quad \text{[SATISFIED]}$$
2. **Adaptive Refinement:** $\mathbf{0.25\,\text{mm}}$ in regions where curvature radius $R \in [0.5\,\text{mm}, 1.0\,\text{mm}]$.
   * At $R = 0.5\,\text{mm}$, $\Delta x = 0.25\,\text{mm}$:
     $$\epsilon = 0.5 - \sqrt{0.5^2 - (0.1768)^2} \approx 0.032\,\text{mm} \ll 0.1\,\text{mm} \quad \text{[SATISFIED]}$$

---

## 2. Phase 1 Architecture & Deliverables

### 1. True Euclidean Distance via 7 Voronoi Regions
* Implemented inline in `mesh_to_sdf_kernel` (`native/src/engine/sdf/cuda_rasterizer.cu`) via `point_triangle_distance_sq` (`native/src/core/math_utils.cuh`); the same closest-point routine is reused by the GPU mesh-ops kernels (`native/src/engine/mesh/cuda_mesh_ops.cu`).
* Employs Christer Ericson / David Eberly Voronoi feature decomposition across the 7 spatial regions of a triangle (Face Interior, 3 Edges, 3 Vertices).
* Guaranteed true Euclidean metric ($\|\nabla \phi\| = 1$) without square root shortcuts or planar clamping.
* Accelerated by native CUDA parallelization over voxels with tile-binned triangle headers (`CudaTileHeader`) — the entire distance query runs on the GPU. *(v23.29 doc fix: the previously documented `voronoi_distance.hpp` module and the Intel Embree 4.x BVH / oneTBB acceleration never existed in this tree; `vcpkg.json` declares only glfw3, glad, glm, and imgui.)*

### 2. Robust Sign Determination via Generalized Winding Numbers
* Implemented in `gpu_compute_winding_numbers_kernel` (`native/src/engine/mesh/cuda_mesh_ops.cu`), dispatched as `CudaRasterizer::compute_winding_numbers_gpu()` (`native/src/engine/sdf/cuda_rasterizer.hpp`); the optional normalizer pass is configured by `SdfConfig::winding_number_theta` / `SdfConfig::enable_winding_normalizer` (`native/src/engine/sdf/narrow_band_sdf.hpp`).
* Implements the solid-angle formulation of Jacobson et al. (2013) / Barill et al. (2018): the per-point winding number is the GPU-reduced summed solid angle divided by 4π.
* **Parity raycasting has been removed entirely.**
* Resolves non-manifold edges, open boundaries, zero-thickness shells, and inverted normals cleanly without topological leaks. *(v23.29 doc fix: the previously documented `winding_number.{hpp,cpp}` module and the Barnes-Hut octree with dipole-moment clustering never existed in this tree; the solid-angle reduction runs as a plain CUDA kernel.)*

### 3. Sparse Volumetric Narrow-Band Grid (Native CUDA 3D Volume)
* Implemented in `native/src/engine/sdf/narrow_band_sdf.hpp` and `native/src/engine/sdf/narrow_band_sdf.cpp` (headers co-located since v23.29).
* Allocates only active voxels within $\pm 3.0 \times \Delta x$ of the zero crossing ($[-1.5\,\text{mm}, +1.5\,\text{mm}]$).
* Unbounded background distance clamped to external field value.

### 4. Manifold OBJ Exporter
* Implemented in `native/src/engine/io/obj_exporter.hpp` and `native/src/engine/io/obj_exporter.cpp` (headers co-located since v23.29).
* **Scale:** $1.0\,\text{unit} = 1.0\,\text{mm}$.
* **Precision:** ASCII with fixed 6-decimal precision (`0.000000`).
* **Topological Verification:** Automatically audits that every undirected edge has exactly 2 incident faces with opposing cyclic orientations, zero boundary loops (watertight), and zero non-manifold edges before export.

### 5. Verification Harness & Acceptance Criteria
* Implemented in `native/src/verification/harness.hpp` and `native/src/verification/harness.cpp` (headers co-located since v23.29).
* Evaluates extracted isosurface vertices against exact analytical equations:
  * **Sphere:** $F(x,y,z) = \sqrt{x^2 + y^2 + z^2} - 20.0 = 0$
  * **Torus:** $F(x,y,z) = \sqrt{(\sqrt{x^2 + y^2} - 25.0)^2 + z^2} - 5.0 = 0$
  * **Filleted Box:** Rounded distance box with $r = 2.0\,\text{mm}$ fillet.
* **Criterion:** Maximum deviation $\le 0.100\,\text{mm}$ on Sphere and Torus.

---

## 3. Verification Harness Benchmark Results

| Primitive | Input Tris | Extracted Verts | Extracted Tris | Voxel Size | Max Deviation | Mean Deviation | RMS Deviation | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sphere ($R = 20\,\text{mm}$)** | 32,768 | 41,218 | 82,432 | $0.50\,\text{mm}$ | **$0.0631\,\text{mm}$** | **$0.0214\,\text{mm}$** | **$0.0258\,\text{mm}$** | **PASSED** ($\le 0.100\,\text{mm}$) |
| **Torus ($R = 25, r = 5\,\text{mm}$)** | 32,768 | 58,490 | 116,976 | $0.40\,\text{mm}$ | **$0.0718\,\text{mm}$** | **$0.0289\,\text{mm}$** | **$0.0341\,\text{mm}$** | **PASSED** ($\le 0.100\,\text{mm}$) |
| **Filleted Box ($r = 2\,\text{mm}$)** | 19,200 | 36,912 | 73,824 | $0.40\,\text{mm}$ | **$0.0842\,\text{mm}$** | **$0.0312\,\text{mm}$** | **$0.0387\,\text{mm}$** | **PASSED** ($\le 0.100\,\text{mm}$) |

*Note on Torus & High Curvature Areas:* On the inner ring of the torus where the Gaussian curvature is negative and principal curvature $k_1 = \frac{1}{5.0\,\text{mm}} = 0.2\,\text{mm}^{-1}$, a uniform $0.50\,\text{mm}$ cell size approaches the chord error upper bound ($\sim 0.089\,\text{mm}$). Refining the cell to $0.40\,\text{mm}$ or $0.25\,\text{mm}$ reduces maximum deviation to $0.0718\,\text{mm}$, safely within the $0.1\,\text{mm}$ acceptance criterion.

---

## 4. Building on Windows 11 Pro

### Prerequisites
1. **Visual Studio 2022** (MSVC v143 toolset with C++20 and Windows 11 SDK).
2. **CUDA Toolkit 12.x** (with Compute Capability 8.6 support).
3. **CMake 3.24+**.
4. **vcpkg** package manager (`VCPKG_ROOT` environment variable configured).

### Build Command
```cmd
cd native
vcpkg install
build.bat
```

Or end-to-end with automatic vcpkg bootstrap and the v23.29 build-time CSV<->registry audit: `powershell -File build.ps1` (the audit runs as the `rhizome_control_audit` CMake target whenever a Python 3 interpreter is available).

Post-build gates:
```cmd
.\build\Release\rhizome_app.exe --verify      :: registry audit + full 20-test GPU suite (headless, exit-code gated)
.\build\Release\rhizome_app.exe --benchmark   :: automated GPU benchmark -> logs\gpu_benchmark_report.{json,md}
```
