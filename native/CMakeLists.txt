cmake_minimum_required(VERSION 3.24)
project(RhizomeNative LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA 12 targeting Compute Capability 8.6 & 8.9 (NVIDIA Ampere: RTX 30xx, Ada Lovelace: RTX 40xx)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86 89)

# MSVC Compiler flags
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:/MP>
        $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
        $<$<COMPILE_LANGUAGE:C,CXX>:/O2>
        $<$<COMPILE_LANGUAGE:C,CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus>
        $<$<COMPILE_LANGUAGE:C,CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>
        $<$<COMPILE_LANGUAGE:C,CXX>:/fp:precise>
    )
    add_compile_definitions(
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

if(CMAKE_CUDA_COMPILER)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# Find dependencies via vcpkg / CMake Find Modules
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OPENGL_INCLUDE_DIR}
)

# Core Library: Mesh, Distance, CUDA SDF & Root Mesher, Space Colonization, Verifier
# v23.29 Phase 2: core/ layer added (AppState SSOT, UnifiedLogger, growth
#   controller, algorithm/preset I/O) — migrated from src/app/ per AGENTS.md.
# v23.29 Phase 3: engine/ domain tree (sdf/colonization/mesh/io) + viewport
#   CUDA services under ui/viewport/ (compiled here — all CUDA stays in one
#   NVCC target) + verification CUDA co-located in verification/. Kernels and
#   host functions migrated VERBATIM from v23.28 src/sdf|mesh|colonization.
# v23.29 Phase 4: legacy include/ tree deleted — src/ is the sole include
#   root (co-located headers, AGENTS.md Sec 2 target tree).
add_library(rhizome_core STATIC
    # core (state management, logger, math utilities, orchestration)
    src/core/app_state.cpp
    src/core/unified_logger.cpp
    src/core/growth_controller.cpp
    src/core/algorithm_io.cpp
    # engine/sdf (SDF generation, rasterization, narrow band, root field)
    src/engine/sdf/cuda_rasterizer.cu
    src/engine/sdf/narrow_band_sdf.cpp
    src/engine/sdf/root_field.cpp
    # engine/colonization (space colonization, volumetric roots, Murray)
    src/engine/colonization/colonization.cpp
    src/engine/colonization/cuda_colonization.cu
    src/engine/colonization/cuda_colonization_volumetric.cu
    # engine/mesh (two-pass marching cubes, smoothing, repair, primitives)
    src/engine/mesh/cuda_marching_cubes.cu
    src/engine/mesh/cuda_mesh_ops.cu
    src/engine/mesh/cuda_mesh_topology.cu
    src/engine/mesh/cuda_mesh_primitives.cu
    # engine/io (single-pass streaming parsers + GPU format encoding)
    src/engine/io/obj_parser.cpp
    src/engine/io/stl_parser.cpp
    src/engine/io/obj_exporter.cpp
    src/engine/io/cuda_mesh_io.cu
    # viewport GPU services (CUDA compilation stays in core; domain in ui/)
    src/ui/viewport/cuda_viewport_gl.cu
    src/ui/viewport/cuda_viewport_packers.cu
    # verification (test harness, analytical evaluation, GPU benchmark)
    src/verification/harness.cpp
    src/verification/harness_vessel_tests.cpp
    src/verification/harness_eikonal_cuda_tests.cpp
    src/verification/harness_volumetric_tests.cpp
    src/verification/cuda_verification.cu
    src/verification/gpu_benchmark_runner.cu
)

target_link_libraries(rhizome_core PUBLIC
    glm::glm
)

# Native Desktop Application (Dear ImGui + OpenGL 3.3 Viewport + CLI Harness)
# v23.29 Phase 4: monolithic src/app/* split into ui/panels (growth, styles,
#   scene, verification) + ui/widgets (widget factory + 138-control
#   ControlDefinition registry); viewport_renderer relocated to ui/viewport/.
add_executable(rhizome_app
    src/main.cpp
    src/ui/widgets/widget_factory.cpp
    src/ui/widgets/control_registry.cpp
    src/ui/panels/panel_growth.cpp
    src/ui/panels/panel_styles.cpp
    src/ui/panels/panel_scene.cpp
    src/ui/panels/panel_verification.cpp
    src/ui/viewport/viewport_renderer.cpp
)

target_link_libraries(rhizome_app PRIVATE
    rhizome_core
    glfw
    glad::glad
    imgui::imgui
    OpenGL::GL
)

# v24.02 DIAG1 (2026-10-23, post-closure diagnostics; user-reported GUI crash):
# emit a linker symbol map next to the build root. Link-time OUTPUT only —
# zero codegen impact, binary layout unchanged, so a map produced by a rebuild
# of these same sources also resolves "Fault offset" RVAs recorded by Windows
# Event Log against earlier builds of the same tree. Consumed by
# native/tools/resolve_map_offsets.py. Records: V2402_PHASE2_REPORT.md Sec 12,
# TRACK_B_DIAG1_NOTE.md (user steps).
if(MSVC)
    target_link_options(rhizome_app PRIVATE
        "/MAP:${CMAKE_BINARY_DIR}/rhizome_app.map"
    )
endif()

# v23.29 Phase 5: CSV <-> control registry build-time audit (AGENTS.md Sec 1.3).
# Fails the build whenever param_controls_complete_list.csv and the
# 138-control ControlDefinition registry drift apart (label / section /
# category / kind / unit / slider range / logging flag / resolver count /
# section tiling / JSON-Lines monitor catalog). Requires a Python 3
# interpreter; when absent the audit is skipped with a warning — the same
# contract is enforced at runtime by --verify via
# control_registry_integrity_check() (see src/main.cpp).
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_FOUND)
    set(RHZ_CONTROL_CSV   ${CMAKE_CURRENT_SOURCE_DIR}/../param_controls_complete_list.csv)
    set(RHZ_CONTROL_REG   ${CMAKE_CURRENT_SOURCE_DIR}/src/ui/widgets/control_registry.cpp)
    set(RHZ_CONTROL_LOG   ${CMAKE_CURRENT_SOURCE_DIR}/src/core/unified_logger.cpp)
    set(RHZ_CONTROL_STAMP ${CMAKE_CURRENT_BINARY_DIR}/control_registry_audit.stamp)
    add_custom_command(
        OUTPUT ${RHZ_CONTROL_STAMP}
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/audit_control_registry.py
                --csv ${RHZ_CONTROL_CSV}
                --registry ${RHZ_CONTROL_REG}
                --logger ${RHZ_CONTROL_LOG}
                --stamp ${RHZ_CONTROL_STAMP}
        DEPENDS ${RHZ_CONTROL_CSV} ${RHZ_CONTROL_REG} ${RHZ_CONTROL_LOG}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/audit_control_registry.py
        COMMENT "Rhizome: auditing CSV <-> 138-control registry consistency"
        VERBATIM)
    add_custom_target(rhizome_control_audit DEPENDS ${RHZ_CONTROL_STAMP})
    add_dependencies(rhizome_app rhizome_control_audit)
else()
    message(WARNING "Python3 interpreter not found: CSV<->registry build-time audit SKIPPED — run native/tools/audit_control_registry.py manually ('python native/tools/audit_control_registry.py --csv param_controls_complete_list.csv --registry native/src/ui/widgets/control_registry.cpp --logger native/src/core/unified_logger.cpp'). --verify enforces registry integrity at runtime.")
endif()

# Enable CUDA integration for GPU acceleration
set_target_properties(rhizome_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

if(MSVC)
    target_link_options(rhizome_app PRIVATE /NODEFAULTLIB:LIBCMT)
endif()

message(STATUS "Configured RhizomeNative for Windows 11 MSVC C++20 + CUDA 12 (sm_86, sm_89)")
message(STATUS "Engine: 100% Native NVIDIA CUDA GPU Pipeline (OpenVDB eliminated)")
