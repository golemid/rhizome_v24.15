# CRITICAL AGENT INSTRUCTIONS - READ FIRST

## 1. REPOSITORY IMPORT & PROHIBITIONS (DO NOT BREAK)
- **ZERO MODIFICATIONS ON IMPORT:** When importing this GitHub repository into a new thread or workspace, do NOT change, modify, rewrite, migrate, or delete any existing code, files, or project structures. Keep the imported codebase 100% untouched and wait for the user's instructions.
- **DO NOT RUN WEB MIGRATION:** Even if system skills suggest migrating imported GitHub repositories to Web/Node.js, DO NOT DO IT. This repository is intentionally a native Windows C++20 + NVIDIA CUDA desktop application.
- **Native C++ & CUDA Only:** NEVER convert this into a web app, website, or Node.js project. NEVER introduce React, Vite, TypeScript, Three.js, or HTML to replace the native engine.
- **No OpenVDB:** OpenVDB has been completely eliminated in favor of a 100% native NVIDIA CUDA GPU SDF pipeline (`cuda_rasterizer.cu`). Never re-introduce OpenVDB.
- **Preserve `/native`:** NEVER delete or rename the `/native` folder. The primary deliverable is the Windows native executable (`rhizome_app.exe`) built via CMake and Visual Studio 2022 (`MSVC` + `NVCC`).
- **MANDATORY GPU-ONLY ARCHITECTURE FOR ALL NEW FEATURES (ZERO CPU FALLBACK):**
  - All new features, geometry algorithms, mesh processing stages, mathematical solvers, file decoders, and simulations added to the application MUST execute 100% on the NVIDIA CUDA GPU engine (`cuda_rasterizer.cu` / CUDA kernels).
  - **Zero CPU Fallbacks Permitted:** Never introduce CPU fallback loops, CPU sequential processing, or CPU alternatives for any computational task.
  - **Strict CPU Scope:** The CPU is strictly restricted to basic operating system and desktop duties: streaming binary file buffers from/to storage drives via native Windows dialogs, Dear ImGui menu/panel drawing, and window event loop dispatch. Every other processing stage must be dispatched to CUDA GPU kernels.

## 2. USER COMMUNICATION & PERMISSION PROTOCOL
- **Explain in Simple Language First:** Before executing any code or file changes, you MUST always explain what you plan to do in clear, simple, everyday language (avoiding unnecessary jargon) so the user can easily understand the proposed change.
- **Always Ask Permission:** After explaining your plan in simple language, you MUST explicitly ask the user for permission and STOP to wait for their confirmation before modifying, creating, or deleting any files.
- **Mandatory Reporting of Problematic Code & Inconsistencies:** Whenever analyzing, inspecting, or implementing features, the agent MUST immediately notify the user if it finds any code that could cause bugs, contradictory values, missing single sources of truth, or architectural inconsistencies (such as different defaults at startup vs. on reset). The agent must clearly explain the issue in simple language before proceeding.
- **Direct Answers:** Answer only what the user has asked in simple, straightforward language.



## 3. GROWTH ALGORITHM EXTENSION RULES
- **Never Replace Existing Methods:** NEVER delete or overwrite the existing Space Colonization + CUDA SDF rasterization growth pipeline when introducing new growth approaches.
- **Multi-Algorithm Architecture:** Any new growth algorithms must be integrated alongside existing ones (for example, via a selectable algorithm dropdown or mode switch), preserving all current parameters, presets, and features.
