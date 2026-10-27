param (
    [switch]$TestOnly,
    [string]$VcpkgPath = $env:VCPKG_ROOT
)

$ErrorActionPreference = "Stop"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Rhizome Native Engine - Windows 11 Build & Verification Script    " -ForegroundColor Cyan
Write-Host "  Stack: C++20 | MSVC 2022 | CUDA 12 (sm_86) | Native CUDA GPU Pipeline" -ForegroundColor Cyan
Write-Host "  v23.29: CSV<->registry build-time audit wired into rhizome_app    " -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# 1. Verify CMake is installed
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] CMake is not found in your PATH." -ForegroundColor Red
    Write-Host "Please install CMake from https://cmake.org/download/ (select 'Add CMake to system PATH')." -ForegroundColor Yellow
    exit 1
}

# 2. Locate or bootstrap vcpkg automatically
if (-not $VcpkgPath -or -not (Test-Path "$VcpkgPath\vcpkg.exe")) {
    # Check if vcpkg is in PATH
    $vcpkgCmd = Get-Command vcpkg -ErrorAction SilentlyContinue
    if ($vcpkgCmd) {
        $VcpkgPath = Split-Path (Split-Path $vcpkgCmd.Source)
    } elseif (Test-Path "..\vcpkg\vcpkg.exe") {
        $VcpkgPath = (Resolve-Path "..\vcpkg").Path
    } elseif (Test-Path "C:\vcpkg\vcpkg.exe") {
        $VcpkgPath = "C:\vcpkg"
    } else {
        Write-Host "`n[INFO] vcpkg package manager not detected." -ForegroundColor Yellow
        Write-Host "Bootstrapping vcpkg into local folder '..\vcpkg'..." -ForegroundColor Yellow
        
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            Write-Host "[ERROR] Git was not found. Please install Git from https://git-scm.com/ or clone vcpkg manually." -ForegroundColor Red
            exit 1
        }
        
        git clone https://github.com/microsoft/vcpkg.git "..\vcpkg"
        & "..\vcpkg\bootstrap-vcpkg.bat" -disableMetrics
        $VcpkgPath = (Resolve-Path "..\vcpkg").Path
        [Environment]::SetEnvironmentVariable("VCPKG_ROOT", $VcpkgPath, "User")
        Write-Host "[SUCCESS] vcpkg bootstrapped at $VcpkgPath" -ForegroundColor Green
    }
}

$vcpkgExe = Join-Path $VcpkgPath "vcpkg.exe"
$toolchainFile = Join-Path $VcpkgPath "scripts\buildsystems\vcpkg.cmake"

Write-Host "`n[1/4] Installing C++ Dependencies via vcpkg..." -ForegroundColor Cyan
Write-Host "Using vcpkg: $vcpkgExe" -ForegroundColor Gray
& $vcpkgExe install --triplet x64-windows

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Dependency installation failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 3. Configure with Visual Studio 2022
Write-Host "`n[2/4] Configuring with CMake and Visual Studio 2022 (x64)..." -ForegroundColor Cyan
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

cmake -B build -S . -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_TOOLCHAIN_FILE="$toolchainFile" `
    -DCMAKE_CUDA_ARCHITECTURES=86

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] CMake configuration failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 4. Build Release configuration
Write-Host "`n[3/4] Compiling Release Build..." -ForegroundColor Cyan
cmake --build build --config Release --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Compilation failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 5. Run Verification Harness (headless, exit-code gated — Phase 5)
Write-Host "`n[4/4] Executing Verification Harness (--verify: 138-control registry audit + 20-test GPU suite)..." -ForegroundColor Cyan
$exePath = ".\build\Release\rhizome_app.exe"

if (-not (Test-Path $exePath)) {
    Write-Host "[ERROR] Executable not found at $exePath" -ForegroundColor Red
    exit 1
}

& $exePath --verify

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Verification harness reported failures (registry audit or tolerance violations)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[SUCCESS] Verification Passed! (registry audit: 138/138 controls; suite: 20/20 tests)" -ForegroundColor Green

if (-not $TestOnly) {
    Write-Host "Launching 3D Viewport GUI..." -ForegroundColor Green
    Start-Process $exePath
}
