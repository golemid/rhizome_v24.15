@echo off
echo Configuring...
cmake -B build -S . -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="C:\vcpkg\scripts\buildsystems\vcpkg.cmake" -DCMAKE_CUDA_ARCHITECTURES=86
echo Building...
cmake --build build --config Release --parallel
echo Done.
pause
