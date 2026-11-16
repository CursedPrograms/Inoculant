#!/bin/sh
# Builds Inoculant (the window) and inoculant-cli on Linux: needs cmake, a C++17 compiler and libsdl2-dev, and
# ONNX Runtime's libonnxruntime.so next to the programs (or on the library path). On Windows: build.bat.
set -e
cd "$(dirname "$0")"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cp -r lang assets build/ 2>/dev/null || true
echo "Built: build/Inoculant and build/inoculant-cli"
