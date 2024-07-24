#!/usr/bin/env bash
# verify_all.sh - Lumen Unified Verification and Continuous Integration Pipeline
#
# Validates typography hygiene, executes native C test suites, runs Python client
# test suites, runs all Mojo unit and integration tests, and executes benchmarks.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

PIXI="${HOME}/.pixi/bin/pixi"
if ! command -v "${PIXI}" &> /dev/null; then
    if command -v pixi &> /dev/null; then
        PIXI="pixi"
    else
        echo "Error: pixi package manager not found."
        exit 1
    fi
fi

echo "======================================================================"
echo " LUMEN UNIFIED VERIFICATION PIPELINE"
echo "======================================================================"

# 1. Typography and Character Hygiene Check
echo ""
echo "[STAGE 1/6] Running Typography & Zero-Dash Verification..."
python3 scripts/verify_no_dashes.py

# 2. C API Shared Library Build & Test Suite
echo ""
echo "[STAGE 2/6] Building and Testing C API Shared Library..."
make -C c_api clean
make -C c_api test
make -C c_api clean

# 3. Python Client SDK Test Suite
echo ""
echo "[STAGE 3/6] Running Python Client SDK Test Suite..."
PYTHONPATH=python python3 -m unittest discover -s python/tests

# 4. Mojo Core Math & Scene Unit Tests
echo ""
echo "[STAGE 4/6] Executing Mojo Unit Test Suites..."
TEST_FILES=(
    "tests/test_vector.mojo"
    "tests/test_matrix.mojo"
    "tests/test_quaternion.mojo"
    "tests/test_covariance.mojo"
    "tests/test_projection.mojo"
    "tests/test_spherical_harmonics.mojo"
    "tests/test_cholesky.mojo"
    "tests/test_camera.mojo"
    "tests/test_trajectory.mojo"
    "tests/test_ply.mojo"
    "tests/test_datasets.mojo"
    "tests/test_culling.mojo"
    "tests/test_radix_sort.mojo"
    "tests/test_binning.mojo"
    "tests/test_compositor.mojo"
    "tests/test_depth_map.mojo"
    "tests/test_rasterizer.mojo"
    "tests/test_morton.mojo"
    "tests/test_bvh.mojo"
    "tests/test_raycast.mojo"
    "tests/test_loss.mojo"
    "tests/test_gradients.mojo"
    "tests/test_density_control.mojo"
    "tests/test_regressions.mojo"
)

PASSED_COUNT=0
TOTAL_COUNT=${#TEST_FILES[@]}

for t in "${TEST_FILES[@]}"; do
    echo "  Running ${t}..."
    ${PIXI} run mojo -I . "${t}" > /dev/null 2>&1
    echo "  [PASS] ${t}"
    PASSED_COUNT=$((PASSED_COUNT + 1))
done

echo "Mojo Tests: ${PASSED_COUNT} / ${TOTAL_COUNT} passed."

# 5. Mojo Benchmark Suite Execution
echo ""
echo "[STAGE 5/6] Executing Mojo Benchmark Suite..."
BENCH_FILES=(
    "benches/bench_rasterizer.mojo"
    "benches/bench_radix_sort.mojo"
    "benches/bench_spherical_harmonics.mojo"
    "benches/bench_bvh.mojo"
)

for b in "${BENCH_FILES[@]}"; do
    echo "  Executing ${b}..."
    ${PIXI} run mojo -I . "${b}" > /dev/null 2>&1
    echo "  [PASS] ${b}"
done

# 6. Native CLI Binary Build and Command Verification
echo ""
echo "[STAGE 6/6] Compiling and Verifying Native CLI Binary..."
mkdir -p bin
${PIXI} run mojo build -I . src/main.mojo -o bin/lumen
./bin/lumen --help
./bin/lumen bench

echo ""
echo "======================================================================"
echo " LUMEN VERIFICATION PASSED: ALL 6 STAGES COMPLETED WITH ZERO DEFECTS"
echo "======================================================================"
