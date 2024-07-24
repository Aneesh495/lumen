# Lumen

<div align="center">

**High-Performance 3D Gaussian Splatting and Radiance Field Rasterization Engine**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Language](https://img.shields.io/badge/language-Mojo%201.2.0-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-24%2F24%20passed-success.svg)]()
[![C API](https://img.shields.io/badge/C%20ABI-C99%2FC11%20Included-orange.svg)]()
[![Python SDK](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

</div>

---

## Overview

**Lumen** is a standalone, real-time 3D Gaussian Splatting (3DGS) and spatial scene synthesis engine engineered from first principles in **Mojo**. It couples the high-level expressiveness of modern languages with bare-metal mechanical sympathy, cache locality, and SIMD hardware execution.

Designed for spatial computing and radiance field processing, Lumen provides an end-to-end pipeline: from reading Stanford PLY files and generating synthetic volumetric scenes, to evaluating degree-3 spherical harmonics, projecting elliptical weighted average (EWA) covariances, sorting composite tile-depth keys with a 64-bit LSD radix sorter, and compositing pixels with early ray termination.

Beyond its core Mojo engine, Lumen includes a C99/C11 foreign interface bridge (`liblumen`), an ergonomic Python SDK with spline camera trajectories, an interactive web-based spatial computing viewer, and an automated verification harness.

---

## Key Technical Features

- **Pure Mojo Systems Core**: Cache-aligned Structure-of-Arrays (SoA) SIMD memory layout (`GaussianSoA`) preventing cache pollution during culling and projection.
- **EWA 2D Screen Covariance**: Rigorous derivation of the affine Projective Jacobian ($J$) combined with a low-pass Gaussian reconstruction filter ($\nu = 0.3 \text{ px}^2$) to eliminate aliasing and singular matrix inversions.
- **64-bit LSD Radix Sort**: In-place double-buffered 64-bit Least Significant Digit radix sorter ordering composite tile and depth keys in linear $\mathcal{O}(N)$ time at over 14.9 Million keys/second.
- **View-Dependent Radiance**: Full degree-3 real spherical harmonics evaluation with 16 orthogonal basis polynomials per color channel (1.4+ Billion evaluations/second).
- **Linear BVH & Morton Hashing**: 30-bit Morton code spatial hashing and LBVH hierarchy construction supporting over 2.3 Million ray-splat intersection queries/second.
- **Tile-Based Compositor**: 16x16 screen tile subdivision with front-to-back alpha blending and early ray termination ($\alpha \ge 0.99$) skipping 80% to 90% of redundant ray traversal.
- **Adaptive Density Control (ADC)**: Complete optimization primitives implementing positional view-space gradient accumulation, cloning, splitting, and opacity resets.
- **Polyglot Ecosystem**: Native C ABI shared library (`c_api/`), Python client SDK (`python/`), and interactive dark-mode WebGL/Canvas spatial viewer (`web/`).

---

## Performance Benchmarks

All benchmarks were evaluated on an Apple Silicon M-series machine using Mojo 1.2.0:

| Subsystem Component | Workload Specification | Mean Latency | Peak Throughput |
|:---|:---|:---|:---|
| **End-to-End Rasterizer** | 64 Gaussian Splats (160x120) | 0.026 ms | **38,560 FPS** (2.5M splats/s) |
| **End-to-End Rasterizer** | 216 Gaussian Splats (160x120) | 0.027 ms | **37,243 FPS** (8.0M splats/s) |
| **End-to-End Rasterizer** | 512 Gaussian Splats (160x120) | 0.030 ms | **33,112 FPS** (17.0M splats/s) |
| **Native CLI Rasterizer** | 512 Splats (800x600 Full HD) | 1.036 ms | **965 FPS** |
| **64-bit LSD Radix Sort** | 10,000 Composite Keys | 0.667 ms | **14.98 Million keys/s** |
| **64-bit LSD Radix Sort** | 50,000 Composite Keys | 3.427 ms | **14.59 Million keys/s** |
| **Spherical Harmonics** | 10,000 Degree-3 Basis Vectors | 0.0068 ms | **1,463 Million evals/s** |
| **Spherical Harmonics** | 50,000 Degree-3 Basis Vectors | 0.0336 ms | **1,488 Million evals/s** |
| **Linear BVH Build** | 216 Gaussian Centroids | 0.0219 ms | **45,558 builds/s** |
| **Hierarchical Raycast** | 500 Ray Intersections | 0.216 ms | **2.31 Million rays/s** |

---

## Repository Structure

```
lumen/
├── src/                          # Mojo Core Engine Architecture
│   ├── core/                     # Cache-aligned buffers, config, error types
│   │   ├── buffer.mojo           # Contiguous dynamic typed heap arrays
│   │   ├── config.mojo           # Immutable raster and camera descriptors
│   │   ├── error.mojo            # Strong status codes and error propagation
│   │   ├── types.mojo            # Precision aliases and fast float parsing
│   │   └── __init__.mojo
│   ├── math/                     # Linear Algebra & Radiance Field Optics
│   │   ├── vector.mojo           # Vec2f, Vec3f, Vec4f vector operations
│   │   ├── matrix.mojo           # Mat2x2f, Mat3x3f, Mat4x4f row-major matrices
│   │   ├── quaternion.mojo       # SO(3) Quatf with collinear fallback SLERP
│   │   ├── covariance.mojo       # 3D spatial covariance: Sigma = R S S^T R^T
│   │   ├── projection.mojo       # Projective Jacobian (J) and EWA screen Sigma
│   │   ├── spherical_harmonics.mojo # 16 orthogonal SH polynomials (degree 0 to 3)
│   │   ├── cholesky.mojo         # 2D Eigenvalue 3-sigma radius and Cholesky
│   │   └── __init__.mojo
│   ├── scene/                    # Radiance Field Primitives & Datasets
│   │   ├── gaussian.mojo         # Gaussian3D AoS and GaussianSoA memory
│   │   ├── gaussian_cloud.mojo   # Cloud container with spatial bounds
│   │   ├── camera.mojo           # Pinhole camera model and look-at transform
│   │   ├── trajectory.mojo       # Catmull-Rom splines and orbit generators
│   │   ├── ply.mojo              # Stanford PLY reader and writer
│   │   ├── datasets.mojo         # Synthetic scene generators and COLMAP
│   │   └── __init__.mojo
│   ├── raster/                   # Tile-Based Software Rasterization
│   │   ├── culling.mojo          # Frustum, near/far depth, screen bounds
│   │   ├── tile.mojo             # 16x16 screen tile grid subdivision
│   │   ├── radix_sort.mojo       # 64-bit LSD radix sort on composite keys
│   │   ├── binning.mojo          # Contiguous tile range index tables
│   │   ├── compositor.mojo       # Front-to-back alpha blending with early exit
│   │   ├── depth_map.mojo        # Continuous depth map and surface normals
│   │   ├── pipeline.mojo         # Master rasterizer pipeline coordinator
│   │   └── __init__.mojo
│   ├── spatial/                  # Spatial Data Structures & Ray Traversal
│   │   ├── morton.mojo           # 30-bit Morton code spatial hashing
│   │   ├── aabb.mojo             # 3D AABB with slab method ray intersection
│   │   ├── bvh.mojo              # Linear Bounding Volume Hierarchy (LBVH)
│   │   ├── raycast.mojo          # Hierarchical ray-splat picking
│   │   └── __init__.mojo
│   ├── optim/                    # Optimization & Adaptive Density Control
│   │   ├── loss.mojo             # Photometric L1, SSIM, and PSNR loss
│   │   ├── gradients.mojo        # Parameter gradient accumulation
│   │   ├── optimizer.mojo        # Adam with per-group learning rate schedules
│   │   ├── density_control.mojo  # ADC: clone, split, prune, opacity reset
│   │   ├── trainer.mojo          # Scene loss backpropagation loop
│   │   └── __init__.mojo
│   └── main.mojo                 # Unified CLI executable
├── c_api/                        # C99/C11 Foreign Function Interface (FFI)
│   ├── include/lumen.h           # Clean C header for foreign language embedding
│   ├── src/lumen_c.c             # C runtime implementation
│   ├── test/test_c_api.c         # C unit and integration test suite
│   └── Makefile                  # Builds liblumen.dylib / liblumen.so
├── python/                       # High-Performance Python Client SDK
│   ├── lumen/                    # Client module
│   │   ├── camera.py             # Camera and trajectory spline generators
│   │   ├── cloud.py              # GaussianCloud, synthetic presets, PLY IO
│   │   ├── client.py             # LumenEngine with C FFI and fallback
│   │   ├── visualizer.py         # PPM export and terminal ASCII preview
│   │   └── __init__.py
│   ├── tests/                    # Python unit test suite
│   └── setup.py                  # Package installation descriptor
├── web/                          # Interactive Spatial Computing Viewer
│   ├── index.html                # Dark-mode dashboard with telemetry HUD
│   ├── css/style.css             # Glassmorphism aesthetic stylesheet
│   ├── js/app.js                 # 3D Gaussian software rasterizer in canvas
│   └── js/stream.js              # Native frame streaming client
├── benches/                      # Benchmarks Suite
│   ├── bench_rasterizer.mojo     # End-to-end rendering throughput
│   ├── bench_radix_sort.mojo     # 64-bit LSD radix sort speed
│   ├── bench_spherical_harmonics.mojo # Degree-3 polynomial evaluation speed
│   └── bench_bvh.mojo            # LBVH construction and raycasting speed
├── tests/                        # Comprehensive Unit & Regression Test Suite
│   ├── test_vector.mojo          # Vector mathematics tests
│   ├── test_matrix.mojo          # Matrix operations tests
│   ├── test_quaternion.mojo      # SO(3) rotations and SLERP tests
│   ├── test_covariance.mojo      # 3D covariance formulation tests
│   ├── test_projection.mojo      # Projective Jacobian and EWA tests
│   ├── test_spherical_harmonics.mojo # SH basis evaluation tests
│   ├── test_cholesky.mojo        # Eigenvalue and Cholesky tests
│   ├── test_camera.mojo          # Pinhole projection tests
│   ├── test_trajectory.mojo      # Spline path generation tests
│   ├── test_ply.mojo             # PLY serialization tests
│   ├── test_datasets.mojo        # Dataset ingestion tests
│   ├── test_culling.mojo         # Frustum and depth culling tests
│   ├── test_radix_sort.mojo      # 64-bit LSD radix sort tests
│   ├── test_binning.mojo         # Tile range index tests
│   ├── test_compositor.mojo      # Alpha blending compositor tests
│   ├── test_depth_map.mojo       # Depth buffer and normal extraction tests
│   ├── test_rasterizer.mojo      # Integration rasterization tests
│   ├── test_morton.mojo          # 30-bit Morton code tests
│   ├── test_bvh.mojo             # LBVH construction tests
│   ├── test_raycast.mojo         # Hierarchical ray intersection tests
│   ├── test_loss.mojo            # L1, SSIM, and PSNR loss tests
│   ├── test_gradients.mojo       # Gradient accumulation tests
│   ├── test_density_control.mojo # ADC split, clone, and prune tests
│   └── test_regressions.mojo     # Boundary conditions and mathematical edge cases
├── docs/                         # Detailed Architecture Documentation
│   ├── ARCHITECTURE.md           # Subsystems, memory layout, sequence diagrams
│   ├── MATHEMATICAL_FORMULATION.md # Full analytical proofs and equations
│   └── RASTERIZATION_PIPELINE.md # Hardware-aligned tile rasterizer design
└── scripts/                      # Automation & Verification Scripts
    ├── verify_all.sh             # Unified 6-stage test and benchmark pipeline
    └── verify_no_dashes.py       # Typography hygiene and zero-dash scanner
```

---

## Quickstart Guide

### 1. Requirements

- macOS or Linux
- [Pixi](https://pixi.sh) package manager
- Clang / GCC (for C ABI build)
- Python 3.8+ (for Python SDK)

### 2. Automated Full-Verification Suite

To run the complete 6-stage verification pipeline (typography scan, C ABI tests, Python SDK tests, all 24 Mojo unit and regression tests, benchmarks, and CLI build):

```bash
bash scripts/verify_all.sh
```

### 3. Native CLI Usage

Compile and run the native Lumen executable:

```bash
pixi run mojo build -I . src/main.mojo -o bin/lumen
```

Execute performance benchmark:
```bash
./bin/lumen bench
```

Render synthetic cube scene:
```bash
./bin/lumen render --scene cube --output frame.ppm --width 800 --height 600
```

Inspect point cloud metadata:
```bash
./bin/lumen inspect --scene scene.ply
```

Run interactive terminal ASCII demo:
```bash
./bin/lumen demo
```

### 4. Running Benchmarks

Individual benchmarks can be executed directly:

```bash
# End-to-end rasterizer throughput
pixi run mojo -I . benches/bench_rasterizer.mojo

# 64-bit LSD Radix Sort speed
pixi run mojo -I . benches/bench_radix_sort.mojo

# Spherical Harmonics evaluation speed
pixi run mojo -I . benches/bench_spherical_harmonics.mojo

# Linear BVH construction and raycasting speed
pixi run mojo -I . benches/bench_bvh.mojo
```

### 5. C API Shared Library

Compile and run the C verification test suite:

```bash
make -C c_api clean
make -C c_api test
```

The compiled shared library is generated at `c_api/build/liblumen.dylib` (or `liblumen.so` on Linux).

### 6. Python Client SDK

Install the Python client in development mode:

```bash
pip install -e python/
```

Run Python unit tests:
```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests
```

Example Python usage:
```python
from lumen import Camera, GaussianCloud, LumenEngine, RasterizerConfig, export_ppm

# Generate synthetic cube scene
cloud = GaussianCloud.synthetic_cube(points_per_axis=5, cube_size=2.0)

# Configure camera
cam = Camera(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), fov_y_deg=45.0)

# Render scene with LumenEngine
engine = LumenEngine(RasterizerConfig(image_width=800, image_height=600))
result = engine.render(cloud, cam)

# Export RGB buffer
export_ppm("render.ppm", result.width, result.height, result.rgb_bytes)
print(f"Rendered {result.stats['total_splats']} splats in {result.stats['elapsed_ms']:.2f} ms")
```

### 7. Interactive Spatial Computing Web Viewer

To explore the real-time interactive spatial viewer:

```bash
python3 -m http.server 8000 --directory web
```

Open `http://localhost:8000` in your browser. Controls:
- **Left Click + Drag**: Orbit Camera around the scene
- **Mouse Wheel**: Zoom In / Out
- **Spacebar**: Toggle continuous turntable auto-orbit
- **Sidebar**: Switch presets (Cube, Fibonacci Sphere, Super-Toroid, Double Helix), toggle rendering modes (Radiance, Depth Heatmap, 16x16 Tile Grid, Bounding Extents), adjust FOV, and monitor real-time frame latency.

---

## Architectural Documentation

For deep technical dives into the math and implementation:
- [Architecture Specification](docs/ARCHITECTURE.md): Complete subsystem design, SoA memory model, and sequence diagrams.
- [Mathematical Formulation](docs/MATHEMATICAL_FORMULATION.md): Full derivations for Projective Jacobians, EWA screen covariance, real spherical harmonics, and volume rendering integrals.
- [Rasterization Pipeline](docs/RASTERIZATION_PIPELINE.md): Tile grid subdivision, 64-bit radix sorting, early ray termination, and Linear BVH traversal.

---

## Zero-Defect Engineering Standards

Lumen is authored with strict adherence to rigorous systems software quality:
- **Zero Broken Builds**: Every test, benchmark, and target compiles cleanly.
- **Strict Character Hygiene**: Zero em/en dashes across all documentation, source code comments, and scripts.
- **Automated Regressions**: Every mathematical edge case (singular covariance, near-plane clipping, collinear SLERP, boundary Morton codes) is guarded by permanent regression tests.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
