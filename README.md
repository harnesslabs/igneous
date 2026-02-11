# Igneous

**A Data-Oriented Geometric Algebra Engine for Computational Topology and Geometry**

Igneous is a high-performance C++23 library designed to bridge the gap between **Geometric Algebra (GA)** and **Discrete Computational Geometry**. Unlike traditional object-oriented geometry libraries that rely on pointer chasing, Igneous employs strict **Data-Oriented Design (DOD)** principles.

It provides a unified buffer architecture for storing CGA/PGA primitives and an optimized Compressed Sparse Row (CSR) graph system for topological navigation. This architecture enables real-time geometric flows, curvature analysis, and physics simulations on meshes exceeding 1 million vertices entirely on the CPU.

## Key Features

* **Arbitrary Signature Algebra:** Supports `Euclidean3D`, `PGA3D`, `Minkowski`, and custom `Cl(p,q,r)` signatures.
* **SIMD-First Architecture:** Built on top of `xsimd` to leverage AVX/NEON instruction sets for batched geometric products.
* **Zero-Copy Topology:** Topological connectivity (Coboundaries) is constructed in milliseconds using a flat CSR memory layout.
* **Math Kernels:**
* Discrete Differential Geometry operators (Gaussian , Mean , Laplacian ).
* Geometric Flows (Mean Curvature Flow, Heat Diffusion).
* Robust geometric predicates using multivector logic.



This is a great way to visualize progress. By calculating the delta between the two runs, we can explicitly show the **~9-14% performance gain** achieved by the packed buffer optimization.

Here is the updated **Performance** section for your README.

---

## Performance

Benchmarks were conducted on procedurally generated "Wavy Grid" meshes to evaluate scalability. The engine maintains interactive framerates for physics simulations even at high vertex counts.

**System Specifications:**

* **Machine:** MacBook Pro (14-inch, Nov 2023)
* **Chip:** Apple M3 Max
* **Memory:** 36 GB Unified Memory

### Current Benchmarks (v0.2 - Packed Buffers)

*Optimization: Switched `GeometryBuffer` to packed SoA float storage to reduce memory bandwidth.*

| Mesh Size | Vertices | Faces | Topology Build | Geometry Kernel (H, K) | Physics Kernel (Flow) | Sim FPS | Speedup* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Grid 100x100** | 10,000 | 19,602 | 0.084 ms | 1.060 ms | 0.107 ms | **857 Hz** | **+6.9%** |
| **Grid 250x250** | 62,500 | 124,002 | 0.502 ms | 6.354 ms | 0.636 ms | **143 Hz** | **+14.4%** |
| **Grid 500x500** | 250,000 | 498,002 | 1.914 ms | 24.459 ms | 2.564 ms | **37 Hz** | **+8.8%** |
| **Grid 1k x 1k** | 1,000,000 | 1,996,002 | 8.261 ms | 96.654 ms | 10.212 ms | **9.4 Hz** | **+9.3%** |

<small>*Speedup compared to initial baseline (commit `028b9c7`).</small>

### Baseline Benchmarks (v0.1 - Sparse Multivectors)

*Initial implementation using full sparse Multivector storage.*

| Mesh Size | Vertices | Faces | Topology Build | Geometry Kernel (H, K) | Physics Kernel (Flow) | Sim FPS |
| --- | --- | --- | --- | --- | --- | --- |
| **Grid 100x100** | 10,000 | 19,602 | 0.074 ms | 1.151 ms | 0.095 ms | **802 Hz** |
| **Grid 250x250** | 62,500 | 124,002 | 0.445 ms | 7.113 ms | 0.830 ms | **125 Hz** |
| **Grid 500x500** | 250,000 | 498,002 | 1.967 ms | 26.746 ms | 2.679 ms | **34 Hz** |
| **Grid 1k x 1k** | 1,000,000 | 1,996,002 | 7.437 ms | 105.208 ms | 11.187 ms | **8.6 Hz** |

* **Topology:** Time to build the Vertex  Face adjacency graph (CSR).
* **Geometry:** Time to compute Angle Deficit (Gaussian) and Edge-Mean (Mean) curvature for the entire mesh.
* **Physics:** Time to compute Laplacian flow vectors and integrate positions ().

## Quick Start

Igneous is a header-only library requiring a C++23 compliant compiler.

**Dependencies:**

* `xsimd` (SIMD intrinsics)
* `fmt` (Formatting)
* `range-v3` (Optional, for views)

### Installation

```cmake
# CMakeLists.txt
add_subdirectory(igneous)
target_link_libraries(your_app PRIVATE igneous)

```

### Example: Mean Curvature Flow

This example loads a mesh, normalizes it to the unit cube, and runs a geometric smoothing simulation, exporting the curvature field as a heatmap for each frame.

```cpp
#include <igneous/igneous.hpp>
#include <format>

using namespace igneous;
using Sig = Euclidean3D; // Algebra Signature

int main() {
    // 1. Initialize
    Mesh<Sig> mesh;
    io::load_obj(mesh, "assets/bunny.obj");

    // 2. Pre-process
    ops::normalize(mesh); // Center and scale to unit cube

    // 3. Simulation Loop
    double dt = 0.01;
    for (int i = 0; i < 100; ++i) {
        
        // Compute Discrete Curvature Fields
        auto [H, K] = ops::compute_curvature(mesh);

        // Export Visualization
        // Saves a heatmap of Mean Curvature (H) with sigma-clipping
        io::export_heatmap(mesh, H, std::format("out/frame_{:03}.obj", i), 2.0);

        // Integrate Physics
        ops::integrate_mean_curvature_flow(mesh, dt);
    }
}

```

## Architecture

The engine architecture separates **Data** from **Operations** to maximize cache coherency.

1. **Geometry Buffer (`GeometryBuffer<T>`)**: Utilizes a Structure-of-Arrays (SoA) layout where efficient. The base type `Multivector<Sig>` ensures mathematical operations respect the geometric algebra signature at compile time.
2. **Topology Buffer (`TopologyBuffer`)**: Connectivity is stored in a Directed Acyclic Graph (DAG) logic but flattened into integer arrays. The vertex-to-face adjacency is computed on-demand using a linear counting sort, generating a CSR structure that allows  access to the 1-ring neighborhood without pointer chasing.
3. **Kernels (`ops::`)**: All computation is performed by stateless kernels in `igneous::ops`. These functions accept raw buffers and perform parallel-friendly gather/scatter operations.

## License

MIT License