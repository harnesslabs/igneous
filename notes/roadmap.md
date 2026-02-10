# 🗺️ Project Igneous Roadmap

**Goal:** High-performance Computational Geometry engine using Conformal Geometric Algebra (CGA) and Discrete Morse Theory (DMT).

## ✅ Phase 1: The Kernel (Completed)
- [x] **Clifford Algebra Engine**
    - [x] Compile-time optimizations (Template Metaprogramming).
    - [x] SIMD Vectorization (xsimd, NEON/AVX).
    - [x] Structure of Arrays (SoA) layout.
- [x] **Memory System**
    - [x] Linear Allocator (`MemoryArena`).
    - [x] `std::pmr` compatibility.
- [x] **Basic Testing**
    - [x] Doctest integration.
    - [x] Micro-benchmarks.

## 🚧 Phase 2: The Topology (Current Focus)
**Objective:** Represent a mesh not just as a bag of triangles, but as a Hasse Diagram (Graph of incidence).
- [ ] **Complex Simplex Storage**
    - Store `Boundary` (d-1) and `Coboundary` (d+1) relations efficiently.
- [ ] **Mesh Loading**
    - Simple `.obj` or `.ply` parser.
    - Convert raw triangles into Simplices (0, 1, 2).
- [ ] **Hasse Diagram Construction**
    - Build the graph of connections (Points <-> Lines <-> Faces).

## 🔮 Phase 3: Conformal Geometric Algebra (CGA)
**Objective:** Upgrade from standard Vector Algebra (VGA) to CGA (5D).
- [ ] **Implement Cl(4, 1)**
    - Add `Signature<4, 1>` to the kernel.
- [ ] **Geometric Primitives**
    - Implement `Point`, `Line`, `Plane`, `Circle`, `Sphere` generation.
- [ ] **Motors & Transformations**
    - Rotors (Rotation) and Translators (Motion) applied to the whole mesh.

## 🏔️ Phase 4: Discrete Morse Theory (DMT)
**Objective:** Analyze the topology of the mesh.
- [ ] **Scalar Field Generation**
    - Assign values (height, curvature) to every vertex.
- [ ] **Gradient Vector Field**
    - Compute discrete gradients (pairing simplices).
- [ ] **Morse-Smale Complex**
    - Identify Critical Points (Minima, Maxima, Saddles).
- [ ] **Simplification**
    - Cancel pairs of critical points to smooth the mesh topology.

## 🎨 Phase 5: Visualization
- [ ] **Export to glTF**
    - Visualize the mesh, vector fields, and critical points in Blender/Web.