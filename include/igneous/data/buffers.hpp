#pragma once
#include <cstdint>
#include <igneous/core/algebra.hpp>
#include <span>
#include <vector>

namespace igneous::data {

using igneous::core::IsSignature;
using igneous::core::Multivector;

/**
 * @brief Aligned allocator to ensure SIMD compatibility for Multivectors.
 */
template <typename T> struct AlignedAllocator {
  using value_type = T;
  static constexpr size_t alignment = xsimd::default_arch::alignment();

  T *allocate(std::size_t n) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, n * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<T *>(ptr);
  }

  void deallocate(T *p, std::size_t) { free(p); }
};

/**
 * @brief GeometryBuffer: A Data-Oriented container for CGA primitives.
 * * Instead of storing objects, we store flat, contiguous arrays of blades.
 * This allows the CPU to stream geometry into SIMD registers without cache
 * misses.
 */
template <typename Field, IsSignature Sig> struct GeometryBuffer {
  // 0-Simplices: Points (1-blades in CGA)
  // Layout: [P0, P1, P2, P3, ... Pn]
  std::vector<Multivector<Field, Sig>,
              AlignedAllocator<Multivector<Field, Sig>>>
      points;

  // 1-Simplices: Edges/Point-Pairs (2-blades in CGA)
  // Storing the dual bivector allows for instant intersection tests.
  std::vector<Multivector<Field, Sig>,
              AlignedAllocator<Multivector<Field, Sig>>>
      edges;

  // 2-Simplices: Faces/Circles (3-blades in CGA)
  // The trivector represents the circumcircle and surface orientation.
  std::vector<Multivector<Field, Sig>,
              AlignedAllocator<Multivector<Field, Sig>>>
      faces;

  void reserve(size_t v, size_t e, size_t f) {
    points.reserve(v);
    edges.reserve(e);
    faces.reserve(f);
  }

  void clear() {
    points.clear();
    edges.clear();
    faces.clear();
  }
};

struct TopologyBuffer {
  // ========================================================================
  // 1. BOUNDARY (Downwards: Face -> Vertices)
  // ========================================================================

  // Flattened Triangle Indices: [v0, v1, v2, v0, v1, v2, ...]
  std::vector<uint32_t> faces_to_vertices;

  inline uint32_t get_vertex_for_face(size_t face_idx, int corner) const {
    return faces_to_vertices[face_idx * 3 + corner];
  }

  size_t num_faces() const { return faces_to_vertices.size() / 3; }

  // ========================================================================
  // 2. COBOUNDARY (Upwards: Vertex -> Faces)
  // ========================================================================
  // Implemented as Compressed Sparse Row (CSR).
  // This replaces "std::vector<std::vector<int>>" with two flat arrays.

  // Where does the list for Vertex i start?
  // Range for vertex i is: [ offsets[i], offsets[i+1] )
  std::vector<uint32_t> coboundary_offsets;

  // The actual face indices packed together.
  std::vector<uint32_t> coboundary_data;

  // --- ACCESSOR ---
  // Returns a view of the face indices connected to the given vertex.
  std::span<const uint32_t> get_faces_for_vertex(uint32_t vertex_idx) const {
    if (vertex_idx + 1 >= coboundary_offsets.size())
      return {};

    uint32_t start = coboundary_offsets[vertex_idx];
    uint32_t end = coboundary_offsets[vertex_idx + 1];
    return {&coboundary_data[start], end - start};
  }

  // --- BUILDER ---
  // Constructs the reverse lookup map (Vertex -> Face)
  // This must be called after loading the OBJ and before processing curvature.
  void build_coboundaries(size_t num_vertices) {
    // 1. Resize Offsets
    // We need N+1 offsets to define N ranges.
    coboundary_offsets.assign(num_vertices + 1, 0);

    // 2. PASS 1: Count Valences (Histogram)
    // How many faces touch each vertex?
    for (uint32_t v_idx : faces_to_vertices) {
      if (v_idx < num_vertices) {
        coboundary_offsets[v_idx + 1]++;
      }
    }

    // 3. PASS 2: Prefix Sum (Scan)
    // Convert counts to start indices.
    // offsets[i] becomes the cumulative sum of counts before i.
    for (size_t i = 0; i < num_vertices; ++i) {
      coboundary_offsets[i + 1] += coboundary_offsets[i];
    }

    // 4. Allocation
    // The total size is exactly the number of face corners (num_faces * 3).
    coboundary_data.resize(faces_to_vertices.size());

    // 5. PASS 3: Scatter (Fill)
    // We need a temporary cursor to track where we are writing for each vertex.
    // We can reuse the 'offsets' array logic, but we need a copy to not corrupt
    // the read-offsets. Actually, standard trick: use a temp 'write_head'
    // vector.
    std::vector<uint32_t> write_head = coboundary_offsets;

    size_t num_f = num_faces();
    for (uint32_t f = 0; f < num_f; ++f) {
      for (int c = 0; c < 3; ++c) {
        uint32_t v = faces_to_vertices[f * 3 + c];
        if (v < num_vertices) {
          // Look up where to write for this vertex
          uint32_t pos = write_head[v];
          coboundary_data[pos] = f; // Store the Face Index
          write_head[v]++;          // Advance the cursor
        }
      }
    }
    // Done! 'coboundary_offsets' is now ready for reading.
  }

  void clear() {
    faces_to_vertices.clear();
    coboundary_offsets.clear();
    coboundary_data.clear();
  }
};
} // namespace igneous::data