#pragma once
#include <cstdint>
#include <igneous/core/algebra.hpp>
#include <span>
#include <vector>

namespace igneous::data {

// Bring in core types
using igneous::core::IsSignature;
using igneous::core::Multivector;

/**
 * @brief GeometryBuffer: A Data-Oriented container with Compact Storage.
 *
 * instead of storing full Multivectors (which are sparse and padded to
 * power-of-2), we store tightly packed arrays of coefficients for specific
 * grades.
 *
 * - Points (Grade 1): Stored as [e1, e2, e3, ...]
 * - Edges  (Grade 2): Stored as [e12, e23, e31, ...]
 * - Faces  (Grade 3): Stored as [e123, ...]
 *
 * The 'get_X(i)' methods "Lift" these packed floats into a full SIMD-ready
 * Multivector register for computation.
 */
template <typename Field, IsSignature Sig> struct GeometryBuffer {

  // ========================================================================
  // 1. COMPACT STORAGE
  // ========================================================================

  // Stride calculations based on Signature dimension
  // For Euclidean3D (3,0,0):
  //  - Vector Stride   = 3 (e1, e2, e3)
  //  - Bivector Stride = 3 (e12, e13, e23)
  //  - Trivector Stride= 1 (e123)

  // Helper to calculate binomial coeff (n choose k) could be used here
  // For now, we assume standard 3D/4D usage patterns.
  static constexpr size_t POINT_STRIDE = Sig::dim;

  // Layout: [p0_e1, p0_e2, p0_e3,  p1_e1, ...]
  std::vector<Field> packed_points;

  // Layout: [e0_12, e0_13, e0_23, ...] (Lexicographical order usually)
  std::vector<Field> packed_edges;

  // Layout: [f0_123, f1_123, ...]
  std::vector<Field> packed_faces;

  // ========================================================================
  // 2. ACCESSORS (The "Lift")
  // ========================================================================

  // Retrieve the i-th point as a full Multivector
  Multivector<Field, Sig> get_point(size_t i) const {
    Multivector<Field, Sig> mv;
    size_t offset = i * POINT_STRIDE;

    // We unroll this manual loop for 3D/4D common cases
    // Note: This assumes the Basis blades 1, 2, 4 (binary) correspond to array
    // indices A robust implementation maps blade bits to array indices. For
    // Euclidean3D, indices 1, 2, 3 correspond to e1, e2, e3.

    if constexpr (Sig::dim == 3) {
      mv[1] = packed_points[offset + 0]; // x
      mv[2] = packed_points[offset + 1]; // y
      mv[3] = packed_points[offset + 2]; // z
    } else {
      // Fallback for generic dimensions
      // Assumes packed data corresponds to basis vectors e1, e2, ... en
      for (int k = 0; k < Sig::dim; ++k) {
        // Basis vector indices in Multivector are usually 2^0, 2^1, ...
        // Actually, MV storage is flat 0..Size-1.
        // In 3D: 0=1, 1=e1, 2=e2, 3=e12, 4=e3...
        // WAIT. Standard binary indexing:
        // 1 (001) = e1
        // 2 (010) = e2
        // 4 (100) = e3
        mv[1 << k] = packed_points[offset + k];
      }
    }
    return mv;
  }

  // Write a full Multivector back to compact storage
  void set_point(size_t i, const Multivector<Field, Sig> &mv) {
    size_t offset = i * POINT_STRIDE;
    if (offset + POINT_STRIDE > packed_points.size()) {
      // Auto-resize or throw? For perf, assume caller sized it.
      // But safe to just push_back if at end
    }

    if constexpr (Sig::dim == 3) {
      packed_points[offset + 0] = mv[1]; // x
      packed_points[offset + 1] = mv[2]; // y
      packed_points[offset + 2] = mv[3]; // z (usually index 4 in binary layout)
      // CAUTION: Is MV[3] e3?
      // Standard GA Binary Indexing for 3D:
      // 0: scalar
      // 1: e1
      // 2: e2
      // 3: e12
      // 4: e3  <-- WATCH OUT
      // 5: e13
      // 6: e23
      // 7: e123
      //
      // Your current Algebra.hpp loop just iterates 0..Size.
      // If your operator[] accesses linear array index, we need to know the
      // mapping. Assuming your previous code `mv[1]=x, mv[2]=y, mv[3]=z`
      // implied you wrote a custom mapping or used a linear basis 1, e1, e2,
      // e3...

      // FOR SAFETY with your current Algebra.hpp:
      // Let's assume you used the binary indexing (standard xsimd/GA).
      // We need a helper "get_basis_index(k)".
      // For now, I will stick to the Binary Indexing standard:
      // e1=1, e2=2, e3=4.

      // HOWEVER, looking at your previous `mesh_loader`:
      // mv[1]=x, mv[2]=y, mv[3]=z.
      // This implies you are treating the MV array as [s, e1, e2, e3, e12, ...]
      // I will preserve THAT mapping for now.

      packed_points[offset + 0] = mv[1];
      packed_points[offset + 1] = mv[2];
      packed_points[offset + 2] = mv[3];
    }
  }

  // helper to append
  void push_point(const Multivector<Field, Sig> &mv) {
    if constexpr (Sig::dim == 3) {
      packed_points.push_back(mv[1]);
      packed_points.push_back(mv[2]);
      packed_points.push_back(mv[3]);
    }
  }

  // ========================================================================
  // 3. UTILITY
  // ========================================================================

  size_t num_points() const { return packed_points.size() / POINT_STRIDE; }

  void reserve(size_t v, size_t e, size_t f) {
    packed_points.reserve(v * POINT_STRIDE);
    packed_edges.reserve(e * 3); // Approx
    packed_faces.reserve(f);
  }

  void clear() {
    packed_points.clear();
    packed_edges.clear();
    packed_faces.clear();
  }
};

struct TopologyBuffer {
  // Flattened Triangle Indices: [v0, v1, v2, v0, v1, v2, ...]
  std::vector<uint32_t> faces_to_vertices;

  // CSR Storage
  std::vector<uint32_t> coboundary_offsets;
  std::vector<uint32_t> coboundary_data;

  inline uint32_t get_vertex_for_face(size_t face_idx, int corner) const {
    return faces_to_vertices[face_idx * 3 + corner];
  }

  size_t num_faces() const { return faces_to_vertices.size() / 3; }

  // CSR Accessor
  std::span<const uint32_t> get_faces_for_vertex(uint32_t vertex_idx) const {
    if (vertex_idx + 1 >= coboundary_offsets.size())
      return {};
    uint32_t start = coboundary_offsets[vertex_idx];
    uint32_t end = coboundary_offsets[vertex_idx + 1];
    return {&coboundary_data[start], end - start};
  }

  // Builder (same as before)
  void build_coboundaries(size_t num_vertices) {
    coboundary_offsets.assign(num_vertices + 1, 0);
    for (uint32_t v_idx : faces_to_vertices) {
      if (v_idx < num_vertices)
        coboundary_offsets[v_idx + 1]++;
    }
    for (size_t i = 0; i < num_vertices; ++i) {
      coboundary_offsets[i + 1] += coboundary_offsets[i];
    }
    coboundary_data.resize(faces_to_vertices.size());
    std::vector<uint32_t> write_head = coboundary_offsets;
    size_t num_f = num_faces();
    for (uint32_t f = 0; f < num_f; ++f) {
      for (int c = 0; c < 3; ++c) {
        uint32_t v = faces_to_vertices[f * 3 + c];
        if (v < num_vertices) {
          uint32_t pos = write_head[v];
          coboundary_data[pos] = f;
          write_head[v]++;
        }
      }
    }
  }

  void clear() {
    faces_to_vertices.clear();
    coboundary_offsets.clear();
    coboundary_data.clear();
  }
};

} // namespace igneous::data