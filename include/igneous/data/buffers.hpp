#pragma once
#include <cstdint>
#include <igneous/core/algebra.hpp>
#include <span>
#include <vector>

namespace igneous::data {

// Bring in core types
using igneous::core::IsSignature;
using igneous::core::Multivector;

using igneous::core::Multivector;
using igneous::core::Packet; // The SIMD Batch (e.g. float x 8)
using igneous::core::WideMultivector;

template <typename Field, typename Sig> struct GeometryBuffer {
  // ========================================================================
  // 1. SoA STORAGE (Structure of Arrays)
  // ========================================================================
  // Instead of [x0, y0, z0, x1...], we store:
  // components[0]: [x0, x1, x2, x3...]
  // components[1]: [y0, y1, y2, y3...]
  // components[2]: [z0, z1, z2, z3...]
  // This layout is "SIMD-Native".
  std::array<std::vector<Field>, Sig::dim> components;

  // ========================================================================
  // 2. SCALAR ACCESSORS (For random access / topology)
  // ========================================================================

  Multivector<Field, Sig> get_point(size_t i) const {
    Multivector<Field, Sig> mv;
    // Map SoA arrays back to Multivector basis indices (1, 2, 4...)
    for (int k = 0; k < Sig::dim; ++k) {
      mv[1 << k] = components[k][i];
    }
    return mv;
  }

  void set_point(size_t i, const Multivector<Field, Sig> &mv) {
    for (int k = 0; k < Sig::dim; ++k) {
      components[k][i] = mv[1 << k];
    }
  }

  void push_point(const Multivector<Field, Sig> &mv) {
    for (int k = 0; k < Sig::dim; ++k) {
      components[k].push_back(mv[1 << k]);
    }
  }

  // ========================================================================
  // 3. WIDE ACCESSORS (The "Vectorized" Lift)
  // ========================================================================
  // Returns a SIMD batch of 8 vertices at once!

  WideMultivector<Sig> get_batch(size_t i) const {
    WideMultivector<Sig> wide_mv;
    for (int k = 0; k < Sig::dim; ++k) {
      // Aligned load of 8 floats from the k-th component array
      wide_mv[1 << k] = xsimd::load_unaligned(&components[k][i]);
    }
    return wide_mv;
  }

  void set_batch(size_t i, const WideMultivector<Sig> &wide_mv) {
    for (int k = 0; k < Sig::dim; ++k) {
      wide_mv[1 << k].store_unaligned(&components[k][i]);
    }
  }

  // ========================================================================
  // 4. UTILITY
  // ========================================================================
  size_t num_points() const { return components[0].size(); }

  void reserve(size_t v, size_t, size_t) {
    for (auto &vec : components)
      vec.reserve(v);
  }

  void clear() {
    for (auto &vec : components)
      vec.clear();
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