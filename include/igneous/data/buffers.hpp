#pragma once
#include <igneous/core/algebra.hpp>
#include <igneous/core/blades.hpp>
#include <vector>

namespace igneous::data {

using igneous::core::IsSignature;
using igneous::core::Multivector;
using igneous::core::Vec3;

template <typename Field, typename Sig> struct GeometryBuffer {
  // Stride per vertex (e.g., 3 for Euclidean3D, 4 for PGA)
  static constexpr size_t STRIDE = Sig::dim;

  // ========================================================================
  // 1. INTERLEAVED STORAGE (Array of Structs)
  // ========================================================================
  // Layout: [x0, y0, z0, x1, y1, z1, ...]
  // This is optimal for random access (Topology/Curvature).
  std::vector<Field> packed_data;

  // ========================================================================
  // 2. BLADE ACCESSORS (Fastest - No Padding)
  // ========================================================================

  // Directly returns a Vec3 struct (12 bytes).
  // This bypasses Multivector construction entirely.
  igneous::core::Vec3 get_vec3(size_t i) const {
    size_t offset = i * STRIDE;
    // Assume standard layout: first 3 components are spatial x, y, z
    return {packed_data[offset], packed_data[offset + 1],
            packed_data[offset + 2]};
  }

  void set_vec3(size_t i, const igneous::core::Vec3 &v) {
    size_t offset = i * STRIDE;
    packed_data[offset] = v.x;
    packed_data[offset + 1] = v.y;
    packed_data[offset + 2] = v.z;
    // Note: For PGA/CGA, we leave the higher dimensions (w, n_inf) untouched.
    // This is exactly what we want for spatial flow.
  }

  // ========================================================================
  // 3. MULTIVECTOR ACCESSORS (Generic Lift)
  // ========================================================================

  Multivector<Field, Sig> get_point(size_t i) const {
    Multivector<Field, Sig> mv;
    size_t offset = i * STRIDE;
    // Map interleaved array to Basis Indices (1, 2, 4...)
    for (int k = 0; k < Sig::dim; ++k) {
      mv[1 << k] = packed_data[offset + k];
    }
    return mv;
  }

  void set_point(size_t i, const Multivector<Field, Sig> &mv) {
    size_t offset = i * STRIDE;
    for (int k = 0; k < Sig::dim; ++k) {
      packed_data[offset + k] = mv[1 << k];
    }
  }

  void push_point(const Multivector<Field, Sig> &mv) {
    for (int k = 0; k < Sig::dim; ++k) {
      packed_data.push_back(mv[1 << k]);
    }
  }

  // ========================================================================
  // 4. UTILITY
  // ========================================================================
  size_t num_points() const { return packed_data.size() / STRIDE; }

  void reserve(size_t v, size_t, size_t) { packed_data.reserve(v * STRIDE); }

  void resize(size_t v) { packed_data.resize(v * STRIDE); }

  void clear() { packed_data.clear(); }
};

} // namespace igneous::data