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

  // Layout: [x0, y0, z0, x1, y1, z1, ...]
  // This is optimal for random access (Topology/Curvature).
  std::vector<Field> packed_data;
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
  }

  void push_point(const igneous::core::Vec3 &v) {
    packed_data.push_back(v.x);
    packed_data.push_back(v.y);
    packed_data.push_back(v.z);
  }

  size_t num_points() const { return packed_data.size() / STRIDE; }

  void reserve(size_t v, size_t, size_t) { packed_data.reserve(v * STRIDE); }

  void resize(size_t v) { packed_data.resize(v * STRIDE); }

  void clear() { packed_data.clear(); }
};

} // namespace igneous::data