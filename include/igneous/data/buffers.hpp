#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include <igneous/core/blades.hpp>

namespace igneous::data {

template <typename Field, typename Sig> struct GeometryBuffer {
  static_assert(Sig::dim >= 3, "GeometryBuffer requires at least 3 dimensions");

  std::vector<Field> x;
  std::vector<Field> y;
  std::vector<Field> z;

  [[nodiscard]] size_t num_points() const { return x.size(); }

  void reserve(size_t vertices) {
    x.reserve(vertices);
    y.reserve(vertices);
    z.reserve(vertices);
  }

  void resize(size_t vertices) {
    x.resize(vertices);
    y.resize(vertices);
    z.resize(vertices);
  }

  void clear() {
    x.clear();
    y.clear();
    z.clear();
  }

  [[nodiscard]] core::Vec3 get_vec3(size_t i) const { return {x[i], y[i], z[i]}; }

  void set_vec3(size_t i, const core::Vec3 &v) {
    x[i] = v.x;
    y[i] = v.y;
    z[i] = v.z;
  }

  void push_point(const core::Vec3 &v) {
    x.push_back(v.x);
    y.push_back(v.y);
    z.push_back(v.z);
  }

  [[nodiscard]] std::span<const Field> x_span() const { return x; }
  [[nodiscard]] std::span<const Field> y_span() const { return y; }
  [[nodiscard]] std::span<const Field> z_span() const { return z; }

  [[nodiscard]] std::span<Field> x_span() { return x; }
  [[nodiscard]] std::span<Field> y_span() { return y; }
  [[nodiscard]] std::span<Field> z_span() { return z; }

  [[nodiscard]] std::array<std::span<const Field>, 3> xyz_spans() const {
    return {x_span(), y_span(), z_span()};
  }
};

} // namespace igneous::data
