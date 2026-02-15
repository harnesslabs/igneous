#pragma once

#include <array>
#include <span>
#include <string>
#include <vector>

#include <igneous/core/blades.hpp>
#include <igneous/data/structure.hpp>
#include <igneous/data/structures/discrete_exterior_calculus.hpp>
#include <igneous/data/structures/diffusion_geometry.hpp>

namespace igneous::data {

template <Structure StructureT = DiscreteExteriorCalculus> struct Space {
  using StructureType = StructureT;

  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;

  StructureT structure;
  std::string name;

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

  [[nodiscard]] std::span<const float> x_span() const { return x; }
  [[nodiscard]] std::span<const float> y_span() const { return y; }
  [[nodiscard]] std::span<const float> z_span() const { return z; }

  [[nodiscard]] std::span<float> x_span() { return x; }
  [[nodiscard]] std::span<float> y_span() { return y; }
  [[nodiscard]] std::span<float> z_span() { return z; }

  [[nodiscard]] std::array<std::span<const float>, 3> xyz_spans() const {
    return {x_span(), y_span(), z_span()};
  }

  [[nodiscard]] bool is_valid() const {
    if (num_points() == 0) {
      return false;
    }

    if constexpr (SurfaceStructure<StructureT>) {
      return structure.num_faces() > 0;
    }

    return true;
  }

  void clear() {
    x.clear();
    y.clear();
    z.clear();
    structure.clear();
    name.clear();
  }
};

} // namespace igneous::data
