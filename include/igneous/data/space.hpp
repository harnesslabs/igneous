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

/**
 * \brief Geometry container parameterized by a single structure type.
 *
 * `Space` owns:
 * - flattened SoA geometry (`x`, `y`, `z`)
 * - one structure instance (`structure`)
 * - optional user-facing name (`name`)
 *
 * The structure is intentionally explicit and built by callers when needed.
 */
template <Structure StructureT = DiscreteExteriorCalculus> struct Space {
  /// \brief Alias for the instantiated structure type.
  using StructureType = StructureT;

  /// \brief X coordinates for all points.
  std::vector<float> x;
  /// \brief Y coordinates for all points.
  std::vector<float> y;
  /// \brief Z coordinates for all points.
  std::vector<float> z;

  /// \brief Structure data associated with this geometry.
  StructureT structure;
  /// \brief Optional descriptive identifier.
  std::string name;

  /**
   * \brief Number of stored points.
   * \return Number of points in the geometry arrays.
   */
  [[nodiscard]] size_t num_points() const { return x.size(); }

  /**
   * \brief Reserve geometry capacity for `vertices` points.
   * \param vertices Number of points to reserve capacity for.
   */
  void reserve(size_t vertices) {
    x.reserve(vertices);
    y.reserve(vertices);
    z.reserve(vertices);
  }

  /**
   * \brief Resize geometry arrays to `vertices` points.
   * \param vertices New point count.
   */
  void resize(size_t vertices) {
    x.resize(vertices);
    y.resize(vertices);
    z.resize(vertices);
  }

  /**
   * \brief Read a 3D point as `core::Vec3`.
   * \param i Point index.
   * \return Point coordinates at index `i`.
   */
  [[nodiscard]] core::Vec3 get_vec3(size_t i) const { return {x[i], y[i], z[i]}; }

  /**
   * \brief Overwrite a 3D point from `core::Vec3`.
   * \param i Point index.
   * \param v Replacement coordinate value.
   */
  void set_vec3(size_t i, const core::Vec3 &v) {
    x[i] = v.x;
    y[i] = v.y;
    z[i] = v.z;
  }

  /**
   * \brief Append a new point to all coordinate channels.
   * \param v Point to append.
   */
  void push_point(const core::Vec3 &v) {
    x.push_back(v.x);
    y.push_back(v.y);
    z.push_back(v.z);
  }

  /**
   * \brief Immutable view of X coordinates.
   * \return Span over `x`.
   */
  [[nodiscard]] std::span<const float> x_span() const { return x; }
  /**
   * \brief Immutable view of Y coordinates.
   * \return Span over `y`.
   */
  [[nodiscard]] std::span<const float> y_span() const { return y; }
  /**
   * \brief Immutable view of Z coordinates.
   * \return Span over `z`.
   */
  [[nodiscard]] std::span<const float> z_span() const { return z; }

  /**
   * \brief Mutable view of X coordinates.
   * \return Mutable span over `x`.
   */
  [[nodiscard]] std::span<float> x_span() { return x; }
  /**
   * \brief Mutable view of Y coordinates.
   * \return Mutable span over `y`.
   */
  [[nodiscard]] std::span<float> y_span() { return y; }
  /**
   * \brief Mutable view of Z coordinates.
   * \return Mutable span over `z`.
   */
  [[nodiscard]] std::span<float> z_span() { return z; }

  /**
   * \brief Immutable grouped axis spans in X/Y/Z order.
   * \return Array `{x_span(), y_span(), z_span()}`.
   */
  [[nodiscard]] std::array<std::span<const float>, 3> xyz_spans() const {
    return {x_span(), y_span(), z_span()};
  }

  /**
   * \brief Lightweight validity check for geometry + structure presence.
   *
   * This does not verify deep structure invariants; it checks whether geometry
   * is non-empty and, for surface structures, faces exist.
   * \return `true` when geometry/structure satisfy minimal checks.
   */
  [[nodiscard]] bool is_valid() const {
    if (num_points() == 0) {
      return false;
    }

    if constexpr (SurfaceStructure<StructureT>) {
      return structure.num_faces() > 0;
    }

    return true;
  }

  /// \brief Clear geometry, structure, and name.
  void clear() {
    x.clear();
    y.clear();
    z.clear();
    structure.clear();
    name.clear();
  }
};

} // namespace igneous::data
