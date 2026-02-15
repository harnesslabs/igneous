#pragma once

#include <concepts>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace igneous::data {

/**
 * \brief Concept for a structure object carried by `data::Space`.
 *
 * A valid structure type must define:
 * - a build input type (`T::Input`)
 * - a topological dimension (`T::DIMENSION`)
 * - explicit lifecycle methods (`build`, `clear`)
 * - a neighborhood query (`get_neighborhood`)
 */
template <typename T>
concept Structure = requires(T& t, const T& ct, uint32_t idx) {
  typename T::Input;
  { T::DIMENSION } -> std::convertible_to<int>;
  { t.build(std::declval<typename T::Input>()) } -> std::same_as<void>;
  { t.clear() } -> std::same_as<void>;
  { ct.get_neighborhood(idx) } -> std::convertible_to<std::span<const uint32_t>>;
};

/**
 * \brief Refinement of `Structure` for surface/triangle-based connectivity.
 *
 * Surface structures expose face and vertex adjacency needed by DEC operators.
 */
template <typename T>
concept SurfaceStructure =
    Structure<T> && requires(T& t, const T& ct, size_t f_idx, size_t v_idx, int corner) {
      { ct.num_faces() } -> std::convertible_to<size_t>;
      { ct.get_vertex_for_face(f_idx, corner) } -> std::convertible_to<uint32_t>;
      { ct.get_faces_for_vertex(v_idx) } -> std::convertible_to<std::span<const uint32_t>>;
      { ct.get_vertex_neighbors(v_idx) } -> std::convertible_to<std::span<const uint32_t>>;
      { t.faces_to_vertices } -> std::same_as<std::vector<uint32_t>&>;
      { ct.faces_to_vertices } -> std::same_as<const std::vector<uint32_t>&>;
    };

} // namespace igneous::data
