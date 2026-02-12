#pragma once
#include <concepts>
#include <cstdint>
#include <span>
#include <vector>

namespace igneous::data {

// ==============================================================================
// 1. CONCEPTS
// ==============================================================================

// The Base Concept: Any structure that manages connectivity for vertices.
template <typename T>
concept Topology = requires(T &t, const T &ct, size_t n_verts, uint32_t idx) {
  // Must declare its dimension
  { T::DIMENSION } -> std::convertible_to<int>;

  // Must support lifecycle management
  // TODO: `build_connectivity` was previusly `build_coboundaries`
  { t.build_connectivity(n_verts) } -> std::same_as<void>;
  { t.clear() } -> std::same_as<void>;

  // Must support generic neighborhood queries (even if empty for Points)
  // We require it returns something range-like or a span
  {
    ct.get_neighborhood(idx)
  } -> std::convertible_to<std::span<const uint32_t>>;
};

// The Surface Concept: A Topology that specifically implies Faces/Polygons.
// Algorithms like "Curvature" or "OBJ Face Parsing" require this.
template <typename T>
concept SurfaceTopology =
    Topology<T> &&
    requires(T &t, const T &ct, size_t f_idx, size_t v_idx, int corner) {
      { ct.num_faces() } -> std::convertible_to<size_t>;
      // Must allow looking up vertices of a face
      {
        ct.get_vertex_for_face(f_idx, corner)
      } -> std::convertible_to<uint32_t>;

      {
        ct.get_faces_for_vertex(v_idx)
      } -> std::convertible_to<std::span<const uint32_t>>;

      // Must expose the raw index vector (mutable and const)
      // This fixes the static_assert error by checking the *expression* type
      // (reference)
      { t.faces_to_vertices } -> std::same_as<std::vector<uint32_t> &>;
      { ct.faces_to_vertices } -> std::same_as<const std::vector<uint32_t> &>;
    };

// ==============================================================================
// 2. IMPLEMENTATIONS
// ==============================================================================

// --- Triangle Topology (Satisfies SurfaceTopology) ---
struct TriangleTopology {
  static constexpr int DIMENSION = 2;

  std::vector<uint32_t> faces_to_vertices;
  std::vector<uint32_t> coboundary_offsets;
  std::vector<uint32_t> coboundary_data;

  size_t num_faces() const { return faces_to_vertices.size() / 3; }

  inline uint32_t get_vertex_for_face(size_t face_idx, int corner) const {
    return faces_to_vertices[face_idx * 3 + corner];
  }

  std::span<const uint32_t> get_faces_for_vertex(uint32_t vertex_idx) const {
    if (vertex_idx + 1 >= coboundary_offsets.size())
      return {};

    uint32_t start = coboundary_offsets[vertex_idx];
    uint32_t end = coboundary_offsets[vertex_idx + 1];
    return {&coboundary_data[start], end - start};
  }

  std::span<const uint32_t> get_neighborhood(uint32_t vertex_idx) const {
    if (vertex_idx + 1 >= coboundary_offsets.size())
      return {};
    uint32_t start = coboundary_offsets[vertex_idx];
    uint32_t end = coboundary_offsets[vertex_idx + 1];
    return {&coboundary_data[start], end - start};
  }

  void build_connectivity(size_t num_vertices) {
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

// --- Point Topology (Satisfies Topology, but NOT SurfaceTopology) ---
struct PointTopology {
  static constexpr int DIMENSION = 0;

  std::span<const uint32_t> get_neighborhood(uint32_t) const {
    return {}; // No connectivity
  }

  void build_connectivity(size_t) { /* No-op */ }

  void clear() {}
};

// Validate concepts at compile time
static_assert(Topology<TriangleTopology>);
static_assert(SurfaceTopology<TriangleTopology>);
static_assert(Topology<PointTopology>);
static_assert(!SurfaceTopology<PointTopology>);

} // namespace igneous::data