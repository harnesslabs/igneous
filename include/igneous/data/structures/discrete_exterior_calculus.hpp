#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <igneous/core/parallel.hpp>
#include <igneous/data/structure.hpp>

namespace igneous::data {

/**
 * \brief Triangle-surface structure used by DEC-style operators.
 *
 * Connectivity is stored in CSR-style arrays for face incidence and
 * vertex-neighbor traversal.
 */
struct DiscreteExteriorCalculus {
  /// \brief Topological dimension of this structure.
  static constexpr int DIMENSION = 2;

  /// \brief Input triangle index buffer (`[f0v0, f0v1, f0v2, ...]`).
  std::vector<uint32_t> faces_to_vertices;

  /// \brief Split face corner arrays for cache-friendly face traversal.
  std::vector<uint32_t> face_v0;
  /// \brief Split face corner arrays for cache-friendly face traversal.
  std::vector<uint32_t> face_v1;
  /// \brief Split face corner arrays for cache-friendly face traversal.
  std::vector<uint32_t> face_v2;

  /// \brief CSR offsets for face incidence per vertex.
  std::vector<uint32_t> vertex_face_offsets;
  /// \brief CSR payload of face indices per vertex.
  std::vector<uint32_t> vertex_face_data;

  /// \brief CSR offsets for 1-ring neighbors per vertex.
  std::vector<uint32_t> vertex_neighbor_offsets;
  /// \brief CSR payload of neighbor vertex indices.
  std::vector<uint32_t> vertex_neighbor_data;

  /// \brief Parameters used by `build`.
  struct Input {
    /// \brief Number of vertices in the associated geometry.
    size_t num_vertices = 0;
    /// \brief Whether to build vertex-neighbor CSR in addition to face incidence.
    bool build_vertex_neighbors = true;
  };

  /**
   * \brief Number of triangles in the structure.
   * \return Triangle count.
   */
  [[nodiscard]] size_t num_faces() const {
    if (!face_v0.empty()) {
      return face_v0.size();
    }
    return faces_to_vertices.size() / 3;
  }

  /**
   * \brief Return the vertex index for a face corner (`corner` in `[0,2]`).
   * \param face_idx Face index.
   * \param corner Corner index (`0`, `1`, or `2`).
   * \return Vertex index for the requested corner.
   */
  [[nodiscard]] uint32_t get_vertex_for_face(size_t face_idx, int corner) const {
    if (corner == 0) {
      return face_v0[face_idx];
    }
    if (corner == 1) {
      return face_v1[face_idx];
    }
    return face_v2[face_idx];
  }

  /**
   * \brief Faces incident to vertex `vertex_idx`.
   * \param vertex_idx Vertex index.
   * \return Span of incident face indices.
   */
  [[nodiscard]] std::span<const uint32_t> get_faces_for_vertex(uint32_t vertex_idx) const {
    if (vertex_idx + 1 >= vertex_face_offsets.size()) {
      return {};
    }

    const uint32_t begin = vertex_face_offsets[vertex_idx];
    const uint32_t end = vertex_face_offsets[vertex_idx + 1];
    return {&vertex_face_data[begin], end - begin};
  }

  /**
   * \brief 1-ring neighbors for vertex `vertex_idx`.
   * \param vertex_idx Vertex index.
   * \return Span of neighboring vertex indices.
   */
  [[nodiscard]] std::span<const uint32_t> get_vertex_neighbors(uint32_t vertex_idx) const {
    if (vertex_idx + 1 >= vertex_neighbor_offsets.size()) {
      return {};
    }

    const uint32_t begin = vertex_neighbor_offsets[vertex_idx];
    const uint32_t end = vertex_neighbor_offsets[vertex_idx + 1];
    return {&vertex_neighbor_data[begin], end - begin};
  }

  /**
   * \brief Generic neighborhood accessor required by `data::Structure`.
   * \param vertex_idx Vertex index.
   * \return Same data as `get_faces_for_vertex(vertex_idx)`.
   */
  [[nodiscard]] std::span<const uint32_t> get_neighborhood(uint32_t vertex_idx) const {
    return get_faces_for_vertex(vertex_idx);
  }

  /**
   * \brief Build CSR structures from `faces_to_vertices`.
   *
   * If `face_v*` arrays are already provided, they are reused.
   * \param input Build configuration and vertex count.
   */
  void build(Input input) {
    const size_t num_vertices = input.num_vertices;

    if (!faces_to_vertices.empty()) {
      const size_t n_faces = faces_to_vertices.size() / 3;
      face_v0.resize(n_faces);
      face_v1.resize(n_faces);
      face_v2.resize(n_faces);

      core::parallel_for_index(
          0, static_cast<int>(n_faces),
          [&](int face_idx) {
            const size_t f = static_cast<size_t>(face_idx);
            const size_t base = f * 3;
            face_v0[f] = faces_to_vertices[base + 0];
            face_v1[f] = faces_to_vertices[base + 1];
            face_v2[f] = faces_to_vertices[base + 2];
          },
          32768);
    }

    const size_t n_faces = num_faces();

    vertex_face_offsets.assign(num_vertices + 1, 0);
    for (size_t f = 0; f < n_faces; ++f) {
      const uint32_t a = face_v0[f];
      const uint32_t b = face_v1[f];
      const uint32_t c = face_v2[f];

      if (a < num_vertices)
        vertex_face_offsets[a + 1]++;
      if (b < num_vertices)
        vertex_face_offsets[b + 1]++;
      if (c < num_vertices)
        vertex_face_offsets[c + 1]++;
    }

    for (size_t i = 0; i < num_vertices; ++i) {
      vertex_face_offsets[i + 1] += vertex_face_offsets[i];
    }

    vertex_face_data.resize(n_faces * 3);
    std::vector<uint32_t> face_write_head = vertex_face_offsets;

    for (uint32_t f = 0; f < n_faces; ++f) {
      const uint32_t corners[3] = {face_v0[f], face_v1[f], face_v2[f]};
      for (uint32_t v : corners) {
        if (v < num_vertices) {
          const uint32_t pos = face_write_head[v]++;
          vertex_face_data[pos] = f;
        }
      }
    }

    if (!input.build_vertex_neighbors) {
      vertex_neighbor_offsets.clear();
      vertex_neighbor_data.clear();
      return;
    }

    vertex_neighbor_offsets.assign(num_vertices + 1, 0);

    constexpr size_t kSmallMeshNeighborThreshold = 50000;
    if (num_vertices < kSmallMeshNeighborThreshold) {
      std::vector<uint32_t> seen_neighbor_stamp(num_vertices, 0);
      uint32_t stamp = 1;

      for (uint32_t v = 0; v < num_vertices; ++v) {
        if (stamp == 0) {
          std::fill(seen_neighbor_stamp.begin(), seen_neighbor_stamp.end(), 0);
          stamp = 1;
        }

        uint32_t count = 0;
        const uint32_t face_begin = vertex_face_offsets[v];
        const uint32_t face_end = vertex_face_offsets[v + 1];
        for (uint32_t idx = face_begin; idx < face_end; ++idx) {
          const uint32_t f = vertex_face_data[idx];
          const uint32_t corners[3] = {face_v0[f], face_v1[f], face_v2[f]};
          for (uint32_t u : corners) {
            if (u >= num_vertices || u == v || seen_neighbor_stamp[u] == stamp) {
              continue;
            }
            seen_neighbor_stamp[u] = stamp;
            ++count;
          }
        }

        vertex_neighbor_offsets[v + 1] = count;
        ++stamp;
      }

      for (size_t i = 0; i < num_vertices; ++i) {
        vertex_neighbor_offsets[i + 1] += vertex_neighbor_offsets[i];
      }

      vertex_neighbor_data.resize(vertex_neighbor_offsets[num_vertices]);

      std::fill(seen_neighbor_stamp.begin(), seen_neighbor_stamp.end(), 0);
      stamp = 1;
      for (uint32_t v = 0; v < num_vertices; ++v) {
        if (stamp == 0) {
          std::fill(seen_neighbor_stamp.begin(), seen_neighbor_stamp.end(), 0);
          stamp = 1;
        }

        uint32_t write = vertex_neighbor_offsets[v];
        const uint32_t face_begin = vertex_face_offsets[v];
        const uint32_t face_end = vertex_face_offsets[v + 1];
        for (uint32_t idx = face_begin; idx < face_end; ++idx) {
          const uint32_t f = vertex_face_data[idx];
          const uint32_t corners[3] = {face_v0[f], face_v1[f], face_v2[f]};
          for (uint32_t u : corners) {
            if (u >= num_vertices || u == v || seen_neighbor_stamp[u] == stamp) {
              continue;
            }
            seen_neighbor_stamp[u] = stamp;
            vertex_neighbor_data[write++] = u;
          }
        }
        ++stamp;
      }
      return;
    }

    const auto gather_unique_neighbors = [&](uint32_t v, std::vector<uint32_t>& neighbors) {
      neighbors.clear();
      const uint32_t face_begin = vertex_face_offsets[v];
      const uint32_t face_end = vertex_face_offsets[v + 1];

      const auto try_push = [&](uint32_t u) {
        if (u >= num_vertices || u == v) {
          return;
        }
        for (uint32_t existing : neighbors) {
          if (existing == u) {
            return;
          }
        }
        neighbors.push_back(u);
      };

      const size_t candidate_hint =
          static_cast<size_t>(std::max<uint32_t>(1, face_end - face_begin) * 2);
      if (neighbors.capacity() < candidate_hint) {
        neighbors.reserve(candidate_hint);
      }

      for (uint32_t idx = face_begin; idx < face_end; ++idx) {
        const uint32_t f = vertex_face_data[idx];
        try_push(face_v0[f]);
        try_push(face_v1[f]);
        try_push(face_v2[f]);
      }
    };

    core::parallel_for_index(
        0, static_cast<int>(num_vertices),
        [&](int vertex_idx) {
          const uint32_t v = static_cast<uint32_t>(vertex_idx);
          thread_local std::vector<uint32_t> neighbors;
          gather_unique_neighbors(v, neighbors);
          vertex_neighbor_offsets[static_cast<size_t>(v) + 1] =
              static_cast<uint32_t>(neighbors.size());
        },
        32768);

    for (size_t i = 0; i < num_vertices; ++i) {
      vertex_neighbor_offsets[i + 1] += vertex_neighbor_offsets[i];
    }

    vertex_neighbor_data.resize(vertex_neighbor_offsets[num_vertices]);

    core::parallel_for_index(
        0, static_cast<int>(num_vertices),
        [&](int vertex_idx) {
          const uint32_t v = static_cast<uint32_t>(vertex_idx);
          thread_local std::vector<uint32_t> neighbors;
          gather_unique_neighbors(v, neighbors);

          uint32_t write = vertex_neighbor_offsets[v];
          for (uint32_t u : neighbors) {
            vertex_neighbor_data[write++] = u;
          }
        },
        32768);
  }

  /// \brief Drop all connectivity storage.
  void clear() {
    faces_to_vertices.clear();
    face_v0.clear();
    face_v1.clear();
    face_v2.clear();
    vertex_face_offsets.clear();
    vertex_face_data.clear();
    vertex_neighbor_offsets.clear();
    vertex_neighbor_data.clear();
  }
};

static_assert(Structure<DiscreteExteriorCalculus>);
static_assert(SurfaceStructure<DiscreteExteriorCalculus>);

} // namespace igneous::data
