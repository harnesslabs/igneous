#pragma once

#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <nanoflann.hpp>
#include <span>
#include <vector>

namespace igneous::data {

template <typename T>
concept Topology = requires(T &t, const T &ct, uint32_t idx) {
  typename T::Input;
  { T::DIMENSION } -> std::convertible_to<int>;
  { t.build(std::declval<typename T::Input>()) } -> std::same_as<void>;
  { t.clear() } -> std::same_as<void>;
  { ct.get_neighborhood(idx) } -> std::convertible_to<std::span<const uint32_t>>;
};

template <typename T>
concept SurfaceTopology =
    Topology<T> && requires(T &t, const T &ct, size_t f_idx, size_t v_idx, int corner) {
      { ct.num_faces() } -> std::convertible_to<size_t>;
      { ct.get_vertex_for_face(f_idx, corner) } -> std::convertible_to<uint32_t>;
      { ct.get_faces_for_vertex(v_idx) } -> std::convertible_to<std::span<const uint32_t>>;
      { ct.get_vertex_neighbors(v_idx) } -> std::convertible_to<std::span<const uint32_t>>;
      { t.faces_to_vertices } -> std::same_as<std::vector<uint32_t> &>;
      { ct.faces_to_vertices } -> std::same_as<const std::vector<uint32_t> &>;
    };

struct TriangleTopology {
  static constexpr int DIMENSION = 2;

  // Ingest representation
  std::vector<uint32_t> faces_to_vertices;

  // Hot-path face storage
  std::vector<uint32_t> face_v0;
  std::vector<uint32_t> face_v1;
  std::vector<uint32_t> face_v2;

  // Vertex -> incident faces CSR
  std::vector<uint32_t> vertex_face_offsets;
  std::vector<uint32_t> vertex_face_data;

  // Vertex -> unique neighboring vertices CSR
  std::vector<uint32_t> vertex_neighbor_offsets;
  std::vector<uint32_t> vertex_neighbor_data;

  struct Input {
    size_t num_vertices = 0;
    bool build_vertex_neighbors = true;
  };

  [[nodiscard]] size_t num_faces() const {
    if (!face_v0.empty()) {
      return face_v0.size();
    }
    return faces_to_vertices.size() / 3;
  }

  [[nodiscard]] uint32_t get_vertex_for_face(size_t face_idx, int corner) const {
    switch (corner) {
    case 0:
      return face_v0[face_idx];
    case 1:
      return face_v1[face_idx];
    default:
      return face_v2[face_idx];
    }
  }

  [[nodiscard]] std::span<const uint32_t> get_faces_for_vertex(uint32_t vertex_idx) const {
    if (vertex_idx + 1 >= vertex_face_offsets.size()) {
      return {};
    }

    const uint32_t begin = vertex_face_offsets[vertex_idx];
    const uint32_t end = vertex_face_offsets[vertex_idx + 1];
    return {&vertex_face_data[begin], end - begin};
  }

  [[nodiscard]] std::span<const uint32_t> get_vertex_neighbors(uint32_t vertex_idx) const {
    if (vertex_idx + 1 >= vertex_neighbor_offsets.size()) {
      return {};
    }

    const uint32_t begin = vertex_neighbor_offsets[vertex_idx];
    const uint32_t end = vertex_neighbor_offsets[vertex_idx + 1];
    return {&vertex_neighbor_data[begin], end - begin};
  }

  [[nodiscard]] std::span<const uint32_t> get_neighborhood(uint32_t vertex_idx) const {
    return get_faces_for_vertex(vertex_idx);
  }

  void build(Input input) {
    const size_t num_vertices = input.num_vertices;

    if (!faces_to_vertices.empty()) {
      const size_t n_faces = faces_to_vertices.size() / 3;
      face_v0.resize(n_faces);
      face_v1.resize(n_faces);
      face_v2.resize(n_faces);

      for (size_t f = 0; f < n_faces; ++f) {
        const size_t base = f * 3;
        face_v0[f] = faces_to_vertices[base + 0];
        face_v1[f] = faces_to_vertices[base + 1];
        face_v2[f] = faces_to_vertices[base + 2];
      }
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

    std::vector<uint64_t> directed_edges;
    directed_edges.reserve(n_faces * 6);

    const auto push_edge = [&](uint32_t src, uint32_t dst) {
      if (src < num_vertices && dst < num_vertices && src != dst) {
        directed_edges.push_back((static_cast<uint64_t>(src) << 32) | dst);
      }
    };

    for (size_t f = 0; f < n_faces; ++f) {
      const uint32_t a = face_v0[f];
      const uint32_t b = face_v1[f];
      const uint32_t c = face_v2[f];
      push_edge(a, b);
      push_edge(a, c);
      push_edge(b, a);
      push_edge(b, c);
      push_edge(c, a);
      push_edge(c, b);
    }

    std::sort(directed_edges.begin(), directed_edges.end());
    directed_edges.erase(std::unique(directed_edges.begin(), directed_edges.end()), directed_edges.end());

    vertex_neighbor_offsets.assign(num_vertices + 1, 0);
    for (uint64_t edge : directed_edges) {
      const uint32_t src = static_cast<uint32_t>(edge >> 32);
      vertex_neighbor_offsets[src + 1]++;
    }

    for (size_t i = 0; i < num_vertices; ++i) {
      vertex_neighbor_offsets[i + 1] += vertex_neighbor_offsets[i];
    }

    vertex_neighbor_data.resize(directed_edges.size());
    std::vector<uint32_t> neighbor_write_head = vertex_neighbor_offsets;

    for (uint64_t edge : directed_edges) {
      const uint32_t src = static_cast<uint32_t>(edge >> 32);
      const uint32_t dst = static_cast<uint32_t>(edge & 0xFFFFFFFFu);
      const uint32_t pos = neighbor_write_head[src]++;
      vertex_neighbor_data[pos] = dst;
    }
  }

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

struct PointTopology {
  static constexpr int DIMENSION = 0;

  struct Input {};

  [[nodiscard]] std::span<const uint32_t> get_neighborhood(uint32_t) const { return {}; }

  void build(Input) {}
  void clear() {}
};

struct DiffusionTopology {
  static constexpr int DIMENSION = 0;

  struct Input {
    std::span<const float> x;
    std::span<const float> y;
    std::span<const float> z;
    float bandwidth = 0.01f;
    int k_neighbors = 32;
  };

  Eigen::SparseMatrix<float> P;
  Eigen::VectorXf mu;
  Eigen::MatrixXf eigen_basis;

  [[nodiscard]] size_t num_primitives() const { return static_cast<size_t>(P.rows()); }
  [[nodiscard]] std::span<const uint32_t> get_neighborhood(uint32_t) const { return {}; }

  void clear() {
    P.resize(0, 0);
    mu.resize(0);
    eigen_basis.resize(0, 0);
  }

  struct PointCloudAdaptor {
    std::span<const float> x;
    std::span<const float> y;
    std::span<const float> z;

    [[nodiscard]] size_t kdtree_get_point_count() const { return x.size(); }

    [[nodiscard]] float kdtree_get_pt(size_t idx, size_t dim) const {
      if (dim == 0)
        return x[idx];
      if (dim == 1)
        return y[idx];
      return z[idx];
    }

    template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
  };

  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, PointCloudAdaptor, 3>;

  void build(Input input) {
    const size_t n_verts = input.x.size();
    if (n_verts == 0 || input.y.size() != n_verts || input.z.size() != n_verts) {
      clear();
      return;
    }

    const int k_neighbors = std::max(1, std::min(input.k_neighbors, static_cast<int>(n_verts)));
    const float t_sq = std::max(input.bandwidth, 1e-8f);

    PointCloudAdaptor adaptor{input.x, input.y, input.z};
    KDTree tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(n_verts * static_cast<size_t>(k_neighbors));

    std::vector<uint32_t> ret_index(static_cast<size_t>(k_neighbors));
    std::vector<float> out_dist_sqr(static_cast<size_t>(k_neighbors));
    Eigen::VectorXf D = Eigen::VectorXf::Zero(static_cast<int>(n_verts));

    for (size_t i = 0; i < n_verts; ++i) {
      const float query_pt[3] = {input.x[i], input.y[i], input.z[i]};
      const size_t num_results = tree.knnSearch(query_pt, static_cast<size_t>(k_neighbors), ret_index.data(), out_dist_sqr.data());

      for (size_t k = 0; k < num_results; ++k) {
        const uint32_t j = ret_index[k];
        const float dist_sq = out_dist_sqr[k];

        const float k_val = std::exp(-dist_sq / t_sq);
        if (k_val < 1e-8f) {
          continue;
        }

        triplets.emplace_back(static_cast<int>(i), static_cast<int>(j), k_val);
        D[static_cast<int>(i)] += k_val;
      }
    }

    P.resize(static_cast<int>(n_verts), static_cast<int>(n_verts));
    P.setFromTriplets(triplets.begin(), triplets.end());

    for (int outer = 0; outer < P.outerSize(); ++outer) {
      for (Eigen::SparseMatrix<float>::InnerIterator it(P, outer); it; ++it) {
        const float denom = std::max(D[it.row()], 1e-12f);
        it.valueRef() /= denom;
      }
    }

    mu = D;
    const float mu_sum = mu.sum();
    if (mu_sum > 1e-12f) {
      mu /= mu_sum;
    } else {
      mu = Eigen::VectorXf::Constant(static_cast<int>(n_verts), 1.0f / static_cast<float>(n_verts));
    }

    if (std::getenv("IGNEOUS_BENCH_MODE") == nullptr) {
      std::cout << "[Diffusion] Built connectivity for " << n_verts
                << " points.\n";
    }
  }
};

static_assert(Topology<TriangleTopology>);
static_assert(Topology<DiffusionTopology>);
static_assert(SurfaceTopology<TriangleTopology>);
static_assert(!SurfaceTopology<DiffusionTopology>);

} // namespace igneous::data
