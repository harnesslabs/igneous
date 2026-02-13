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

  std::vector<uint32_t> faces_to_vertices;

  std::vector<uint32_t> face_v0;
  std::vector<uint32_t> face_v1;
  std::vector<uint32_t> face_v2;

  std::vector<uint32_t> vertex_face_offsets;
  std::vector<uint32_t> vertex_face_data;

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

    vertex_neighbor_offsets.assign(num_vertices + 1, 0);
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

  [[nodiscard]] std::span<const uint32_t> get_neighborhood(uint32_t) const {
    return {};
  }

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

  using SparseMatrixT = Eigen::SparseMatrix<float>;
  using RowSparseMatrixT = Eigen::SparseMatrix<float, Eigen::RowMajor>;

  SparseMatrixT P;
  Eigen::VectorXf mu;
  Eigen::MatrixXf eigen_basis;

  // Direct CSR view for hot diffusion operators.
  std::vector<int> markov_row_offsets;
  std::vector<int> markov_col_indices;
  std::vector<float> markov_values;

  [[nodiscard]] size_t num_primitives() const {
    return static_cast<size_t>(P.rows());
  }
  [[nodiscard]] std::span<const uint32_t> get_neighborhood(uint32_t) const {
    return {};
  }

  void clear() {
    P.resize(0, 0);
    mu.resize(0);
    eigen_basis.resize(0, 0);
    markov_row_offsets.clear();
    markov_col_indices.clear();
    markov_values.clear();
  }

  struct PointCloudAdaptor {
    std::span<const float> x;
    std::span<const float> y;
    std::span<const float> z;

    [[nodiscard]] size_t kdtree_get_point_count() const { return x.size(); }

    [[nodiscard]] float kdtree_get_pt(size_t idx, size_t dim) const {
      if (dim == 0) {
        return x[idx];
      }
      if (dim == 1) {
        return y[idx];
      }
      return z[idx];
    }

    template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
  };

  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, PointCloudAdaptor,
      3>;

  void build(Input input) {
    const size_t n_verts = input.x.size();
    if (n_verts == 0 || input.y.size() != n_verts || input.z.size() != n_verts) {
      clear();
      return;
    }

    const int k_neighbors =
        std::max(1, std::min(input.k_neighbors, static_cast<int>(n_verts)));
    const float t_sq = std::max(input.bandwidth, 1e-8f);

    PointCloudAdaptor adaptor{input.x, input.y, input.z};
    KDTree tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(32));
    tree.buildIndex();

    const size_t row_capacity = n_verts * static_cast<size_t>(k_neighbors);

    markov_row_offsets.assign(n_verts + 1, 0);
    markov_col_indices.clear();
    markov_values.clear();
    markov_col_indices.reserve(row_capacity);
    markov_values.reserve(row_capacity);

    std::vector<uint32_t> ret_index(static_cast<size_t>(k_neighbors));
    std::vector<float> out_dist_sqr(static_cast<size_t>(k_neighbors));
    mu.resize(static_cast<int>(n_verts));
    mu.setZero();

    for (size_t i = 0; i < n_verts; ++i) {
      markov_row_offsets[i] = static_cast<int>(markov_col_indices.size());
      const float query_pt[3] = {input.x[i], input.y[i], input.z[i]};
      const size_t num_results = tree.knnSearch(
          query_pt, static_cast<size_t>(k_neighbors), ret_index.data(),
          out_dist_sqr.data());

      float row_mass = 0.0f;
      for (size_t k = 0; k < num_results; ++k) {
        const float dist_sq = out_dist_sqr[k];

        const float k_val = std::exp(-dist_sq / t_sq);
        if (k_val < 1e-8f) {
          out_dist_sqr[k] = 0.0f;
          continue;
        }

        out_dist_sqr[k] = k_val;
        row_mass += k_val;
      }

      if (row_mass <= 1e-12f) {
        markov_col_indices.push_back(static_cast<int>(i));
        markov_values.push_back(1.0f);
        mu[static_cast<int>(i)] = 1.0f;
        markov_row_offsets[i + 1] = static_cast<int>(markov_col_indices.size());
        continue;
      }

      const float inv_row_mass = 1.0f / row_mass;
      for (size_t k = 0; k < num_results; ++k) {
        const float k_val = out_dist_sqr[k];
        if (k_val <= 0.0f) {
          continue;
        }

        const uint32_t j = ret_index[k];
        const float normalized = k_val * inv_row_mass;

        markov_col_indices.push_back(static_cast<int>(j));
        markov_values.push_back(normalized);
      }

      mu[static_cast<int>(i)] = row_mass;
      markov_row_offsets[i + 1] = static_cast<int>(markov_col_indices.size());
    }

    markov_row_offsets[n_verts] = static_cast<int>(markov_col_indices.size());

    const int nnz = static_cast<int>(markov_col_indices.size());
    RowSparseMatrixT P_row(static_cast<int>(n_verts), static_cast<int>(n_verts));
    P_row.resizeNonZeros(nnz);
    std::copy(markov_row_offsets.begin(), markov_row_offsets.end(),
              P_row.outerIndexPtr());
    std::copy(markov_col_indices.begin(), markov_col_indices.end(),
              P_row.innerIndexPtr());
    std::copy(markov_values.begin(), markov_values.end(), P_row.valuePtr());
    P_row.finalize();

    P = P_row;
    P.makeCompressed();

    const float mu_sum = mu.sum();
    if (mu_sum > 1e-12f) {
      mu /= mu_sum;
    } else {
      mu = Eigen::VectorXf::Constant(static_cast<int>(n_verts),
                                     1.0f / static_cast<float>(n_verts));
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
