#pragma once
#include <Eigen/Sparse>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <nanoflann.hpp>
#include <span>
#include <vector>

namespace igneous::data {

// ==============================================================================
// 1. CONCEPTS
// ==============================================================================

template <typename T>
concept Topology = requires(T &t, const T &ct, size_t n_verts, uint32_t idx) {
  // Must declare what input data it needs
  typename T::Input;

  // Must declare its dimension
  { T::DIMENSION } -> std::convertible_to<int>;

  // Must support lifecycle management
  { t.build(std::declval<typename T::Input>()) } -> std::same_as<void>;
  { t.clear() } -> std::same_as<void>;

  // Must support generic neighborhood queries
  {
    ct.get_neighborhood(idx)
  } -> std::convertible_to<std::span<const uint32_t>>;
};

template <typename T>
concept SurfaceTopology =
    Topology<T> &&
    requires(T &t, const T &ct, size_t f_idx, size_t v_idx, int corner) {
      { ct.num_faces() } -> std::convertible_to<size_t>;
      {
        ct.get_vertex_for_face(f_idx, corner)
      } -> std::convertible_to<uint32_t>;
      {
        ct.get_faces_for_vertex(v_idx)
      } -> std::convertible_to<std::span<const uint32_t>>;
      { t.faces_to_vertices } -> std::same_as<std::vector<uint32_t> &>;
      { ct.faces_to_vertices } -> std::same_as<const std::vector<uint32_t> &>;
    };

// ==============================================================================
// 2. IMPLEMENTATIONS
// ==============================================================================

// --- Triangle Topology ---
struct TriangleTopology {
  static constexpr int DIMENSION = 2;

  std::vector<uint32_t> faces_to_vertices;
  std::vector<uint32_t> coboundary_offsets;
  std::vector<uint32_t> coboundary_data;

  struct Input {
    size_t num_vertices;
  };

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
    return get_faces_for_vertex(vertex_idx);
  }

  void build(Input input) {
    size_t num_vertices = input.num_vertices;
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

// --- Point Topology ---
struct PointTopology {
  static constexpr int DIMENSION = 0;

  struct Input {};

  std::span<const uint32_t> get_neighborhood(uint32_t) const { return {}; }
  void build(Input) {}
  void clear() {}
};

// --- Diffusion Topology ---
struct DiffusionTopology {
  static constexpr int DIMENSION = 0;

  struct Input {
    std::span<const float> coords;
    float bandwidth = 0.01f;
    int k_neighbors = 32;
  };

  // 1. The Markov Chain (P)
  Eigen::SparseMatrix<float> P;

  // 2. The Measure (mu)
  Eigen::VectorXf mu;

  // 3. Eigenfunctions (Optional)
  Eigen::MatrixXf eigen_basis;

  // Interface
  size_t num_primitives() const { return 0; }
  std::span<const uint32_t> get_neighborhood(uint32_t) const { return {}; }

  void clear() {
    P.resize(0, 0);
    mu.resize(0);
  }

  // --- Internal Helpers for Nanoflann ---
  struct PointCloudAdaptor {
    std::span<const float> points;
    size_t num_points;

    PointCloudAdaptor(std::span<const float> p, size_t n)
        : points(p), num_points(n) {}

    inline size_t kdtree_get_point_count() const { return num_points; }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
      return points[idx * 3 + dim];
    }

    template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
  };

  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, PointCloudAdaptor,
      3 /* dim */
      >;

  // --- Implementation ---
  void build(Input input) {
    size_t n_verts = input.coords.size() / 3;
    if (n_verts == 0)
      return;

    // 1. Build KD-Tree
    // FIX: Use input.coords
    PointCloudAdaptor adaptor(input.coords, n_verts);
    KDTree tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    // 2. Prepare Sparse Matrix Triplets
    std::vector<Eigen::Triplet<float>> triplets;
    // FIX: Use input.k_neighbors
    triplets.reserve(n_verts * input.k_neighbors);

    std::vector<uint32_t> ret_index(input.k_neighbors);
    std::vector<float> out_dist_sqr(input.k_neighbors);

    Eigen::VectorXf D = Eigen::VectorXf::Zero(n_verts);

    // FIX: Use input.bandwidth
    // Using fixed bandwidth as per standard starting point
    float t_sq = input.bandwidth;

    for (size_t i = 0; i < n_verts; ++i) {
      float query_pt[3] = {input.coords[i * 3], input.coords[i * 3 + 1],
                           input.coords[i * 3 + 2]};

      size_t num_results = tree.knnSearch(&query_pt[0], input.k_neighbors,
                                          &ret_index[0], &out_dist_sqr[0]);

      for (size_t k = 0; k < num_results; ++k) {
        size_t j = ret_index[k];
        float dist_sq = out_dist_sqr[k];

        // Gaussian Kernel
        float k_val = std::exp(-dist_sq / t_sq);

        if (k_val < 1e-8f)
          continue;

        triplets.emplace_back(i, j, k_val);
        D[i] += k_val;
      }
    }

    // 4. Construct Markov Chain P
    P.resize(n_verts, n_verts);
    P.setFromTriplets(triplets.begin(), triplets.end());

    // Normalize rows to sum to 1
    for (int k = 0; k < P.outerSize(); ++k) {
      for (typename Eigen::SparseMatrix<float>::InnerIterator it(P, k); it;
           ++it) {
        it.valueRef() /= D[it.row()];
      }
    }

    // 5. Compute Measure mu
    // Represents stationary distribution
    mu = D / D.sum();

    std::cout << "[Diffusion] Built connectivity for " << n_verts
              << " points.\n";
  }
};

// Validate concepts
static_assert(Topology<TriangleTopology>);
static_assert(Topology<DiffusionTopology>);
static_assert(SurfaceTopology<TriangleTopology>);
static_assert(!SurfaceTopology<DiffusionTopology>);

} // namespace igneous::data