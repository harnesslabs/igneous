#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <nanoflann.hpp>
#include <span>
#include <vector>

#include <igneous/core/gpu.hpp>
#include <igneous/core/parallel.hpp>
#include <igneous/data/structure.hpp>

namespace igneous::data {

/**
 * \brief Point-cloud diffusion structure with kNN Markov geometry.
 *
 * This structure builds:
 * - row-stochastic Markov CSR (`markov_*`)
 * - symmetric kernel CSR (`symmetric_*`) for spectral solves
 * - stationary density (`mu`)
 * - auxiliary geometric fields used by downstream operators
 */
struct DiffusionGeometry {
  /// \brief Dimension marker used by the `Structure` concept.
  static constexpr int DIMENSION = 0;

  /// \brief Parameters for diffusion graph construction.
  struct Input {
    /// \brief X coordinates of points.
    std::span<const float> x;
    /// \brief Y coordinates of points.
    std::span<const float> y;
    /// \brief Z coordinates of points.
    std::span<const float> z;
    /// \brief Number of nearest neighbors per row.
    int k_neighbors = 32;
    /// \brief Number of neighbors used for local bandwidth estimation.
    int knn_bandwidth = 8;
    /// \brief Density-adaptive bandwidth exponent.
    float bandwidth_variability = -0.5f;
    /// \brief Drift tuning parameter from the reference construction.
    float c = 0.0f;
    /// \brief Center `carre_du_champ` with row means instead of self values.
    bool use_mean_centres = true;
  };

  /// \brief Stationary measure induced by the symmetric kernel.
  Eigen::VectorXf mu;
  /// \brief Spectral basis produced by diffusion eigensolves.
  Eigen::MatrixXf eigen_basis;
  /// \brief Per-point local diffusion timescale.
  Eigen::VectorXf local_bandwidths;
  /// \brief Row sums of the symmetric kernel.
  Eigen::VectorXf symmetric_row_sums;
  /// \brief Markov-averaged embedding coordinates.
  Eigen::MatrixXf immersion_coords;
  /// \brief Runtime toggle for centered `carre_du_champ`.
  bool use_mean_centres = true;
  /// \brief Effective k used by the latest build.
  int knn_k = 0;

  /// \brief Dense kNN column indices (`n * k`).
  std::vector<int> knn_indices;
  /// \brief Dense kNN distances (`n * k`).
  std::vector<float> knn_distances;
  /// \brief Dense row-stochastic kernel values (`n * k`).
  std::vector<float> knn_kernel;

  /// \brief CSR row offsets for Markov transition matrix.
  std::vector<int> markov_row_offsets;
  /// \brief CSR column indices for Markov transition matrix.
  std::vector<int> markov_col_indices;
  /// \brief CSR values for Markov transition matrix.
  std::vector<float> markov_values;

  /// \brief CSR row offsets for symmetric kernel matrix.
  std::vector<int> symmetric_row_offsets;
  /// \brief CSR column indices for symmetric kernel matrix.
  std::vector<int> symmetric_col_indices;
  /// \brief CSR values for symmetric kernel matrix.
  std::vector<float> symmetric_values;

  /**
   * \brief Number of points represented by the CSR graph.
   * \return Row count inferred from `markov_row_offsets`.
   */
  [[nodiscard]] size_t num_primitives() const {
    if (markov_row_offsets.empty()) {
      return 0;
    }
    return markov_row_offsets.size() - 1;
  }
  /**
   * \brief Neighborhood accessor required by `data::Structure`.
   * \param vertex_idx Unused.
   * \return Empty span (diffusion neighborhoods are held in CSR arrays).
   */
  [[nodiscard]] std::span<const uint32_t> get_neighborhood(uint32_t vertex_idx) const {
    (void)vertex_idx;
    return {};
  }

  /// \brief Clear all diffusion data and invalidate any GPU-side cache.
  void clear() {
    core::gpu::invalidate_markov_cache(this);
    mu.resize(0);
    eigen_basis.resize(0, 0);
    local_bandwidths.resize(0);
    symmetric_row_sums.resize(0);
    immersion_coords.resize(0, 0);
    use_mean_centres = true;
    knn_k = 0;
    knn_indices.clear();
    knn_distances.clear();
    knn_kernel.clear();
    markov_row_offsets.clear();
    markov_col_indices.clear();
    markov_values.clear();
    symmetric_row_offsets.clear();
    symmetric_col_indices.clear();
    symmetric_values.clear();
  }

  /// \brief nanoflann adaptor over the SoA coordinate spans.
  struct PointCloudAdaptor {
    /// \brief X coordinate channel.
    std::span<const float> x;
    /// \brief Y coordinate channel.
    std::span<const float> y;
    /// \brief Z coordinate channel.
    std::span<const float> z;

    /// \brief Number of points available to the k-d tree.
    [[nodiscard]] size_t kdtree_get_point_count() const { return x.size(); }

    /// \brief Coordinate accessor required by nanoflann.
    [[nodiscard]] float kdtree_get_pt(size_t idx, size_t dim) const {
      if (dim == 0) {
        return x[idx];
      }
      if (dim == 1) {
        return y[idx];
      }
      return z[idx];
    }

    /// \brief Bounding box callback (unused for this adaptor).
    template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
  };

  /// \brief nanoflann 3D L2 tree type.
  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, PointCloudAdaptor,
      3>;

  /**
   * \brief Candidate epsilon sweep used during kernel tuning.
   * \return Monotonic list of epsilon values.
   */
  static std::vector<float> build_epsilons() {
    std::vector<float> epsilons;
    epsilons.reserve(80);
    for (float exponent = -10.0f; exponent < 10.0f; exponent += 0.25f) {
      epsilons.push_back(std::pow(2.0f, exponent));
    }
    return epsilons;
  }

  /**
   * \brief Tune kernel scale and intrinsic dimension surrogate.
   * \param entries Squared-distance-like kernel arguments.
   * \param n Number of points.
   * \param k Number of neighbors per point.
   * \param epsilons Candidate kernel scales.
   * \return Pair `(epsilon, dim_surrogate)`.
   */
  static std::pair<float, float> tune_kernel(const std::vector<float> &entries,
                                             int n, int k,
                                             const std::vector<float> &epsilons) {
    const int e_count = static_cast<int>(epsilons.size());
    std::vector<float> averages(static_cast<size_t>(e_count), 0.0f);
    const float inv_nk = 1.0f / static_cast<float>(std::max(1, n * k));

    for (int e = 0; e < e_count; ++e) {
      const float epsilon = std::max(epsilons[static_cast<size_t>(e)], 1e-12f);
      float sum = 0.0f;
      for (float value : entries) {
        sum += std::exp(-value / epsilon);
      }
      averages[static_cast<size_t>(e)] = sum * inv_nk;
    }

    int best_idx = 0;
    float best_criterion = -std::numeric_limits<float>::infinity();
    for (int i = 0; i + 1 < e_count; ++i) {
      const float avg0 = std::max(averages[static_cast<size_t>(i)], 1e-30f);
      const float avg1 = std::max(averages[static_cast<size_t>(i + 1)], 1e-30f);
      const float eps0 = std::max(epsilons[static_cast<size_t>(i)], 1e-12f);
      const float eps1 = std::max(epsilons[static_cast<size_t>(i + 1)], 1e-12f);
      const float denom = std::log(eps1) - std::log(eps0);
      if (std::abs(denom) < 1e-12f) {
        continue;
      }
      const float criterion = (std::log(avg1) - std::log(avg0)) / denom;
      if (criterion > best_criterion) {
        best_criterion = criterion;
        best_idx = i;
      }
    }

    const float epsilon =
        std::max(epsilons[static_cast<size_t>(best_idx)], 1e-12f);
    const float dim = 2.0f * best_criterion;
    return {epsilon, dim};
  }

  /**
   * \brief Compute per-point local bandwidth estimates from kNN distances.
   * \param nbr_distances Dense kNN distance matrix in row-major flat storage.
   * \param n Number of points.
   * \param k Number of neighbors per point.
   * \param knn_bandwidth Number of neighbors used for RMS estimate.
   * \param bandwidths_out Output vector of local bandwidths.
   */
  static void compute_local_bandwidths(const std::vector<float> &nbr_distances,
                                       int n, int k, int knn_bandwidth,
                                       Eigen::VectorXf &bandwidths_out) {
    bandwidths_out.resize(n);
    bandwidths_out.setOnes();
    if (k <= 1) {
      return;
    }

    const int end = std::max(2, std::min(knn_bandwidth, k));
    const int count = std::max(1, end - 1);
    for (int i = 0; i < n; ++i) {
      const size_t base = static_cast<size_t>(i) * static_cast<size_t>(k);
      float accum = 0.0f;
      for (int idx = 1; idx < end; ++idx) {
        const float d = nbr_distances[base + static_cast<size_t>(idx)];
        accum += d * d;
      }
      const float rms = std::sqrt(accum / static_cast<float>(count));
      bandwidths_out[i] = std::max(rms, 1e-8f);
    }
  }

  /**
   * \brief Median helper used for robust bandwidth normalization.
   * \param values Input values (consumed by value).
   * \return Median value (or `1.0f` when empty).
   */
  static float vector_median(std::vector<float> values) {
    if (values.empty()) {
      return 1.0f;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<long>(mid),
                     values.end());
    return values[mid];
  }

  /**
   * \brief Build diffusion CSR structures from point coordinates.
   * \param input Build parameters and coordinate spans.
   */
  void build(Input input) {
    core::gpu::invalidate_markov_cache(this);
    const size_t n_verts = input.x.size();
    if (n_verts == 0 || input.y.size() != n_verts || input.z.size() != n_verts) {
      clear();
      return;
    }

    const int n = static_cast<int>(n_verts);
    const int k_neighbors =
        std::max(1, std::min(input.k_neighbors, static_cast<int>(n_verts)));
    const int knn_bandwidth = std::max(2, std::min(input.knn_bandwidth, k_neighbors));
    use_mean_centres = input.use_mean_centres;
    knn_k = k_neighbors;

    PointCloudAdaptor adaptor{input.x, input.y, input.z};
    KDTree tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(32));
    tree.buildIndex();

    const size_t row_stride = static_cast<size_t>(k_neighbors);
    const size_t row_capacity = static_cast<size_t>(n) * row_stride;

    knn_indices.assign(row_capacity, 0);
    knn_distances.assign(row_capacity, 0.0f);
    knn_kernel.assign(row_capacity, 0.0f);

    core::parallel_for_index(
        0, n,
        [&](int row_idx) {
          const size_t i = static_cast<size_t>(row_idx);
          const size_t base = i * row_stride;
          thread_local std::vector<uint32_t> ret_index;
          thread_local std::vector<float> out_dist_sqr;
          ret_index.assign(static_cast<size_t>(k_neighbors), 0u);
          out_dist_sqr.assign(static_cast<size_t>(k_neighbors), 0.0f);

          const float query_pt[3] = {input.x[i], input.y[i], input.z[i]};
          const size_t found = tree.knnSearch(query_pt, static_cast<size_t>(k_neighbors),
                                              ret_index.data(),
                                              out_dist_sqr.data());
          const size_t usable = std::min(found, static_cast<size_t>(k_neighbors));
          for (size_t k = 0; k < usable; ++k) {
            knn_indices[base + k] = static_cast<int>(ret_index[k]);
            knn_distances[base + k] = std::sqrt(std::max(0.0f, out_dist_sqr[k]));
          }
          for (size_t k = usable; k < static_cast<size_t>(k_neighbors); ++k) {
            knn_indices[base + k] = row_idx;
            knn_distances[base + k] = 0.0f;
          }
        },
        64);

    Eigen::VectorXf bandwidths_A;
    compute_local_bandwidths(knn_distances, n, k_neighbors, knn_bandwidth,
                             bandwidths_A);

    std::vector<float> kernel_entries_A(row_capacity, 0.0f);
    for (int i = 0; i < n; ++i) {
      const size_t base = static_cast<size_t>(i) * row_stride;
      const float bw_i = std::max(bandwidths_A[i], 1e-8f);
      for (int k = 0; k < k_neighbors; ++k) {
        const int j = knn_indices[base + static_cast<size_t>(k)];
        const float bw_j = std::max(bandwidths_A[j], 1e-8f);
        const float dist = knn_distances[base + static_cast<size_t>(k)];
        kernel_entries_A[base + static_cast<size_t>(k)] =
            (dist * dist) / (bw_j * bw_i);
      }
    }

    const std::vector<float> epsilons = build_epsilons();
    const auto [epsilon_A, dim_A] =
        tune_kernel(kernel_entries_A, n, k_neighbors, epsilons);

    constexpr float kPi = 3.14159265358979323846f;
    const float kernel_A_norm =
        std::pow(kPi * std::max(epsilon_A, 1e-12f), dim_A * 0.5f);
    Eigen::VectorXf density_estimate_A(n);
    density_estimate_A.setZero();
    for (int i = 0; i < n; ++i) {
      const size_t base = static_cast<size_t>(i) * row_stride;
      float row_sum = 0.0f;
      for (int k = 0; k < k_neighbors; ++k) {
        row_sum += std::exp(-kernel_entries_A[base + static_cast<size_t>(k)] /
                            std::max(epsilon_A, 1e-12f));
      }
      const float bw = std::max(bandwidths_A[i], 1e-8f);
      density_estimate_A[i] =
          (row_sum / std::max(kernel_A_norm, 1e-12f)) /
          (static_cast<float>(n) * std::pow(bw, dim_A));
      density_estimate_A[i] = std::max(density_estimate_A[i], 1e-12f);
    }

    Eigen::VectorXf bandwidths_B(n);
    for (int i = 0; i < n; ++i) {
      bandwidths_B[i] =
          std::pow(density_estimate_A[i], input.bandwidth_variability);
    }
    std::vector<float> bw_values(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      bw_values[static_cast<size_t>(i)] = bandwidths_B[i];
    }
    const float bw_median = std::max(vector_median(std::move(bw_values)), 1e-8f);
    bandwidths_B /= bw_median;

    std::vector<float> kernel_entries_B(row_capacity, 0.0f);
    for (int i = 0; i < n; ++i) {
      const size_t base = static_cast<size_t>(i) * row_stride;
      const float bw_i = std::max(bandwidths_B[i], 1e-8f);
      for (int k = 0; k < k_neighbors; ++k) {
        const int j = knn_indices[base + static_cast<size_t>(k)];
        const float bw_j = std::max(bandwidths_B[j], 1e-8f);
        const float dist = knn_distances[base + static_cast<size_t>(k)];
        kernel_entries_B[base + static_cast<size_t>(k)] =
            (dist * dist) / (bw_j * bw_i);
      }
    }

    const auto [epsilon_B, dim_B] =
        tune_kernel(kernel_entries_B, n, k_neighbors, epsilons);

    Eigen::VectorXf density_estimate_B(n);
    density_estimate_B.setZero();
    for (int i = 0; i < n; ++i) {
      const size_t base = static_cast<size_t>(i) * row_stride;
      float row_sum = 0.0f;
      for (int k = 0; k < k_neighbors; ++k) {
        row_sum += std::exp(-kernel_entries_B[base + static_cast<size_t>(k)] /
                            std::max(epsilon_B, 1e-12f));
      }
      const float bw = std::max(bandwidths_B[i], 1e-8f);
      density_estimate_B[i] = row_sum / std::pow(bw, dim_B);
      density_estimate_B[i] = std::max(density_estimate_B[i], 1e-12f);
    }

    const float alpha = 1.0f - (input.c * 0.5f) +
                        input.bandwidth_variability * ((dim_B * 0.5f) + 1.0f);
    Eigen::VectorXf density_alpha(n);
    for (int i = 0; i < n; ++i) {
      density_alpha[i] = std::pow(density_estimate_B[i], -alpha);
    }

    markov_row_offsets.resize(static_cast<size_t>(n) + 1);
    for (int i = 0; i <= n; ++i) {
      markov_row_offsets[static_cast<size_t>(i)] = i * k_neighbors;
    }
    markov_col_indices = knn_indices;
    markov_values.assign(row_capacity, 0.0f);

    for (int i = 0; i < n; ++i) {
      const size_t base = static_cast<size_t>(i) * row_stride;
      float row_sum = 0.0f;
      for (int k = 0; k < k_neighbors; ++k) {
        const int j = knn_indices[base + static_cast<size_t>(k)];
        const float kernel_B =
            std::exp(-kernel_entries_B[base + static_cast<size_t>(k)] /
                     std::max(epsilon_B, 1e-12f));
        const float val = kernel_B * density_alpha[i] * density_alpha[j];
        markov_values[base + static_cast<size_t>(k)] = val;
        row_sum += val;
      }

      if (row_sum <= 1e-30f) {
        int self_idx = -1;
        for (int k = 0; k < k_neighbors; ++k) {
          if (knn_indices[base + static_cast<size_t>(k)] == i) {
            self_idx = k;
            break;
          }
        }
        if (self_idx < 0) {
          self_idx = 0;
        }
        for (int k = 0; k < k_neighbors; ++k) {
          markov_values[base + static_cast<size_t>(k)] =
              (k == self_idx) ? 1.0f : 0.0f;
        }
        knn_kernel[base + static_cast<size_t>(self_idx)] = 1.0f;
        for (int k = 0; k < k_neighbors; ++k) {
          if (k != self_idx) {
            knn_kernel[base + static_cast<size_t>(k)] = 0.0f;
          }
        }
        continue;
      }

      const float inv_row_sum = 1.0f / row_sum;
      for (int k = 0; k < k_neighbors; ++k) {
        const size_t idx = base + static_cast<size_t>(k);
        markov_values[idx] *= inv_row_sum;
        knn_kernel[idx] = markov_values[idx];
      }
    }

    std::vector<Eigen::Triplet<float>> markov_triplets;
    markov_triplets.reserve(row_capacity);
    for (int i = 0; i < n; ++i) {
      const int begin = markov_row_offsets[static_cast<size_t>(i)];
      const int end = markov_row_offsets[static_cast<size_t>(i) + 1];
      for (int idx = begin; idx < end; ++idx) {
        markov_triplets.emplace_back(i, markov_col_indices[static_cast<size_t>(idx)],
                                     markov_values[static_cast<size_t>(idx)]);
      }
    }

    Eigen::SparseMatrix<float, Eigen::RowMajor> P_row(n, n);
    P_row.setFromTriplets(markov_triplets.begin(), markov_triplets.end());
    P_row.makeCompressed();

    Eigen::SparseMatrix<float, Eigen::RowMajor> Pt_row = P_row.transpose();
    Pt_row.makeCompressed();

    P_row += Pt_row;
    P_row *= 0.5f;
    P_row.makeCompressed();

    const Eigen::VectorXf ones = Eigen::VectorXf::Ones(n);
    symmetric_row_sums = P_row * ones;
    symmetric_row_sums =
        symmetric_row_sums.array().max(1e-12f).matrix();

    const float mu_sum = symmetric_row_sums.sum();
    if (mu_sum > 1e-12f) {
      mu = symmetric_row_sums / mu_sum;
    } else {
      mu = Eigen::VectorXf::Constant(n, 1.0f / static_cast<float>(n));
    }

    const int *outer = P_row.outerIndexPtr();
    const int *inner = P_row.innerIndexPtr();
    const float *values = P_row.valuePtr();
    const int nnz_sym = P_row.nonZeros();
    symmetric_row_offsets.assign(outer, outer + (n + 1));
    symmetric_col_indices.assign(inner, inner + nnz_sym);
    symmetric_values.assign(values, values + nnz_sym);

    local_bandwidths.resize(n);
    for (int i = 0; i < n; ++i) {
      const float bw = std::max(bandwidths_B[i], 1e-8f);
      local_bandwidths[i] = (std::max(epsilon_B, 1e-12f) * bw * bw) / 4.0f;
    }

    immersion_coords.resize(n, 3);
    for (int i = 0; i < n; ++i) {
      float rx = 0.0f;
      float ry = 0.0f;
      float rz = 0.0f;
      const int begin = markov_row_offsets[static_cast<size_t>(i)];
      const int end = markov_row_offsets[static_cast<size_t>(i) + 1];
      for (int idx = begin; idx < end; ++idx) {
        const int j = markov_col_indices[static_cast<size_t>(idx)];
        const float w = markov_values[static_cast<size_t>(idx)];
        rx += w * input.x[static_cast<size_t>(j)];
        ry += w * input.y[static_cast<size_t>(j)];
        rz += w * input.z[static_cast<size_t>(j)];
      }
      immersion_coords(i, 0) = rx;
      immersion_coords(i, 1) = ry;
      immersion_coords(i, 2) = rz;
    }

    if (std::getenv("IGNEOUS_BENCH_MODE") == nullptr) {
      std::cout << "[Diffusion] Built reference-style kernel for " << n_verts
                << " points (k=" << k_neighbors << ", knn_bw=" << knn_bandwidth
                << ").\n";
    }
  }
};

static_assert(Structure<DiffusionGeometry>);
static_assert(!SurfaceStructure<DiffusionGeometry>);

} // namespace igneous::data
