#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cassert>
#include <span>
#include <type_traits>

#include <igneous/core/gpu.hpp>
#include <igneous/core/parallel.hpp>
#include <igneous/data/mesh.hpp>

namespace igneous::ops {

template <typename MeshT> struct GeometryWorkspace {
  std::array<Eigen::VectorXf, 3> coords;
  std::array<std::array<Eigen::VectorXf, 3>, 3> gamma_coords;
  Eigen::VectorXf weights;
  Eigen::VectorXf gamma_tmp;
};

template <typename MeshT> struct DiffusionWorkspace {
  Eigen::VectorXf ping;
  Eigen::VectorXf pong;
};

template <typename MeshT>
void fill_coordinate_vectors(const MeshT &mesh,
                             std::array<Eigen::VectorXf, 3> &coords) {
  const size_t n_verts = mesh.geometry.num_points();
  for (int d = 0; d < 3; ++d) {
    coords[d].resize(static_cast<int>(n_verts));
  }

  if constexpr (requires { mesh.topology.immersion_coords; }) {
    const auto &immersion = mesh.topology.immersion_coords;
    if (immersion.rows() == static_cast<int>(n_verts) && immersion.cols() >= 3) {
      for (size_t i = 0; i < n_verts; ++i) {
        const int idx = static_cast<int>(i);
        coords[0][idx] = immersion(idx, 0);
        coords[1][idx] = immersion(idx, 1);
        coords[2][idx] = immersion(idx, 2);
      }
      return;
    }
  }

  for (size_t i = 0; i < n_verts; ++i) {
    const auto p = mesh.geometry.get_vec3(i);
    const int idx = static_cast<int>(i);
    coords[0][idx] = p.x;
    coords[1][idx] = p.y;
    coords[2][idx] = p.z;
  }
}

template <typename MeshT>
void fill_data_coordinate_vectors(const MeshT &mesh,
                                  std::array<Eigen::VectorXf, 3> &coords) {
  const size_t n_verts = mesh.geometry.num_points();
  for (int d = 0; d < 3; ++d) {
    coords[d].resize(static_cast<int>(n_verts));
  }

  for (size_t i = 0; i < n_verts; ++i) {
    const auto p = mesh.geometry.get_vec3(i);
    const int idx = static_cast<int>(i);
    coords[0][idx] = p.x;
    coords[1][idx] = p.y;
    coords[2][idx] = p.z;
  }
}

template <typename MeshT>
void carre_du_champ(const MeshT &mesh, Eigen::Ref<const Eigen::VectorXf> f,
                    Eigen::Ref<const Eigen::VectorXf> h, [[maybe_unused]] float bandwidth,
                    Eigen::Ref<Eigen::VectorXf> gamma_out) {
  [[maybe_unused]] const int expected_size =
      static_cast<int>(mesh.geometry.num_points());
  assert(gamma_out.size() == expected_size);
  gamma_out.setZero();

  if constexpr (requires {
                  mesh.topology.markov_row_offsets;
                  mesh.topology.markov_col_indices;
                  mesh.topology.markov_values;
                }) {
    const auto &row_offsets = mesh.topology.markov_row_offsets;
    const auto &col_indices = mesh.topology.markov_col_indices;
    const auto &weights = mesh.topology.markov_values;

    assert(row_offsets.size() == static_cast<size_t>(expected_size) + 1);

    const float *f_data = f.data();
    const float *h_data = h.data();
    Eigen::VectorXf means_f = Eigen::VectorXf::Zero(expected_size);
    Eigen::VectorXf means_h = Eigen::VectorXf::Zero(expected_size);
    const bool use_mean_centres = [&]() {
      if constexpr (requires { mesh.topology.use_mean_centres; }) {
        return mesh.topology.use_mean_centres;
      }
      return false;
    }();

    if (use_mean_centres) {
      for (int i = 0; i < expected_size; ++i) {
        const int begin = row_offsets[static_cast<size_t>(i)];
        const int end = row_offsets[static_cast<size_t>(i) + 1];
        const int *cols = col_indices.data() + begin;
        const float *w = weights.data() + begin;
        float mean_f = 0.0f;
        float mean_h = 0.0f;
        for (int idx = 0; idx < (end - begin); ++idx) {
          mean_f += w[idx] * f_data[cols[idx]];
          mean_h += w[idx] * h_data[cols[idx]];
        }
        means_f[i] = mean_f;
        means_h[i] = mean_h;
      }
    }

    for (int i = 0; i < expected_size; ++i) {
      const int begin = row_offsets[static_cast<size_t>(i)];
      const int end = row_offsets[static_cast<size_t>(i) + 1];
      const int count = end - begin;
      const int *cols = col_indices.data() + begin;
      const float *w = weights.data() + begin;
      const float center_f = use_mean_centres ? means_f[i] : f_data[i];
      const float center_h = use_mean_centres ? means_h[i] : h_data[i];

      float acc = 0.0f;
      int idx = 0;
      for (; idx + 3 < count; idx += 4) {
        const int j0 = cols[idx + 0];
        const int j1 = cols[idx + 1];
        const int j2 = cols[idx + 2];
        const int j3 = cols[idx + 3];
        acc += w[idx + 0] * (f_data[j0] - center_f) * (h_data[j0] - center_h);
        acc += w[idx + 1] * (f_data[j1] - center_f) * (h_data[j1] - center_h);
        acc += w[idx + 2] * (f_data[j2] - center_f) * (h_data[j2] - center_h);
        acc += w[idx + 3] * (f_data[j3] - center_f) * (h_data[j3] - center_h);
      }
      for (; idx < count; ++idx) {
        const int j = cols[idx];
        acc += w[idx] * (f_data[j] - center_f) * (h_data[j] - center_h);
      }

      float denom = 2.0f;
      if constexpr (requires { mesh.topology.local_bandwidths; }) {
        if (mesh.topology.local_bandwidths.size() == expected_size) {
          denom = 2.0f * std::max(mesh.topology.local_bandwidths[i], 1e-8f);
        }
      }
      gamma_out[i] = acc / denom;
    }
  } else {
    const auto &P = mesh.topology.P;
    for (int outer = 0; outer < P.outerSize(); ++outer) {
      for (typename std::remove_cvref_t<decltype(P)>::InnerIterator it(P, outer);
           it; ++it) {
        const int i = it.row();
        const int j = it.col();
        const float w = it.value();
        gamma_out[i] += w * (f[j] - f[i]) * (h[j] - h[i]);
      }
    }
    gamma_out *= 0.5f;
  }
}

template <typename MeshT>
void apply_markov_transition(const MeshT &mesh,
                             Eigen::Ref<const Eigen::VectorXf> input,
                             Eigen::Ref<Eigen::VectorXf> output) {
  const int expected_size = static_cast<int>(mesh.geometry.num_points());
  assert(input.size() == expected_size);
  assert(output.size() == expected_size);

  if constexpr (requires {
                  mesh.topology.markov_row_offsets;
                  mesh.topology.markov_col_indices;
                  mesh.topology.markov_values;
                }) {
    const auto &row_offsets = mesh.topology.markov_row_offsets;
    const auto &col_indices = mesh.topology.markov_col_indices;
    const auto &weights = mesh.topology.markov_values;

    assert(row_offsets.size() == static_cast<size_t>(expected_size) + 1);
    assert(input.data() != output.data());

    const bool use_gpu =
        core::compute_backend_from_env() == core::ComputeBackend::Gpu &&
        (core::gpu::gpu_force_enabled() || expected_size >= core::gpu::gpu_min_rows());
    if (use_gpu) {
      if (core::gpu::apply_markov_transition(
              static_cast<const void *>(&mesh.topology),
              std::span<const int>(row_offsets.data(), row_offsets.size()),
              std::span<const int>(col_indices.data(), col_indices.size()),
              std::span<const float>(weights.data(), weights.size()),
              std::span<const float>(input.data(), static_cast<size_t>(expected_size)),
              std::span<float>(output.data(), static_cast<size_t>(expected_size)))) {
        return;
      }
    }

    const float *input_data = input.data();
    float *output_data = output.data();

    for (int i = 0; i < expected_size; ++i) {
      const int begin = row_offsets[static_cast<size_t>(i)];
      const int end = row_offsets[static_cast<size_t>(i) + 1];
      const int count = end - begin;
      const int *cols = col_indices.data() + begin;
      const float *w = weights.data() + begin;

      float acc = 0.0f;
      int idx = 0;
      for (; idx + 3 < count; idx += 4) {
        acc += w[idx + 0] * input_data[cols[idx + 0]];
        acc += w[idx + 1] * input_data[cols[idx + 1]];
        acc += w[idx + 2] * input_data[cols[idx + 2]];
        acc += w[idx + 3] * input_data[cols[idx + 3]];
      }

      for (; idx < count; ++idx) {
        acc += w[idx] * input_data[cols[idx]];
      }

      output_data[i] = acc;
    }
  } else {
    output.noalias() = mesh.topology.P * input;
  }
}

template <typename MeshT>
void apply_markov_transition_steps(const MeshT &mesh,
                                   Eigen::Ref<const Eigen::VectorXf> input,
                                   int steps,
                                   Eigen::Ref<Eigen::VectorXf> output,
                                   DiffusionWorkspace<MeshT> &workspace) {
  const int expected_size = static_cast<int>(mesh.geometry.num_points());
  assert(input.size() == expected_size);
  assert(output.size() == expected_size);

  if (steps <= 0) {
    output = input;
    return;
  }

  if constexpr (requires {
                  mesh.topology.markov_row_offsets;
                  mesh.topology.markov_col_indices;
                  mesh.topology.markov_values;
                }) {
    const auto &row_offsets = mesh.topology.markov_row_offsets;
    const auto &col_indices = mesh.topology.markov_col_indices;
    const auto &weights = mesh.topology.markov_values;

    assert(row_offsets.size() == static_cast<size_t>(expected_size) + 1);
    const long long row_step_work =
        static_cast<long long>(expected_size) * static_cast<long long>(steps);
    const bool use_gpu =
        core::compute_backend_from_env() == core::ComputeBackend::Gpu &&
        (core::gpu::gpu_force_enabled() ||
         expected_size >= core::gpu::gpu_min_rows() ||
         row_step_work >= core::gpu::gpu_min_row_steps());
    if (use_gpu) {
      if (core::gpu::apply_markov_transition_steps(
              static_cast<const void *>(&mesh.topology),
              std::span<const int>(row_offsets.data(), row_offsets.size()),
              std::span<const int>(col_indices.data(), col_indices.size()),
              std::span<const float>(weights.data(), weights.size()),
              std::span<const float>(input.data(), static_cast<size_t>(expected_size)),
              steps,
              std::span<float>(output.data(), static_cast<size_t>(expected_size)))) {
        return;
      }
    }
  }

  if (steps == 1) {
    apply_markov_transition(mesh, input, output);
    return;
  }

  workspace.ping = input;
  workspace.pong.resize(expected_size);

  Eigen::VectorXf *src = &workspace.ping;
  Eigen::VectorXf *dst = &workspace.pong;
  for (int step = 0; step < steps; ++step) {
    apply_markov_transition(mesh, *src, *dst);
    std::swap(src, dst);
  }

  output = *src;
}

template <typename MeshT>
void apply_markov_transition_steps(const MeshT &mesh,
                                   Eigen::Ref<const Eigen::VectorXf> input,
                                   int steps,
                                   Eigen::Ref<Eigen::VectorXf> output) {
  DiffusionWorkspace<MeshT> workspace;
  apply_markov_transition_steps(mesh, input, steps, output, workspace);
}

template <typename MeshT>
Eigen::MatrixXf compute_1form_gram_matrix(const MeshT &mesh, float bandwidth,
                                          GeometryWorkspace<MeshT> &workspace) {
  const int n_basis = mesh.topology.eigen_basis.cols();
  const int n_total = n_basis * 3;

  Eigen::MatrixXf G = Eigen::MatrixXf::Zero(n_total, n_total);

  fill_coordinate_vectors(mesh, workspace.coords);

  const int n_verts = static_cast<int>(mesh.geometry.num_points());
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      workspace.gamma_coords[a][b].resize(n_verts);
      carre_du_champ(mesh, workspace.coords[a], workspace.coords[b], bandwidth,
                     workspace.gamma_coords[a][b]);
      if (a != b) {
        workspace.gamma_coords[b][a] = workspace.gamma_coords[a][b];
      }
    }
  }

  const auto &U = mesh.topology.eigen_basis;
  const auto &mu = mesh.topology.mu;

  core::parallel_for_index(
      0, n_basis,
      [&](int i) {
        thread_local Eigen::VectorXf weights_local;
        weights_local.resize(n_verts);

        for (int k = i; k < n_basis; ++k) {
          weights_local = U.col(i).cwiseProduct(U.col(k)).cwiseProduct(mu);

          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
              const float val = weights_local.dot(workspace.gamma_coords[a][b]);
              const int row = i * 3 + a;
              const int col = k * 3 + b;

              G(row, col) = val;
              if (row != col) {
                G(col, row) = val;
              }
            }
          }
        }
      },
      8);

  return G;
}

template <typename MeshT>
Eigen::MatrixXf compute_1form_gram_matrix(const MeshT &mesh, float bandwidth) {
  GeometryWorkspace<MeshT> workspace;
  return compute_1form_gram_matrix(mesh, bandwidth, workspace);
}

} // namespace igneous::ops
