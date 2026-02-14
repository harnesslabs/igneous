#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>

#include <igneous/ops/geometry.hpp>
#include <igneous/ops/hodge.hpp>
#include <igneous/ops/spectral.hpp>

#include "support/synthetic_meshes.hpp"
#include "support/test_env.hpp"

static float gram_inner(const Eigen::VectorXf &a, const Eigen::VectorXf &b,
                        const Eigen::MatrixXf &G) {
  return a.dot(G * b);
}

TEST_CASE("Hodge decomposition reconstructs and yields near-orthogonal parts") {
  igneous::test_support::configure_deterministic_test_env();
  auto mesh = igneous::test_support::make_torus_cloud(1200, 0.05f, 24);
  igneous::ops::compute_eigenbasis(mesh, 24);

  igneous::ops::GeometryWorkspace<decltype(mesh)> geom_ws;
  const Eigen::MatrixXf G =
      igneous::ops::compute_1form_gram_matrix(mesh, 0.05f, geom_ws);
  igneous::ops::HodgeWorkspace<decltype(mesh)> hodge_ws;
  const Eigen::MatrixXf D =
      igneous::ops::compute_weak_exterior_derivative(mesh, 0.05f, hodge_ws);
  const Eigen::MatrixXf E =
      igneous::ops::compute_curl_energy_matrix(mesh, 0.05f, hodge_ws);
  const Eigen::MatrixXf L = igneous::ops::compute_hodge_laplacian_matrix(D, E);
  auto [evals, evecs] = igneous::ops::compute_hodge_spectrum(L, G);

  const Eigen::VectorXf alpha = evecs.col(0);
  const auto decomposition =
      igneous::ops::compute_hodge_decomposition_1form(mesh, alpha, 0.05f);

  const Eigen::VectorXf reconstructed = decomposition.exact_component +
                                        decomposition.harmonic_component +
                                        decomposition.coexact_component;
  const float rec_rel_err =
      (reconstructed - alpha).norm() / std::max(1.0f, alpha.norm());
  CHECK(rec_rel_err <= 1e-3f);

  const float ex_ha =
      std::abs(gram_inner(decomposition.exact_component,
                          decomposition.harmonic_component, G));
  const float ex_co =
      std::abs(gram_inner(decomposition.exact_component,
                          decomposition.coexact_component, G));
  const float ha_co =
      std::abs(gram_inner(decomposition.harmonic_component,
                          decomposition.coexact_component, G));

  const float ex_norm =
      std::sqrt(std::max(gram_inner(decomposition.exact_component,
                                    decomposition.exact_component, G),
                         1e-12f));
  const float ha_norm =
      std::sqrt(std::max(gram_inner(decomposition.harmonic_component,
                                    decomposition.harmonic_component, G),
                         1e-12f));
  const float co_norm =
      std::sqrt(std::max(gram_inner(decomposition.coexact_component,
                                    decomposition.coexact_component, G),
                         1e-12f));

  CHECK(ex_ha <= 2e-3f * ex_norm * ha_norm + 1e-5f);
  CHECK(ex_co <= 2e-3f * ex_norm * co_norm + 1e-5f);
  CHECK(ha_co <= 2e-3f * ha_norm * co_norm + 1e-5f);

  const float harmonic_energy =
      gram_inner(decomposition.harmonic_component,
                 L * decomposition.harmonic_component, G);
  CHECK(harmonic_energy <= 1e-2f);
  CHECK(decomposition.exact_potential.size() == mesh.topology.eigen_basis.cols());
}
