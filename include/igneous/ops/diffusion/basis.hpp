#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace igneous::ops::diffusion {

/// \brief Dense lookup tables for wedge-product index expansion.
struct WedgeProductIndexData {
  /// \brief Output component index for each wedge term.
  std::vector<int> target_indices;
  /// \brief Left operand component index for each wedge term.
  std::vector<int> left_indices;
  /// \brief Right operand component index for each wedge term.
  std::vector<int> right_indices;
  /// \brief Orientation sign for each wedge term.
  std::vector<int> signs;
  /// \brief Number of target basis components.
  int n_targets = 0;
  /// \brief Number of left/right split patterns.
  int n_splits = 0;
};

/// \brief Child-index tables used by weak exterior derivative assembly.
struct Kp1ChildrenAndSignsData {
  /// \brief Basis indices for k-forms.
  std::vector<std::vector<int>> idx_k;
  /// \brief Basis indices for (k+1)-forms.
  std::vector<std::vector<int>> idx_kp1;
  /// \brief Child mapping from each (k+1)-index to k-indices.
  std::vector<std::vector<int>> children;
  /// \brief Alternating signs applied during assembly.
  std::vector<int> signs;
};

/**
 * \brief Binomial coefficient helper (`n choose k`).
 * \param n Universe size.
 * \param k Selection size.
 * \return Binomial coefficient value.
 */
inline int binomial_coeff(int n, int k) {
  if (k < 0 || k > n) {
    return 0;
  }
  if (k == 0 || k == n) {
    return 1;
  }
  k = std::min(k, n - k);
  long long result = 1;
  for (int i = 1; i <= k; ++i) {
    result = (result * (n - k + i)) / i;
  }
  return static_cast<int>(result);
}

/**
 * \brief Recursive generator for lexicographic combinations.
 * \param n Universe size.
 * \param k Selection size.
 * \param start First candidate value.
 * \param current Current partial combination.
 * \param out Destination list of combinations.
 */
inline void combinations_recursive(int n, int k, int start,
                                   std::vector<int> &current,
                                   std::vector<std::vector<int>> &out) {
  if (static_cast<int>(current.size()) == k) {
    out.push_back(current);
    return;
  }

  const int remaining = k - static_cast<int>(current.size());
  for (int value = start; value <= n - remaining; ++value) {
    current.push_back(value);
    combinations_recursive(n, k, value + 1, current, out);
    current.pop_back();
  }
}

/**
 * \brief Enumerate ordered wedge-basis index combinations for degree `k`.
 * \param d Ambient dimension.
 * \param k Exterior degree.
 * \return Lexicographically ordered basis-index combinations.
 */
inline std::vector<std::vector<int>> get_wedge_basis_indices(int d, int k) {
  if (k < 0 || k > d) {
    return {};
  }
  std::vector<std::vector<int>> out;
  if (k == 0) {
    out.push_back({});
    return out;
  }

  out.reserve(static_cast<size_t>(binomial_coeff(d, k)));
  std::vector<int> current;
  current.reserve(static_cast<size_t>(k));
  combinations_recursive(d, k, 0, current, out);
  return out;
}

/**
 * \brief Lexicographic rank of a sorted combination.
 * \param comb Sorted combination indices.
 * \param d Ambient dimension.
 * \return Zero-based rank among all `k`-combinations.
 */
inline int lex_rank_combination(const std::vector<int> &comb, int d) {
  const int k = static_cast<int>(comb.size());
  if (k == 0) {
    return 0;
  }

  int rank = 0;
  int prev = -1;
  for (int i = 0; i < k; ++i) {
    const int value = comb[static_cast<size_t>(i)];
    for (int x = prev + 1; x < value; ++x) {
      rank += binomial_coeff(d - 1 - x, k - 1 - i);
    }
    prev = value;
  }
  return rank;
}

/**
 * \brief Precompute child indices and signs for `d_k` style expansions.
 * \param d Ambient dimension.
 * \param k Exterior degree.
 * \return Tables used to assemble `(k+1)` from `k` components.
 */
inline Kp1ChildrenAndSignsData kp1_children_and_signs(int d, int k) {
  if (k < 1 || k > d - 1) {
    throw std::invalid_argument("kp1_children_and_signs requires 1 <= k <= d-1");
  }

  Kp1ChildrenAndSignsData out;
  out.idx_k = get_wedge_basis_indices(d, k);
  out.idx_kp1 = get_wedge_basis_indices(d, k + 1);
  out.signs.resize(static_cast<size_t>(k + 1), 1);
  for (int r = 0; r < k + 1; ++r) {
    out.signs[static_cast<size_t>(r)] = (r % 2 == 0) ? 1 : -1;
  }

  out.children.resize(out.idx_kp1.size(), std::vector<int>(static_cast<size_t>(k + 1), 0));
  for (size_t row = 0; row < out.idx_kp1.size(); ++row) {
    const auto &parent = out.idx_kp1[row];
    for (int r = 0; r < k + 1; ++r) {
      std::vector<int> child;
      child.reserve(static_cast<size_t>(k));
      for (int c = 0; c < k + 1; ++c) {
        if (c == r) {
          continue;
        }
        child.push_back(parent[static_cast<size_t>(c)]);
      }
      out.children[row][static_cast<size_t>(r)] = lex_rank_combination(child, d);
    }
  }

  return out;
}

/**
 * \brief Return complement indices of `subset` in `[0, total)`.
 * \param total Universe size.
 * \param subset Sorted subset indices.
 * \return Sorted complement indices.
 */
inline std::vector<int> complement_indices(int total, const std::vector<int> &subset) {
  std::vector<int> out;
  out.reserve(static_cast<size_t>(total - static_cast<int>(subset.size())));
  int cursor = 0;
  for (int i = 0; i < total; ++i) {
    if (cursor < static_cast<int>(subset.size()) && subset[static_cast<size_t>(cursor)] == i) {
      ++cursor;
      continue;
    }
    out.push_back(i);
  }
  return out;
}

/**
 * \brief Precompute sparse mapping from `(k1,k2)` components to `k1+k2` wedge components.
 *
 * The output drives fast pointwise wedge assembly in diffusion-form operators.
 * \param d Ambient dimension.
 * \param k1 Left exterior degree.
 * \param k2 Right exterior degree.
 * \return Index/sign mapping for wedge assembly.
 */
inline WedgeProductIndexData get_wedge_product_indices(int d, int k1, int k2) {
  WedgeProductIndexData out;
  const int k_total = k1 + k2;
  if (k_total > d || k1 < 0 || k2 < 0) {
    return out;
  }

  const auto targets = get_wedge_basis_indices(d, k_total);
  const auto local_I = get_wedge_basis_indices(k_total, k1);

  out.n_targets = static_cast<int>(targets.size());
  out.n_splits = static_cast<int>(local_I.size());

  const size_t n_terms = static_cast<size_t>(out.n_targets) * static_cast<size_t>(out.n_splits);
  out.target_indices.reserve(n_terms);
  out.left_indices.reserve(n_terms);
  out.right_indices.reserve(n_terms);
  out.signs.reserve(n_terms);

  for (int target_idx = 0; target_idx < out.n_targets; ++target_idx) {
    const auto &target = targets[static_cast<size_t>(target_idx)];
    for (int split = 0; split < out.n_splits; ++split) {
      const auto &local_left = local_I[static_cast<size_t>(split)];
      const auto local_right = complement_indices(k_total, local_left);

      std::vector<int> left;
      std::vector<int> right;
      left.reserve(static_cast<size_t>(k1));
      right.reserve(static_cast<size_t>(k2));

      for (int idx : local_left) {
        left.push_back(target[static_cast<size_t>(idx)]);
      }
      for (int idx : local_right) {
        right.push_back(target[static_cast<size_t>(idx)]);
      }

      const int parity =
          (std::accumulate(local_left.begin(), local_left.end(), 0) -
           (k1 * (k1 - 1) / 2)) &
          1;
      const int sign = (parity == 0) ? 1 : -1;

      out.target_indices.push_back(target_idx);
      out.left_indices.push_back(lex_rank_combination(left, d));
      out.right_indices.push_back(lex_rank_combination(right, d));
      out.signs.push_back(sign);
    }
  }

  return out;
}

} // namespace igneous::ops::diffusion
