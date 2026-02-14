#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vector>

#include <igneous/ops/diffusion/basis.hpp>

TEST_CASE("Wedge basis indices match lexicographic combinations") {
  const auto idx = igneous::ops::get_wedge_basis_indices(4, 2);
  REQUIRE(idx.size() == 6);

  CHECK(idx[0] == std::vector<int>({0, 1}));
  CHECK(idx[1] == std::vector<int>({0, 2}));
  CHECK(idx[2] == std::vector<int>({0, 3}));
  CHECK(idx[3] == std::vector<int>({1, 2}));
  CHECK(idx[4] == std::vector<int>({1, 3}));
  CHECK(idx[5] == std::vector<int>({2, 3}));
}

TEST_CASE("kp1 children and signs follow Laplace expansion semantics") {
  const auto info = igneous::ops::kp1_children_and_signs(4, 2);
  REQUIRE(info.signs.size() == 3);
  CHECK(info.signs[0] == 1);
  CHECK(info.signs[1] == -1);
  CHECK(info.signs[2] == 1);

  for (size_t row = 0; row < info.idx_kp1.size(); ++row) {
    const auto &parent = info.idx_kp1[row];
    for (int r = 0; r < 3; ++r) {
      std::vector<int> child;
      child.reserve(2);
      for (int c = 0; c < 3; ++c) {
        if (c != r) {
          child.push_back(parent[static_cast<size_t>(c)]);
        }
      }
      const int expected = igneous::ops::lex_rank_combination(child, 4);
      CHECK(info.children[row][static_cast<size_t>(r)] == expected);
    }
  }
}

TEST_CASE("Wedge product index mapping produces anti-symmetric split for 1-forms") {
  const auto map = igneous::ops::get_wedge_product_indices(3, 1, 1);
  REQUIRE(map.n_targets == 3);
  REQUIRE(map.n_splits == 2);
  REQUIRE(map.target_indices.size() == 6);

  // Target [0,1] contributes with (0,1) and (1,0) with opposite signs.
  CHECK(map.target_indices[0] == 0);
  CHECK(map.left_indices[0] == 0);
  CHECK(map.right_indices[0] == 1);
  CHECK(map.signs[0] == 1);

  CHECK(map.target_indices[1] == 0);
  CHECK(map.left_indices[1] == 1);
  CHECK(map.right_indices[1] == 0);
  CHECK(map.signs[1] == -1);
}
