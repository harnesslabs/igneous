#include <chrono>
#include <fmt/core.h>
#include <igneous/algebra.hpp>
#include <random>
#include <vector>

using namespace igneous;
using Clock = std::chrono::high_resolution_clock;

// Define the algebra to test (e.g., Euclidean 3D)
using Algebra = Multivector<double, Euclidean3D>;

int main() {
  constexpr int ITERATIONS = 10'000'000;

  // 1. Setup Random Data
  // We generate data at runtime so the compiler can't cheat by pre-calculating.
  std::vector<Algebra> lhs(ITERATIONS);
  std::vector<Algebra> rhs(ITERATIONS);

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  std::uniform_int_distribution<unsigned int> blade_dist(0, 7);

  fmt::print("Generating {} random multivectors...\n", ITERATIONS);
  for (int i = 0; i < ITERATIONS; ++i) {
    // Create random sparse vectors (simulating real physics use case)
    lhs[i] = Algebra::from_blade(blade_dist(gen), dist(gen));
    rhs[i] = Algebra::from_blade(blade_dist(gen), dist(gen));
  }

  // Dummy accumulator to prevent optimization
  double checksum = 0.0;

  // --- Benchmark 1: Naive ---
  fmt::print("Benchmarking Naive Implementation... ");
  auto start = Clock::now();

  for (int i = 0; i < ITERATIONS; ++i) {
    auto result = lhs[i].multiply_naive(rhs[i]);
    checksum += result[0]; // Touch memory
  }

  auto end = Clock::now();
  auto dur_naive =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  fmt::print("{} ms\n", dur_naive);

  // --- Benchmark 2: Template Unrolled ---
  fmt::print("Benchmarking TMP Unrolled Implementation... ");
  start = Clock::now();

  for (int i = 0; i < ITERATIONS; ++i) {
    auto result = lhs[i] * rhs[i]; // Uses operator* (Unrolled)
    checksum += result[0];
  }

  end = Clock::now();
  auto dur_tmpl =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  fmt::print("{} ms\n", dur_tmpl);

  // --- Results ---
  fmt::print("\nSpeedup: {:.2f}x\n", (double)dur_naive / (double)dur_tmpl);
  fmt::print("Checksum: {} (ignore this)\n", checksum);

  return 0;
}