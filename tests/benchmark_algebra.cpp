#include <chrono>
#include <fmt/core.h>
#include <igneous/algebra.hpp>
#include <random>
#include <vector>

using namespace igneous;
using Clock = std::chrono::high_resolution_clock;

// 1. Standard Scalar Algebra
using Algebra = Multivector<double, Euclidean3D>;

// 2. Wide SIMD Algebra (float)
using WideAlgebra = WideMultivector<Euclidean3D>;

int main() {
  // Increased to 100 million to make SIMD measureable
  constexpr int ITERATIONS = 10'000'000;

  // --- Setup Random Data ---
  // Using vectors to ensure memory is allocated before timing
  std::vector<Algebra> lhs(ITERATIONS);
  std::vector<Algebra> rhs(ITERATIONS);

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  std::uniform_int_distribution<unsigned int> blade_dist(0, 7);

  fmt::print("Generating {} random multivectors...\n", ITERATIONS);
  for (int i = 0; i < ITERATIONS; ++i) {
    lhs[i] = Algebra::from_blade(blade_dist(gen), dist(gen));
    rhs[i] = Algebra::from_blade(blade_dist(gen), dist(gen));
  }

  // The "Sink" variable that prevents optimization
  double checksum = 0.0;

  // --- Benchmark 1: Naive ---
  fmt::print("Benchmarking Naive Implementation... ");
  auto start = Clock::now();

  for (int i = 0; i < ITERATIONS; ++i) {
    auto result = lhs[i].multiply_naive(rhs[i]);
    checksum += result[0];
  }

  auto end = Clock::now();
  auto dur_naive =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  fmt::print("{} ms\n", dur_naive);

  // --- Benchmark 2: Unrolled (Scalar) ---
  fmt::print("Benchmarking TMP Unrolled Implementation... ");
  start = Clock::now();

  for (int i = 0; i < ITERATIONS; ++i) {
    auto result = lhs[i] * rhs[i];
    checksum += result[0];
  }

  end = Clock::now();
  auto dur_tmpl =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  fmt::print("{} ms\n", dur_tmpl);

  // --- Benchmark 3: Wide SIMD ---
  constexpr int BATCH_SIZE = Packet::size;
  int BATCH_COUNT = ITERATIONS / BATCH_SIZE;

  std::vector<WideAlgebra> lhs_wide(BATCH_COUNT);
  std::vector<WideAlgebra> rhs_wide(BATCH_COUNT);

  // Init Wide Data
  for (int i = 0; i < BATCH_COUNT; ++i) {
    lhs_wide[i] = WideAlgebra::from_blade(blade_dist(gen), Packet(1.0f));
    rhs_wide[i] = WideAlgebra::from_blade(blade_dist(gen), Packet(1.0f));
  }

  fmt::print("Benchmarking Wide SIMD (Size {})... ", BATCH_SIZE);
  start = Clock::now();

  Packet wide_checksum(0.0f);

  for (int i = 0; i < BATCH_COUNT; ++i) {
    auto result = lhs_wide[i] * rhs_wide[i];
    // We accumulate into a SIMD register
    wide_checksum += result[0];
  }

  // CRITICAL FIX: Extract a value and add to global checksum.
  // This forces the loop to run because 'checksum' is printed below.
  // .get(0) grabs the first float from the SIMD packet.
  checksum += wide_checksum.get(0);

  end = Clock::now();
  auto dur_wide =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  fmt::print("{} ms\n", dur_wide);

  // --- Final Report ---
  fmt::print("\n--- Speedup Report ---\n");
  fmt::print("Scalar Unrolled vs Naive: {:.2f}x\n",
             (double)dur_naive / (double)dur_tmpl);
  fmt::print("Wide SIMD vs Scalar Unrolled: {:.2f}x\n",
             (double)dur_tmpl / (double)dur_wide);
  fmt::print("Wide SIMD vs Naive: {:.2f}x\n",
             (double)dur_naive / (double)dur_wide);

  // This print is what keeps the optimizer honest!
  fmt::print("\nChecksum: {} (ignore)\n", checksum);

  return 0;
}