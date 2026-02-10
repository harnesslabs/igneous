#pragma once
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <utility>

// Portable SIMD Intrinsics (NEON/AVX/SSE)
#include <xsimd/xsimd.hpp>

namespace igneous {

// ========================================================================
// 1. SIGNATURE & METRIC
// ========================================================================

// Represents Cl(p, q, r)
template <int P, int Q, int R = 0>
  requires(P >= 0) && (Q >= 0) && (R >= 0)
struct Signature {
  static constexpr int p = P;
  static constexpr int q = Q;
  static constexpr int r = R;
  static constexpr int dim = P + Q + R;
  static constexpr size_t size = 1ULL << dim;

  // Safety check: Prevent stack overflows
  static_assert(dim <= 16, "Algebra dimension too large for stack allocation.");
};

// Common Signatures
using Euclidean2D = Signature<2, 0>;
using Euclidean3D = Signature<3, 0>;
using Minkowski = Signature<1, 3>;
using PGA3D = Signature<3, 0, 1>;

// Metric Helper
template <typename Sig> constexpr int get_basis_metric(int index) {
  if (index < Sig::p)
    return 1;
  if (index < Sig::p + Sig::q)
    return -1;
  return 0;
}

// Computes the sign/metric for basis blade multiplication a * b
// Returns: 1, -1, or 0 (if metric is degenerate)
template <typename Sig>
constexpr int geometric_product_sign(unsigned int a, unsigned int b) {
  int sign = 1;

  // 1. Canonical Reordering (Count swaps)
  // We shift 'a' down and check intersections with 'b'.
  // If 'a' has bit i and 'b' has bit j with i > j, that's a swap.
  unsigned int a_temp = a >> 1;
  int swaps = 0;
  while (a_temp != 0) {
    swaps += std::popcount(a_temp & b);
    a_temp >>= 1;
  }
  if ((swaps % 2) != 0)
    sign = -sign;

  // 2. Metric Contraction (Square basis vectors)
  // Bits present in BOTH a and b are squared.
  unsigned int intersection = a & b;
  while (intersection != 0) {
    int i = std::countr_zero(intersection);
    sign *= get_basis_metric<Sig>(i);
    intersection &= intersection - 1;
  }
  return sign;
}

// ========================================================================
// 2. CAYLEY LOOKUP TABLE
// ========================================================================
// optimization for the runtime "naive" implementation.
// It pre-calculates all signs at compile time to avoid bit-twiddling at
// runtime.
template <typename Sig> struct CayleyTable {
  static constexpr size_t N = Sig::size;
  std::array<int, N * N> signs{};

  consteval CayleyTable() {
    for (unsigned int i = 0; i < N; ++i) {
      for (unsigned int j = 0; j < N; ++j) {
        signs[i * N + j] = geometric_product_sign<Sig>(i, j);
      }
    }
  }

  constexpr int get(unsigned int i, unsigned int j) const {
    return signs[i * N + j];
  }
};

// ========================================================================
// 3. MULTIVECTOR
// ========================================================================

// Concept to ensure valid signature
template <typename T>
concept IsSignature = requires {
  { T::p } -> std::convertible_to<int>;
  { T::dim } -> std::convertible_to<int>;
};

template <typename Field, IsSignature Sig> struct Multivector {
  static constexpr size_t Size = Sig::size;

  // The lookup table for naive multiplication
  static constexpr CayleyTable<Sig> table{};

  // Data Storage (Aligned for SIMD safety)
  // xsimd handles the alignment logic automatically
  alignas(xsimd::default_arch::alignment()) std::array<Field, Sig::size> data;

  // --- Constructors ---
  constexpr Multivector() : data{0} {}

  // Variadic Constructor: Multivector(1.0, 2.0, ...)
  template <std::same_as<Field>... Args>
    requires(sizeof...(Args) == Sig::size)
  constexpr Multivector(Args... args) : data{args...} {}

  // Factory: Multivector::from_blade(3, 2.5) -> 2.5 * e12
  static constexpr Multivector from_blade(unsigned int bitmap, Field scale) {
    Multivector mv;
    if (bitmap < Sig::size)
      mv.data[bitmap] = scale;
    return mv;
  }

  // --- Accessors ---
  constexpr Field operator[](size_t i) const { return data[i]; }
  constexpr Field &operator[](size_t i) { return data[i]; }

  // --- Basic Arithmetic ---
  constexpr Multivector operator+(const Multivector &other) const {
    Multivector result;
    for (size_t i = 0; i < Size; ++i)
      result.data[i] = data[i] + other.data[i];
    return result;
  }

  constexpr Multivector operator-(const Multivector &other) const {
    Multivector result;
    for (size_t i = 0; i < Size; ++i)
      result.data[i] = data[i] - other.data[i];
    return result;
  }

  // =========================================================
  // IMPLEMENTATION A: NAIVE (Runtime Loop with Table)
  // =========================================================
  constexpr Multivector multiply_naive(const Multivector &other) const {
    Multivector result;
    for (size_t i = 0; i < Size; ++i) {
      // Sparsity Check
      // Note: std::abs is not defined for xsimd types, so this works best for
      // scalars
      if constexpr (std::is_arithmetic_v<Field>) {
        if (std::abs(data[i]) < 1e-9)
          continue;
      }

      for (size_t j = 0; j < Size; ++j) {
        if constexpr (std::is_arithmetic_v<Field>) {
          if (std::abs(other.data[j]) < 1e-9)
            continue;
        }

        unsigned int target_bit = i ^ j;

        // FAST LOOKUP (No bit hacks here)
        int sign = table.get(i, j);

        if (sign != 0) {
          if (sign == 1)
            result.data[target_bit] += data[i] * other.data[j];
          else
            result.data[target_bit] -= data[i] * other.data[j];
        }
      }
    }
    return result;
  }

  // =========================================================
  // IMPLEMENTATION B: OPTIMIZED (Compile-Time Gather)
  // =========================================================
private:
  // Calculates one component of the result (TargetK)
  // by finding all pairs (I, J) that sum to it.
  template <size_t TargetK, size_t I>
  constexpr void accumulate_dot_product(Field &accumulator,
                                        const Multivector &other) const {
    // Logic: I ^ J = TargetK  ==>  J = I ^ TargetK
    constexpr size_t J = I ^ TargetK;

    // Compile-Time Sign Calculation
    constexpr int sign = geometric_product_sign<Sig>(I, J);

    if constexpr (sign != 0) {
      // Unrolling logic:
      // accumulator += this[I] * other[J] * sign
      // We separate +/- to support SIMD types that don't cast to int easily
      if constexpr (sign == 1) {
        accumulator += data[I] * other.data[J];
      } else {
        accumulator -= data[I] * other.data[J];
      }
    }
  }

  // Unrolls the dot product for a single target component
  template <size_t TargetK, size_t... Is>
  constexpr void compute_component(Multivector &result,
                                   const Multivector &other,
                                   std::index_sequence<Is...>) const {
    Field sum = Field(0); // Initialize zero (works for float or Packet)
    // Fold expression: Sum all contributions
    (accumulate_dot_product<TargetK, Is>(sum, other), ...);
    result.data[TargetK] = sum;
  }

  // Unrolls the loop over all target components
  template <size_t... Ks>
  constexpr Multivector unroll_targets(const Multivector &other,
                                       std::index_sequence<Ks...>) const {
    Multivector result;
    // For each target component K, compute its value
    (compute_component<Ks>(result, other, std::make_index_sequence<Size>{}),
     ...);
    return result;
  }

public:
  constexpr Multivector operator*(const Multivector &other) const {
    // Entry point for the unroller
    return unroll_targets(other, std::make_index_sequence<Size>{});
  }
};

// ========================================================================
// 4. WIDE TYPES (Structure of Arrays)
// ========================================================================

// Auto-detect the best SIMD batch size (NEON on Mac, AVX on Intel)
using Packet = xsimd::batch<float>;

// A Bundle of Multivectors stored in SoA format
// Operations on this type happen in parallel automatically
template <typename Sig> using WideMultivector = Multivector<Packet, Sig>;

} // namespace igneous