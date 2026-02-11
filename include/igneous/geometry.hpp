// geometry.hpp
#pragma once
#include <igneous/algebra.hpp>
#include <memory>
#include <vector>

namespace igneous {

/**
 * @brief Aligned allocator to ensure SIMD compatibility for Multivectors.
 */
template <typename T> struct AlignedAllocator {
  using value_type = T;
  static constexpr size_t alignment = xsimd::default_arch::alignment();

  T *allocate(std::size_t n) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, n * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<T *>(ptr);
  }

  void deallocate(T *p, std::size_t) { free(p); }
};

/**
 * @brief GeometryBuffer: A Data-Oriented container for CGA primitives.
 * * Instead of storing objects, we store flat, contiguous arrays of blades.
 * This allows the CPU to stream geometry into SIMD registers without cache
 * misses.
 */
template <typename Field, IsSignature Sig> struct GeometryBuffer {
  // 0-Simplices: Points (1-blades in CGA)
  // Layout: [P0, P1, P2, P3, ... Pn]
  std::vector<Multivector<Field, Sig>,
              AlignedAllocator<Multivector<Field, Sig>>>
      points;

  // 1-Simplices: Edges/Point-Pairs (2-blades in CGA)
  // Storing the dual bivector allows for instant intersection tests.
  std::vector<Multivector<Field, Sig>,
              AlignedAllocator<Multivector<Field, Sig>>>
      edges;

  // 2-Simplices: Faces/Circles (3-blades in CGA)
  // The trivector represents the circumcircle and surface orientation.
  std::vector<Multivector<Field, Sig>,
              AlignedAllocator<Multivector<Field, Sig>>>
      faces;

  void reserve(size_t v, size_t e, size_t f) {
    points.reserve(v);
    edges.reserve(e);
    faces.reserve(f);
  }

  void clear() {
    points.clear();
    edges.clear();
    faces.clear();
  }
};

} // namespace igneous