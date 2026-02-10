#pragma once
#include <cstdint>
#include <igneous/algebra.hpp>
#include <igneous/memory.hpp>
#include <optional>
#include <vector>

namespace igneous {

// A Handle is just an index into the Arena.
// We use handles instead of pointers so we can re-organize memory later
// without breaking references (and it saves 4 bytes on 64-bit systems if we use
// uint32).
struct SimplexHandle {
  uint32_t index = UINT32_MAX;

  bool is_valid() const { return index != UINT32_MAX; }

  bool operator==(const SimplexHandle &other) const = default;
};

// The Simplex: The fundamental atom of geometry.
// Templated on the Algebra so we can use VGA, PGA, or STA.
template <typename Algebra> struct Simplex {
  // 1. The Geometry (Where is it?)
  // e.g., a Point is 'e0 + x*e1 + y*e2' in PGA
  Algebra geometry;

  // 2. The Topology (What is it connected to?)
  // This is a "Boundary Operator".
  // A generic simplex stores its boundary as a list of handles to lower-dim
  // simplices. e.g., A Line (1-Simplex) stores handles to 2 Points
  // (0-Simplices). We use a small inline vector optimization or just a handle
  // for now. For efficiency, let's store boundary indices directly. (Simplified
  // for now: Just storing the geometry)

  // 3. Properties
  uint8_t dimension; // 0=Point, 1=Line, 2=Triangle

  // Constructors
  Simplex() : geometry(), dimension(0) {}
  Simplex(Algebra g, uint8_t dim) : geometry(g), dimension(dim) {}
};

// The Container: Holds all simplices in a contiguous Arena
template <typename Algebra> class SimplexMesh {
public:
  using SimplexType = Simplex<Algebra>;

private:
  // The Arena for raw memory storage
  MemoryArena arena;

  // We keep a direct pointer list for fast iteration,
  // but the actual data lives inside the 'arena'.
  // This vector is standard heap, but it points to arena memory.
  std::vector<SimplexType *> elements;

public:
  explicit SimplexMesh(size_t size_bytes = 1024 * 1024) : arena(size_bytes) {
    // Reserve space for pointers to avoid reallocations
    elements.reserve(10000);
  }

  // Create a new Simplex (0-Simplex, 1-Simplex, etc.)
  // Returns a Handle (Index) to it.
  SimplexHandle add(const Algebra &geom, uint8_t dim) {
    // 1. Allocate in Arena (Super fast!)
    void *mem = arena.allocate(sizeof(SimplexType), alignof(SimplexType));

    // 2. Construct in place
    SimplexType *s = new (mem) SimplexType(geom, dim);

    // 3. Store pointer and return index
    uint32_t index = static_cast<uint32_t>(elements.size());
    elements.push_back(s);

    return {index};
  }

  // Accessor
  SimplexType &get(SimplexHandle h) { return *elements[h.index]; }

  const SimplexType &get(SimplexHandle h) const { return *elements[h.index]; }

  // Iterators (for range-based loops)
  auto begin() { return elements.begin(); }
  auto end() { return elements.end(); }
  size_t size() const { return elements.size(); }
};

} // namespace igneous