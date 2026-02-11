#pragma once
#include <cstdint>
#include <igneous/algebra.hpp>
#include <igneous/memory.hpp>
#include <memory_resource>
#include <stdexcept>
#include <vector>

namespace igneous {

// A Handle is a type-safe index into the Arena.
struct SimplexHandle {
  uint32_t index = UINT32_MAX;

  constexpr bool is_valid() const { return index != UINT32_MAX; }
  constexpr bool operator==(const SimplexHandle &other) const = default;
};

// The Atom of Geometry.
// Represents a 0-Simplex (Point), 1-Simplex (Edge), 2-Simplex (Triangle), etc.
template <typename Algebra> struct Simplex {
  // 1. Geometry (Position/Orientation in CGA/VGA)
  Algebra geometry;

  // 2. Properties
  uint8_t dimension; // 0, 1, 2, 3...

  // 3. Topology (The Hasse Diagram Connections)
  // We use std::pmr::vector so the dynamic lists live inside our MemoryArena.

  // Boundary: Lower-dimensional faces (e.g., Edge -> {Vertex A, Vertex B})
  std::pmr::vector<SimplexHandle> boundary;

  // Coboundary: Higher-dimensional parents (e.g., Vertex A -> {Edge 1,
  // Edge 2...})
  std::pmr::vector<SimplexHandle> coboundary;

  // Constructor requires the arena allocator
  Simplex(Algebra g, uint8_t dim, std::pmr::memory_resource *mem)
      : geometry(g), dimension(dim), boundary(mem), coboundary(mem) {}
};

// The Container: Manages the Arena and the Topological Graph
template <typename Algebra> class SimplexMesh {
public:
  using SimplexType = Simplex<Algebra>;

private:
  MemoryArena arena;
  std::pmr::polymorphic_allocator<std::byte> allocator;

  // Direct pointers for fast iteration.
  // The pointers point into 'arena', but this vector itself is on the heap.
  std::vector<SimplexType *> elements;

public:
  explicit SimplexMesh(size_t size_bytes = 1024 * 1024 * 64)
      : arena(size_bytes), allocator(&arena) {
    elements.reserve(100000);
  }

  size_t size() const { return elements.size(); }

  SimplexType &get(SimplexHandle h) {
    if (h.index >= elements.size())
      throw std::out_of_range("Invalid handle");
    return *elements[h.index];
  }

  const SimplexType &get(SimplexHandle h) const {
    if (h.index >= elements.size())
      throw std::out_of_range("Invalid handle");
    return *elements[h.index];
  }

  // --- Hasse Diagram Builders ---

  // Level 0: Add a Vertex (0-Simplex)
  SimplexHandle add_vertex(const Algebra &point) {
    return create_simplex(point, 0, {});
  }

  // Level 1: Add an Edge (1-Simplex) between two vertices
  SimplexHandle add_edge(const Algebra &line, SimplexHandle a,
                         SimplexHandle b) {
    return create_simplex(line, 1, {a, b});
  }

  // Level 2: Add a Triangle (2-Simplex) defined by 3 edges
  SimplexHandle add_triangle(const Algebra &face, SimplexHandle e1,
                             SimplexHandle e2, SimplexHandle e3) {
    return create_simplex(face, 2, {e1, e2, e3});
  }

private:
  // Internal helper that handles the heavy lifting of graph connections
  SimplexHandle
  create_simplex(const Algebra &geom, uint8_t dim,
                 std::initializer_list<SimplexHandle> boundaries) {
    // 1. Allocate Simplex in Arena
    void *mem = arena.allocate(sizeof(SimplexType), alignof(SimplexType));
    SimplexType *s = new (mem) SimplexType(geom, dim, &arena);

    // 2. Store Handle
    uint32_t new_idx = static_cast<uint32_t>(elements.size());
    elements.push_back(s);
    SimplexHandle new_handle = {new_idx};

    // 3. Link Topology (The "Hasse" logic)
    // For every boundary handle passed in...
    for (auto &lower_handle : boundaries) {
      // A. Add it to *my* boundary list
      s->boundary.push_back(lower_handle);

      // B. Add *me* to *its* coboundary list (Back-link)
      // This is the crucial step standard engines skip!
      get(lower_handle).coboundary.push_back(new_handle);
    }

    return new_handle;
  }
};

} // namespace igneous