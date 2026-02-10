#include <chrono>
#include <fmt/core.h>
#include <igneous/algebra.hpp>
#include <igneous/memory.hpp>
#include <vector>

using namespace igneous;
using Clock = std::chrono::high_resolution_clock;

using Algebra = Multivector<double, Euclidean3D>;

int main() {
  constexpr int OBJECT_COUNT = 1'000'000;

  // --- Benchmark 1: Standard Heap Allocation ---
  fmt::print("Benchmarking Standard 'new' (Heap)...\n");
  auto start = Clock::now();

  // We use a vector of pointers to simulate creating distinct objects
  std::vector<Algebra *> heap_objects;
  heap_objects.reserve(OBJECT_COUNT);

  for (int i = 0; i < OBJECT_COUNT; ++i) {
    // This hits the OS allocator (malloc) every time
    heap_objects.push_back(new Algebra());
  }

  auto end = Clock::now();
  auto dur_heap =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  fmt::print("Time: {} ms\n", dur_heap);

  // Cleanup heap objects (slow!)
  for (auto *ptr : heap_objects)
    delete ptr;

  // --- Benchmark 2: Arena Allocation ---
  fmt::print("Benchmarking Arena Allocation...\n");

  // Create an arena big enough (Size of Algebra * Count + some padding)
  // Algebra is 8 doubles (64 bytes). 1M * 64B = ~64MB.
  igneous::MemoryArena arena(OBJECT_COUNT * sizeof(Algebra) * 2);

  // We use std::pmr::vector which uses our arena
  std::pmr::vector<Algebra *> arena_objects(&arena);
  arena_objects.reserve(OBJECT_COUNT);

  start = Clock::now();

  for (int i = 0; i < OBJECT_COUNT; ++i) {
    // This uses our do_allocate (pointer bump)
    void *mem = arena.allocate(sizeof(Algebra));
    // Placement new (construct object in that memory)
    arena_objects.push_back(new (mem) Algebra());
  }

  end = Clock::now();
  auto dur_arena =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  fmt::print("Time: {} ms\n", dur_arena);

  fmt::print("Speedup: {:.2f}x\n", (double)dur_heap / (double)dur_arena);

  return 0;
}