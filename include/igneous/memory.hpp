#pragma once
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <vector>

namespace igneous {

// A simple linear allocator (Arena).
// It grabs a big chunk of memory upfront and hands it out sequentially.
// Deallocation only happens when the entire arena is reset or destroyed.
class MemoryArena : public std::pmr::memory_resource {
private:
  std::vector<std::byte> buffer; // The backing storage
  std::byte *ptr = nullptr;      // Current allocation pointer
  std::size_t offset = 0;        // Current offset in bytes

public:
  // Reserve a big block of memory (default 1MB for now)
  explicit MemoryArena(std::size_t size_bytes = 1024 * 1024) {
    buffer.resize(size_bytes);
    ptr = buffer.data();
  }

  // Reset the arena (wipe all allocations instantly)
  void reset() { offset = 0; }

  // Get usage statistics
  std::size_t used_bytes() const { return offset; }
  std::size_t total_bytes() const { return buffer.size(); }

protected:
  // Implementation of do_allocate (required by std::pmr)
  void *do_allocate(std::size_t bytes, std::size_t alignment) override {
    // Calculate padding needed for alignment
    std::size_t padding = 0;
    std::uintptr_t current_addr =
        reinterpret_cast<std::uintptr_t>(ptr + offset);

    if (alignment > 0) {
      std::size_t mask = alignment - 1;
      if (current_addr & mask) {
        padding = alignment - (current_addr & mask);
      }
    }

    if (offset + padding + bytes > buffer.size()) {
      throw std::bad_alloc(); // Arena is full!
    }

    void *result = ptr + offset + padding;
    offset += padding + bytes;
    return result;
  }

  // Linear allocators don't support individual deallocation.
  // We just ignore calls to deallocate.
  void do_deallocate(void *p, std::size_t bytes,
                     std::size_t alignment) override {
    // No-op. Memory is freed when Arena is reset or destroyed.
    (void)p;
    (void)bytes;
    (void)alignment;
  }

  // Comparison (two arenas are equal if they are the same object)
  bool
  do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }
};

} // namespace igneous