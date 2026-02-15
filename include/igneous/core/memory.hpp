#pragma once
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <vector>

namespace igneous::core {

/**
 * \brief Monotonic arena allocator backed by a contiguous byte buffer.
 *
 * Individual deallocation is intentionally unsupported; callers reset the arena
 * when all allocations can be discarded together.
 */
class MemoryArena : public std::pmr::memory_resource {
private:
  /// \brief Owned backing storage.
  std::vector<std::byte> buffer;
  /// \brief Base pointer into `buffer`.
  std::byte *ptr = nullptr;
  /// \brief Current linear allocation offset in bytes.
  std::size_t offset = 0;

public:
  /**
   * \brief Construct arena with fixed capacity (`1 MiB` default).
   * \param size_bytes Backing buffer size in bytes.
   */
  explicit MemoryArena(std::size_t size_bytes = 1024 * 1024) {
    buffer.resize(size_bytes);
    ptr = buffer.data();
  }

  /// \brief Reset all allocations in constant time.
  void reset() { offset = 0; }

  /**
   * \brief Number of bytes currently allocated from this arena.
   * \return Used bytes.
   */
  std::size_t used_bytes() const { return offset; }
  /**
   * \brief Total arena capacity in bytes.
   * \return Total bytes.
   */
  std::size_t total_bytes() const { return buffer.size(); }

protected:
  /**
   * \brief `std::pmr` allocation entry point.
   * \param bytes Requested size in bytes.
   * \param alignment Requested alignment.
   * \return Pointer to allocated storage.
   */
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
      throw std::bad_alloc();
    }

    void *result = ptr + offset + padding;
    offset += padding + bytes;
    return result;
  }

  /**
   * \brief No-op for monotonic allocation strategy.
   * \param p Ignored.
   * \param bytes Ignored.
   * \param alignment Ignored.
   */
  void do_deallocate(void *p, std::size_t bytes,
                     std::size_t alignment) override {
    (void)p;
    (void)bytes;
    (void)alignment;
  }

  /**
   * \brief Memory resources compare equal only by object identity.
   * \param other Resource to compare against.
   * \return `true` if both references point to the same object.
   */
  bool
  do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }
};

} // namespace igneous::core
