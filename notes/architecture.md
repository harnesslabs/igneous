# 🏗️ System Architecture

## 1. The Kernel (`igneous::Multivector`)
We prioritize **Throughput** over Latency.
- **Problem:** GA operations involve sparse matrix multiplies that branch heavily.
- **Solution:** We assume a "Dense" layout for small dimensions (up to 5D) and use template metaprogramming to unroll loops.

## 2. SIMD Strategy (`igneous::WideMultivector`)
We use a **Structure of Arrays (SoA)** approach.
Instead of:
```cpp
struct Multivector { float components[32]; }; // Array of Structs (AoS)
```
We use:
```cpp
struct WideMultivector { xsimd::batch<float> components[32]; }; // Struct of Arrays (SoA)
```
- **Benefit:** A single instruction processes 4 (NEON) or 8 (AVX) multivectors at once.

- **Usage:** Always batch operations. Don't multiply one vector; multiply a generic stream of them.

## 3. Memory Management (`igneous::MemoryArena`)
Geometry is allocation-heavy.

- **Constraint:** `malloc`/`free` are non-deterministic and slow.

- **Solution:** `MemoryArena` allocates a 1GB block at startup.

- **Allocation cost**: `ptr += size` (1 CPU cycle).

- **Deallocation cost**: `0` (We reset the whole arena at the end of the frame).