#include <igneous/core/gpu.hpp>

#include <Metal/Metal.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace igneous::core::gpu {
namespace {

constexpr const char* kMetalDiffusionSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void csr_matvec(device const int *row_offsets [[buffer(0)]],
                       device const int *col_indices [[buffer(1)]],
                       device const float *weights [[buffer(2)]],
                       device const float *input [[buffer(3)]],
                       device float *output [[buffer(4)]],
                       constant uint &row_count [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid >= row_count) {
    return;
  }

  const int begin = row_offsets[gid];
  const int end = row_offsets[gid + 1];
  float acc = 0.0f;
  for (int idx = begin; idx < end; ++idx) {
    acc += weights[idx] * input[col_indices[idx]];
  }
  output[gid] = acc;
}

kernel void csr_carre_du_champ(device const int *row_offsets [[buffer(0)]],
                               device const int *col_indices [[buffer(1)]],
                               device const float *weights [[buffer(2)]],
                               device const float *f [[buffer(3)]],
                               device const float *h [[buffer(4)]],
                               device float *output [[buffer(5)]],
                               constant float &inv_2t [[buffer(6)]],
                               constant uint &row_count [[buffer(7)]],
                               uint gid [[thread_position_in_grid]]) {
  if (gid >= row_count) {
    return;
  }

  const int begin = row_offsets[gid];
  const int end = row_offsets[gid + 1];
  const float fi = f[gid];
  const float hi = h[gid];

  float acc = 0.0f;
  for (int idx = begin; idx < end; ++idx) {
    const int col = col_indices[idx];
    acc += weights[idx] * (f[col] - fi) * (h[col] - hi);
  }

  output[gid] = acc * inv_2t;
}
)METAL";

struct CsrBuffers {
  size_t row_count = 0;
  size_t nnz = 0;
  id<MTLBuffer> row_offsets = nil;
  id<MTLBuffer> col_indices = nil;
  id<MTLBuffer> weights = nil;
};

class MetalDiffusionRuntime {
public:
  static MetalDiffusionRuntime& instance() {
    static MetalDiffusionRuntime runtime;
    return runtime;
  }

  [[nodiscard]] bool is_available() const {
    return ready_;
  }

  void invalidate_markov_cache(const void* cache_key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    csr_cache_.erase(cache_key);
  }

  [[nodiscard]] bool
  apply_markov_transition(const void* cache_key, std::span<const int> row_offsets,
                          std::span<const int> col_indices, std::span<const float> weights,
                          std::span<const float> input, std::span<float> output) {
    if (!ready_ || row_offsets.size() < 2) {
      return false;
    }

    const size_t row_count = row_offsets.size() - 1;
    if (input.size() != row_count || output.size() != row_count ||
        col_indices.size() != weights.size()) {
      return false;
    }

    id<MTLBuffer> rows_buffer = nil;
    id<MTLBuffer> cols_buffer = nil;
    id<MTLBuffer> weights_buffer = nil;
    if (!prepare_csr_buffers(cache_key, row_offsets, col_indices, weights, rows_buffer, cols_buffer,
                             weights_buffer)) {
      return false;
    }

    const size_t vector_bytes = row_count * sizeof(float);
    id<MTLBuffer> input_buffer = [device_ newBufferWithLength:vector_bytes
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [device_ newBufferWithLength:vector_bytes
                                                       options:MTLResourceStorageModeShared];
    if (input_buffer == nil || output_buffer == nil) {
      return false;
    }

    std::memcpy([input_buffer contents], input.data(), vector_bytes);

    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    if (command_buffer == nil) {
      return false;
    }

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return false;
    }

    const uint32_t row_count_u32 = static_cast<uint32_t>(row_count);
    [encoder setComputePipelineState:matvec_pipeline_];
    [encoder setBuffer:rows_buffer offset:0 atIndex:0];
    [encoder setBuffer:cols_buffer offset:0 atIndex:1];
    [encoder setBuffer:weights_buffer offset:0 atIndex:2];
    [encoder setBuffer:input_buffer offset:0 atIndex:3];
    [encoder setBuffer:output_buffer offset:0 atIndex:4];
    [encoder setBytes:&row_count_u32 length:sizeof(row_count_u32) atIndex:5];
    dispatch_rows(encoder, matvec_pipeline_, row_count);
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] != MTLCommandBufferStatusCompleted) {
      return false;
    }

    std::memcpy(output.data(), [output_buffer contents], vector_bytes);
    return true;
  }

  [[nodiscard]] bool
  apply_markov_transition_steps(const void* cache_key, std::span<const int> row_offsets,
                                std::span<const int> col_indices, std::span<const float> weights,
                                std::span<const float> input, int steps, std::span<float> output) {
    if (!ready_ || row_offsets.size() < 2 || steps <= 0) {
      return false;
    }

    const size_t row_count = row_offsets.size() - 1;
    if (input.size() != row_count || output.size() != row_count ||
        col_indices.size() != weights.size()) {
      return false;
    }

    if (steps == 1) {
      return apply_markov_transition(cache_key, row_offsets, col_indices, weights, input, output);
    }

    id<MTLBuffer> rows_buffer = nil;
    id<MTLBuffer> cols_buffer = nil;
    id<MTLBuffer> weights_buffer = nil;
    if (!prepare_csr_buffers(cache_key, row_offsets, col_indices, weights, rows_buffer, cols_buffer,
                             weights_buffer)) {
      return false;
    }

    const size_t vector_bytes = row_count * sizeof(float);
    id<MTLBuffer> buffer_a = [device_ newBufferWithLength:vector_bytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> buffer_b = [device_ newBufferWithLength:vector_bytes
                                                  options:MTLResourceStorageModeShared];
    if (buffer_a == nil || buffer_b == nil) {
      return false;
    }

    std::memcpy([buffer_a contents], input.data(), vector_bytes);

    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    if (command_buffer == nil) {
      return false;
    }

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return false;
    }

    const uint32_t row_count_u32 = static_cast<uint32_t>(row_count);
    [encoder setComputePipelineState:matvec_pipeline_];
    [encoder setBuffer:rows_buffer offset:0 atIndex:0];
    [encoder setBuffer:cols_buffer offset:0 atIndex:1];
    [encoder setBuffer:weights_buffer offset:0 atIndex:2];
    [encoder setBytes:&row_count_u32 length:sizeof(row_count_u32) atIndex:5];

    for (int step = 0; step < steps; ++step) {
      const bool even_step = (step % 2) == 0;
      id<MTLBuffer> src = even_step ? buffer_a : buffer_b;
      id<MTLBuffer> dst = even_step ? buffer_b : buffer_a;
      [encoder setBuffer:src offset:0 atIndex:3];
      [encoder setBuffer:dst offset:0 atIndex:4];
      dispatch_rows(encoder, matvec_pipeline_, row_count);
    }
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] != MTLCommandBufferStatusCompleted) {
      return false;
    }

    id<MTLBuffer> final_buffer = (steps % 2 == 0) ? buffer_a : buffer_b;
    std::memcpy(output.data(), [final_buffer contents], vector_bytes);
    return true;
  }

  [[nodiscard]] bool carre_du_champ(const void* cache_key, std::span<const int> row_offsets,
                                    std::span<const int> col_indices,
                                    std::span<const float> weights, std::span<const float> f,
                                    std::span<const float> h, float inv_2t,
                                    std::span<float> output) {
    if (!ready_ || row_offsets.size() < 2) {
      return false;
    }

    const size_t row_count = row_offsets.size() - 1;
    if (f.size() != row_count || h.size() != row_count || output.size() != row_count ||
        col_indices.size() != weights.size()) {
      return false;
    }

    id<MTLBuffer> rows_buffer = nil;
    id<MTLBuffer> cols_buffer = nil;
    id<MTLBuffer> weights_buffer = nil;
    if (!prepare_csr_buffers(cache_key, row_offsets, col_indices, weights, rows_buffer, cols_buffer,
                             weights_buffer)) {
      return false;
    }

    const size_t vector_bytes = row_count * sizeof(float);
    id<MTLBuffer> f_buffer = [device_ newBufferWithLength:vector_bytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> h_buffer = [device_ newBufferWithLength:vector_bytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [device_ newBufferWithLength:vector_bytes
                                                       options:MTLResourceStorageModeShared];
    if (f_buffer == nil || h_buffer == nil || output_buffer == nil) {
      return false;
    }

    std::memcpy([f_buffer contents], f.data(), vector_bytes);
    std::memcpy([h_buffer contents], h.data(), vector_bytes);

    id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
    if (command_buffer == nil) {
      return false;
    }

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return false;
    }

    const uint32_t row_count_u32 = static_cast<uint32_t>(row_count);
    [encoder setComputePipelineState:carre_pipeline_];
    [encoder setBuffer:rows_buffer offset:0 atIndex:0];
    [encoder setBuffer:cols_buffer offset:0 atIndex:1];
    [encoder setBuffer:weights_buffer offset:0 atIndex:2];
    [encoder setBuffer:f_buffer offset:0 atIndex:3];
    [encoder setBuffer:h_buffer offset:0 atIndex:4];
    [encoder setBuffer:output_buffer offset:0 atIndex:5];
    [encoder setBytes:&inv_2t length:sizeof(inv_2t) atIndex:6];
    [encoder setBytes:&row_count_u32 length:sizeof(row_count_u32) atIndex:7];
    dispatch_rows(encoder, carre_pipeline_, row_count);
    [encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] != MTLCommandBufferStatusCompleted) {
      return false;
    }

    std::memcpy(output.data(), [output_buffer contents], vector_bytes);
    return true;
  }

private:
  MetalDiffusionRuntime() {
    device_ = MTLCreateSystemDefaultDevice();
    if (device_ == nil) {
      return;
    }

    queue_ = [device_ newCommandQueue];
    if (queue_ == nil) {
      return;
    }

    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:kMetalDiffusionSource];
    id<MTLLibrary> library = [device_ newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
      return;
    }

    id<MTLFunction> matvec_fn = [library newFunctionWithName:@"csr_matvec"];
    if (matvec_fn == nil) {
      return;
    }

    matvec_pipeline_ = [device_ newComputePipelineStateWithFunction:matvec_fn error:&error];
    if (matvec_pipeline_ == nil) {
      return;
    }

    id<MTLFunction> carre_fn = [library newFunctionWithName:@"csr_carre_du_champ"];
    if (carre_fn == nil) {
      return;
    }

    carre_pipeline_ = [device_ newComputePipelineStateWithFunction:carre_fn error:&error];
    if (carre_pipeline_ == nil) {
      return;
    }

    ready_ = true;
  }

  [[nodiscard]] bool prepare_csr_buffers(const void* cache_key, std::span<const int> row_offsets,
                                         std::span<const int> col_indices,
                                         std::span<const float> weights, id<MTLBuffer>& rows_buffer,
                                         id<MTLBuffer>& cols_buffer,
                                         id<MTLBuffer>& weights_buffer) {
    const void* key =
        (cache_key != nullptr) ? cache_key : static_cast<const void*>(row_offsets.data());
    const size_t row_count = row_offsets.size() - 1;
    const size_t nnz = weights.size();

    std::lock_guard<std::mutex> lock(cache_mutex_);
    CsrBuffers& entry = csr_cache_[key];

    if (entry.row_count != row_count || entry.nnz != nnz || entry.row_offsets == nil ||
        entry.col_indices == nil || entry.weights == nil) {
      const size_t row_bytes = row_offsets.size() * sizeof(int);
      const size_t col_bytes = col_indices.size() * sizeof(int);
      const size_t weight_bytes = weights.size() * sizeof(float);

      entry.row_offsets = [device_ newBufferWithLength:row_bytes
                                               options:MTLResourceStorageModeShared];
      entry.col_indices = [device_ newBufferWithLength:col_bytes
                                               options:MTLResourceStorageModeShared];
      entry.weights = [device_ newBufferWithLength:weight_bytes
                                           options:MTLResourceStorageModeShared];
      if (entry.row_offsets == nil || entry.col_indices == nil || entry.weights == nil) {
        csr_cache_.erase(key);
        return false;
      }

      std::memcpy([entry.row_offsets contents], row_offsets.data(), row_bytes);
      std::memcpy([entry.col_indices contents], col_indices.data(), col_bytes);
      std::memcpy([entry.weights contents], weights.data(), weight_bytes);
      entry.row_count = row_count;
      entry.nnz = nnz;
    }

    rows_buffer = entry.row_offsets;
    cols_buffer = entry.col_indices;
    weights_buffer = entry.weights;
    return true;
  }

  static void dispatch_rows(id<MTLComputeCommandEncoder> encoder,
                            id<MTLComputePipelineState> pipeline, size_t row_count) {
    const NSUInteger width =
        std::max<NSUInteger>(1, std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256));
    const MTLSize threads_per_group = MTLSizeMake(width, 1, 1);
    const MTLSize threads_per_grid = MTLSizeMake(static_cast<NSUInteger>(row_count), 1, 1);
    [encoder dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_group];
  }

  bool ready_ = false;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLComputePipelineState> matvec_pipeline_ = nil;
  id<MTLComputePipelineState> carre_pipeline_ = nil;

  std::mutex cache_mutex_;
  std::unordered_map<const void*, CsrBuffers> csr_cache_;
};

} // namespace

bool available() {
  return MetalDiffusionRuntime::instance().is_available();
}

void invalidate_markov_cache(const void* cache_key) {
  if (cache_key == nullptr) {
    return;
  }
  MetalDiffusionRuntime::instance().invalidate_markov_cache(cache_key);
}

bool apply_markov_transition(const void* cache_key, std::span<const int> row_offsets,
                             std::span<const int> col_indices, std::span<const float> weights,
                             std::span<const float> input, std::span<float> output) {
  return MetalDiffusionRuntime::instance().apply_markov_transition(
      cache_key, row_offsets, col_indices, weights, input, output);
}

bool apply_markov_transition_steps(const void* cache_key, std::span<const int> row_offsets,
                                   std::span<const int> col_indices, std::span<const float> weights,
                                   std::span<const float> input, int steps,
                                   std::span<float> output) {
  return MetalDiffusionRuntime::instance().apply_markov_transition_steps(
      cache_key, row_offsets, col_indices, weights, input, steps, output);
}

bool carre_du_champ(const void* cache_key, std::span<const int> row_offsets,
                    std::span<const int> col_indices, std::span<const float> weights,
                    std::span<const float> f, std::span<const float> h, float inv_2t,
                    std::span<float> output) {
  return MetalDiffusionRuntime::instance().carre_du_champ(cache_key, row_offsets, col_indices,
                                                          weights, f, h, inv_2t, output);
}

} // namespace igneous::core::gpu
