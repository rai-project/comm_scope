#include <cassert>
#include <cuda_runtime.h>

#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudaMemcpyPeerAsync_GPUToGPUPeer"

auto Comm_cudaMemcpyPeerAsync_GPUToGPUPeer = [](benchmark::State &state,
                                                const int srcGpu,
                                                const int dstGpu) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  OR_SKIP_AND_RETURN(cuda_reset_device(srcGpu), "failed to reset src CUDA device");
  OR_SKIP_AND_RETURN(cuda_reset_device(dstGpu), "failed to reset dst CUDA device");

  void *src = nullptr;
  void *dst = nullptr;
  cudaStream_t stream;
  cudaError_t err;
  cudaEvent_t start, stop;

  OR_SKIP_AND_RETURN(cudaSetDevice(srcGpu), "failed to set src device");
  OR_SKIP_AND_RETURN(cudaMalloc(&src, bytes), "failed to perform cudaMalloc");
  defer(cudaFree(src));
  OR_SKIP_AND_RETURN(cudaMemset(src, 0, bytes), "failed to perform src cudaMemset");
  OR_SKIP_AND_RETURN(cudaStreamCreate(&stream), "failed to create stream");
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), "couldn't create start event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), "couldn't create stop event");
  defer(cudaEventDestroy(start));
  defer(cudaEventDestroy(stop));
  if (srcGpu != dstGpu) {
    err = cudaDeviceEnablePeerAccess(dstGpu, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError("failed to ensure peer access");
      return;
    }
  }

  OR_SKIP_AND_RETURN(cudaSetDevice(dstGpu), "failed to set dst device");
  OR_SKIP_AND_RETURN(cudaMalloc(&dst, bytes), "failed to perform cudaMalloc");
  defer(cudaFree(dst));
  OR_SKIP_AND_RETURN(cudaMemset(dst, 0, bytes), "failed to perform dst cudaMemset");
  if (srcGpu != dstGpu) {
    err = cudaDeviceEnablePeerAccess(srcGpu, 0);
    if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
      state.SkipWithError("failed to ensure peer access");
      return;
    }
  }

  OR_SKIP_AND_RETURN(cudaSetDevice(srcGpu), "failed to set src device");
  for (auto _ : state) {
    OR_SKIP_AND_BREAK(cudaEventRecord(start, stream), "failed to record start");
    OR_SKIP_AND_BREAK(cudaMemcpyPeerAsync(dst, dstGpu, src, srcGpu, bytes, stream),
            "failed to memcpy");
    OR_SKIP_AND_BREAK(cudaEventRecord(stop, stream), "failed to stop");
    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop), "failed to synchronize");

    float ms = 0.0f;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&ms, start, stop),
            NAME "failed to compute elapsed time");
    state.SetIterationTime(ms / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["srcGpu"] = srcGpu;
  state.counters["dstGpu"] = dstGpu;
};

static void registerer() {
  for (size_t i = 0; i < unique_cuda_device_ids().size(); ++i) {
    for (size_t j = 0; j < unique_cuda_device_ids().size(); ++j) {
      auto srcGpu = unique_cuda_device_ids()[i];
      auto dstGpu = unique_cuda_device_ids()[j];
      int s2d, d2s;
      if (!PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&s2d, srcGpu, dstGpu)) &&
          !PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&d2s, dstGpu, srcGpu))) {
        if (s2d && d2s) {
          std::string name = std::string(NAME) + "/" + std::to_string(srcGpu) + "/" +
                 std::to_string(dstGpu);
          benchmark::RegisterBenchmark(name.c_str(),
                                       Comm_cudaMemcpyPeerAsync_GPUToGPUPeer,
                                       srcGpu, dstGpu)
              ->SMALL_ARGS()
              ->UseManualTime();
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
