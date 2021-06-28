#include "scope/scope.hpp"

#include "args.hpp"

#define NAME "Comm_cudaMemcpyAsync_GPUToGPUPeer"

auto Comm_Memcpy_GPUToGPUPeer = [](benchmark::State &state, const int src_gpu,
                                   const int dst_gpu) {
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));

  OR_SKIP_AND_RETURN(cuda_reset_device(src_gpu),
                     "failed to reset src CUDA device");
  OR_SKIP_AND_RETURN(cuda_reset_device(dst_gpu),
                     "failed to reset dst CUDA device");

  char *src = nullptr;
  char *dst = nullptr;

  OR_SKIP_AND_RETURN(cudaSetDevice(src_gpu), "failed to set src device");
  OR_SKIP_AND_RETURN(cudaMalloc(&src, bytes), "failed to perform cudaMalloc");
  defer(cudaFree(src));
  OR_SKIP_AND_RETURN(cudaMemset(src, 0, bytes),
                     "failed to perform src cudaMemset");
  cudaError_t err = cudaDeviceEnablePeerAccess(dst_gpu, 0);
  if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
    state.SkipWithError("failed to ensure peer access");
    return;
  }

  OR_SKIP_AND_RETURN(cudaSetDevice(dst_gpu), "failed to set dst device");
  OR_SKIP_AND_RETURN(cudaMalloc(&dst, bytes), "failed to perform cudaMalloc");
  defer(cudaFree(dst));
  OR_SKIP_AND_RETURN(cudaMemset(dst, 0, bytes),
                     "failed to perform dst cudaMemset");
  err = cudaDeviceEnablePeerAccess(src_gpu, 0);
  if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
    state.SkipWithError("failed to ensure peer access");
    return;
  }

  cudaEvent_t start, stop;
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), "couldn't create stop event");
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), "couldn't create start event");

  for (auto _ : state) {
    OR_SKIP_AND_BREAK(cudaEventRecord(start, NULL), "failed to record start");
    OR_SKIP_AND_BREAK(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice),
        "failed to memcpy");
    OR_SKIP_AND_BREAK(cudaEventRecord(stop, NULL), "failed to stop");
    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop), "failed to synchronize");

    float msecTotal = 0.0f;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&msecTotal, start, stop),
                      "failed to compute elapsed time");
    state.SetIterationTime(msecTotal / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["src_gpu"] = src_gpu;
  state.counters["dst_gpu"] = dst_gpu;
};

static void registerer() {
  std::string name;
  for (size_t i = 0; i < unique_cuda_device_ids().size(); ++i) {
    for (size_t j = 0; j < unique_cuda_device_ids().size(); ++j) {
      auto src_gpu = unique_cuda_device_ids()[i];
      auto dst_gpu = unique_cuda_device_ids()[j];
      int s2d, d2s;
      if (!PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&s2d, src_gpu, dst_gpu)) &&
          !PRINT_IF_ERROR(cudaDeviceCanAccessPeer(&d2s, dst_gpu, src_gpu))) {
        if (s2d && d2s) {
          name = std::string(NAME) + "/" + std::to_string(src_gpu) + "/" +
                 std::to_string(dst_gpu);
          benchmark::RegisterBenchmark(name.c_str(), Comm_Memcpy_GPUToGPUPeer,
                                       src_gpu, dst_gpu)
              ->SMALL_ARGS()
              ->UseManualTime();
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
