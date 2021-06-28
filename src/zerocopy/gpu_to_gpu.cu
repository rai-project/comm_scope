
#include "scope/scope.hpp"

#include "args.hpp"
#include "kernels.hu"

#define NAME "Comm_ZeroCopy_GPUToGPU"

typedef enum {
  READ,
  WRITE,
} AccessType;

static std::string to_string(const AccessType &a) {
  if (a == READ) {
    return "_Read";
  } else {
    return "_Write";
  }
}

auto Comm_ZeroCopy_GPUToGPU = [](benchmark::State &state, const int gpu0,
                                 const int gpu1, const AccessType access_type) {
  LOG(debug, "Entered {}", NAME);

  const size_t pageSize = page_size();
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  void *ptr = nullptr;

  OR_SKIP_AND_RETURN(cuda_reset_device(gpu0), "");
  OR_SKIP_AND_RETURN(cuda_reset_device(gpu1), "");

  OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), "");
  {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu1, 0);
    if (cudaErrorPeerAccessAlreadyEnabled != err) {
      OR_SKIP_AND_RETURN(err, "");
    }
  }
  OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), "");
  {
    cudaError_t err = cudaDeviceEnablePeerAccess(gpu0, 0);
    if (cudaErrorPeerAccessAlreadyEnabled != err) {
      OR_SKIP_AND_RETURN(err, "");
    }
  }

  if (READ == access_type) {
    OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), "");
    OR_SKIP_AND_RETURN(cudaMalloc(&ptr, bytes), "");
    OR_SKIP_AND_RETURN(cudaMemset(ptr, 0, bytes), "");
    OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), "");
  } else {
    OR_SKIP_AND_RETURN(cudaSetDevice(gpu1), "");
    OR_SKIP_AND_RETURN(cudaMalloc(&ptr, bytes), "");
    OR_SKIP_AND_RETURN(cudaMemset(ptr, 0, bytes), "");
    OR_SKIP_AND_RETURN(cudaSetDevice(gpu0), "");
  }
  defer(cudaFree(ptr));

  cudaEvent_t start, stop;
  OR_SKIP_AND_RETURN(cudaEventCreate(&start), "");
  OR_SKIP_AND_RETURN(cudaEventCreate(&stop), "");
  defer(cudaEventDestroy(start));
  defer(cudaEventDestroy(stop));

  for (auto _ : state) {
    OR_SKIP_AND_BREAK(cudaEventRecord(start), "");
    if (READ == access_type) {
      // READ: gpu1 reads from gpu0 (gpu0 is src, gpu1 is dst)
      gpu_read<256><<<256, 256>>>(static_cast<int32_t *>(ptr),
                                       static_cast<int32_t *>(nullptr), bytes);
    } else {
      // WRITE: gpu0 writes to gpu1 (gpu0 is src, gpu1 is dst)
      gpu_write<256><<<256, 256>>>(static_cast<int32_t *>(ptr), bytes);
    }
    OR_SKIP_AND_BREAK(cudaEventRecord(stop), "");
    OR_SKIP_AND_BREAK(cudaEventSynchronize(stop), "");

    float millis;
    OR_SKIP_AND_BREAK(cudaEventElapsedTime(&millis, start, stop), "");
    state.SetIterationTime(millis / 1000);
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters["bytes"] = bytes;
  state.counters["gpu0"] = gpu0;
  state.counters["gpu1"] = gpu1;
};

static void registerer() {

  LOG(debug, "Registering {} benchmarks", NAME);
  for (auto workload : {READ, WRITE}) {
    for (auto src_gpu : unique_cuda_device_ids()) {
      for (auto dst_gpu : unique_cuda_device_ids()) {
        {

          int s2d = false;
          int d2s = false;
          if (!PRINT_IF_ERROR(
                  cudaDeviceCanAccessPeer(&s2d, src_gpu, dst_gpu)) &&
              !PRINT_IF_ERROR(
                  cudaDeviceCanAccessPeer(&d2s, dst_gpu, src_gpu))) {
            if (s2d && d2s) {
              std::string name(NAME);
              name += to_string(workload) + "/" + std::to_string(src_gpu) +
                      "/" + std::to_string(dst_gpu);
              benchmark::RegisterBenchmark(name.c_str(), Comm_ZeroCopy_GPUToGPU,
                                           src_gpu, dst_gpu, workload)
                  ->ARGS()
                  ->UseManualTime();
            } else {
              LOG(debug,
                  "{} can't run on devices {} {}: peer access not available",
                  NAME, src_gpu, dst_gpu);
            }
          }
        }
      }
    }
  }
}

SCOPE_AFTER_INIT(registerer, NAME);
