#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define CUDA_RUNTIME(ans)                 \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
{

  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

    /*if (abort) */ exit(1);
  }
}

#define execKernel(kernel, gridSize, blockSize, deviceId, verbose, ...)                       \
  {                                                                                           \
    dim3 grid(gridSize);                                                                      \
    dim3 block(blockSize);                                                                    \
                                                                                              \
    CUDA_RUNTIME(cudaSetDevice(deviceId));                                                    \
    if (verbose)                                                                              \
      Log(debug, "Launching %s with nblocks: %u, blockDim: %u", #kernel, gridSize, blockSize); \
    kernel<<<grid, block>>>(__VA_ARGS__);                                                     \
    CUDA_RUNTIME(cudaGetLastError());                                                         \
    CUDA_RUNTIME(cudaDeviceSynchronize());                                                    \
  }

#define execKernel2(kernel, gridSize, blockSize, deviceId, verbose, ...) \
  {                                                                      \
    float singleKernelTime;                                              \
    cudaEvent_t start, end;                                              \
    CUDA_RUNTIME(cudaEventCreate(&start));                               \
    CUDA_RUNTIME(cudaEventCreate(&end));                                 \
    dim3 grid(gridSize);                                                 \
    dim3 block(blockSize);                                               \
                                                                         \
    CUDA_RUNTIME(cudaSetDevice(deviceId));                               \
    CUDA_RUNTIME(cudaEventRecord(start));                                \
    kernel<<<grid, block>>>(__VA_ARGS__);                                \
    CHECK_KERNEL(#kernel)                                                \
    CUDA_RUNTIME(cudaPeekAtLastError());                                 \
    CUDA_RUNTIME(cudaEventRecord(end));                                  \
                                                                         \
    CUDA_RUNTIME(cudaEventSynchronize(start));                           \
    CUDA_RUNTIME(cudaEventSynchronize(end));                             \
    CUDA_RUNTIME(cudaDeviceSynchronize());                               \
    CUDA_RUNTIME(cudaEventElapsedTime(&singleKernelTime, start, end));   \
                                                                         \
    {                                                                    \
    }                                                                    \
  }
