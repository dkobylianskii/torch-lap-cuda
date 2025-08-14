#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>
#include <ATen/cuda/CUDAContext.h>

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

#define execKernel(kernel, exec_gridSize, exec_nwarps, stream, verbose, ...)                               \
  {                                                                                                        \
    dim3 grid(exec_gridSize);                                                                              \
    dim3 block(32, exec_nwarps);                                                                           \
    if (verbose)                                                                                           \
      Log(debug, "Launching %s with nblocks: %u, blockDim: (%u, 32)", #kernel, exec_gridSize, exec_nwarps);\
    kernel<<<grid, block, 0, stream>>>(__VA_ARGS__);                                                       \
    CUDA_RUNTIME(cudaGetLastError());                                                                      \
  }
