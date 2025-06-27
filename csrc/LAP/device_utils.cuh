#pragma once
#include "../include/utils.cuh"
#include "../include/defs.cuh"

#define checkpoint()                                   \
  {                                                    \
    __syncthreads();                                   \
    if (__DEBUG__D)                                    \
    {                                                  \
      if (threadIdx.x == 0)                            \
        printf("Reached %s:%u\n", __FILE__, __LINE__); \
    }                                                  \
    __syncthreads();                                   \
  }

enum MemoryLoc
{
  INTERNAL,
  EXTERNAL
};

template <typename data = int>
struct TILED_HANDLE
{
  MemoryLoc memoryloc;
  data *cost;
  data *slack;
  data *min_in_rows;
  data *min_in_cols;
  data *objective;

  size_t *zeros;
  int *row_of_star_at_column;
  int *column_of_star_at_row; // In unified memory
  int *cover_row, *cover_column;
  int *column_of_prime_at_row, *row_of_green_at_column;
  uint *tail;

  data *max_in_mat_row, *max_in_mat_col, *d_min_in_mat;
  int row_mask;

  void clear()
  {
    // CUDA_RUNTIME(cudaFree(cost));  //Already cleared to save memory
    if (memoryloc == INTERNAL)
    {
      CUDA_RUNTIME(cudaFree(min_in_rows));
      CUDA_RUNTIME(cudaFree(min_in_cols));
      CUDA_RUNTIME(cudaFree(row_of_star_at_column));
      CUDA_RUNTIME(cudaFree(objective));
    }
    CUDA_RUNTIME(cudaFree(slack));
    CUDA_RUNTIME(cudaFree(zeros));
    CUDA_RUNTIME(cudaFree(column_of_star_at_row));
    CUDA_RUNTIME(cudaFree(cover_row));
    CUDA_RUNTIME(cudaFree(cover_column));
    CUDA_RUNTIME(cudaFree(column_of_prime_at_row));
    CUDA_RUNTIME(cudaFree(row_of_green_at_column));

    CUDA_RUNTIME(cudaFree(max_in_mat_row));
    CUDA_RUNTIME(cudaFree(max_in_mat_col));
    CUDA_RUNTIME(cudaFree(d_min_in_mat));
    CUDA_RUNTIME(cudaFree(tail));
  };
};

template <typename data = int>
struct GLOBAL_HANDLE
{
  data *cost;
  data *slack;
  data *min_in_rows;
  data *min_in_cols;

  size_t *zeros;
  int *row_of_star_at_column;
  int *column_of_star_at_row; // In unified memory
  int *cover_row, *cover_column;
  int *column_of_prime_at_row, *row_of_green_at_column;

  data *max_in_mat_row, *max_in_mat_col, *d_min_in_mat, *objective;
  int row_mask;

  void clear()
  {
    CUDA_RUNTIME(cudaFree(cost)); // Already cleared to save memory
    CUDA_RUNTIME(cudaFree(slack));
    CUDA_RUNTIME(cudaFree(min_in_rows));
    CUDA_RUNTIME(cudaFree(min_in_cols));

    CUDA_RUNTIME(cudaFree(zeros));
    CUDA_RUNTIME(cudaFree(row_of_star_at_column));
    CUDA_RUNTIME(cudaFree(column_of_star_at_row));
    CUDA_RUNTIME(cudaFree(cover_row));
    CUDA_RUNTIME(cudaFree(cover_column));
    CUDA_RUNTIME(cudaFree(column_of_prime_at_row));
    CUDA_RUNTIME(cudaFree(row_of_green_at_column));

    CUDA_RUNTIME(cudaFree(max_in_mat_row));
    CUDA_RUNTIME(cudaFree(max_in_mat_col));
    CUDA_RUNTIME(cudaFree(d_min_in_mat));
    CUDA_RUNTIME(cudaFree(objective));
  };
};

struct SHARED_HANDLE
{
  int zeros_size, n_matches;
  bool goto_5, repeat_kernel;
};

template <int NPART = 1>
struct BLOCK_HANDLE
{
  int zeros_size[NPART], n_matches[NPART];
  bool goto_5[NPART], repeat_kernel[NPART];
};

struct WARP_HANDLE
{
  int &zeros_size, &n_matches;
  bool &goto_5, &repeat_kernel;
};

void memstatus(const char *message)
{
  size_t t, f;
  float total, free;
  cuMemGetInfo(&f, &t);
  total = (t * 1.0) / (1024 * 1024);
  free = (f * 1.0) / (1024 * 1024);
  Log(info, "%s occupied memory: %.1f MB", message, total - free);
}

struct CUDAContext
{
  uint32_t max_threads_per_SM;
  uint32_t num_SMs;
  uint32_t shared_mem_size_per_block;
  uint32_t shared_mem_size_per_sm;

  CUDAContext()
  {
    /*get the maximal number of threads in an SM*/
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); /*currently 0th device*/
    max_threads_per_SM = prop.maxThreadsPerMultiProcessor;
    Log(debug, "Max threads per SM %u", max_threads_per_SM);

    // Log(LogPriorityEnum::info, "Shared MemPerBlock: %zu, PerSM: %zu", prop.sharedMemPerBlock, prop.sharedMemPerMultiprocessor);
    shared_mem_size_per_block = prop.sharedMemPerBlock;
    shared_mem_size_per_sm = prop.sharedMemPerMultiprocessor;
    num_SMs = prop.multiProcessorCount;
  }

  uint32_t GetConCBlocks(uint32_t block_size)
  {
    auto conc_blocks_per_SM = max_threads_per_SM / block_size; /*assume regs are not limited*/
    Log(LogPriorityEnum::info, "#SMs: %d, con blocks/SM: %d", num_SMs, conc_blocks_per_SM);
    return conc_blocks_per_SM;
  }
};