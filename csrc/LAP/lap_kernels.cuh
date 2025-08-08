#pragma once
#include "../include/utils.cuh"
#include "device_utils.cuh"
#include <cub/cub.cuh>

#define FULLWARP 0xffffffff
#define blockSize 1024

#define fundef template <typename data = int> \
__forceinline__ __device__

__device__ __constant__ size_t SIZE;
__device__ __constant__ size_t NPROB;

fundef void init(GLOBAL_HANDLE<data> &gh) // with single block
{
  // initializations
  // for step 2
  for (size_t i = threadIdx.x; i < SIZE; i += blockDim.x)
  {
    gh.cover_row[i] = 0;
    gh.column_of_star_at_row[i] = -1;
    gh.cover_column[i] = 0;
    gh.row_of_star_at_column[i] = -1;
    gh.zeros_sizes[i] = 0;
  }
}

fundef bool near_zero(data val)
{
  return ((val < eps) && (val > -eps));
}

fundef void row_sub(GLOBAL_HANDLE<data> &gh) 
{
  typedef cub::BlockReduce<data, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ data row_min;
  for (size_t row = 0; row < SIZE; row++) {
    data thread_min = (data)MAX_DATA;
    for (size_t col = threadIdx.x; col < SIZE; col += blockDim.x)
      thread_min = min(thread_min, gh.slack[row * SIZE + col]);

    thread_min = BlockReduce(temp_storage).Reduce(thread_min, cub::Min());
    if (threadIdx.x == 0)
      row_min = thread_min;
    __syncthreads();

    for (size_t col = threadIdx.x; col < SIZE; col += blockDim.x)
      gh.slack[row * SIZE + col] -= row_min;
  }
}

fundef void col_sub_and_compress_matrix(GLOBAL_HANDLE<data> &gh) 
{
  for (size_t col = threadIdx.x; col < SIZE; col += blockDim.x) {
    data col_min = (data)MAX_DATA;
    for (size_t row = 0; row < SIZE; row++)
      col_min = min(col_min, gh.slack[row * SIZE + col]);
    
    for (size_t row = 0; row < SIZE; row++) {
      size_t i = row * SIZE + col;
      data x = gh.slack[i] - col_min;
      gh.slack[i] = x;
      if (x == 0)
      // if (near_zero(gh.slack[i]))
      {
        int z_col = atomicAdd(&gh.zeros_sizes[row], 1);
        gh.zeros[row * SIZE + z_col] = col;
      }
    }
  }
}

fundef void step_2(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh)
{
  for (int row = threadIdx.x; row < SIZE; row += blockDim.x) 
  {
    for (int z_col = 0; z_col < gh.zeros_sizes[row]; z_col++) 
    {
      int col = gh.zeros[row * SIZE + z_col];
      if (!atomicExch((int *)&(gh.cover_column[col]), 1)) 
      {
        gh.row_of_star_at_column[col] = row;
        gh.column_of_star_at_row[row] = col;
        break;
      }
    }
  }
}

fundef void step_3(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh) // For single block
{
  if (threadIdx.x == 0)
    sh.n_matches = 0;
  __syncthreads();

  for (size_t i = threadIdx.x; i < SIZE; i += blockDim.x)
  {
    gh.cover_row[i] = 0;
    gh.cover_column[i] = (gh.row_of_star_at_column[i] >= 0);
    if (gh.cover_column[i])
      atomicAdd((int *)&sh.n_matches, 1);
  }
}

// STEP 4
// Find a noncovered zero and prime it. If there is no starred
// zero in the row containing this primed zero, go to Step 5.
// Otherwise, cover this row and uncover the column containing
// the starred zero. Continue in this manner until there are no
// uncovered zeros left. Save the smallest uncovered value and
// Go to Step 6.

fundef void step_4_init(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh)
{
  for (size_t i = threadIdx.x; i < SIZE; i += blockDim.x)
  {
    gh.column_of_prime_at_row[i] = -1;
    gh.row_of_green_at_column[i] = -1;
  }
}

fundef void step_4(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh)
{
  // volatile int* v_cover_row = gh.cover_row;
  typedef cub::BlockReduce<data, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage_found;
  __shared__ typename BlockReduce::TempStorage temp_storage_goto;
  __shared__ int warp_uncovered_found;
  volatile int* v_cover_column = gh.cover_column;
  int goto_5_flag = false;
  do
  {
    int uncovered_found = false;
    for (size_t row = threadIdx.x; row < SIZE; row += blockDim.x)
    {
      if (gh.cover_row[row])
        continue;
      int star_col = gh.column_of_star_at_row[row];
      for (size_t z_col = 0; z_col < gh.zeros_sizes[row]; z_col++)
      {
        size_t col = gh.zeros[row * SIZE + z_col];
        if (!v_cover_column[col])
        {
          uncovered_found = true;
          gh.column_of_prime_at_row[row] = col;

          if (star_col >= 0)
          {
            // there was a starred zero in this column → toggle covers & keep looping
            gh.cover_row[row] = 1;
            // atomicExch((int*)&v_cover_column[star_col], 0); // Atomic clear
            v_cover_column[star_col] = 0;
            // __threadfence_block();
            break;
          }
          else
          {
            // no starred zero in this column → we're ready to augment
            goto_5_flag = true;
          }
        }
      }
    }
    goto_5_flag = BlockReduce(temp_storage_goto).Reduce(goto_5_flag, cub::Max());
    uncovered_found = BlockReduce(temp_storage_found).Reduce(uncovered_found, cub::Max());
    if (threadIdx.x == 0) 
    {
      warp_uncovered_found = uncovered_found;
      sh.goto_5 = goto_5_flag;
    }
    __syncthreads();
  } while (warp_uncovered_found && !sh.goto_5);
}

fundef void min_reduce_kernel1(GLOBAL_HANDLE<data> &gh)
{
  typedef cub::BlockReduce<data, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  // strided reductions for each thread
  data thread_min = (data)MAX_DATA;
  // for (size_t row = 0; row < SIZE; row++)
  // {
  //   if (gh.cover_row[row])
  //     continue;
  //   for (size_t col = threadIdx.x; col < SIZE; col += blockDim.x)
  //   {
  //     data x = gh.slack[row * SIZE + col];
  //     if (!gh.cover_column[col])
  //       thread_min = min(thread_min, x);
  //   }
  // }
  for (size_t i = threadIdx.x; i < SIZE * SIZE; i += blockDim.x)
  {
    int row = i / SIZE;
    int col = i % SIZE;
    if (!gh.cover_row[row] && !gh.cover_column[col])
      thread_min = min(thread_min, gh.slack[i]);
  }

  thread_min = BlockReduce(temp_storage).Reduce(thread_min, cub::Min());
  if (threadIdx.x == 0)
    gh.d_min_in_mat[0] = thread_min;
}

/* STEP 5:
Construct a series of alternating primed and starred zeros as
follows:
Let Z0 represent the uncovered primed zero found in Step 4.
Let Z1 denote the starred zero in the column of Z0(if any).
Let Z2 denote the primed zero in the row of Z1(there will always
be one). Continue until the series terminates at a primed zero
that has no starred zero in its column. Unstar each starred
zero of the series, star each primed zero of the series, erase
all primes and uncover every line in the matrix. Return to Step 3.*/

// Eliminates joining paths
fundef void step_5a(GLOBAL_HANDLE<data> gh)
{
  // size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  for (size_t i = threadIdx.x; i < SIZE; i += blockDim.x)
  {
    int r_Z0, c_Z0;

    c_Z0 = gh.column_of_prime_at_row[i];
    if (c_Z0 >= 0 && gh.column_of_star_at_row[i] < 0) // if primed zero does not share a row with a starred zero
    {
      gh.row_of_green_at_column[c_Z0] = i; // mark the column as green

      while ((r_Z0 = gh.row_of_star_at_column[c_Z0]) >= 0) // get row of star in the same column
      {
        c_Z0 = gh.column_of_prime_at_row[r_Z0]; // get column of prime in the same column as current star (should always exist)
        gh.row_of_green_at_column[c_Z0] = r_Z0; // mark the column as green
      }
    }
  }
}

// // Applies the alternating paths
fundef void step_5b(GLOBAL_HANDLE<data> &gh)
{
  for (size_t c = threadIdx.x; c < SIZE; c += blockDim.x)
  {
    int row = gh.row_of_green_at_column[c];
    int col = c;

    if (row >= 0 && gh.row_of_star_at_column[c] < 0)
    {
      int next_col = gh.column_of_star_at_row[row];
      while (1) {
        gh.column_of_star_at_row[row] = col;
        gh.row_of_star_at_column[col] = row;
        if (next_col < 0) break;
        col = next_col;
        row = gh.row_of_green_at_column[col];
        next_col = gh.column_of_star_at_row[row];
      };
    }
  }
}

fundef void step_6_add_sub_fused_compress_matrix(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh) // For single block
{
  // STEP 6:
  /*STEP 6: Add the minimum uncovered value to every element covered by both a row and column, 
  and subtract it from every uncovered element.
  Return to Step 4 without altering any stars, primes, or covered lines. */
  // const size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;

  for (size_t i = threadIdx.x; i < SIZE; i += blockDim.x)
    gh.zeros_sizes[i] = 0;
  __syncthreads();

  data offset = gh.d_min_in_mat[0];

  // for (size_t row = 0; row < SIZE; row++) {
  //   int cover_row = gh.cover_row[row];
  //   for (size_t col = threadIdx.x; col < SIZE; col += blockDim.x)
  //   {
  //     size_t i = row * SIZE + col;
  //     data x = gh.slack[i] + (cover_row + gh.cover_column[col] - 1) * offset;
  //     gh.slack[i] = x;
  //     if (x == 0)
  //     {
  //       int z_col = atomicAdd(&gh.zeros_sizes[row], 1);
  //       gh.zeros[row * SIZE + z_col] = col;
  //     }
  //   }
  // }
  for (size_t i = threadIdx.x; i < SIZE * SIZE; i += blockDim.x)
  {
    int row = i / SIZE;
    int col = i % SIZE;
    data x = gh.slack[i] + (gh.cover_row[row] + gh.cover_column[col] - 1) * offset;
    gh.slack[i] = x;
    if (x == 0)
    {
      int z_col = atomicAdd(&gh.zeros_sizes[row], 1);
      gh.zeros[row * SIZE + z_col] = col;
    }
  }
}

fundef void set_handles(TILED_HANDLE<data> &th, GLOBAL_HANDLE<data> &gh, uint &problemID)
{
  if (threadIdx.x == 0)
  {
    size_t b = blockIdx.x;
    problemID = atomicAdd(th.tail, 1);
    // problemID = b;
    if (problemID < NPROB)
    {
      // gh.SIZE = SIZE;
      // External memory
      gh.slack = &th.slack[(size_t)problemID * SIZE * SIZE];
      gh.column_of_star_at_row = &th.column_of_star_at_row[(size_t)problemID * SIZE];
      gh.row_of_star_at_column = &th.row_of_star_at_column[(size_t)problemID * SIZE];

      // Internal memory
      gh.zeros = &th.zeros[b * SIZE * SIZE];
      gh.zeros_sizes = &th.zeros_sizes[b * SIZE];
      gh.cover_row = &th.cover_row[b * SIZE];
      gh.cover_column = &th.cover_column[b * SIZE];
      gh.column_of_prime_at_row = &th.column_of_prime_at_row[b * SIZE];
      gh.row_of_green_at_column = &th.row_of_green_at_column[b * SIZE];
      gh.d_min_in_mat = &th.d_min_in_mat[b * 1];
    }
  }
}

fundef void BHA(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh, const uint problemID = 0)
{
  init(gh);
  row_sub(gh);
  __syncthreads();
  col_sub_and_compress_matrix(gh);
  __syncthreads();
  step_2(gh, sh);

  while (1)
  {
    // if (threadIdx.x == 0) printf("Start step 3 ");
    step_3(gh, sh);
    __syncthreads();
    // if (threadIdx.x == 0) printf("End step 3 ");
    if (sh.n_matches >= SIZE)
      return;
    step_4_init(gh, sh);

    while (1)
    {
      __syncthreads();
      // if (threadIdx.x == 0) printf("Start step 4 ");
      step_4(gh, sh);
      __syncthreads();
      if (sh.goto_5)
        break;

      min_reduce_kernel1(gh);
      // __syncthreads();

      // if (threadIdx.x == 0) printf("End step 4 ");

      // if (gh.d_min_in_mat[0] <= 0)
      // {
      //   __syncthreads();
      //   if (threadIdx.x == 0)
      //   {
      //     printf("minimum element in problemID %u is non positive: %f\n", problemID, (float)gh.d_min_in_mat[0]);
      //   }
      //   return;
      // }

      // if (threadIdx.x == 0) printf("Start step 6 ");

      step_6_add_sub_fused_compress_matrix(gh, sh);
      // if (threadIdx.x == 0) printf("End step 6 ");
    }
    // if (threadIdx.x == 0) printf("Start step 5a ");
    step_5a(gh);
    // if (threadIdx.x == 0) printf("Start step 5b ");
    __syncthreads();
    step_5b(gh);
    // if (threadIdx.x == 0) printf("End step 5 ");
  }
}

template <typename data>
__global__ void THA(TILED_HANDLE<data> th)
{
  __shared__ GLOBAL_HANDLE<data> gh;
  __shared__ SHARED_HANDLE sh;
  __shared__ uint problemID;
  // checkpoint();
  while (1)
  {
    set_handles(th, gh, problemID);
    __syncthreads();
    if (problemID >= NPROB)
      return;
    BHA<data>(gh, sh, problemID);
  }
  return;
}