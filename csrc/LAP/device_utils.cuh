#pragma once
#include "../include/utils.cuh"
#include "../include/defs.cuh"
#include <ATen/ATen.h>

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

template <typename data = int>
struct TILED_HANDLE
{
  data *slack;
  data *min_in_rows, *min_in_cols;

  int64_t *zeros;
  int *zeros_sizes;
  int *row_of_star_at_column, *column_of_star_at_row;
  int *cover_row, *cover_column;
  int *column_of_prime_at_row, *row_of_green_at_column;

  int *tail;
  data *d_min_in_mat;
};

template <typename data = int>
struct GLOBAL_HANDLE
{
  data *slack;
  data *min_in_rows, *min_in_cols;

  int64_t *zeros;
  int *zeros_sizes;
  int *row_of_star_at_column, *column_of_star_at_row;
  int *cover_row, *cover_column;
  int *column_of_prime_at_row, *row_of_green_at_column;

  data *d_min_in_mat;
};

struct SHARED_HANDLE
{
  int n_matches;
  bool goto_5, repeat_step_4;
};
