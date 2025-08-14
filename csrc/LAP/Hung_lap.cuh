#include "../include/defs.cuh"
#include "../include/logger.cuh"
#include "lap_kernels.cuh"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

template <typename data>
class TLAP
{
private:
  int gridSize;
  at::Tensor cost_matrix;

  at::Tensor zeros;
  at::Tensor zeros_sizes;
  at::Tensor row_of_star_at_column, column_of_star_at_row;
  at::Tensor cover_row, cover_column;
  at::Tensor column_of_prime_at_row, row_of_green_at_column;
  // at::Tensor row_of_prime_at_column, column_of_green_at_row;
  at::Tensor tail;
  at::Tensor d_min_in_mat;

public:
  // Blank constructor
  TILED_HANDLE<data> th;
  TLAP(at::Tensor cost_matrix_)
      : cost_matrix(cost_matrix_)
  {
    int n_problem = cost_matrix.size(0);
    int n_rows = cost_matrix.size(1);
    int n_cols = cost_matrix.size(2);
    
    gridSize = n_problem;

    // external memory
    th.slack = cost_matrix.data_ptr<data>();

    auto input_dtype = torch::CppTypeToScalarType<data>();
    auto options = torch::TensorOptions().device(cost_matrix.device()).requires_grad(false);
    column_of_star_at_row = at::empty({n_problem, n_rows}, options.dtype(torch::kInt32));
    th.column_of_star_at_row = column_of_star_at_row.data_ptr<int>();
    row_of_star_at_column = at::empty({n_problem, n_cols}, options.dtype(torch::kInt32));
    th.row_of_star_at_column = row_of_star_at_column.data_ptr<int>();

    // internal memory
    zeros = at::empty({gridSize, n_rows, n_cols}, options.dtype(torch::kLong));
    // organised by column, with n_cols striding
    th.zeros = zeros.data_ptr<int64_t>();
    zeros_sizes = at::empty({gridSize, n_rows}, options.dtype(torch::kInt32));
    th.zeros_sizes = zeros_sizes.data_ptr<int>();
    // zeros_sizes = at::empty({gridSize}, options.dtype(torch::kInt32));
    // th.zeros_sizes = zeros_sizes.data_ptr<int>();

    cover_row = at::empty({gridSize, n_rows}, options.dtype(torch::kInt32));
    th.cover_row = cover_row.data_ptr<int>();
    cover_column = at::empty({gridSize, n_cols}, options.dtype(torch::kInt32));
    th.cover_column = cover_column.data_ptr<int>();
    column_of_prime_at_row = at::empty({gridSize, n_rows}, options.dtype(torch::kInt32));
    th.column_of_prime_at_row = column_of_prime_at_row.data_ptr<int>();
    row_of_green_at_column = at::empty({gridSize, n_cols}, options.dtype(torch::kInt32));
    th.row_of_green_at_column = row_of_green_at_column.data_ptr<int>();

    d_min_in_mat = at::empty({gridSize}, options.dtype(input_dtype));
    th.d_min_in_mat = d_min_in_mat.data_ptr<data>();
    tail = at::zeros({1}, options.dtype(torch::kInt32));
    th.tail = tail.data_ptr<int>();

    CUDA_RUNTIME(cudaMemcpyToSymbol(NPROB, &n_problem, sizeof(NPROB)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE, &n_rows, sizeof(SIZE)));
  };

  at::Tensor solve()
  {
    at::cuda::CUDAStream cuda_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = cuda_stream.stream();
    CUDA_RUNTIME(cudaStreamSynchronize(stream));
    execKernel((THA<data>), gridSize, N_WARPS, stream, true, th);
    CUDA_RUNTIME(cudaStreamSynchronize(stream));
    return column_of_star_at_row;
  };
};