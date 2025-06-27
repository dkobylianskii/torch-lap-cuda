#include "../include/defs.cuh"
#include "../include/logger.cuh"
#include "lap_kernels.cuh"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

template <typename data>
class TLAP
{
private:
  uint nprob_;
  int dev_, maxtile;
  size_t size_, h_nrows, h_ncols;
  data *Tcost_;
  uint num_blocks_4, num_blocks_reduction;

public:
  // Blank constructor
  TILED_HANDLE<data> th;
  TLAP(uint nproblem, size_t size, int dev = 0)
      : nprob_(nproblem), dev_(dev), size_(size)
  {
    th.memoryloc = EXTERNAL;
    allocate(nproblem, size, dev);
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }
  TLAP(uint nproblem, data *tcost, size_t size, int dev = 0)
      : nprob_(nproblem), Tcost_(tcost), dev_(dev), size_(size)
  {
    th.memoryloc = INTERNAL;
    allocate(nproblem, size, dev);
    th.cost = Tcost_;
    // initialize slack
    CUDA_RUNTIME(cudaMemcpy(th.slack, Tcost_, nproblem * size * size * sizeof(data), cudaMemcpyDeviceToDevice));
    CUDA_RUNTIME(cudaDeviceSynchronize());
  };
  // destructor
  ~TLAP()
  {
    th.clear();
  }

  void solve()
  {
    if (th.memoryloc == EXTERNAL)
    {
      Log(critical, "Unassigned external memory, exiting...");
      return;
    }
    int nblocks = maxtile;
    Log(debug, "nblocks: %d\n", nblocks);
    execKernel((THA<data, nthr>), nblocks, nthr, dev_, true, th);
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  void solve(data *costs, int *row_ass, data *row_duals, data *col_duals, data *obj)
  {
    if (th.memoryloc == INTERNAL)
    {
      Log(debug, "Doubly assigned external memory, exiting...");
      return;
    }
    th.cost = costs;
    th.row_of_star_at_column = row_ass;
    th.min_in_rows = row_duals;
    th.min_in_cols = col_duals;
    th.objective = obj;
    int nblocks = maxtile;
    CUDA_RUNTIME(cudaMemcpy(th.slack, th.cost, nprob_ * size_ * size_ * sizeof(data), cudaMemcpyDefault));
    Log(debug, "nblocks from external solve: %d\n", nblocks);

    execKernel((THA<data, nthr>), nblocks, nthr, dev_, true, th);
  }

  void allocate(uint nproblem, size_t size, int dev)
  {
    h_nrows = size;
    h_ncols = size;
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaMemcpyToSymbol(NPROB, &nprob_, sizeof(NPROB)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE, &size, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(nrows, &h_nrows, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(ncols, &h_ncols, sizeof(SIZE)));

    int max_active_blocks = 1;
    CUDAContext context;
    int num_SMs = context.num_SMs;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                  THA<data, nthr>,
                                                  nthr, 0);
    max_active_blocks *= num_SMs;
    maxtile = MIN(nproblem, max_active_blocks);
    Log(debug, "Grid dimension %d, max active blocks %d", maxtile, max_active_blocks);

    // external memory
    CUDA_RUNTIME(cudaMalloc((void **)&th.slack, nproblem * size * size * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.column_of_star_at_row, nproblem * h_nrows * sizeof(int)));

    // internal memory
    CUDA_RUNTIME(cudaMalloc((void **)&th.zeros, maxtile * h_nrows * h_ncols * sizeof(size_t)));
    CUDA_RUNTIME(cudaMemset(th.zeros, 0, maxtile * h_nrows * h_ncols * sizeof(size_t)));

    CUDA_RUNTIME(cudaMalloc((void **)&th.cover_row, maxtile * h_nrows * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.cover_column, maxtile * h_ncols * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.column_of_prime_at_row, maxtile * h_nrows * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.row_of_green_at_column, maxtile * h_ncols * sizeof(int)));

    CUDA_RUNTIME(cudaMalloc((void **)&th.max_in_mat_row, maxtile * h_nrows * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.max_in_mat_col, maxtile * h_ncols * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.d_min_in_mat, maxtile * 1 * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.tail, 1 * sizeof(uint)));

    CUDA_RUNTIME(cudaMemset(th.tail, 0, sizeof(uint)));
    // CUDA_RUNTIME(cudaDeviceSynchronize());
    if (th.memoryloc == INTERNAL)
    {
      CUDA_RUNTIME(cudaMalloc((void **)&th.min_in_rows, maxtile * h_nrows * sizeof(data)));
      CUDA_RUNTIME(cudaMalloc((void **)&th.min_in_cols, maxtile * h_ncols * sizeof(data)));
      CUDA_RUNTIME(cudaMalloc((void **)&th.row_of_star_at_column, maxtile * h_ncols * sizeof(int)));
      CUDA_RUNTIME(cudaMallocManaged((void **)&th.objective, nproblem * 1 * sizeof(data)));
    }
  }
};