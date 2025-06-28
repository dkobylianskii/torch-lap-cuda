#include <assert.h>
#include <cmath>
#include <cuda.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include "LAP/Hung_lap.cuh"
#include "include/defs.cuh"
#include "include/Timer.h"

template <typename scalar_t>
torch::Tensor solve(scalar_t* cost_matrix_ptr, uint batch_size, uint dim, uint device_idx) {
  // Create LAP solver instance
  Timer t;
  TLAP<scalar_t> *solver = new TLAP<scalar_t>(batch_size, cost_matrix_ptr, dim, device_idx);
  auto time = t.elapsed();
  Log(debug, "LAP creation time %f s", time);

  // Solve the assignment problem
  t.reset();
  solver->solve();
  time = t.elapsed();
  Log(debug, "LAP solving time %f s", time);

  // Get assignments from column_of_star_at_row
  auto options =
    torch::TensorOptions()
      .dtype(torch::kInt32)
      .device(torch::kCUDA, device_idx)
      .requires_grad(false);
  torch::Tensor assignments_tensor = torch::empty({batch_size, dim}, options);
  int32_t *assignments_data = assignments_tensor.data_ptr<int32_t>();
  // cudaMemcpy(assignments_data, solver->th.row_of_star_at_column,
  //            batch_size * dim * sizeof(int32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpy(assignments_data, solver->th.column_of_star_at_row,
             batch_size * dim * sizeof(int32_t), cudaMemcpyDeviceToDevice);
  delete solver;
  return assignments_tensor;
}

// Convert PyTorch tensor to raw pointer and call LAP solver
torch::Tensor solve_lap(torch::Tensor const& cost_matrix) {
  // Ensure the input is on CUDA
  TORCH_CHECK(cost_matrix.is_cuda(), "Input tensor must be on CUDA");
  // Ensure the input is a 3D tensor
  TORCH_CHECK(cost_matrix.dim() == 3, "Input must be a 3D tensor");
  // Ensure the input is a square matrix
  TORCH_CHECK(cost_matrix.size(1) == cost_matrix.size(2),
              "Input must be a batch of square matrices");

  const auto batch_size = cost_matrix.size(0);
  const auto dim = cost_matrix.size(1);
  const auto device_idx = cost_matrix.device().index();

  void *cost_matrix_ptr = static_cast<void *>(cost_matrix.data_ptr());
  // Get raw pointer to tensor data
  return AT_DISPATCH_ALL_TYPES(cost_matrix.scalar_type(), "solve_lap", [&] {
    return solve<scalar_t>(reinterpret_cast<scalar_t *>(cost_matrix_ptr), batch_size, dim, device_idx);
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("solve_lap", &solve_lap,
        "Solve Linear Assignment Problem using Hungarian algorithm on GPU",
        py::arg("cost_matrix"));
}