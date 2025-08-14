#include <assert.h>
#include <cmath>
#include <cuda.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include "LAP/Hung_lap.cuh"
#include "include/defs.cuh"
#include "include/Timer.h"

template <typename scalar_t>
torch::Tensor solve(at::Tensor cost_matrix) {
  // Create LAP solver instance
  Timer t;
  TLAP<scalar_t> solver(cost_matrix);
  auto time = t.elapsed();
  Log(debug, "LAP creation time %f s", time);

  // Solve the assignment problem
  t.reset();
  torch::Tensor assignments_tensor = solver.solve();
  time = t.elapsed();
  Log(debug, "LAP solving time %f s", time);

  return assignments_tensor;
}

// Convert PyTorch tensor to raw pointer and call LAP solver
torch::Tensor solve_lap(torch::Tensor const& cost_matrix, at::Device device) {
  // Ensure the target LAP solver device is a CUDA device
  TORCH_CHECK(device.is_cuda(), "Expected a CUDA device (use CPU-based LAP solvers instead)");
  // Ensure the input is a 3D tensor
  TORCH_CHECK(cost_matrix.dim() == 3, "Input must be a 3D tensor");
  // Ensure the input is a square matrix
  TORCH_CHECK(cost_matrix.size(1) == cost_matrix.size(2),
              "Input must be a batch of square matrices");

  auto dtype = cost_matrix.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16 ||
      dtype == at::kFloat || dtype == at::kDouble ||
      dtype == at::kInt || dtype == at::kLong,
      "solve_lap: expected cost_matrix to have dtype Half, BFloat16, Float, Double, or any Integer type, but got ",
      dtype
  );

  at::Tensor local_costs, assignments_tensor;

  at::cuda::CUDAGuard guard(device);
  if (cost_matrix.device() == device)
    local_costs = cost_matrix.clone();
  else
    local_costs = cost_matrix.cpu().to(device);
  
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kInt32, torch::kInt64, cost_matrix.scalar_type(),
      "solve_lap", [&] { assignments_tensor = solve<scalar_t>(local_costs); });

  if (cost_matrix.device() == device)
    assignments_tensor = assignments_tensor.clone();
  else
    assignments_tensor = assignments_tensor.cpu().to(cost_matrix.device());
  return assignments_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("solve_lap", &solve_lap,
        "Solve Linear Assignment Problem using Hungarian algorithm on GPU");
}