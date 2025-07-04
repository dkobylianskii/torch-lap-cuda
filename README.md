# CUDA LAP Solver

A fast CUDA implementation of the Linear Assignment Problem (LAP) solver for PyTorch. This project provides GPU-accelerated HyLAC algorithm implementation that can efficiently handle batched inputs.

Based on the HyLAC code https://github.com/Nagi-Research-Group/HyLAC/tree/Block-LAP
Please cite the original work if you use this code in your research:  https://doi.org/10.1016/j.jpdc.2024.104838 

## Features

- Fast CUDA-based implementation of the LAP solver 
- Batched processing support for multiple cost matrices
- Seamless integration with PyTorch
- Currently supports `torch.float32` and `torch.int32` data types.

## Requirements

- Python >= 3.9
- CUDA >= 10.0
- PyTorch
- NVIDIA GPU with compute capability >= 7.5

## Installation

To install the package, you can use pip:

```bash
pip install torch-lap-cuda --no-build-isolation
```

You can install the package directly from source:

```bash
git clone https://github.com/dkobylianskii/torch-lap-cuda.git
cd torch-lap-cuda
pip install .
```

## Usage

Here's a simple example of how to use the LAP solver:

```python
import torch
from torch_lap_cuda import solve_lap

# Create a random cost matrix (batch_size x N x N)
batch_size = 128
size = 256
cost_matrix = torch.randn((batch_size, size, size), device="cuda")

# Solve the assignment problem
# assignments shape will be (batch_size, size)
# Each batch element contains the column indices for optimal assignment
assignments = solve_lap(cost_matrix)

# Calculate total costs
batch_idxs = torch.arange(batch_size, device=assignments.device).unsqueeze(1)
row_idxs = torch.arange(size, device=assignments.device).unsqueeze(0)
total_cost = cost_matrix[batch_idxs, row_idxs, assignments].sum()
```

The solver also supports 2D inputs for single matrices:

```python
# Single cost matrix (N x N)
cost_matrix = torch.randn((size, size), device="cuda")
assignments = solve_lap(cost_matrix)  # Shape: (size,)
```

## Input Requirements

- Cost matrices must be on a CUDA device
- Input can be either 2D (N x N) or 3D (batch_size x N x N) 
- Matrices must be square
- Supports both torch.float32 and torch.int32 dtypes

## Benchmarks

Tests were performed on an NVIDIA A6000 Ada GPU with CUDA 12.5 and PyTorch 2.6.0.

To run the benchmarks, execute:

```bash
python tests/benchmark.py
```

### Benchmark for uniform random distribution:

| Batch Size | Dimension  |    Scipy     |   LAP CUDA   | Speedup  |
|------------|------------|--------------|--------------|----------|
|     1      |     64     |   0.000088   |   0.003411   |  0.03  x |
|     1      |    256     |   0.001776   |   0.025034   |  0.07  x |
|     1      |    512     |   0.008163   |   0.172786   |  0.05  x |
|     64     |     64     |   0.005553   |   0.002980   |  1.86  x |
|     64     |    256     |   0.106160   |   0.029023   |  3.66  x |
|     64     |    512     |   0.489652   |   0.211794   |  2.31  x |
|    256     |     64     |   0.022495   |   0.002992   |  7.52  x |
|    256     |    256     |   0.421585   |   0.046722   |  9.02  x |
|    256     |    512     |   1.969822   |   0.506008   |  3.89  x |

### Benchmark for normal random distribution:

| Batch Size | Dimension  |    Scipy     |   LAP CUDA   | Speedup  |
|------------|------------|--------------|--------------|----------|
|     1      |     64     |   0.000083   |   0.002496   |  0.03  x |
|     1      |    256     |   0.001071   |   0.022604   |  0.05  x |
|     1      |    512     |   0.006488   |   0.165791   |  0.04  x |
|     64     |     64     |   0.005152   |   0.002456   |  2.10  x |
|     64     |    256     |   0.090533   |   0.029366   |  3.08  x |
|     64     |    512     |   0.418793   |   0.210599   |  1.99  x |
|    256     |     64     |   0.020670   |   0.002933   |  7.05  x |
|    256     |    256     |   0.373914   |   0.047726   |  7.83  x |
|    256     |    512     |   1.660714   |   0.515324   |  3.22  x |

### Benchmark for integer random distribution:

| Batch Size | Dimension  |    Scipy     |   LAP CUDA   | Speedup  |
|------------|------------|--------------|--------------|----------|
|     1      |     64     |   0.000060   |   0.001939   |  0.03  x |
|     1      |    256     |   0.001453   |   0.002154   |  0.67  x |
|     1      |    512     |   0.005005   |   0.002502   |  2.00  x |
|     64     |     64     |   0.005075   |   0.001886   |  2.69  x |
|     64     |    256     |   0.086510   |   0.003483   |  24.84 x |
|     64     |    512     |   0.362696   |   0.005855   |  61.95 x |
|    256     |     64     |   0.020354   |   0.002204   |  9.24  x |
|    256     |    256     |   0.341607   |   0.005341   |  63.96 x |
|    256     |    512     |   1.466242   |   0.016548   |  88.61 x |

## Testing

To run the test suite:

```bash
pytest tests/
```