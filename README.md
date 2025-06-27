# CUDA LAP Solver

A fast CUDA implementation of the Linear Assignment Problem (LAP) solver for PyTorch. This project provides GPU-accelerated HyLAC algorithm implementation that can efficiently handle batched inputs.

Based on the HyLAC code https://github.com/Nagi-Research-Group/HyLAC/tree/Block-LAP
Please cite the original work if you use this code in your research:  https://doi.org/10.1016/j.jpdc.2024.104838 

## Features

- Fast CUDA-based implementation of the LAP solver 
- Batched processing support for multiple cost matrices
- Seamless integration with PyTorch

## Requirements

- Python >= 3.9
- CUDA >= 10.0
- PyTorch
- NVIDIA GPU with compute capability >= 7.5

## Installation

You can install the package directly from source:

```bash
git clone https://github.com/dkobylianskii/lap_cuda.git
cd lap_cuda
pip install .
```

## Usage

Here's a simple example of how to use the LAP solver:

```python
import torch
from lap_cuda import solve_lap

# Create a random cost matrix (batch_size x N x N)
batch_size = 128
size = 256
cost_matrix = torch.randn((batch_size, size, size), device="cuda")

# Solve the assignment problem
assignments = solve_lap(cost_matrix)

# assignments shape will be (batch_size, size)
# Each batch element contains the row indices for optimal assignment
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

For a 128x256x256 random cost matrix:
```
CUDA implementation: ~0.12 seconds
SciPy implementation: ~0.23 seconds
```

## Testing

To run the test suite:

```bash
pytest tests/
```