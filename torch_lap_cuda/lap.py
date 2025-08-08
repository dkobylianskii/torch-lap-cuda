import torch
import warnings
import torch_lap_cuda_lib
from typing import Union


@torch.no_grad()
def solve_lap(cost_matrix: torch.Tensor, device: Union[torch.device, str, int] = None) -> torch.Tensor:
    """
    Solve the Linear Assignment Problem using GPU-accelerated Hungarian algorithm.

    Args:
        cost_matrix (torch.Tensor): A square matrix of costs (BxNxN) on GPU.
        The first dimension is the batch size, and the last two dimensions are the cost matrix.
        If the input is 2D, it is assumed to be a single batch (B=1).

    Returns:
        torch.Tensor: Assignment vector where index i contains the column assigned to row i.
    """
    squeeze_ = False
    if not isinstance(cost_matrix, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    if cost_matrix.device.type != "cuda":
        raise ValueError("Input tensor must be on CUDA device")
    if cost_matrix.dim() != 3:
        warnings.warn(
            f"Input tensor should be 3D (batch_size, size, size), but got {cost_matrix.dim()}D"
            " - assuming batch_size=1 for 2D input.",
            UserWarning,
        )
        squeeze_ = True
        cost_matrix = cost_matrix.unsqueeze(0)
    if cost_matrix.size(1) != cost_matrix.size(2):
        raise ValueError("Input tensor must be square (size, size) for each batch")
    if device is None:
        device = cost_matrix.device

    assignments = torch_lap_cuda_lib.solve_lap(cost_matrix, torch.device(device))
    assignments = assignments.long()
    return assignments.squeeze(0) if squeeze_ else assignments
