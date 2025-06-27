import pytest
import torch
import numpy as np
from lap import solve_lap
from scipy.optimize import linear_sum_assignment

@pytest.mark.parametrize("batch_size", [1, 64, 128])
@pytest.mark.parametrize("size", [1, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("random_type", ['rand', 'randn', 'randint'])
def test_solve_lap(batch_size, size, dtype, random_type):
    if random_type == 'rand':
        cost_matrix = (1e3 * torch.rand((batch_size, size, size), device='cuda')).to(dtype)
    elif random_type == 'randn':
        cost_matrix = torch.randn((batch_size, size, size), device='cuda').to(dtype)
    elif random_type == 'randint':
        cost_matrix = torch.randint(0, 1024, (batch_size, size, size), device='cuda').to(dtype)
    
    assignments = solve_lap(cost_matrix)
    
    assert assignments.shape == (batch_size, size)
    assert assignments.dtype == torch.int32
    assert assignments.device.type == 'cuda'

def test_batch_unsqueeze():
    # Test with a 2D tensor to ensure it gets unsqueezed correctly
    cost_matrix = torch.rand((256, 256), dtype=torch.float32, device='cuda')
    with pytest.warns(UserWarning):
        assignments = solve_lap(cost_matrix)
    
    assert assignments.shape == (256)
    assert assignments.dtype == torch.int32
    assert assignments.device.type == 'cuda'

def test_invalid_input():
    # Test with a non-tensor input
    with pytest.raises(TypeError):
        solve_lap([1, 2, 3])
    
    # Test with a tensor on CPU
    cost_matrix = torch.rand((256, 256), dtype=torch.float32)
    with pytest.raises(ValueError):
        solve_lap(cost_matrix)
    
    # Test with a non-square matrix
    cost_matrix = torch.rand((256, 512), dtype=torch.float32, device='cuda')
    with pytest.raises(ValueError):
        solve_lap(cost_matrix)

def test_match_equivalence():
    # Test that the assignments are equivalent to SciPy's linear_sum_assignment
    batch_size = 1
    size = 256
    cost_matrix = torch.rand((batch_size, size, size), dtype=torch.float32, device='cuda')
    
    assignments = solve_lap(cost_matrix)
    
    cost_matrix_numpy = cost_matrix.cpu().numpy()
    _, row_idx = linear_sum_assignment(cost_matrix_numpy[0])

    assert torch.equal(assignments[0], torch.tensor(row_idx, dtype=torch.int32, device='cuda'))

@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0])
def test_solver_with_target_padding(scale):
    rng = np.random.default_rng()

    for _ in range(10):
        # n_objects = rng.integers(5, 10)
        # n_valid_objects = rng.integers(n_objects // 4, n_objects // 2)
        # n_valid_objects = rng.integers(n_objects // 4, n_objects // 2)
        n_objects = 5
        n_valid_objects = 3
        cost = rng.random((1, n_objects, n_objects)) * scale  # Add batch dimension
        object_valid_mask = np.zeros((1, n_objects), dtype=bool)  # Add batch dimension
        object_valid_mask[0, :n_valid_objects] = True
        cost[0, :, ~object_valid_mask[0]] = 1e8  # Set padding costs
        print(cost)

        # Convert to torch tensors
        cost_tensor = torch.from_numpy(cost).float()

        col_idx, row_idx = linear_sum_assignment(cost_tensor[0, :, :n_valid_objects].cpu().numpy()) 
        assignments = solve_lap(cost_tensor.cuda()).cpu().numpy()[0]
        print("col_idx:", col_idx)
        print("row_idx:", row_idx)
        print("assignments:", assignments)
        print(cost[0][col_idx[:n_valid_objects], row_idx[:n_valid_objects]].sum())
        print(cost[0][col_idx[:n_valid_objects], assignments[:n_valid_objects]].sum())
        assert np.all(row_idx[:n_valid_objects] == assignments[:n_valid_objects]), (
            f"Mismatch for n_objects={n_objects}, n_valid_objects={n_valid_objects}"
        )