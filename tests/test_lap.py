import pytest
import torch
import numpy as np
from lap_cuda import solve_lap
from scipy.optimize import linear_sum_assignment


@pytest.mark.parametrize("batch_size", [1, 64, 128])
@pytest.mark.parametrize("size", [1, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("random_type", ["rand", "randn", "randint"])
def test_solve_lap(batch_size, size, dtype, random_type):
    if random_type == "rand":
        cost_matrix = (1e3 * torch.rand((batch_size, size, size), device="cuda")).to(
            dtype
        )
    elif random_type == "randn":
        cost_matrix = torch.randn((batch_size, size, size), device="cuda").to(dtype)
    elif random_type == "randint":
        cost_matrix = torch.randint(
            0, 1024, (batch_size, size, size), device="cuda"
        ).to(dtype)

    assignments = solve_lap(cost_matrix)

    assert assignments.shape == (batch_size, size)
    assert assignments.dtype == torch.int64
    assert assignments.device.type == "cuda"


def test_batch_unsqueeze():
    # Test with a 2D tensor to ensure it gets unsqueezed correctly
    cost_matrix = torch.rand((256, 256), dtype=torch.float32, device="cuda")
    with pytest.warns(UserWarning):
        assignments = solve_lap(cost_matrix)

    assert assignments.shape == torch.Size([256])
    assert assignments.dtype == torch.int64
    assert assignments.device.type == "cuda"


def test_invalid_input():
    # Test with a non-tensor input
    with pytest.raises(TypeError):
        solve_lap([1, 2, 3])

    # Test with a tensor on CPU
    cost_matrix = torch.rand((256, 256), dtype=torch.float32)
    with pytest.raises(ValueError):
        solve_lap(cost_matrix)

    # Test with a non-square matrix
    cost_matrix = torch.rand((256, 512), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        solve_lap(cost_matrix)


def test_match_equivalence():
    # Test that the assignments are equivalent to SciPy's linear_sum_assignment
    batch_size = 1
    size = 256
    cost_matrix = torch.rand(
        (batch_size, size, size), dtype=torch.float32, device="cuda"
    )

    assignments = solve_lap(cost_matrix)

    cost_matrix_numpy = cost_matrix.cpu().numpy()
    _, col_idx = linear_sum_assignment(cost_matrix_numpy[0])

    assert torch.equal(
        assignments[0], torch.tensor(col_idx, dtype=torch.int32, device="cuda")
    )


@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0])
def test_solver_with_target_padding(scale):
    rng = np.random.default_rng(2)

    for _ in range(10):
        n_objects = rng.integers(150, 250)
        n_valid_objects = rng.integers(n_objects // 4, n_objects // 2)

        cost = rng.random((1, n_objects, n_objects)) * scale  # Add batch dimension
        object_valid_mask = np.zeros((1, n_objects), dtype=bool)  # Add batch dimension
        object_valid_mask[0, :n_valid_objects] = True

        # Convert to torch tensors
        cost_tensor = torch.from_numpy(cost).float()

        row_idx_scipy, col_idx_scipy = linear_sum_assignment(
            cost_tensor[0, :, :n_valid_objects].cpu().numpy()
        )

        cost_tensor_ = cost_tensor.clone()[0]
        cost_tensor_[:, n_valid_objects:] = (
            max(0, cost_tensor_.max()) + 2
        )  # Set padding costs

        col_idx_cuda = solve_lap(cost_tensor_.cuda()).cpu().numpy()
        row_idx_cuda = np.arange(n_objects)
        row_idx_cuda = row_idx_cuda[col_idx_cuda < n_valid_objects]
        col_idx_cuda = col_idx_cuda[col_idx_cuda < n_valid_objects]

        row_col_scipy = np.stack((row_idx_scipy, col_idx_scipy), axis=1)
        row_col_cuda = np.stack((row_idx_cuda, col_idx_cuda), axis=1)

        row_col_scipy = np.sort(row_col_scipy, axis=1)
        row_col_cuda = np.sort(row_col_cuda, axis=1)

        assert np.array_equal(row_col_scipy, row_col_cuda), (
            f"Mismatch in assignments for n_objects={n_objects}, "
            f"n_valid_objects={n_valid_objects}. "
            f"Scipy assignments: {row_col_scipy}, "
            f"CUDA assignments: {row_col_cuda}"
        )


@pytest.mark.parametrize("batch_size", [1, 3, 4, 64, 128])
@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0])
def test_solver_batch_with_target_padding(batch_size, scale):
    rng = np.random.default_rng(42)

    n_objects = rng.integers(150, 250)
    cost = rng.random((batch_size, n_objects, n_objects)) * scale  # Add batch dimension

    n_valid_objects = torch.tensor(
        rng.integers(n_objects // 4, n_objects // 2, size=batch_size), dtype=torch.int32
    )
    object_valid_mask = torch.zeros(
        (batch_size, n_objects), dtype=bool
    )  # Add batch dimension
    for i in range(batch_size):
        object_valid_mask[i, : n_valid_objects[i]] = True

    # Convert to torch tensors
    cost_tensor = torch.from_numpy(cost).float()
    row_idx_scipy, col_idx_scipy = [], []
    for b in range(batch_size):
        row_idx, col_idx = linear_sum_assignment(
            cost_tensor[b, :, : n_valid_objects[b]].cpu().numpy()
        )
        row_idx_scipy.append(row_idx)
        col_idx_scipy.append(col_idx)

    cost_tensor_ = cost_tensor.clone()
    for b in range(batch_size):
        cost_tensor_[b, :, n_valid_objects[b] :] = max(0, cost_tensor_[b].max()) + 5

    col_idx_cuda = solve_lap(cost_tensor_.cuda()).cpu()
    row_idx_cuda = torch.arange(n_objects).unsqueeze(0).repeat(batch_size, 1)

    for b in range(batch_size):
        row_col_scipy = np.stack((row_idx_scipy[b], col_idx_scipy[b]), axis=1)
        row_col_cuda = np.stack((row_idx_cuda[b], col_idx_cuda[b]), axis=1)
        row_col_cuda = row_col_cuda[row_col_cuda[:, 1] < n_valid_objects[b].numpy()]

        row_col_scipy = np.sort(row_col_scipy, axis=1)
        row_col_cuda = np.sort(row_col_cuda, axis=1)
        assert np.array_equal(row_col_scipy, row_col_cuda), (
            f"Mismatch in assignments for n_objects={n_objects}, "
            f"batch_size={batch_size}, batch_index={b}, "
            f"n_valid_objects={n_valid_objects}. "
            f"Scipy assignments: {row_col_scipy}, "
            f"CUDA assignments: {row_col_cuda}"
        )
