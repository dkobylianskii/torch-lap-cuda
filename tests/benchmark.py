import time

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from lap_cuda import solve_lap

N_TRIALS = 20
BATCH_SIZES = [1, 64, 256]
DIMS = [64, 256, 512]

RANDOM_FUNCS = {
    "uniform": lambda size: torch.rand(size, dtype=torch.float32),
    "normal": lambda size: torch.randn(size, dtype=torch.float32),
    "integer": lambda size: torch.randint(0, 100, size, dtype=torch.int32),
}


def test_scipy_lap(cost: torch.Tensor):
    """
    Test the scipy linear_sum_assignment function.
    """
    times = []
    for _ in range(N_TRIALS):
        start = time.perf_counter()
        cost_ = cost.cpu().numpy()
        for i in range(cost_.shape[0]):
            cost_i = cost_[i]
            linear_sum_assignment(cost_i)
        times.append(time.perf_counter() - start)
    return np.mean(times)


def test_lap_cuda(cost: torch.Tensor):
    """
    Test the lap_cuda solve_lap function.
    """
    cost = cost.cuda()
    times = []
    for _ in range(N_TRIALS):
        start = time.perf_counter()
        solve_lap(cost)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return np.mean(times)


def main():
    for name, func in RANDOM_FUNCS.items():
        results = {"scipy": {}, "lap_cuda": {}}
        for batch_size in BATCH_SIZES:
            for dim in DIMS:
                cost = func((batch_size, dim, dim))
                # Test scipy
                scipy_time = test_scipy_lap(cost)
                results["scipy"][(batch_size, dim)] = scipy_time
                # Test lap_cuda
                lap_cuda_time = test_lap_cuda(cost)
                results["lap_cuda"][(batch_size, dim)] = lap_cuda_time
        # Print results table
        print(f"\nBenchmark for {name} random distribution:")
        print("|", "-" * 66, "|", sep="")
        print(
            f"| {'Batch Size':^10} | {'Dimension':^10} | {'Scipy':^12} | {'LAP CUDA':^12} | {'Speedup':^8} |"
        )
        print("|", *["-" * el + "|" for el in [12, 12, 14, 14, 10]], sep="")
        for batch_size, dim in sorted(results["scipy"].keys()):
            scipy_t = results["scipy"][(batch_size, dim)]
            cuda_t = results["lap_cuda"][(batch_size, dim)]
            speedup = scipy_t / cuda_t
            print(
                f"| {batch_size:^10d} | {dim:^10d} | {scipy_t:^12.6f} | {cuda_t:^12.6f} | {speedup:^7.2f}x |"
            )
        print("|", "-" * 66, "|", sep="")


if __name__ == "__main__":
    main()
