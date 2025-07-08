import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch_lap_cuda import solve_lap

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

N_TRIALS = 20
BATCH_SIZES = [1, 64, 256]
DIMS = [64, 256, 512]

RANDOM_FUNCS = {
    "uniform": lambda size: torch.rand(size, dtype=torch.float32),
    "normal": lambda size: torch.randn(size, dtype=torch.float32),
    "integer": lambda size: torch.randint(0, 100, size, dtype=torch.int32),
}

FUNC_LABELS = {
    "uniform": "Uniform (0, 1)",
    "normal": "Normal (0, 1)",
    "integer": "Integer (0, 100)",
}

LAP_LABELS = {
    "scipy": "SciPy LAP",
    "torch_lap_cuda": "Torch LAP CUDA",
    "scipy_mp": "SciPy LAP (MP)",
    "scipy_mt": "SciPy LAP (MT)",
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


def test_scipy_lap_multiprocessing(cost: torch.Tensor):
    """
    Test the scipy linear_sum_assignment function.
    """
    times = []
    for _ in range(N_TRIALS):
        start = time.perf_counter()
        cost_ = cost.cpu().numpy()
        with Pool(32) as pool:
            # Use pool.map to parallelize the linear_sum_assignment calls
            _ = pool.map(linear_sum_assignment, cost_)
        times.append(time.perf_counter() - start)
    return np.mean(times)


def test_scipy_lap_multithreading(cost: torch.Tensor):
    """
    Test the scipy linear_sum_assignment function.
    """
    times = []
    for _ in range(N_TRIALS):
        start = time.perf_counter()
        cost_ = cost.cpu().numpy()
        with ThreadPool(32) as pool:
            # Use pool.map to parallelize the linear_sum_assignment calls
            _ = pool.map(linear_sum_assignment, cost_)
        times.append(time.perf_counter() - start)
    return np.mean(times)


def test_torch_lap_cuda(cost: torch.Tensor):
    """
    Test the torch_lap_cuda solve_lap function.
    """
    cost = cost.cuda()
    times = []
    for _ in range(N_TRIALS):
        start = time.perf_counter()
        solve_lap(cost)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return np.mean(times)


def plot_bar_chart(ax, data, width=0.4):
    labels_ = []
    for i, (lap_name, lap_data) in enumerate(data.items()):
        ax.bar(
            i,
            lap_data,
            width=width,
        )
        ax.annotate(
            f"x{(data['scipy'] / lap_data):.2f}",
            (i, lap_data * 1.05),
            fontsize=10,
            horizontalalignment="center",
        )
        labels_.append(LAP_LABELS[lap_name])
    ax.set_xticks(
        range(len(labels_)),
        labels=labels_,
        fontsize=10,
        fontweight="bold",
    )


def main():
    n_rows = len(BATCH_SIZES)
    n_cols = len(DIMS)
    ax_height = 4
    ax_width = 7
    print(
        f"Running benchmarks for {list(RANDOM_FUNCS.keys())} random functions and {BATCH_SIZES} batch sizes."
    )
    for name, func in RANDOM_FUNCS.items():
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * ax_width, n_rows * ax_height)
        )
        axs = axs.flatten()
        ax_idx = 0
        fig.suptitle(
            f"Benchmark for {FUNC_LABELS[name]} distribution", fontsize=16, y=1.01
        )
        results = {"scipy": {}, "torch_lap_cuda": {}, "scipy_mp": {}, "scipy_mt": {}}
        for batch_size in BATCH_SIZES:
            for dim in DIMS:
                cost = func((batch_size, dim, dim))
                # Test scipy
                scipy_time = test_scipy_lap(cost)
                results["scipy"][(batch_size, dim)] = scipy_time
                # Test scipy with multiprocessing
                scipy_mp_time = test_scipy_lap_multiprocessing(cost)
                results["scipy_mp"][(batch_size, dim)] = scipy_mp_time
                # Test scipy with multithreading
                scipy_mt_time = test_scipy_lap_multithreading(cost)
                results["scipy_mt"][(batch_size, dim)] = scipy_mt_time
                # Test torch_lap_cuda
                torch_lap_cuda_time = test_torch_lap_cuda(cost)
                results["torch_lap_cuda"][(batch_size, dim)] = torch_lap_cuda_time
                plot_bar_chart(
                    axs[ax_idx],
                    {
                        "scipy": scipy_time,
                        "scipy_mp": scipy_mp_time,
                        "scipy_mt": scipy_mt_time,
                        "torch_lap_cuda": torch_lap_cuda_time,
                    },
                    width=0.4,
                )
                axs[ax_idx].set_title(
                    f"Batch: {batch_size}, Dim: {dim}",
                    fontsize=12,
                    fontweight="bold",
                )
                axs[ax_idx].set_ylabel("Time (s)", fontsize=10)
                axs[ax_idx].set_xlabel("LAP Method", fontsize=10)
                axs[ax_idx].set_ylim(
                    0,
                    max(
                        results["scipy"][(batch_size, dim)],
                        results["scipy_mp"][(batch_size, dim)],
                        results["scipy_mt"][(batch_size, dim)],
                        results["torch_lap_cuda"][(batch_size, dim)],
                    )
                    * 1.1,
                )
                ax_idx += 1
        # Adjust layout
        fig.tight_layout()
        fig.savefig(f"figs/benchmark_{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        # Print results table
        print(f"\nBenchmark for {name} random distribution:")
        print("|", "-" * 96, "|", sep="")
        print(
            f"| {'Batch Size':^10} | {'Dimension':^10} | {'Scipy':^12} | {'Scipy (MP)':^12} | {'Scipy (MT)':^12} | {'LAP CUDA':^12} | {'Speedup':^8} |"
        )
        print("|", *["-" * el + "|" for el in [12, 12, 14, 14, 14, 14, 10]], sep="")
        for batch_size, dim in sorted(results["scipy"].keys()):
            scipy_t = results["scipy"][(batch_size, dim)]
            scipy_mp_t = results["scipy_mp"][(batch_size, dim)]
            scipy_mt_t = results["scipy_mt"][(batch_size, dim)]
            cuda_t = results["torch_lap_cuda"][(batch_size, dim)]
            speedup = scipy_t / cuda_t
            print(
                f"| {batch_size:^10d} | {dim:^10d} | {scipy_t:^12.6f} | {scipy_mp_t:^12.6f} | {scipy_mt_t:^12.6f} | {cuda_t:^12.6f} | {speedup:^7.2f}x |"
            )
        print("|", "-" * 96, "|", sep="")


if __name__ == "__main__":
    main()
