from collections.abc import Callable
from typing import Any

import cutlass.cute.testing as cute_testing
import torch


def bench_cute_kernel(
    kernel: Callable[[], Any],
    *,
    warmup_iterations: int = 10,
    iterations: int = 100,
    device: Any | None = None,
) -> tuple[float, Any]:
    """Benchmark a CUDA kernel-launching callable with CuTe's CUPTI benchmark path."""
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    if device is None:
        device = torch.cuda.current_device()

    result: Any = None

    def run_kernel() -> None:
        nonlocal result
        result = kernel()

    with torch.cuda.device(device):
        avg_time_us = cute_testing.benchmark(
            run_kernel,
            kernel_arguments=cute_testing.JitArguments(),
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            use_cupti=True,
        )
    return avg_time_us, result
