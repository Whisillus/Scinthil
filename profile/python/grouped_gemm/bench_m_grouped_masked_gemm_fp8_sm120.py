import argparse

import cutlass.cute as cute
import torch

from scinthil.grouped_gemm.sm120 import m_grouped_masked_gemm_fp8_torch_sm120
from scinthil.testing import (
    bench_cute_kernel,
    get_m_grouped_gemm_masked_fp8_metrics,
    get_m_grouped_masked_gemm_deepgemm_case,
    make_torch_tensor,
    quantize_fp8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SM120 FP8 M-grouped masked GEMM")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--block-swizzle-factor", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--init", choices=("zeros", "empty", "rand"), default="rand")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def benchmark_case(
    args: argparse.Namespace,
    device: torch.device,
    case_index: int,
    groups: int,
    max_m: int,
    m: int,
    n: int,
    k: int,
) -> None:
    valid_m = groups * m
    load_bytes, store_bytes, flops = get_m_grouped_gemm_masked_fp8_metrics(
        groups=groups,
        max_m=max_m,
        m=m,
        n=n,
        k=k,
    )
    torch.manual_seed(args.seed)

    with torch.cuda.device(device):
        a_shape = (groups, max_m, k)
        scale_a_shape = (groups, max_m, k // 128)
        b_shape = (groups, n, k)
        scale_b_shape = (groups, cute.ceil_div(n, 128), k // 128)
        if args.init == "empty":
            a = torch.empty(a_shape, dtype=torch.float8_e4m3fn, device=device)
            scale_a = torch.ones(scale_a_shape, dtype=torch.float32, device=device)
            b = torch.empty(b_shape, dtype=torch.float8_e4m3fn, device=device)
            scale_b = torch.ones(scale_b_shape, dtype=torch.float32, device=device)
        else:
            init_op = torch.rand if args.init == "rand" else torch.zeros
            source_a = make_torch_tensor(a_shape, dtype=torch.bfloat16, device=device, init_op=init_op)
            source_b = make_torch_tensor(b_shape, dtype=torch.bfloat16, device=device, init_op=init_op)
            a, scale_a = quantize_fp8(source_a, (1, 1, 128))
            b, scale_b = quantize_fp8(source_b, (1, 128, 128))
            del source_a, source_b

        token_info = torch.full((groups,), m, dtype=torch.int32, device=device)
        out = torch.empty((groups, max_m, n), dtype=torch.bfloat16, device=device)

        result = m_grouped_masked_gemm_fp8_torch_sm120(
            a,
            scale_a,
            b,
            scale_b,
            token_info,
            out=out,
            block_swizzle_factor=args.block_swizzle_factor,
        )
        torch.cuda.synchronize(device)

        def run_grouped_gemm() -> torch.Tensor:
            return m_grouped_masked_gemm_fp8_torch_sm120(
                a,
                scale_a,
                b,
                scale_b,
                token_info,
                out=out,
                block_swizzle_factor=args.block_swizzle_factor,
            )

        avg_time_us, result = bench_cute_kernel(
            run_grouped_gemm,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            device=device,
        )

    load_tbps = load_bytes / (avg_time_us * 1.0e6)
    store_tbps = store_bytes / (avg_time_us * 1.0e6)
    matmul_tflops = flops / (avg_time_us * 1.0e6)

    print(
        "SM120 FP8 M-grouped masked GEMM benchmark: "
        f"case={case_index}, init={args.init}, G={groups}, Mmax={max_m}, M={m}, N={n}, K={k}, "
        f"valid_M={valid_m}, block_swizzle_factor={args.block_swizzle_factor}"
    )
    print(
        f"device={torch.cuda.get_device_name(device)}, warmup={args.warmup_iterations}, "
        f"iterations={args.iterations}, avg_time_us={avg_time_us:.3f}, "
        f"load_bytes={load_bytes}, load_tbps={load_tbps:.3f}, "
        f"store_bytes={store_bytes}, store_tbps={store_tbps:.3f}, "
        f"flops={flops}, matmul_tflops={matmul_tflops:.3f}, D={tuple(result.shape)}"
    )


def main() -> int:
    args = parse_args()
    if args.block_swizzle_factor <= 0:
        raise ValueError("block-swizzle-factor must be positive")
    if args.device < 0:
        raise ValueError("device must be non-negative")
    if args.warmup_iterations < 0:
        raise ValueError("warmup-iterations must be non-negative")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark SM120 FP8 grouped GEMM")
    if args.device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.device} is not available")

    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise RuntimeError(f"SM120 CUDA device is required, got capability {capability}")

    cases = get_m_grouped_masked_gemm_deepgemm_case()
    if args.case_index is not None and not 0 <= args.case_index < len(cases):
        raise ValueError(f"case-index must be between 0 and {len(cases) - 1}")
    case_indices = range(len(cases)) if args.case_index is None else (args.case_index,)
    for case_index in case_indices:
        benchmark_case(args, device, case_index, *cases[case_index])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
