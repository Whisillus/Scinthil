import argparse

import cutlass
import torch

from scinthil.grouped_gemm.sm120 import m_grouped_masked_gemm_torch_sm120
from scinthil.testing import bench_cute_kernel, get_m_grouped_gemm_masked_metrics, make_torch_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SM120 M-grouped masked GEMM")
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--max-m", type=int, default=4096)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--block-swizzle-factor", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--init", choices=("randn", "zero", "empty"), default="randn")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.groups, args.max_m, args.m, args.n, args.k) <= 0:
        raise ValueError("groups, max-m, M, N, and K must be positive")
    if args.m > args.max_m:
        raise ValueError("M must not exceed max-m")
    if args.k % 16 != 0:
        raise ValueError("K must be divisible by 16")
    if args.block_swizzle_factor <= 0:
        raise ValueError("block-swizzle-factor must be positive")
    if args.device < 0:
        raise ValueError("device must be non-negative")
    if args.warmup_iterations < 0:
        raise ValueError("warmup-iterations must be non-negative")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark SM120 grouped GEMM")
    if args.device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.device} is not available")

    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise RuntimeError(f"SM120 CUDA device is required, got capability {capability}")

    valid_m = args.groups * args.m
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    cutlass_dtype = cutlass.Float16 if args.dtype == "float16" else cutlass.BFloat16
    load_bytes, store_bytes, flops = get_m_grouped_gemm_masked_metrics(
        groups=args.groups,
        max_m=args.max_m,
        m=args.m,
        n=args.n,
        k=args.k,
        dtype=cutlass_dtype,
    )
    init_op = {
        "empty": torch.empty,
        "zero": torch.zeros,
        "randn": torch.randn,
    }[args.init]
    torch.manual_seed(args.seed)

    with torch.cuda.device(device):
        a = make_torch_tensor(
            (args.groups, args.max_m, args.k),
            dtype=dtype,
            device=device,
            init_op=init_op,
        )
        b = make_torch_tensor(
            (args.groups, args.n, args.k),
            dtype=dtype,
            device=device,
            init_op=init_op,
        )
        token_info = torch.full((args.groups,), args.m, dtype=torch.int32, device=device)
        out = torch.empty((args.groups, args.max_m, args.n), dtype=dtype, device=device)

        result = m_grouped_masked_gemm_torch_sm120(
            a,
            b,
            token_info,
            out=out,
            block_swizzle_factor=args.block_swizzle_factor,
        )
        torch.cuda.synchronize(device)

        def run_grouped_gemm() -> torch.Tensor:
            return m_grouped_masked_gemm_torch_sm120(
                a,
                b,
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
        "SM120 M-grouped masked GEMM benchmark: "
        f"dtype={args.dtype}, init={args.init}, G={args.groups}, Mmax={args.max_m}, "
        f"M={args.m}, N={args.n}, K={args.k}, valid_M={valid_m}, "
        f"block_swizzle_factor={args.block_swizzle_factor}"
    )
    print(
        f"device={torch.cuda.get_device_name(device)}, warmup={args.warmup_iterations}, "
        f"iterations={args.iterations}, avg_time_us={avg_time_us:.3f}, "
        f"load_bytes={load_bytes}, load_tbps={load_tbps:.3f}, "
        f"store_bytes={store_bytes}, store_tbps={store_tbps:.3f}, "
        f"flops={flops}, matmul_tflops={matmul_tflops:.3f}, D={tuple(result.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
