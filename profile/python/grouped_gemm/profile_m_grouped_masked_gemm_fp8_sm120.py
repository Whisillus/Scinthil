import argparse

import cutlass.cute as cute
import torch

from scinthil.grouped_gemm.sm120 import m_grouped_masked_gemm_fp8_torch_sm120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch SM120 FP8 M-grouped masked GEMM once")
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--max-m", type=int, default=4096)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--block-swizzle-factor", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.groups, args.max_m, args.m, args.n, args.k) <= 0:
        raise ValueError("groups, max-m, M, N, and K must be positive")
    if args.m > args.max_m:
        raise ValueError("M must not exceed max-m")
    if args.k % 128 != 0:
        raise ValueError("K must be divisible by 128")
    if args.block_swizzle_factor <= 0:
        raise ValueError("block-swizzle-factor must be positive")
    if args.device < 0:
        raise ValueError("device must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to profile SM120 FP8 grouped GEMM")
    if args.device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.device} is not available")

    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise RuntimeError(f"SM120 CUDA device is required, got capability {capability}")

    with torch.cuda.device(device):
        a = torch.empty((args.groups, args.max_m, args.k), dtype=torch.float8_e4m3fn, device=device)
        scale_a = torch.ones((args.groups, args.max_m, args.k // 128), dtype=torch.float32, device=device)
        b = torch.empty((args.groups, args.n, args.k), dtype=torch.float8_e4m3fn, device=device)
        scale_b = torch.ones(
            (args.groups, cute.ceil_div(args.n, 128), args.k // 128),
            dtype=torch.float32,
            device=device,
        )
        token_info = torch.full((args.groups,), args.m, dtype=torch.int32, device=device)
        out = torch.empty((args.groups, args.max_m, args.n), dtype=torch.bfloat16, device=device)
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

    print(
        "SM120 FP8 M-grouped masked GEMM profile launched once: "
        f"G={args.groups}, Mmax={args.max_m}, M={args.m}, N={args.n}, K={args.k}, "
        f"valid_M={args.groups * args.m}, block_swizzle_factor={args.block_swizzle_factor}, "
        f"scale_A={tuple(scale_a.shape)}, scale_B={tuple(scale_b.shape)}, D={tuple(result.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
