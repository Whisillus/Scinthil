import argparse

from scinthil.env import get_cutedsl_dump_dir, get_cutedsl_env  # noqa: E402

cutedsl_env = get_cutedsl_env()
cutedsl_dump_dir = get_cutedsl_dump_dir("grouped_gemm", cutedsl_env["CUTE_DSL_ARCH"])

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402

from scinthil.grouped_gemm.sm120 import m_grouped_masked_gemm_sm120  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile SM120 M-grouped masked GEMM artifacts")
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--max-m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--block-swizzle-factor", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.groups, args.max_m, args.n, args.k) <= 0:
        raise ValueError("groups, max-m, N, and K must be positive")
    if args.k % 16 != 0:
        raise ValueError("K must be divisible by 16")
    if args.block_swizzle_factor <= 0:
        raise ValueError("block-swizzle-factor must be positive")

    dtype = cutlass.Float16 if args.dtype == "float16" else cutlass.BFloat16
    make_tensor = cute.runtime.make_fake_compact_tensor
    tensor_options = {
        "stride_order": (2, 1, 0),
        "assumed_align": 16,
    }
    mA = make_tensor(dtype, (args.groups, args.max_m, args.k), **tensor_options)
    mB = make_tensor(dtype, (args.groups, args.n, args.k), **tensor_options)
    mD = make_tensor(dtype, (args.groups, args.max_m, args.n), **tensor_options)
    mInfo = make_tensor(
        cutlass.Int32,
        (args.groups,),
        stride_order=(0,),
        assumed_align=16,
    )

    compile_options = (
        cute.GPUArch(cutedsl_env["CUTE_DSL_ARCH"]),
        cute.EnableTVMFFI,
    )
    cute.compile[compile_options](
        m_grouped_masked_gemm_sm120,
        mA,
        mB,
        mD,
        mInfo,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        True,
        args.block_swizzle_factor,
    )

    print(
        "SM120 M-grouped masked GEMM artifacts compiled: "
        f"dtype={args.dtype}, G={args.groups}, Mmax={args.max_m}, N={args.n}, K={args.k}, "
        f"block_swizzle_factor={args.block_swizzle_factor}"
    )
    print(f"CuTe DSL artifacts: {cutedsl_dump_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
