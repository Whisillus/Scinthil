import argparse

from scinthil.env import get_cutedsl_dump_dir, get_cutedsl_env  # noqa: E402

cutedsl_env = get_cutedsl_env()
cutedsl_dump_dir = get_cutedsl_dump_dir("flash_attention", cutedsl_env["CUTE_DSL_ARCH"])

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402

from scinthil.flash_attention.sm120.fa_api_sm120 import flash_attention_bshd_sm120  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile SM120 FlashAttention artifacts")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seqlen-q", type=int, default=4096)
    parser.add_argument("--seqlen-kv", type=int, default=4096)
    parser.add_argument("--head-q", type=int, default=32)
    parser.add_argument("--head-kv", type=int, default=8)
    parser.add_argument("--headdim-qk", type=int, default=128)
    parser.add_argument("--headdim-v", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--causal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dtype = cutlass.Float16 if args.dtype == "float16" else cutlass.BFloat16
    make_tensor = cute.runtime.make_fake_compact_tensor
    tensor_options = {
        "stride_order": (3, 2, 1, 0),
        "assumed_align": 16,
    }
    mQ = make_tensor(
        dtype,
        (args.batch_size, args.seqlen_q, args.head_q, args.headdim_qk),
        **tensor_options,
    )
    mK = make_tensor(
        dtype,
        (args.batch_size, args.seqlen_kv, args.head_kv, args.headdim_qk),
        **tensor_options,
    )
    mV = make_tensor(
        dtype,
        (args.batch_size, args.seqlen_kv, args.head_kv, args.headdim_v),
        **tensor_options,
    )
    mO = make_tensor(
        dtype,
        (args.batch_size, args.seqlen_q, args.head_q, args.headdim_v),
        **tensor_options,
    )
    mLSE = make_tensor(
        cutlass.Float32,
        (args.batch_size, args.head_q, args.seqlen_q),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )

    compile_options = (
        cute.GPUArch(cutedsl_env["CUTE_DSL_ARCH"]),
        cute.EnableTVMFFI,
    )
    cute.compile[compile_options](
        flash_attention_bshd_sm120,
        mQ,
        mK,
        mV,
        mO,
        mLSE,
        cutlass.Int32(args.seqlen_q),
        cutlass.Int32(args.seqlen_kv),
        cutlass.Float32(args.headdim_qk**-0.5),
        args.causal,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )

    print(
        "SM120 FlashAttention artifacts compiled: "
        f"dtype={args.dtype}, causal={args.causal}, "
        f"B={args.batch_size}, Sq={args.seqlen_q}, Sk={args.seqlen_kv}, "
        f"Hq={args.head_q}, Hkv={args.head_kv}, Dqk={args.headdim_qk}, Dv={args.headdim_v}"
    )
    print(f"CuTe DSL artifacts: {cutedsl_dump_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
