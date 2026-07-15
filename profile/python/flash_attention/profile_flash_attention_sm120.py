import argparse

import torch

from scinthil.flash_attention.sm120 import flash_attention_bshd_torch_sm120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the SM120 FlashAttention profile once")
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

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    q = torch.empty(
        (args.batch_size, args.seqlen_q, args.head_q, args.headdim_qk),
        dtype=dtype,
        device="cuda",
    )
    k = torch.empty(
        (args.batch_size, args.seqlen_kv, args.head_kv, args.headdim_qk),
        dtype=dtype,
        device="cuda",
    )
    v = torch.empty(
        (args.batch_size, args.seqlen_kv, args.head_kv, args.headdim_v),
        dtype=dtype,
        device="cuda",
    )

    out, lse = flash_attention_bshd_torch_sm120(q, k, v, is_causal=args.causal)
    torch.cuda.synchronize()

    print(
        "SM120 FlashAttention profile launched once: "
        f"dtype={args.dtype}, causal={args.causal}, "
        f"B={args.batch_size}, Sq={args.seqlen_q}, Sk={args.seqlen_kv}, "
        f"Hq={args.head_q}, Hkv={args.head_kv}, Dqk={args.headdim_qk}, Dv={args.headdim_v}, "
        f"O={tuple(out.shape)}, LSE={tuple(lse.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
