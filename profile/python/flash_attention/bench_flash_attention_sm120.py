import argparse

import cutlass
import torch

from scinthil.flash_attention.sm120 import flash_attention_bshd_torch_sm120
from scinthil.testing import bench_cute_kernel, get_flash_attention_metrics, make_torch_tensor
from scinthil.utils import TensorLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SM120 FlashAttention")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seqlen-q", type=int, default=4096)
    parser.add_argument("--seqlen-kv", type=int, default=4096)
    parser.add_argument("--head-q", type=int, default=32)
    parser.add_argument("--head-kv", type=int, default=8)
    parser.add_argument("--headdim-qk", type=int, default=128)
    parser.add_argument("--headdim-v", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--init", choices=("randn", "zero", "empty"), default="randn")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if min(args.batch_size, args.seqlen_q, args.seqlen_kv, args.head_q, args.head_kv) <= 0:
        raise ValueError("batch size, sequence lengths, and head counts must be positive")
    if args.head_q % args.head_kv != 0:
        raise ValueError("head-q must be divisible by head-kv")
    if args.headdim_qk % 64 != 0 or args.headdim_v % 64 != 0:
        raise ValueError("QK and V head dimensions must be divisible by 64")
    if args.device < 0:
        raise ValueError("device must be non-negative")
    if args.warmup_iterations < 0:
        raise ValueError("warmup-iterations must be non-negative")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark SM120 FlashAttention")
    if args.device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.device} is not available")

    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise RuntimeError(f"SM120 CUDA device is required, got capability {capability}")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    init_op = {
        "empty": torch.empty,
        "zero": torch.zeros,
        "randn": torch.randn,
    }[args.init]
    torch.manual_seed(args.seed)

    with torch.cuda.device(device):
        q = make_torch_tensor(
            (args.batch_size, args.seqlen_q, args.head_q, args.headdim_qk),
            dtype=dtype,
            device=device,
            init_op=init_op,
        )
        k = make_torch_tensor(
            (args.batch_size, args.seqlen_kv, args.head_kv, args.headdim_qk),
            dtype=dtype,
            device=device,
            init_op=init_op,
        )
        v = make_torch_tensor(
            (args.batch_size, args.seqlen_kv, args.head_kv, args.headdim_v),
            dtype=dtype,
            device=device,
            init_op=init_op,
        )
        load_bytes, store_bytes, flops = get_flash_attention_metrics(
            q_layout=TensorLayout(tuple(q.shape), tuple(q.stride())),
            k_layout=TensorLayout(tuple(k.shape), tuple(k.stride())),
            v_layout=TensorLayout(tuple(v.shape), tuple(v.stride())),
            dtype=cutlass.Float16 if args.dtype == "float16" else cutlass.BFloat16,
            is_causal=args.causal,
        )

        out, lse = flash_attention_bshd_torch_sm120(q, k, v, is_causal=args.causal)
        torch.cuda.synchronize(device)

        def run_flash_attention() -> tuple[torch.Tensor, torch.Tensor]:
            return flash_attention_bshd_torch_sm120(q, k, v, is_causal=args.causal)

        avg_time_us, result = bench_cute_kernel(
            run_flash_attention,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            device=device,
        )
        out, lse = result

    load_tbps = load_bytes / (avg_time_us * 1.0e6)
    store_tbps = store_bytes / (avg_time_us * 1.0e6)
    matmul_tflops = flops / (avg_time_us * 1.0e6)

    print(
        "SM120 FlashAttention benchmark: "
        f"dtype={args.dtype}, causal={args.causal}, init={args.init}, "
        f"B={args.batch_size}, Sq={args.seqlen_q}, Sk={args.seqlen_kv}, "
        f"Hq={args.head_q}, Hkv={args.head_kv}, Dqk={args.headdim_qk}, Dv={args.headdim_v}"
    )
    print(
        f"device={torch.cuda.get_device_name(device)}, warmup={args.warmup_iterations}, "
        f"iterations={args.iterations}, avg_time_us={avg_time_us:.3f}, "
        f"load_bytes={load_bytes}, load_tbps={load_tbps:.3f}, "
        f"store_bytes={store_bytes}, store_tbps={store_tbps:.3f}, "
        f"flops={flops}, matmul_tflops={matmul_tflops:.3f}, "
        f"O={tuple(out.shape)}, LSE={tuple(lse.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
