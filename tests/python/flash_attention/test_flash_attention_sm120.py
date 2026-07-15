import pytest
import torch
from flash_attention_ref import flash_attention_ref

from scinthil.flash_attention.sm120 import flash_attention_bshd_torch_sm120

pytestmark = [pytest.mark.gpu, pytest.mark.sm120]


@pytest.mark.parametrize("torch_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("seqlen_q", [64, 123, 456])
@pytest.mark.parametrize("seqlen_kv", [64, 123, 456])
@pytest.mark.parametrize("head", [(1, 1), (4, 2)])
@pytest.mark.parametrize("headdim", [(64, 64), (128, 128)])
def test_flash_attention_sm120(
    torch_dtype,
    is_causal: bool,
    seqlen_q: int,
    seqlen_kv: int,
    head: tuple[int, int],
    headdim: tuple[int, int],
) -> None:
    batch = 1
    head_q, head_kv = head
    headdim_qk, headdim_v = headdim
    q = torch.randn((batch, seqlen_q, head_q, headdim_qk), dtype=torch_dtype, device="cuda")
    k = torch.randn((batch, seqlen_kv, head_kv, headdim_qk), dtype=torch_dtype, device="cuda")
    v = torch.randn((batch, seqlen_kv, head_kv, headdim_v), dtype=torch_dtype, device="cuda")
    softmax_scale = headdim_qk**-0.5

    out_actual, lse_actual = flash_attention_bshd_torch_sm120(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
    )
    out_expected, lse_expected = flash_attention_ref(q, k, v, softmax_scale, is_causal)

    atol = rtol = 5.0e-2 if torch_dtype == torch.bfloat16 else 2.0e-2
    torch.testing.assert_close(out_actual.float(), out_expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse_actual, lse_expected, atol=5.0e-3, rtol=5.0e-3)
