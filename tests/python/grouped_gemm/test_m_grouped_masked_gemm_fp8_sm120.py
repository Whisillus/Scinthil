import random

import pytest
import torch

from scinthil.grouped_gemm.sm120 import m_grouped_masked_gemm_fp8_torch_sm120
from scinthil.testing import dequantize_fp8, m_grouped_masked_gemm_ref, quantize_fp8

pytestmark = [pytest.mark.gpu, pytest.mark.sm120]


def get_m_grouped_masked_gemm_fp8_cases() -> (
    tuple[
        tuple[int, int, int, int, int],
        ...,
    ]
):
    # Mirrors DeepGEMM tests/generators.py::enumerate_m_grouped_masked.
    return (
        (32, 4096, 192, 6144, 7168),
        (32, 4096, 192, 7168, 3072),
        (32, 4096, 192, 4096, 4096),
        (32, 4096, 192, 4096, 2048),
        (6, 4096, 1024, 6144, 7168),
        (6, 4096, 1024, 7168, 3072),
        (6, 4096, 1024, 4096, 4096),
        (6, 4096, 1024, 4096, 2048),
        (32, 4096, 20, 6144, 7168),
        (32, 4096, 20, 7168, 3072),
        (32, 4096, 20, 4096, 4096),
        (32, 4096, 20, 4096, 2048),
        (6, 4096, 20, 6144, 7168),
        (6, 4096, 20, 7168, 3072),
        (6, 4096, 20, 4096, 4096),
        (6, 4096, 20, 4096, 2048),
    )


@pytest.mark.parametrize(
    ("groups", "max_m", "expected_m_per_group", "n", "k"),
    get_m_grouped_masked_gemm_fp8_cases(),
)
def test_m_grouped_masked_gemm_fp8_sm120(
    groups: int,
    max_m: int,
    expected_m_per_group: int,
    n: int,
    k: int,
) -> None:
    seed = 0
    torch.manual_seed(seed)
    rng = random.Random(seed)
    token_counts = tuple(int(expected_m_per_group * rng.uniform(0.7, 1.3)) for _ in range(groups))
    assert max(token_counts) <= max_m

    recipe_a = (1, 1, 128)
    recipe_b = (1, 128, 128)
    assert n % recipe_b[1] == 0
    assert k % recipe_a[2] == 0
    assert k % recipe_b[2] == 0
    a = torch.randn((groups, max_m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((groups, n, k), dtype=torch.bfloat16, device="cuda")
    fp8_a, scale_a = quantize_fp8(a, recipe_a)
    fp8_b, scale_b = quantize_fp8(b, recipe_b)
    dequantized_a = dequantize_fp8(fp8_a, scale_a, recipe_a, torch.bfloat16)
    dequantized_b = dequantize_fp8(fp8_b, scale_b, recipe_b, torch.bfloat16)
    token_info = torch.tensor(token_counts, dtype=torch.int32, device="cuda")
    expected = m_grouped_masked_gemm_ref(dequantized_a, dequantized_b, out_dtype=torch.bfloat16)

    out = torch.empty((groups, max_m, n), dtype=torch.bfloat16, device="cuda")
    actual = m_grouped_masked_gemm_fp8_torch_sm120(
        fp8_a,
        scale_a,
        fp8_b,
        scale_b,
        token_info,
        out=out,
    )

    assert actual is out
    atol = rtol = 1.0e-3
    for group_idx, valid_m in enumerate(token_counts):
        torch.testing.assert_close(
            actual[group_idx, :valid_m].float(),
            expected[group_idx, :valid_m].float(),
            atol=atol,
            rtol=rtol,
        )
