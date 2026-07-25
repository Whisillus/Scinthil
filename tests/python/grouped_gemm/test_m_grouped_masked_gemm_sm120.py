import random

import pytest
import torch

from scinthil.grouped_gemm.sm120 import m_grouped_masked_gemm_torch_sm120
from scinthil.testing import get_m_grouped_masked_gemm_deepgemm_case, m_grouped_masked_gemm_ref

pytestmark = [pytest.mark.gpu, pytest.mark.sm120]


@pytest.mark.parametrize(
    ("groups", "max_m", "expected_m_per_group", "n", "k"),
    get_m_grouped_masked_gemm_deepgemm_case(),
)
def test_m_grouped_masked_gemm_sm120(
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

    a = torch.randn((groups, max_m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((groups, n, k), dtype=torch.bfloat16, device="cuda")
    token_info = torch.tensor(token_counts, dtype=torch.int32, device="cuda")
    expected = m_grouped_masked_gemm_ref(a, b)

    out = torch.empty((groups, max_m, n), dtype=torch.bfloat16, device="cuda")
    actual = m_grouped_masked_gemm_torch_sm120(
        a,
        b,
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
