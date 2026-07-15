from dataclasses import FrozenInstanceError

import pytest

from scinthil.utils import TensorLayout


@pytest.mark.parametrize(
    ("shape", "stride", "expected_rank", "expected_cosize"),
    [
        ((2, 3, 4), (12, 4, 1), 3, 24),
        ((2, 3, 4), (20, 5, 1), 3, 34),
        ((2, 3, 4), (0, 4, 1), 3, 12),
        ((2, 3), (0, 0), 2, 1),
    ],
)
def test_tensor_layout_properties(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    expected_rank: int,
    expected_cosize: int,
) -> None:
    layout = TensorLayout(shape, stride)

    assert layout.rank == expected_rank
    assert layout.cosize == expected_cosize


@pytest.mark.parametrize(
    ("shape", "stride"),
    [
        ((), ()),
        ((2, 3), (3,)),
        ((2, 0), (3, 1)),
        ((2, -1), (3, 1)),
        ((2, 3), (3, -1)),
    ],
)
def test_tensor_layout_rejects_invalid_metadata(shape: tuple[int, ...], stride: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        TensorLayout(shape, stride)


def test_tensor_layout_is_immutable() -> None:
    layout = TensorLayout((2, 3), (3, 1))

    with pytest.raises(FrozenInstanceError):
        layout.shape = (6,)
