import torch


def dequantize_fp8(
    fp8_x: torch.Tensor,
    scale: torch.Tensor,
    recipe: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize block-scaled E4M3FN data to FP16 or BF16."""
    if fp8_x.ndim == 0:
        raise ValueError("fp8_x must have at least one dimension")
    if len(recipe) != fp8_x.ndim:
        raise ValueError("fp8_x and recipe must have the same number of dimensions")
    if any(block_size <= 0 for block_size in recipe):
        raise ValueError("recipe block sizes must be positive")
    if any(dim_size <= 0 for dim_size in fp8_x.shape):
        raise ValueError("fp8_x dimensions must be positive")
    if any(dim_size % block_size != 0 for dim_size, block_size in zip(fp8_x.shape, recipe, strict=True)):
        raise ValueError("each fp8_x dimension must be divisible by its recipe block size")
    if fp8_x.dtype != torch.float8_e4m3fn:
        raise TypeError("fp8_x must have dtype torch.float8_e4m3fn")
    if scale.dtype != torch.float32:
        raise TypeError("scale must have dtype torch.float32")
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("dtype must be torch.float16 or torch.bfloat16")
    if scale.device != fp8_x.device:
        raise ValueError("fp8_x and scale must be on the same device")

    expected_scale_shape = tuple(
        dim_size // block_size for dim_size, block_size in zip(fp8_x.shape, recipe, strict=True)
    )
    if scale.shape != expected_scale_shape:
        raise ValueError(f"scale must have shape {expected_scale_shape}")

    expanded_scale = scale
    for dim, block_size in enumerate(recipe):
        expanded_scale = expanded_scale.repeat_interleave(block_size, dim=dim)
    return (fp8_x.float() * expanded_scale).to(dtype).contiguous()


def quantize_fp8(x: torch.Tensor, recipe: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to E4M3FN with one FP32 dequantization scale per recipe block."""
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if len(recipe) != x.ndim:
        raise ValueError("x and recipe must have the same number of dimensions")
    if any(block_size <= 0 for block_size in recipe):
        raise ValueError("recipe block sizes must be positive")
    if any(dim_size <= 0 for dim_size in x.shape):
        raise ValueError("x dimensions must be positive")
    if any(dim_size % block_size != 0 for dim_size, block_size in zip(x.shape, recipe, strict=True)):
        raise ValueError("each x dimension must be divisible by its recipe block size")
    if not x.is_floating_point():
        raise TypeError("x must have a floating-point dtype")

    block_counts = tuple(dim_size // block_size for dim_size, block_size in zip(x.shape, recipe, strict=True))
    blocked_shape = tuple(
        size for block_count, block_size in zip(block_counts, recipe, strict=True) for size in (block_count, block_size)
    )
    block_dims = tuple(range(1, 2 * x.ndim, 2))
    block_amax = x.float().reshape(blocked_shape).abs().amax(dim=block_dims)

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    scale = torch.where(block_amax > 0, block_amax / fp8_max, torch.ones_like(block_amax))

    scale_shape = tuple(size for block_count in block_counts for size in (block_count, 1))
    expanded_scale = scale.reshape(scale_shape).expand(blocked_shape).reshape(x.shape)
    fp8_x = (x.float() / expanded_scale).clamp(min=-fp8_max, max=fp8_max).to(fp8_dtype)
    return fp8_x.contiguous(), scale.contiguous()
