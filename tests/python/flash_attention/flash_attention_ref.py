import torch


def flash_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    k = k.repeat_interleave(q_heads // kv_heads, dim=2)
    v = v.repeat_interleave(q_heads // kv_heads, dim=2)

    q_float = q.float().permute(0, 2, 1, 3)
    k_float = k.float().permute(0, 2, 1, 3)
    v_float = v.float().permute(0, 2, 1, 3)
    scores = torch.matmul(q_float, k_float.transpose(-2, -1)) * softmax_scale

    if is_causal:
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        row = torch.arange(seqlen_q, device=q.device)[:, None]
        col = torch.arange(seqlen_kv, device=q.device)[None, :]
        valid = col <= row + seqlen_kv - seqlen_q
        scores = scores.masked_fill(~valid[None, None, :, :], float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.nan_to_num(torch.softmax(scores, dim=-1))
    out = torch.matmul(probabilities, v_float).permute(0, 2, 1, 3)
    return out, lse
