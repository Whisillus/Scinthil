def flash_attention_bshd_sm120(
    q,
    k,
    v,
    is_causal=False,
    out=None,
    LSE=None,
):
    """Run SM120 FlashAttention on BSHD tensors.

    Shapes:
        q: [batch, seqlen_q, head_q, headdim_qk]
        k: [batch, seqlen_kv, head_kv, headdim_qk]
        v: [batch, seqlen_kv, head_kv, headdim_v]
        out: [batch, seqlen_q, head_q, headdim_v]
        LSE: [batch, head_q, seqlen_q], FP32

    The optional `out` and `LSE` arguments provide caller-owned output buffers.
    """
    raise NotImplementedError("SM120 FlashAttention API is not connected to the kernel yet")
