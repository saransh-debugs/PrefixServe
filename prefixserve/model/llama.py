"""Llama forward pass over a flat, ragged batch. No padding, no batch dim.

Every tensor is [num_tokens, ...]: all requests' new tokens back to back.
Mixing prefill chunks and decodes in one batch is free this way. Padding to
[batch, max_len] would waste compute on pad tokens, and with one 512-token
prefill next to 63 decodes, almost all of it would be padding.

Math follows HF LlamaForCausalLM op for op (RMSNorm upcast, rotate_half
RoPE, llama3 rope scaling), so fp32 outputs match HF to ~1e-6.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from prefixserve.model.attention import AttentionBackend
from prefixserve.model.config import LlamaConfig
from prefixserve.model.weights import LlamaWeights


def rope_inv_freq(cfg: LlamaConfig) -> torch.Tensor:
    """Per-pair rotation frequencies, incl. Llama-3.1's "llama3" scaling.

    Why scaling matters: 3.1 was trained with it. Without it, outputs look
    fine for short prompts and quietly degrade as positions grow, because
    the low-frequency pairs rotate 8x too fast.
    """
    dim = cfg.head_dim
    inv_freq: torch.Tensor = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    rs = cfg.rope_scaling
    if rs is None:
        return inv_freq
    # Three bands by wavelength: long -> divided by factor, short -> unchanged,
    # middle -> blend of the two. Same formula and op order as HF.
    old_ctx = rs.original_max_position_embeddings
    low_freq_wavelen = old_ctx / rs.low_freq_factor
    high_freq_wavelen = old_ctx / rs.high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / rs.factor, inv_freq)
    smooth = (old_ctx / wavelen - rs.low_freq_factor) / (rs.high_freq_factor - rs.low_freq_factor)
    smoothed = (1 - smooth) * scaled / rs.factor + smooth * scaled
    is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


def rope_cache(cfg: LlamaConfig, max_seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for every position up to max_seq_len: [max_seq_len, head_dim].

    Built once. Per step we only gather rows by position. Sized to the
    engine's max_seq_len, not the model's 131k: at 131k this table alone is
    ~130 MB we'd never touch.
    """
    inv_freq = rope_inv_freq(cfg)
    pos = torch.arange(max_seq_len, dtype=torch.int64).float()
    freqs = torch.outer(pos, inv_freq)  # fp32, like HF (which disables autocast here)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(device=device, dtype=dtype), emb.sin().to(device=device, dtype=dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Upcast for the mean of squares. In bf16, summing 4096 squares loses
    # enough precision to shift every downstream logit.
    dtype = x.dtype
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, heads, D]. cos/sin: [T, 1, D]. HF's rotate_half layout: the HF
    checkpoints' q/k weights are permuted for it, so interleaved RoPE would
    load fine and be wrong."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class LlamaModel:
    def __init__(self, cfg: LlamaConfig, weights: LlamaWeights, max_seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.cfg = cfg
        self.w = weights
        self.max_seq_len = max_seq_len
        self.cos, self.sin = rope_cache(cfg, max_seq_len, device, dtype)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,  # [T] int32
        positions: torch.Tensor,  # [T] int32, ABSOLUTE positions (start at num_computed, not 0)
        kv: torch.Tensor,  # [layers, blocks, 2, 16, kv_heads, D]
        attn: AttentionBackend,
        logits_idx: torch.Tensor,  # [S] rows to compute logits for
    ) -> torch.Tensor:
        """Returns fp32 logits [S, vocab], only for the rows in logits_idx.

        Why only those rows: a 512-token prefill chunk needs 1 row of logits
        (its last token), or 0 if the prompt isn't done yet. All 512 would be
        512 x 128k vocab x 4 B = 263 MB and a matmul as big as a whole layer.
        """
        cfg, w = self.cfg, self.w
        t = input_ids.shape[0]
        hq, hkv, d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        eps = cfg.rms_norm_eps

        cos = self.cos.index_select(0, positions)[:, None, :]
        sin = self.sin.index_select(0, positions)[:, None, :]
        h = F.embedding(input_ids, w.embed)

        for i, lw in enumerate(w.layers):
            x = rms_norm(h, lw.input_norm, eps)
            q, k, v = torch.split(F.linear(x, lw.qkv), [cfg.q_size, cfg.kv_size, cfg.kv_size], dim=-1)
            q = apply_rope(q.view(t, hq, d), cos, sin)
            k = apply_rope(k.view(t, hkv, d), cos, sin)
            o = attn.run(q, k, v.view(t, hkv, d), kv[i])
            h = h + F.linear(o.reshape(t, cfg.q_size), lw.o)

            x = rms_norm(h, lw.post_norm, eps)
            gate, up = F.linear(x, lw.gate_up).chunk(2, dim=-1)
            h = h + F.linear(F.silu(gate) * up, lw.down)

        # RMSNorm is per row, so selecting first gives the same result for
        # those rows and skips normalizing the other T - S.
        h = rms_norm(h.index_select(0, logits_idx), w.norm, eps)
        return F.linear(h, w.lm_head).float()
