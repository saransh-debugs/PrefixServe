"""Paged attention backends. Both read and write the same KV pool layout.

Pool layout per layer (FlashInfer "NHD"):
    [num_blocks, 2 (K/V), 16 (page_size), num_kv_heads, head_dim]

A PrefixServe block id IS the page index. No translation table.

Every step is described by the same 4 arrays FlashInfer uses:
    qo_indptr         [B+1]  where each request's new tokens start in the flat batch
    kv_indptr         [B+1]  where each request's pages start in kv_indices
    kv_indices        [P]    all block tables, flattened
    kv_last_page_len  [B]    how many slots of the last page are full AFTER this step

Request i's KV length is (pages - 1) * 16 + last_page_len. Its n new tokens
sit at the END of that, so they go to positions [kv_len - n, kv_len).
Causal attention for new token j sees keys [0, kv_len - n + j].

Two backends:
- FlashInferBackend: production, bf16/fp16 only.
- TorchRefAttention: plain PyTorch, any dtype incl. fp32. Used for the exact
  correctness tier. FlashInfer has no fp32 kernels, so "exact match vs HF in
  fp32" can't run through it. The ref backend derives everything from the
  same 4 arrays, so a bad block table or last_page_len breaks both the same
  way, and the exact tier still catches it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch

from prefixserve._core import BLOCK_SIZE


@dataclass(frozen=True, slots=True)
class AttnMetadata:
    num_tokens: int
    num_seqs: int
    # On the compute device. int32.
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    # Same arrays on the CPU. plan() needs them on the host. Passing the
    # device copies would make it copy them back: a GPU->CPU sync per step.
    qo_indptr_host: torch.Tensor
    kv_indptr_host: torch.Tensor
    kv_indices_host: torch.Tensor
    kv_last_page_len_host: torch.Tensor


class AttentionBackend(Protocol):
    def plan(self, meta: AttnMetadata) -> None:
        """Once per step, before the layer loop."""
        ...

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_layer: torch.Tensor) -> torch.Tensor:
        """Once per layer. Writes k/v ([T, kv_heads, D]) into kv_layer, then
        returns attention output for q ([T, heads, D])."""
        ...


class TorchRefAttention:
    """Slow, obvious, any dtype. Mirrors HF eager attention math."""

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
        self._group = num_heads // num_kv_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._scale = 1.0 / math.sqrt(head_dim)
        self._seqs: list[tuple[int, int, torch.Tensor, int]] = []
        self._write_pages: torch.Tensor | None = None
        self._write_slots: torch.Tensor | None = None

    def plan(self, meta: AttnMetadata) -> None:
        qo = meta.qo_indptr_host.numpy()
        kvp = meta.kv_indptr_host.numpy()
        idx = meta.kv_indices_host.numpy()
        last = meta.kv_last_page_len_host.numpy()
        device = meta.kv_indices.device
        seqs: list[tuple[int, int, torch.Tensor, int]] = []
        wp: list[npt.NDArray[np.int32]] = []
        ws: list[npt.NDArray[np.int64]] = []
        for i in range(meta.num_seqs):
            n = int(qo[i + 1] - qo[i])
            pages = idx[kvp[i] : kvp[i + 1]]
            kv_len = (len(pages) - 1) * BLOCK_SIZE + int(last[i])
            if not 1 <= last[i] <= BLOCK_SIZE or kv_len < n:
                raise ValueError(f"seq {i}: bad paging metadata (kv_len {kv_len}, new {n})")
            pos = np.arange(kv_len - n, kv_len)
            wp.append(pages[pos // BLOCK_SIZE])
            ws.append(pos % BLOCK_SIZE)
            seqs.append((int(qo[i]), int(qo[i + 1]), torch.from_numpy(pages.astype(np.int64)).to(device), kv_len))
        self._seqs = seqs
        self._write_pages = torch.from_numpy(np.concatenate(wp).astype(np.int64)).to(device)
        self._write_slots = torch.from_numpy(np.concatenate(ws).astype(np.int64)).to(device)

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_layer: torch.Tensor) -> torch.Tensor:
        assert self._write_pages is not None and self._write_slots is not None, "plan() first"
        kv_layer[self._write_pages, 0, self._write_slots] = k
        kv_layer[self._write_pages, 1, self._write_slots] = v
        out = torch.empty_like(q)
        hkv, d = self._num_kv_heads, self._head_dim
        for qs, qe, pages, kv_len in self._seqs:
            n = qe - qs
            # [pages, 16, Hkv, D] -> [kv_len, Hkv, D] -> GQA: kv head h // group
            keys = kv_layer[pages, 0].reshape(-1, hkv, d)[:kv_len].repeat_interleave(self._group, dim=1)
            vals = kv_layer[pages, 1].reshape(-1, hkv, d)[:kv_len].repeat_interleave(self._group, dim=1)
            qh = q[qs:qe].transpose(0, 1)  # [H, n, D]
            scores = torch.matmul(qh, keys.permute(1, 2, 0)) * self._scale  # [H, n, kv_len]
            q_pos = torch.arange(kv_len - n, kv_len, device=q.device)[:, None]
            k_pos = torch.arange(kv_len, device=q.device)[None, :]
            scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out[qs:qe] = torch.matmul(probs, vals.transpose(0, 1)).transpose(0, 1)
        return out


class FlashInferBackend:
    """One BatchPrefillWithPagedKVCacheWrapper for the whole mixed batch.

    A decode is just a prefill with qo_len = 1, so prefill chunks and decodes
    go through one kernel per layer. That matches the scheduler, which also
    treats them as one thing.

    plan() does the CPU-side work (splitting the batch into tiles) once per
    step. run() per layer only launches kernels. Calling plan() per layer
    would redo that CPU work 32 times a step.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        workspace_bytes: int = 256 << 20,
    ) -> None:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"FlashInfer needs fp16/bf16, got {dtype}. Use TorchRefAttention for fp32.")
        try:
            import flashinfer  # type: ignore[import-not-found,unused-ignore]
        except ImportError as e:
            raise RuntimeError("flashinfer is not installed; use attention='torch'") from e
        self._fi = flashinfer
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._dtype = dtype
        self._workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        self._prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self._workspace, kv_layout="NHD")
        self._meta: AttnMetadata | None = None
        self._batch_indices: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None

    def plan(self, meta: AttnMetadata) -> None:
        fi = self._fi
        self._meta = meta
        self._prefill.plan(
            meta.qo_indptr_host,
            meta.kv_indptr_host,
            meta.kv_indices,
            meta.kv_last_page_len_host,
            self._num_heads,
            self._num_kv_heads,
            self._head_dim,
            BLOCK_SIZE,
            causal=True,
            q_data_type=self._dtype,
            kv_data_type=self._dtype,
        )
        # Where each new token lands in the paged cache. Same for every
        # layer, so computed once here instead of 32 times.
        seq_lens = fi.get_seq_lens(meta.kv_indptr, meta.kv_last_page_len, BLOCK_SIZE)
        self._batch_indices, self._positions = fi.get_batch_indices_positions(
            meta.qo_indptr, seq_lens, meta.num_tokens
        )

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_layer: torch.Tensor) -> torch.Tensor:
        m = self._meta
        assert m is not None and self._batch_indices is not None and self._positions is not None, "plan() first"
        self._fi.append_paged_kv_cache(
            k,
            v.contiguous(),  # v is a slice of the fused qkv output
            self._batch_indices,
            self._positions,
            kv_layer,
            m.kv_indices,
            m.kv_indptr,
            m.kv_last_page_len,
            kv_layout="NHD",
        )
        out: torch.Tensor = self._prefill.run(q, kv_layer)
        return out
