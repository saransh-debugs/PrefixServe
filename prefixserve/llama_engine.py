"""Real Llama engine: runs a scheduler Batch on the GPU, reading paged KV in place.

Per step:
    1. Build inputs on the CPU from batch.work, in one pinned buffer.
    2. One async host->device copy.
    3. attn.plan() once, then the forward pass (attn.run() once per layer).
    4. Greedy argmax, then ONE .tolist(): the step's only GPU->CPU sync.

The KV pool is allocated once at startup and never grows. Block id from the
C++ BlockManager == page index into the pool.

Threading: step() is called from one worker thread while the event loop
thread waits on it. The scheduler doesn't touch running requests between
schedule_step() and commit() (aborts are deferred to commit), so reading
batch.work here doesn't race.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch

from prefixserve._core import BLOCK_SIZE
from prefixserve.model.attention import AttentionBackend, AttnMetadata, FlashInferBackend, TorchRefAttention
from prefixserve.model.config import LlamaConfig
from prefixserve.model.llama import LlamaModel
from prefixserve.model.weights import load_llama_weights
from prefixserve.scheduler import Batch, Request, WorkItem, blocks_for

logger = logging.getLogger(__name__)

InjectedBug = Literal["rope_relative", "last_page_len"]


class EngineError(RuntimeError):
    """Engine got a batch it can't run. Always a config or scheduler bug."""


@dataclass(frozen=True, slots=True)
class EngineConfig:
    model_path: Path
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    attention: Literal["flashinfer", "torch"] = "flashinfer"
    # Must match the scheduler's limits. They size the pinned input buffer.
    max_num_seqs: int = 64
    max_prefill_tokens: int = 512
    max_seq_len: int = 8192
    # None = size the pool from free GPU memory. Set it to give several
    # engines the same KV capacity in a benchmark.
    num_blocks: int | None = None
    # Left free after the KV pool: cuBLAS workspace, allocator slack, and
    # CUDA graph pools later.
    reserve_bytes: int = 1 << 30
    # Test-only. Lets the test suite prove it catches these bugs.
    inject_bug: InjectedBug | None = None


@dataclass(frozen=True, slots=True)
class StepInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    logits_idx: torch.Tensor
    attn: AttnMetadata
    sample_req_ids: list[str]


class LlamaEngine:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # Set before anything runs a step: the pool-sizing profile run does.
        self.capture_logits = False
        self.last_logits: dict[str, torch.Tensor] = {}
        self.model_cfg = LlamaConfig.from_dir(cfg.model_path)
        if cfg.max_seq_len > self.model_cfg.max_position_embeddings:
            raise ValueError(f"max_seq_len {cfg.max_seq_len} > model limit {self.model_cfg.max_position_embeddings}")
        mc = self.model_cfg

        weights = load_llama_weights(cfg.model_path, mc, self.device, cfg.dtype)
        logger.info("loaded weights: %.2f GB", weights.num_bytes() / 1e9)
        self.model = LlamaModel(mc, weights, cfg.max_seq_len, self.device, cfg.dtype)
        self.attn: AttentionBackend
        if cfg.attention == "flashinfer":
            self.attn = FlashInferBackend(mc.num_heads, mc.num_kv_heads, mc.head_dim, cfg.dtype, self.device)
        else:
            self.attn = TorchRefAttention(mc.num_heads, mc.num_kv_heads, mc.head_dim)

        # Pinned host buffer for all per-step int32 inputs. Pinned memory is
        # what lets the host->device copy be async. Pageable memory forces
        # the driver to stage through a bounce buffer, synchronously.
        self._max_tokens = cfg.max_prefill_tokens + cfg.max_num_seqs
        pages_per_seq = blocks_for(cfg.max_seq_len)
        cap = 2 * self._max_tokens + 4 * (cfg.max_num_seqs + 1) + cfg.max_num_seqs * pages_per_seq
        on_gpu = self.device.type == "cuda"
        self._host = torch.empty(cap, dtype=torch.int32, pin_memory=on_gpu)
        self._host_np = self._host.numpy()
        # Marks when the last copy out of _host finished. We must not overwrite
        # _host before that, or the GPU reads half-old, half-new inputs. Steps
        # that sample sync anyway; mid-prefill steps don't sync at all.
        self._copy_done = torch.cuda.Event() if on_gpu else None  # type: ignore[no-untyped-call]

        self.bytes_per_block = 2 * BLOCK_SIZE * mc.num_kv_heads * mc.head_dim * mc.num_layers * cfg.dtype.itemsize
        self.num_blocks = cfg.num_blocks if cfg.num_blocks is not None else self._profile_num_blocks()
        # One allocation, up front. Growing it later would need a copy of
        # every live block, and could OOM in the middle of serving.
        self.kv = torch.zeros(
            (mc.num_layers, self.num_blocks, 2, BLOCK_SIZE, mc.num_kv_heads, mc.head_dim),
            dtype=cfg.dtype, device=self.device,
        )
        logger.info(
            "KV pool: %d blocks (%d tokens, %.2f GB)",
            self.num_blocks, self.num_blocks * BLOCK_SIZE, self.num_blocks * self.bytes_per_block / 1e9,
        )

    @property
    def max_seq_len(self) -> int:
        return self.cfg.max_seq_len

    @property
    def stop_token_ids(self) -> frozenset[int]:
        return self.model_cfg.stop_token_ids

    # ------------------------------------------------------------------ step

    @torch.inference_mode()
    def step(self, batch: Batch) -> dict[str, int]:
        """Runs one batch. Returns req_id -> next token for items with w.samples."""
        if not batch.work:
            return {}
        return self._run(batch.work, self.kv)

    def _run(self, work: list[WorkItem], kv: torch.Tensor) -> dict[str, int]:
        inp = self._build_inputs(work, num_blocks=kv.shape[1])
        self.attn.plan(inp.attn)
        logits = self.model.forward(inp.input_ids, inp.positions, kv, self.attn, inp.logits_idx)
        if not inp.sample_req_ids:
            return {}  # mid-prefill only: no sync, the CPU can run ahead
        next_ids = torch.argmax(logits, dim=-1)
        if self.capture_logits:
            self.last_logits = {rid: logits[i].cpu() for i, rid in enumerate(inp.sample_req_ids)}
        tokens: list[int] = next_ids.tolist()  # the one GPU->CPU sync per step
        return dict(zip(inp.sample_req_ids, tokens, strict=True))

    def _build_inputs(self, work: list[WorkItem], num_blocks: int) -> StepInputs:
        b = len(work)
        lens = [w.end - w.start for w in work]
        pages = [blocks_for(w.end) for w in work]
        t, p = sum(lens), sum(pages)
        sample_ids = [w.req.req_id for w in work if w.samples]
        s = len(sample_ids)
        if b > self.cfg.max_num_seqs or t > self._max_tokens:
            raise EngineError(f"batch of {b} seqs / {t} tokens is over the engine limits")

        # Layout inside the one buffer. Each piece is a view, so the device
        # side is one copy, then slicing.
        o_ids, o_pos = 0, t
        o_qo = 2 * t
        o_kvp = o_qo + b + 1
        o_last = o_kvp + b + 1
        o_lidx = o_last + b
        o_idx = o_lidx + s
        total = o_idx + p
        if total > self._host_np.size:
            raise EngineError(f"step inputs need {total} ints, buffer has {self._host_np.size}")

        if self._copy_done is not None:
            self._copy_done.synchronize()  # type: ignore[no-untyped-call]
        buf = self._host_np
        ids, pos = buf[o_ids : o_ids + t], buf[o_pos : o_pos + t]
        qo, kvp = buf[o_qo : o_qo + b + 1], buf[o_kvp : o_kvp + b + 1]
        last, lidx, idx = buf[o_last : o_last + b], buf[o_lidx : o_lidx + s], buf[o_idx : o_idx + p]

        qo[0] = kvp[0] = 0
        tok = page = si = 0
        for i, w in enumerate(work):
            req, n, np_ = w.req, lens[i], pages[i]
            if w.end > self.cfg.max_seq_len:
                raise EngineError(f"{req.req_id}: position {w.end} > max_seq_len {self.cfg.max_seq_len}")
            if len(req.block_table) < np_:
                raise EngineError(f"{req.req_id}: {len(req.block_table)} blocks for {w.end} tokens")
            _fill_tokens(ids[tok : tok + n], req, w.start, w.end)
            # Absolute positions. Starting at 0 would rotate a request that
            # reused a cached prefix as if its first new token were token 0.
            first = 0 if self.cfg.inject_bug == "rope_relative" else w.start
            pos[tok : tok + n] = np.arange(first, first + n, dtype=np.int32)
            idx[page : page + np_] = req.block_table[:np_]
            last[i] = w.end - (np_ - 1) * BLOCK_SIZE
            tok += n
            page += np_
            qo[i + 1], kvp[i + 1] = tok, page
            if w.samples:
                lidx[si] = tok - 1  # last new token of this request
                si += 1
        if self.cfg.inject_bug == "last_page_len":
            last[:] = np.where(last > 1, last - 1, last + 1)
        _check_paging(lens, pages, last, idx, num_blocks)

        host = self._host[:total]
        dev = host.to(self.device, non_blocking=True)
        if self._copy_done is not None:
            self._copy_done.record()  # type: ignore[no-untyped-call]

        def h(o: int, n: int) -> torch.Tensor:
            return host[o : o + n]

        def d(o: int, n: int) -> torch.Tensor:
            return dev[o : o + n]

        meta = AttnMetadata(
            num_tokens=t, num_seqs=b,
            qo_indptr=d(o_qo, b + 1), kv_indptr=d(o_kvp, b + 1),
            kv_indices=d(o_idx, p), kv_last_page_len=d(o_last, b),
            qo_indptr_host=h(o_qo, b + 1), kv_indptr_host=h(o_kvp, b + 1),
            kv_indices_host=h(o_idx, p), kv_last_page_len_host=h(o_last, b),
        )
        return StepInputs(
            input_ids=d(o_ids, t), positions=d(o_pos, t), logits_idx=d(o_lidx, s),
            attn=meta, sample_req_ids=sample_ids,
        )

    # --------------------------------------------------------- pool sizing

    def _profile_num_blocks(self) -> int:
        """KV blocks that fit in what's left after weights AND activations.

        Sizing from free memory right after loading weights (the naive way)
        leaves nothing for activations, so the first full-size prefill OOMs
        in the middle of serving. Instead: run one worst-case step on a tiny
        throwaway pool, measure its peak, and leave that much free.
        """
        if self.device.type != "cuda":
            raise ValueError("num_blocks must be set explicitly when not on a GPU")
        cfg, mc = self.cfg, self.model_cfg
        work = _dummy_worst_case(cfg.max_prefill_tokens, cfg.max_num_seqs)
        tmp_blocks = sum(len(w.req.block_table) for w in work)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        tmp = torch.zeros((mc.num_layers, tmp_blocks, 2, BLOCK_SIZE, mc.num_kv_heads, mc.head_dim), dtype=cfg.dtype, device=self.device)
        with torch.inference_mode():
            self._run(work, tmp)
        torch.cuda.synchronize()
        activations = torch.cuda.max_memory_allocated() - base - tmp.numel() * tmp.element_size()
        del tmp
        gc.collect()
        torch.cuda.empty_cache()

        free, _total = torch.cuda.mem_get_info()
        usable = free - activations - cfg.reserve_bytes
        n = int(usable // self.bytes_per_block)
        need = blocks_for(cfg.max_seq_len)
        if n < need:
            raise EngineError(
                f"only {n} KV blocks fit ({usable / 1e9:.2f} GB usable), one max-length request needs {need}"
            )
        logger.info("profile: activations %.0f MB, free %.2f GB -> %d blocks", activations / 1e6, free / 1e9, n)
        return n


def _check_paging(
    lens: list[int], pages: list[int], last: npt.NDArray[np.int32], idx: npt.NDArray[np.int32], num_blocks: int
) -> None:
    """Checks the paging arrays on the CPU before the GPU sees them.

    FlashInfer trusts these arrays. A bad page length or block id there is
    not an error, it's an out-of-bounds write: "illegal memory access",
    which kills the CUDA context and every request in flight with it.
    (Found by the injected last_page_len bug.) Here it's a clean exception
    for one step instead. Cost: a few vectorized compares, microseconds.
    """
    if last.size and (int(last.min()) < 1 or int(last.max()) > BLOCK_SIZE):
        raise EngineError(f"kv_last_page_len out of [1, {BLOCK_SIZE}]")
    kv_len = (np.asarray(pages) - 1) * BLOCK_SIZE + last
    if (kv_len < np.asarray(lens)).any():
        raise EngineError("a request has more new tokens than KV slots")
    if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= num_blocks):
        raise EngineError(f"block id outside the KV pool (0..{num_blocks - 1})")


def _fill_tokens(out: npt.NDArray[np.int32], req: Request, start: int, end: int) -> None:
    """out[:] = (prompt + output)[start:end], without building the full seq.
    req.seq() concatenates the whole thing; per decode step that's O(len)."""
    np_ = len(req.prompt)
    if end <= np_:
        out[:] = req.prompt[start:end]
    elif start >= np_:
        out[:] = req.output[start - np_ : end - np_]
    else:
        k = np_ - start
        out[:k] = req.prompt[start:]
        out[k:] = req.output[: end - np_]


def _dummy_worst_case(max_prefill_tokens: int, max_num_seqs: int) -> list[WorkItem]:
    """One full prefill chunk + (max_num_seqs - 1) decodes, on a throwaway pool."""
    work: list[WorkItem] = []
    next_block = 0

    def req(rid: str, n: int) -> Request:
        nonlocal next_block
        nb = blocks_for(n)
        r = Request(req_id=rid, prompt=np.zeros(n, dtype=np.int32), max_tokens=1, arrival=0)
        r.block_table = list(range(next_block, next_block + nb))
        next_block += nb
        return r

    work.append(WorkItem(req("profile-prefill", max_prefill_tokens), 0, max_prefill_tokens))
    for i in range(max_num_seqs - 1):
        work.append(WorkItem(req(f"profile-decode-{i}", 1), 0, 1))
    return work
