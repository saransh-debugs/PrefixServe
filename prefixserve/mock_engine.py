"""Stand-in for the model. Says how long a batch would take and makes up tokens.

This does NOT measure a GPU. It gives the scheduler a believable cost per
step so its choices (batching, chunking, cache reuse) show up in timing.

Why one cost per batch step, not 50ms per token per request: a GPU runs the
whole batch in one forward pass. If every request paid its own 50ms, batching
would give no speedup and the simulation would show nothing.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from prefixserve.scheduler import Batch
from prefixserve.tokenizer import BYTE_OFFSET, EOS_ID

_TEXT = (
    b"PrefixServe is a mock engine. These tokens are made up, but the timing "
    b"follows the batch: one step per forward pass, plus a cost per prefill token. "
)


@dataclass(frozen=True, slots=True)
class CostModel:
    """Seconds for one batch step = base + per prefill token + per decode seq.

    The numbers are made up. They're roughly the shape of a 7B model on one
    GPU: decode is cheap per request, prefill costs per token, and every step
    has a fixed overhead.
    """

    base_s: float = 0.020
    per_prefill_token_s: float = 0.0001
    per_decode_seq_s: float = 0.0002

    def step_seconds(self, prefill_tokens: int, decode_seqs: int) -> float:
        return self.base_s + prefill_tokens * self.per_prefill_token_s + decode_seqs * self.per_decode_seq_s

    def batch_seconds(self, batch: Batch) -> float:
        prefill = sum(w.end - w.start for w in batch.work if w.end - w.start > 1)
        decode = sum(1 for w in batch.work if w.end - w.start == 1)
        return self.step_seconds(prefill, decode)


class MockEngine:
    # No real KV pool and no length limit: the server's config decides.
    num_blocks: int | None = None
    max_seq_len: int | None = None
    stop_token_ids: frozenset[int] = frozenset({EOS_ID})

    def __init__(self, cost: CostModel | None = None, *, eos_prob: float = 0.0, seed: int = 0) -> None:
        if not 0.0 <= eos_prob <= 1.0:
            raise ValueError("eos_prob must be in [0, 1]")
        self.cost = cost or CostModel()
        self._eos_prob = eos_prob
        self._rng = random.Random(seed)

    def step_seconds(self, batch: Batch) -> float:
        return self.cost.batch_seconds(batch)

    def step(self, batch: Batch) -> dict[str, int]:
        """Blocking, like a real GPU step. The server runs it in a worker
        thread, so this sleep never stalls the event loop."""
        time.sleep(self.step_seconds(batch))
        return self.sample(batch)

    def sample(self, batch: Batch) -> dict[str, int]:
        """One token for every work item that reached the end of its sequence."""
        out: dict[str, int] = {}
        for w in batch.work:
            if not w.samples:
                continue
            rid = w.req.req_id
            if self._eos_prob and self._rng.random() < self._eos_prob:
                out[rid] = EOS_ID
                continue
            # Readable text: walk through _TEXT, one byte per token.
            i = len(w.req.output) % len(_TEXT)
            out[rid] = _TEXT[i] + BYTE_OFFSET
        return out
