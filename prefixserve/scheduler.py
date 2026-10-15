"""Continuous batching scheduler on top of the C++ block manager and radix tree.

One idea keeps this simple. Every request has a token list
``seq = prompt + output`` and a counter ``num_computed`` (how many tokens
already have KV in memory). Each step processes some of ``seq[num_computed:]``.
When that reaches the end of ``seq``, the engine samples one new token.

- Decode = 1 token left to process.
- Prefill = more than 1 left.
- Redoing work after preemption = prefill again.

Same code path for all three, so there are fewer places for bugs.

Not thread-safe, on purpose. One asyncio task owns the scheduler.
"""

from __future__ import annotations

import bisect
import logging
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt

from prefixserve._core import BLOCK_SIZE, BlockManager, RadixTree

logger = logging.getLogger(__name__)

TokenArray = npt.NDArray[np.int32]

DEFAULT_CHUNK_SIZE = 512
SLOW_STEP_SECONDS = 1e-3


class SchedulerError(RuntimeError):
    """Caller broke the schedule -> commit protocol. Always a bug."""


class ReqState(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class FinishReason(str, Enum):
    LENGTH = "length"
    STOP = "stop"
    ABORT = "abort"


class AddResult(Enum):
    ACCEPTED = "accepted"
    QUEUE_FULL = "queue_full"      # HTTP 429
    TOO_LONG = "too_long"          # HTTP 400: can't fit even with the whole pool
    DUPLICATE_ID = "duplicate_id"  # HTTP 409
    INVALID = "invalid"            # HTTP 400


def blocks_for(num_tokens: int) -> int:
    return -(-num_tokens // BLOCK_SIZE)


@dataclass(slots=True, eq=False)
class Request:
    req_id: str
    prompt: TokenArray
    max_tokens: int
    arrival: int  # arrival order. Kept across preemption so priority is kept too.
    state: ReqState = ReqState.WAITING
    output: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)  # we hold 1 ref on each
    num_computed: int = 0
    cached_tokens: int = 0  # tokens reused from the tree, summed over admissions
    num_preemptions: int = 0
    abort_pending: bool = False
    prompt_cached: bool = False  # prompt inserted into the tree at least once

    @property
    def seq_len(self) -> int:
        return len(self.prompt) + len(self.output)

    def seq(self) -> TokenArray:
        # Builds a new array. Only called at admission / insert / finish,
        # never once per decode step.
        if not self.output:
            return self.prompt
        return np.concatenate((self.prompt, np.asarray(self.output, dtype=np.int32)))


@dataclass(frozen=True, slots=True)
class WorkItem:
    req: Request
    start: int  # process seq[start:end]
    end: int

    @property
    def samples(self) -> bool:
        """True if this item reaches the end of seq, so the engine must sample a token."""
        return self.end == self.req.seq_len


@dataclass(frozen=True, slots=True)
class Batch:
    work: list[WorkItem]
    preempted: list[str]

    @property
    def num_tokens(self) -> int:
        return sum(w.end - w.start for w in self.work)


@dataclass(frozen=True, slots=True)
class StepResult:
    new_tokens: dict[str, int]
    finished: dict[str, FinishReason]


@dataclass(frozen=True, slots=True)
class SchedulerStats:
    num_waiting: int
    num_running: int
    free_blocks: int
    cached_blocks: int
    prompt_tokens: int
    cached_prompt_tokens: int
    preemptions: int
    rejected: int

    @property
    def prefix_hit_rate(self) -> float:
        return self.cached_prompt_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


class Scheduler:
    def __init__(
        self,
        num_blocks: int,
        *,
        max_waiting: int = 1024,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        eos_token_id: int | None = None,
        enable_prefix_cache: bool = True,
    ) -> None:
        if max_waiting < 1 or chunk_size < 1:
            raise ValueError("max_waiting and chunk_size must be >= 1")
        self.block_manager = BlockManager(num_blocks)
        self.tree = RadixTree(self.block_manager)
        self._max_waiting = max_waiting
        self._chunk_size = chunk_size
        self._eos = eos_token_id
        # Off = plain paged continuous batching. Only used to measure what the
        # cache buys in the simulation.
        self._use_cache = enable_prefix_cache

        self._requests: dict[str, Request] = {}
        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []  # sorted by arrival, newest last
        self._inflight: Batch | None = None
        self._pending_aborts: list[Request] = []
        self._next_arrival = 0

        self._prompt_tokens = 0
        self._cached_prompt_tokens = 0
        self._preemptions = 0
        self._rejected = 0

    # ------------------------------------------------------------------ API

    def add_request(
        self, req_id: str, prompt: Iterable[int] | TokenArray, max_tokens: int
    ) -> AddResult:
        result = self._validate(req_id, prompt, max_tokens)
        if isinstance(result, AddResult):
            self._rejected += 1
            return result
        req = Request(req_id=req_id, prompt=result, max_tokens=max_tokens, arrival=self._next_arrival)
        self._next_arrival += 1
        self._requests[req_id] = req
        self._waiting.append(req)
        self._prompt_tokens += len(result)
        return AddResult.ACCEPTED

    def schedule_step(self, max_prefill_tokens: int, max_decode_seqs: int) -> Batch:
        """Picks this step's work. Must be followed by commit() before the next call."""
        if self._inflight is not None:
            raise SchedulerError("commit() the previous batch before scheduling another")
        # 0 would let a prefill sit forever with no tokens: a livelock.
        if max_prefill_tokens < 1 or max_decode_seqs < 1:
            raise ValueError("max_prefill_tokens and max_decode_seqs must be >= 1")

        started = time.perf_counter()
        work: list[WorkItem] = []
        preempted: list[str] = []
        budget = max_prefill_tokens

        # 1. Running requests, oldest first.
        #
        # Decodes don't use the prefill budget, so a big prefill can never
        # starve them. Oldest-first matters for memory: _reserve only ever
        # preempts requests newer than the one asking, and those come later
        # in this loop, so a victim is never already in `work`.
        i = 0
        while i < len(self._running):
            req = self._running[i]
            remaining = req.seq_len - req.num_computed
            n = 1 if remaining == 1 else min(self._chunk_size, remaining, budget)
            if n == 0:  # prefill budget used up; this prefill waits a step
                i += 1
                continue
            need = blocks_for(req.num_computed + n) - len(req.block_table)
            if need > 0 and not self._reserve(req, need, preempted):
                continue  # req preempted itself and left _running; same index is the next req
            if remaining > 1:
                budget -= n
            work.append(WorkItem(req, req.num_computed, req.num_computed + n))
            i += 1

        # 2. Admit waiting requests, FIFO.
        while self._waiting and len(self._running) < max_decode_seqs:
            item = self._admit(self._waiting[0], budget)
            if item is None:
                break  # head of line can't start; keep FIFO order, try next step
            self._waiting.popleft()
            if item.end - item.start > 1:
                budget -= item.end - item.start
            work.append(item)

        batch = Batch(work=work, preempted=preempted)
        self._inflight = batch

        elapsed = time.perf_counter() - started
        if elapsed > SLOW_STEP_SECONDS:
            logger.warning("schedule_step took %.2f ms (blocks the event loop)", elapsed * 1e3)
        return batch

    def commit(self, batch: Batch, sampled: Mapping[str, int]) -> StepResult:
        """Applies the engine's results for `batch`. `sampled` maps req_id -> new token
        for every work item where `samples` is True."""
        if batch is not self._inflight:
            raise SchedulerError("commit() got a batch that isn't the one in flight")
        self._inflight = None

        new_tokens: dict[str, int] = {}
        finished: dict[str, FinishReason] = {}

        for w in batch.work:
            req = w.req
            if req.num_computed != w.start:
                raise SchedulerError(f"{req.req_id}: batch is stale")
            reached_end = w.samples
            req.num_computed = w.end
            if not reached_end:
                continue
            if req.req_id not in sampled:
                raise SchedulerError(f"engine returned no token for {req.req_id}")
            token = sampled[req.req_id]

            # Prompt fully computed for the first time: cache it now so other
            # requests with the same prefix can reuse it while this one decodes.
            if not req.prompt_cached:
                self._cache(req)
                req.prompt_cached = True

            req.output.append(token)
            new_tokens[req.req_id] = token
            if req.abort_pending:
                continue  # handled below
            if self._eos is not None and token == self._eos:
                self._finish(req, FinishReason.STOP)
                finished[req.req_id] = FinishReason.STOP
            elif len(req.output) >= req.max_tokens:
                self._finish(req, FinishReason.LENGTH)
                finished[req.req_id] = FinishReason.LENGTH

        # Deferred aborts. Safe now: the batch that was writing into these
        # blocks has finished.
        for req in self._pending_aborts:
            if req.state is ReqState.RUNNING:
                self._finish(req, FinishReason.ABORT)
                finished[req.req_id] = FinishReason.ABORT
        self._pending_aborts.clear()

        return StepResult(new_tokens=new_tokens, finished=finished)

    def free_request(self, req_id: str) -> bool:
        """Aborts a request. Safe to call twice or with an unknown id (returns False)."""
        req = self._requests.get(req_id)
        if req is None or req.abort_pending:
            return False
        if req.state is ReqState.WAITING:
            # Holds no blocks, so it's safe to drop even mid-batch.
            self._waiting.remove(req)
            req.state = ReqState.FINISHED
            del self._requests[req_id]
            return True
        if self._inflight is not None:
            # The batch in flight may still be writing KV into this request's
            # blocks. Freeing now would hand that memory to another request
            # while it's being written: a use-after-free on the GPU.
            req.abort_pending = True
            self._pending_aborts.append(req)
            return True
        self._finish(req, FinishReason.ABORT)
        return True

    def get_request(self, req_id: str) -> Request | None:
        return self._requests.get(req_id)

    def has_work(self) -> bool:
        return bool(self._waiting or self._running)

    def stats(self) -> SchedulerStats:
        return SchedulerStats(
            num_waiting=len(self._waiting),
            num_running=len(self._running),
            free_blocks=self.block_manager.num_free,
            cached_blocks=self.tree.num_cached_blocks,
            prompt_tokens=self._prompt_tokens,
            cached_prompt_tokens=self._cached_prompt_tokens,
            preemptions=self._preemptions,
            rejected=self._rejected,
        )

    # ------------------------------------------------------------ internals

    def _validate(
        self, req_id: str, prompt: Iterable[int] | TokenArray, max_tokens: int
    ) -> TokenArray | AddResult:
        if not req_id or not isinstance(max_tokens, int) or max_tokens < 1:
            return AddResult.INVALID
        try:
            arr = np.asarray(prompt)
        except (TypeError, ValueError, OverflowError):
            return AddResult.INVALID
        # Integer dtype only. Casting floats would silently truncate 1.7 -> 1.
        if arr.ndim != 1 or arr.size == 0 or arr.dtype.kind not in "iu":
            return AddResult.INVALID
        if int(arr.min()) < 0 or int(arr.max()) > np.iinfo(np.int32).max:
            return AddResult.INVALID
        if req_id in self._requests:
            return AddResult.DUPLICATE_ID
        # Needs this many blocks at the end, running alone. If the whole pool
        # is smaller, it could never finish, so reject now, not after it has
        # used up memory.
        if blocks_for(arr.size + max_tokens) > self.block_manager.num_blocks:
            return AddResult.TOO_LONG
        if len(self._waiting) >= self._max_waiting:
            return AddResult.QUEUE_FULL
        # astype copies, so the caller changing their array later can't change ours.
        return arr.astype(np.int32)

    def _admit(self, req: Request, budget: int) -> WorkItem | None:
        seq = req.seq()
        # Leave at least 1 token uncached: the model needs to process
        # something to produce the next token's logits.
        matched, cached = self.tree.match_prefix(seq[:-1]) if self._use_cache else (0, [])
        remaining = len(seq) - matched
        n = 1 if remaining == 1 else min(self._chunk_size, remaining, budget)
        need = blocks_for(matched + n) - len(cached) if n > 0 else 0
        new = self._try_allocate(need) if n > 0 else None
        if new is None:
            # Admission never preempts running requests. Kicking out a running
            # request to start a new one just swaps who waits, and under load
            # it thrashes: both keep redoing work and neither finishes.
            self.block_manager.decref_many(cached)
            return None

        req.block_table = cached + new
        req.num_computed = matched
        req.cached_tokens += matched
        if req.num_preemptions == 0:
            self._cached_prompt_tokens += min(matched, len(req.prompt))
        req.state = ReqState.RUNNING
        bisect.insort(self._running, req, key=lambda r: r.arrival)
        return WorkItem(req, matched, matched + n)

    def _try_allocate(self, n: int) -> list[int] | None:
        if n == 0:
            return []
        blocks = self.block_manager.allocate_many(n)
        if blocks is None:
            self.tree.evict(n - self.block_manager.num_free)
            blocks = self.block_manager.allocate_many(n)
        return blocks

    def _reserve(self, req: Request, n: int, preempted: list[str]) -> bool:
        """Gets n more blocks for a running request. Evicts cache, then preempts
        the newest running request, until it works. Returns False if `req`
        itself had to be preempted.

        Why this can't get stuck: the oldest request can take memory from
        everyone newer, and add_request refused anything that can't fit in
        the whole pool alone. So the oldest always finishes, then the next
        oldest, and so on.
        """
        while True:
            blocks = self._try_allocate(n)
            if blocks is not None:
                req.block_table.extend(blocks)
                return True
            victim = self._running[-1]  # newest
            self._preempt(victim)
            preempted.append(victim.req_id)
            if victim is req:
                return False

    def _preempt(self, req: Request) -> None:
        # Recompute, not swap: there's no CPU copy of the KV. But first cache
        # what it computed. When it comes back, match_prefix usually finds
        # those blocks still there, and the "recompute" is nearly free.
        self._cache(req)
        self.block_manager.decref_many(req.block_table)
        req.block_table = []
        req.num_computed = 0
        req.num_preemptions += 1
        req.state = ReqState.WAITING
        self._running.remove(req)
        self._waiting.appendleft(req)  # it was here first; it goes first
        self._preemptions += 1

    def _finish(self, req: Request, reason: FinishReason) -> None:
        self._cache(req)
        self.block_manager.decref_many(req.block_table)
        req.block_table = []
        req.state = ReqState.FINISHED
        self._running.remove(req)
        del self._requests[req.req_id]
        logger.debug("finished %s: %s", req.req_id, reason.value)

    def _cache(self, req: Request) -> None:
        """Puts the request's computed full blocks into the tree."""
        if self._use_cache and req.num_computed >= BLOCK_SIZE:
            self.tree.insert(req.seq()[: req.num_computed], req.block_table)
