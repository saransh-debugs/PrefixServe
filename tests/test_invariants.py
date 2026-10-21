"""Random add / abort / step sequences against a small pool.

After every step: both C++ structures pass their own checks, and every
block's ref_count equals (1 if the tree caches it) + (how many live requests
hold it). At the end: everything finishes (no livelock) and nothing leaks.
"""

from __future__ import annotations

from collections import Counter

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from prefixserve import Batch, Scheduler

NUM_BLOCKS = 24
EOS = 7
PREFIXES = [list(range(16)), list(range(16)) + list(range(100, 132)), list(range(500, 520))]


class SchedulerMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.s = Scheduler(NUM_BLOCKS, max_waiting=8, chunk_size=24, stop_token_ids={EOS})
        self.ids: set[str] = set()
        self.next_id = 0

    def _sample(self, batch: Batch, data: st.DataObject) -> dict[str, int]:
        return {
            w.req.req_id: data.draw(st.integers(0, 9), label="token") for w in batch.work if w.samples
        }

    @rule(
        prefix=st.sampled_from(PREFIXES),
        tail=st.lists(st.integers(0, 3), max_size=40),
        max_tokens=st.integers(1, 30),
    )
    def add(self, prefix: list[int], tail: list[int], max_tokens: int) -> None:
        rid = f"r{self.next_id}"
        self.next_id += 1
        self.s.add_request(rid, prefix + tail, max_tokens)
        if self.s.get_request(rid) is not None:
            self.ids.add(rid)

    @precondition(lambda self: bool(self.ids))
    @rule(data=st.data())
    def abort(self, data: st.DataObject) -> None:
        rid = data.draw(st.sampled_from(sorted(self.ids)))
        self.s.free_request(rid)
        self._forget_finished()

    @rule(
        data=st.data(),
        budget=st.integers(1, 64),
        max_seqs=st.integers(1, 4),
        abort_mid_batch=st.booleans(),
    )
    def step(self, data: st.DataObject, budget: int, max_seqs: int, abort_mid_batch: bool) -> None:
        running_before = self.s.stats().num_running
        batch = self.s.schedule_step(budget, max_seqs)
        # Admission stops at the cap. (If the cap just went down, requests
        # already running keep running; nothing new is admitted.)
        assert self.s.stats().num_running <= max(max_seqs, running_before)
        if abort_mid_batch and self.ids:
            self.s.free_request(data.draw(st.sampled_from(sorted(self.ids))))
        self.s.commit(batch, self._sample(batch, data))
        self._forget_finished()

    def _forget_finished(self) -> None:
        self.ids = {rid for rid in self.ids if self.s.get_request(rid) is not None}

    @invariant()
    def structures_consistent(self) -> None:
        self.s.block_manager.check_invariants()
        self.s.tree.check_invariants()

    @invariant()
    def ref_counts_match_owners(self) -> None:
        held: Counter[int] = Counter()
        for rid in self.ids:
            req = self.s.get_request(rid)
            assert req is not None
            held.update(req.block_table)
        bm, tree = self.s.block_manager, self.s.tree
        for b in range(NUM_BLOCKS):
            assert bm.ref_count(b) == held[b] + int(tree.is_cached(b)), f"block {b}"

    def teardown(self) -> None:
        # Liveness: with no new requests, everything finishes.
        for _ in range(5000):
            if not self.s.has_work():
                break
            batch = self.s.schedule_step(32, 3)
            self.s.commit(batch, {w.req.req_id: 1 for w in batch.work if w.samples})
        assert not self.s.has_work(), "livelock: requests never finished"
        self.s.tree.evict(NUM_BLOCKS)
        assert self.s.block_manager.num_free == NUM_BLOCKS, "leaked blocks"


TestScheduler = SchedulerMachine.TestCase
TestScheduler.settings = settings(
    max_examples=300,
    stateful_step_count=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
