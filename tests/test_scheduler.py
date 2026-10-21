"""One test per scheduler rule in PLAN.md."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from prefixserve import (
    BLOCK_SIZE,
    AddResult,
    Batch,
    FinishReason,
    Scheduler,
    SchedulerError,
    StepResult,
)

_tok = itertools.count(1000)


def sample(batch: Batch) -> dict[str, int]:
    """Fake engine: a fresh token for every item that reaches the end of its seq."""
    return {w.req.req_id: next(_tok) for w in batch.work if w.samples}


def step(s: Scheduler, budget: int = 2048, max_seqs: int = 8) -> tuple[Batch, StepResult]:
    batch = s.schedule_step(budget, max_seqs)
    return batch, s.commit(batch, sample(batch))


def run_until_done(s: Scheduler, budget: int = 2048, max_seqs: int = 8, cap: int = 10_000) -> list[Batch]:
    batches: list[Batch] = []
    for _ in range(cap):
        if not s.has_work():
            return batches
        batch, _ = step(s, budget, max_seqs)
        batches.append(batch)
    raise AssertionError("scheduler did not finish: possible livelock")


def assert_clean(s: Scheduler) -> None:
    """No request holds a block, and nothing leaked."""
    s.block_manager.check_invariants()
    s.tree.check_invariants()
    s.tree.evict(s.block_manager.num_blocks)
    assert s.block_manager.num_free == s.block_manager.num_blocks


def toks(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


def test_every_decode_runs_every_step() -> None:
    s = Scheduler(64)
    for i in range(4):
        assert s.add_request(f"r{i}", toks(16, 100 * i), max_tokens=10) is AddResult.ACCEPTED
    step(s)  # all 4 prefill
    for _ in range(5):
        batch, _ = step(s)
        assert sorted(w.req.req_id for w in batch.work) == ["r0", "r1", "r2", "r3"]
        assert all(w.end - w.start == 1 for w in batch.work)
    run_until_done(s)
    assert_clean(s)


def test_decodes_ignore_prefill_budget() -> None:
    s = Scheduler(256)
    s.add_request("short", toks(16), max_tokens=50)
    step(s)
    s.add_request("long", toks(2000, 5000), max_tokens=1)
    # Budget of 1 prefill token: the decode still runs.
    batch, _ = step(s, budget=1)
    by_id = {w.req.req_id: w for w in batch.work}
    assert by_id["short"].end - by_id["short"].start == 1
    assert by_id["long"].end - by_id["long"].start == 1


def test_chunked_prefill_512() -> None:
    s = Scheduler(256)
    s.add_request("big", toks(1300), max_tokens=1)
    sizes = []
    for _ in range(3):
        batch, result = step(s)
        sizes.append(batch.work[0].end - batch.work[0].start)
    assert sizes == [512, 512, 276]
    assert result.finished == {"big": FinishReason.LENGTH}


def test_prefill_budget_is_shared() -> None:
    s = Scheduler(256)
    s.add_request("a", toks(1000), max_tokens=1)
    s.add_request("b", toks(1000, 5000), max_tokens=1)
    batch, _ = step(s, budget=600)
    assert [(w.req.req_id, w.end - w.start) for w in batch.work] == [("a", 512), ("b", 88)]
    assert batch.num_tokens == 600


def test_shared_system_prompt_is_reused() -> None:
    s = Scheduler(128)
    system = toks(256, 7000)
    s.add_request("first", system + toks(20), max_tokens=2)
    run_until_done(s)

    free_before = s.block_manager.num_free
    s.add_request("second", system + toks(30, 900), max_tokens=2)
    batch, _ = step(s)
    req = batch.work[0].req
    assert req.cached_tokens == 256
    assert batch.work[0].start == 256  # skipped the cached part
    # Only the 30-token tail needed new blocks.
    assert free_before - s.block_manager.num_free == blocks_used_by_tail(30)
    assert s.stats().prefix_hit_rate > 0
    run_until_done(s)
    assert_clean(s)


def blocks_used_by_tail(n: int) -> int:
    return -(-n // BLOCK_SIZE)


def test_identical_prompt_still_computes_one_token() -> None:
    s = Scheduler(64)
    s.add_request("a", toks(64), max_tokens=1)
    run_until_done(s)
    s.add_request("b", toks(64), max_tokens=1)
    batch, result = step(s)
    w = batch.work[0]
    # Fully cached would mean nothing to compute. Matching seq[:-1] leaves 16.
    assert (w.start, w.end) == (48, 64)
    assert "b" in result.new_tokens
    assert_clean(s)


def test_oom_preempts_newest_and_everything_finishes() -> None:
    s = Scheduler(12)
    for name in ("a", "b", "c"):
        assert s.add_request(name, toks(48, ord(name) * 100), max_tokens=60) is AddResult.ACCEPTED
    batches = run_until_done(s)
    preempted = [rid for b in batches for rid in b.preempted]
    assert preempted, "pool is too small for all three; someone must be preempted"
    assert "a" not in preempted, "the oldest request is never preempted"
    assert s.stats().preemptions == len(preempted)
    assert_clean(s)


def test_preempted_request_rematches_its_own_blocks() -> None:
    s = Scheduler(10)
    s.add_request("a", toks(64, 0), max_tokens=40)
    s.add_request("b", toks(64, 5000), max_tokens=40)
    b = s.get_request("b")
    assert b is not None
    run_until_done(s)
    assert b.num_preemptions >= 1
    # b's first admission matched nothing. Anything cached now came from
    # re-matching the blocks it cached when it was preempted.
    assert b.cached_tokens > 0
    assert_clean(s)


def test_abort_in_flight_is_deferred_to_commit() -> None:
    s = Scheduler(16)
    s.add_request("a", toks(40), max_tokens=10)
    batch = s.schedule_step(512, 8)
    req = batch.work[0].req
    held = list(req.block_table)
    assert s.free_request("a") is True
    assert s.free_request("a") is False  # second call is a no-op
    # Still held: the batch may be writing into them.
    assert all(s.block_manager.ref_count(b) >= 1 for b in held)
    result = s.commit(batch, sample(batch))
    assert result.finished == {"a": FinishReason.ABORT}
    assert s.get_request("a") is None
    assert_clean(s)


def test_abort_when_idle_and_while_waiting() -> None:
    s = Scheduler(16)
    s.add_request("run", toks(20), max_tokens=10)
    step(s)
    assert s.free_request("run") is True
    s.add_request("wait", toks(20), max_tokens=10)
    batch = s.schedule_step(512, 8)
    s.add_request("queued", toks(20, 50), max_tokens=10)
    assert s.free_request("queued") is True  # waiting: no blocks, safe mid-batch
    s.commit(batch, sample(batch))
    assert s.free_request("nope") is False
    s.free_request("wait")
    assert_clean(s)


def test_add_request_rejections() -> None:
    s = Scheduler(8, max_waiting=2)
    assert s.add_request("", toks(4), 1) is AddResult.INVALID
    assert s.add_request("x", [], 1) is AddResult.INVALID
    assert s.add_request("x", toks(4), 0) is AddResult.INVALID
    assert s.add_request("x", [1, -2], 1) is AddResult.INVALID
    assert s.add_request("x", [1.5, 2.0], 1) is AddResult.INVALID  # type: ignore[list-item]
    assert s.add_request("x", [[1, 2]], 1) is AddResult.INVALID  # type: ignore[list-item]
    assert s.add_request("x", [2**40], 1) is AddResult.INVALID
    assert s.add_request("x", toks(8 * BLOCK_SIZE), 1) is AddResult.TOO_LONG
    assert s.add_request("a", toks(4), 1) is AddResult.ACCEPTED
    assert s.add_request("a", toks(4), 1) is AddResult.DUPLICATE_ID
    assert s.add_request("b", np.arange(4, dtype=np.int64), 1) is AddResult.ACCEPTED
    assert s.add_request("c", toks(4), 1) is AddResult.QUEUE_FULL
    assert s.stats().rejected == 10


def test_prompt_is_copied() -> None:
    s = Scheduler(8)
    p = np.arange(20, dtype=np.int32)
    s.add_request("a", p, 1)
    p[:] = 0
    req = s.get_request("a")
    assert req is not None and req.prompt[5] == 5


def test_finish_on_length_and_eos_keeps_prefix_cached() -> None:
    s = Scheduler(32, stop_token_ids={4242})
    s.add_request("len", toks(40), max_tokens=3)
    _, result = step(s)
    assert result.finished == {}
    run_until_done(s)

    s.add_request("eos", toks(40, 300), max_tokens=100)
    batch = s.schedule_step(512, 8)
    result = s.commit(batch, {"eos": 4242})
    assert result.finished == {"eos": FinishReason.STOP}
    assert s.tree.num_cached_blocks >= 4  # both prompts' full blocks stay cached
    assert_clean(s)


def test_protocol_errors() -> None:
    s = Scheduler(16)
    s.add_request("a", toks(20), 5)
    batch = s.schedule_step(512, 8)
    with pytest.raises(SchedulerError):
        s.schedule_step(512, 8)
    with pytest.raises(SchedulerError):
        s.commit(Batch(work=[], preempted=[]), {})
    with pytest.raises(SchedulerError, match="no token"):
        s.commit(batch, {})


def test_bad_step_args() -> None:
    s = Scheduler(16)
    with pytest.raises(ValueError):
        s.schedule_step(0, 8)
    with pytest.raises(ValueError):
        s.schedule_step(512, 0)


def test_running_cap_respected() -> None:
    s = Scheduler(256)
    for i in range(10):
        s.add_request(f"r{i}", toks(16, i * 100), max_tokens=5)
    for _ in range(20):
        step(s, max_seqs=3)
        assert s.stats().num_running <= 3
    run_until_done(s, max_seqs=3)
    assert_clean(s)
