"""Scheduler simulation. NOT a GPU benchmark.

What it does: replays the same multi-tenant workload through three
schedulers on a fake clock. Each batch step moves the clock forward by a
cost from the mock engine's CostModel. Nothing really sleeps, so Python and
OS overhead don't leak into the results, and the same seed gives the same
numbers every run.

What it shows: how the scheduling choices (static vs continuous batching,
prefix cache on/off) change throughput and latency under a given cost model.

What it can't show: real GPU speed. The cost numbers are made up. That's
why it runs three different cost profiles: the conclusions should hold
across all of them, not depend on one lucky setting.

The one real measurement: wall-clock time spent inside schedule_step().

Run:  .venv/bin/python bench/sim.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from prefixserve.mock_engine import CostModel
from prefixserve.scheduler import Scheduler

NUM_BLOCKS = 4096  # 65k tokens of KV for every mode
MAX_SEQS = 64  # max requests running together, every mode
# Picked from the budget sweep below: 512 gives the same throughput as 2048
# with ~2.7x lower p99 time-between-tokens. Bigger budgets make every decode
# wait behind a longer prefill step.
MAX_PREFILL = 512
BUDGET_SWEEP = [256, 512, 1024, 2048]

PROFILES = {
    "default": CostModel(base_s=0.020, per_prefill_token_s=0.0001, per_decode_seq_s=0.0002),
    "prefill-heavy": CostModel(base_s=0.020, per_prefill_token_s=0.0003, per_decode_seq_s=0.0002),
    "decode-heavy": CostModel(base_s=0.030, per_prefill_token_s=0.00005, per_decode_seq_s=0.0005),
}


# ---------------------------------------------------------------- workload


@dataclass(frozen=True, slots=True)
class Item:
    req_id: str
    arrival_s: float
    prompt: list[int]
    output_len: int


def make_workload(n: int, rate: float, seed: int, tenants: int = 8, system_len: int = 768) -> list[Item]:
    """Multi-tenant chat: each tenant has its own long system prompt, and
    every request from that tenant starts with it. Poisson arrivals."""
    rng = random.Random(seed)
    systems = [[rng.randrange(3, 259) for _ in range(system_len)] for _ in range(tenants)]
    t = 0.0
    items = []
    for i in range(n):
        t += rng.expovariate(rate)
        user = [rng.randrange(3, 259) for _ in range(rng.randint(32, 256))]
        items.append(Item(f"r{i}", t, systems[rng.randrange(tenants)] + user, rng.randint(32, 256)))
    return items


# ---------------------------------------------------------------- recording


@dataclass(slots=True)
class Recorder:
    arrival: dict[str, float]
    first: dict[str, float] = field(default_factory=dict)
    last: dict[str, float] = field(default_factory=dict)
    gaps: list[float] = field(default_factory=list)
    finish: dict[str, float] = field(default_factory=dict)
    tokens: int = 0

    def token(self, rid: str, t: float) -> None:
        self.tokens += 1
        if rid not in self.first:
            self.first[rid] = t
        else:
            self.gaps.append(t - self.last[rid])
        self.last[rid] = t


@dataclass(frozen=True, slots=True)
class Result:
    mode: str
    profile: str
    rate: float
    requests: int
    makespan_s: float
    output_tok_per_s: float
    req_per_s: float
    ttft_p50_s: float
    ttft_p99_s: float
    tbt_p50_ms: float
    tbt_p99_ms: float
    prefix_hit_rate: float
    preemptions: int
    sched_us_per_step: float


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def summarize(mode: str, profile: str, rate: float, rec: Recorder, hit: float, pre: int, sched_us: float) -> Result:
    start = min(rec.arrival.values())
    makespan = max(rec.finish.values()) - start
    ttft = [rec.first[r] - rec.arrival[r] for r in rec.arrival]
    return Result(
        mode=mode,
        profile=profile,
        rate=rate,
        requests=len(rec.arrival),
        makespan_s=round(makespan, 2),
        output_tok_per_s=round(rec.tokens / makespan, 1),
        req_per_s=round(len(rec.arrival) / makespan, 2),
        ttft_p50_s=round(pct(ttft, 50), 3),
        ttft_p99_s=round(pct(ttft, 99), 3),
        tbt_p50_ms=round(pct(rec.gaps, 50) * 1e3, 1),
        tbt_p99_ms=round(pct(rec.gaps, 99) * 1e3, 1),
        prefix_hit_rate=round(hit, 3),
        preemptions=pre,
        sched_us_per_step=round(sched_us, 1),
    )


# ---------------------------------------------------------------- modes


def run_continuous(
    wl: list[Item], cost: CostModel, *, cache: bool, profile: str, rate: float, max_prefill: int = MAX_PREFILL
) -> Result:
    s = Scheduler(NUM_BLOCKS, max_waiting=len(wl) + 1, enable_prefix_cache=cache)
    rec = Recorder({it.req_id: it.arrival_s for it in wl})
    t, i, done = 0.0, 0, 0
    sched_wall, steps = 0.0, 0
    while done < len(wl):
        while i < len(wl) and wl[i].arrival_s <= t:
            s.add_request(wl[i].req_id, wl[i].prompt, wl[i].output_len)
            i += 1
        if not s.has_work():
            t = wl[i].arrival_s
            continue
        w0 = time.perf_counter()
        batch = s.schedule_step(max_prefill, MAX_SEQS)
        sched_wall += time.perf_counter() - w0
        steps += 1
        t += cost.batch_seconds(batch) if batch.work else 0.001
        result = s.commit(batch, {w.req.req_id: 100 for w in batch.work if w.samples})
        for rid in result.new_tokens:
            rec.token(rid, t)
        for rid in result.finished:
            rec.finish[rid] = t
            done += 1
    st = s.stats()
    mode = "continuous + prefix cache" if cache else "continuous, no cache"
    if max_prefill != MAX_PREFILL:
        mode += f" (budget {max_prefill})"
    return summarize(mode, profile, rate, rec, st.prefix_hit_rate, st.preemptions, sched_wall / steps * 1e6)


def run_static(wl: list[Item], cost: CostModel, *, profile: str, rate: float) -> Result:
    """Classic static batching: take what's queued, run the whole batch until
    its longest request finishes, then take the next batch.

    - Memory is reserved up front for prompt + max output (no paging).
    - The whole prompt is prefilled in one step (no chunking, no cache).
    - Finished requests keep their slot until the batch ends (padding), so
      they still cost decode compute.
    """
    capacity = NUM_BLOCKS * 16
    rec = Recorder({it.req_id: it.arrival_s for it in wl})
    t, i = 0.0, 0
    queue: list[Item] = []
    while len(rec.finish) < len(wl):
        while i < len(wl) and wl[i].arrival_s <= t:
            queue.append(wl[i])
            i += 1
        if not queue:
            t = wl[i].arrival_s
            continue
        batch: list[Item] = []
        used = 0
        while queue and len(batch) < MAX_SEQS and used + len(queue[0].prompt) + queue[0].output_len <= capacity:
            it = queue.pop(0)
            used += len(it.prompt) + it.output_len
            batch.append(it)

        t += cost.step_seconds(sum(len(it.prompt) for it in batch), 0)
        for it in batch:
            rec.token(it.req_id, t)
        longest = max(it.output_len for it in batch)
        for k in range(1, longest):
            t += cost.step_seconds(0, len(batch))  # padded: every slot computes
            for it in batch:
                if k < it.output_len:
                    rec.token(it.req_id, t)
        for it in batch:
            rec.finish[it.req_id] = t
    return summarize("static batching", profile, rate, rec, 0.0, 0, 0.0)


# ---------------------------------------------------------------- main


def run_all(n: int, rates: list[float], seed: int) -> list[Result]:
    modes: list[Callable[[list[Item], CostModel, str, float], Result]] = [
        lambda wl, c, p, r: run_static(wl, c, profile=p, rate=r),
        lambda wl, c, p, r: run_continuous(wl, c, cache=False, profile=p, rate=r),
        lambda wl, c, p, r: run_continuous(wl, c, cache=True, profile=p, rate=r),
    ]
    results = []
    for profile, cost in PROFILES.items():
        for rate in rates:
            wl = make_workload(n, rate, seed)
            for run in modes:
                res = run(wl, cost, profile, rate)
                results.append(res)
                print(f"{profile:>13} rate={rate:>4} {res.mode:<26} "
                      f"tok/s={res.output_tok_per_s:>8} ttft_p99={res.ttft_p99_s:>7}s "
                      f"tbt_p99={res.tbt_p99_ms:>6}ms", flush=True)
    return results


def run_budget_sweep(n: int, rate: float, seed: int) -> list[Result]:
    """Prefill budget vs latency. Bigger budget = longer steps = decodes wait longer."""
    wl = make_workload(n, rate, seed)
    out = []
    for cache in (False, True):
        for budget in BUDGET_SWEEP:
            res = run_continuous(wl, PROFILES["default"], cache=cache, profile="default", rate=rate, max_prefill=budget)
            res = Result(**{**asdict(res), "mode": f"{'cache' if cache else 'no cache'}, budget {budget}"})
            out.append(res)
            print(f"sweep rate={rate} {res.mode:<22} tok/s={res.output_tok_per_s:>8} "
                  f"tbt_p99={res.tbt_p99_ms:>6}ms ttft_p99={res.ttft_p99_s}s", flush=True)
    return out


def markdown(results: list[Result]) -> str:
    lines = [
        "| Cost profile | Load (req/s) | Mode | Output tok/s | TTFT p50 | TTFT p99 | TBT p50 | TBT p99 | Prefix hit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.profile} | {r.rate:g} | {r.mode} | {r.output_tok_per_s:,.0f} | {r.ttft_p50_s:.2f}s | "
            f"{r.ttft_p99_s:.2f}s | {r.tbt_p50_ms:.0f}ms | {r.tbt_p99_ms:.0f}ms | {r.prefix_hit_rate:.0%} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--requests", type=int, default=500)
    p.add_argument("--rates", type=float, nargs="+", default=[4.0, 8.0, 16.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("results.json"))
    args = p.parse_args()

    results = run_all(args.requests, args.rates, args.seed)
    sweep = run_budget_sweep(args.requests, 8.0, args.seed)
    args.out.write_text(json.dumps({
        "note": "Simulated clock with a made-up cost model. Not a GPU benchmark.",
        "config": {
            "requests": args.requests, "seed": args.seed, "num_blocks": NUM_BLOCKS,
            "max_seqs": MAX_SEQS, "max_prefill_tokens": MAX_PREFILL,
            "profiles": {k: asdict(v) for k, v in PROFILES.items()},
        },
        "results": [asdict(r) for r in results],
        "budget_sweep": [asdict(r) for r in sweep],
    }, indent=2) + "\n")
    print()
    print(markdown(results))
    print()
    print(markdown(sweep))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
