"""Real-model timing: what does the prefix cache save on GPT-2 (CPU)?

Unlike sim.py, these are real wall-clock numbers from a real model.

What it measures:
1. Cold vs warm time-to-first-token for one request. Warm = the tenant's
   system prompt is already in the cache, so only the user's question is
   computed.
2. A multi-tenant run (2 tenants, many requests) through the scheduler,
   cache on vs off: total time, prompt tokens actually computed, TTFT.
   Also checks both runs produce identical text.

What it does NOT measure: batching gains. The engine runs one request at a
time (PagedGPT2Engine from tests/test_cache_correctness.py), so this only
shows the compute saved by reusing cached KV. CPU numbers, not GPU.

Run (needs the torch venv, see README):
    .venv-torch/bin/python bench/gpt2_bench.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from test_cache_correctness import PagedGPT2Engine, PyTorchKVStore  # noqa: E402

from prefixserve import BLOCK_SIZE, Scheduler  # noqa: E402

TENANTS = {
    "travel": (
        "You are the support assistant for Blue Horizon Travel. You help customers with "
        "bookings, cancellations, baggage rules, seat upgrades, and travel insurance. "
        "Always be polite and brief. Never invent prices, dates, flight numbers, or hotel "
        "names. If a question is about a refund, explain that refunds take five to seven "
        "business days and are sent to the original payment method. If a customer is "
        "angry, apologise once and focus on solving the problem. If you do not know the "
        "answer, say so and offer to connect the customer with a human agent. Do not give "
        "legal or medical advice. Keep every answer under three sentences. "
    ) * 2,
    "bank": (
        "You are the virtual assistant for Northwind Bank. You answer questions about "
        "accounts, cards, transfers, loans, and online banking. Never ask for full card "
        "numbers, passwords, or one-time codes. If someone reports fraud, tell them to "
        "freeze their card in the app and call the fraud line right away. Explain fees "
        "clearly and do not guess interest rates. If a question needs account details, "
        "ask the customer to log in to the secure app. Be calm, short, and professional. "
        "Keep every answer under three sentences and avoid jargon whenever possible. "
    ) * 2,
}
QUESTIONS = [
    "How do I change my booking?",
    "Can I get a refund?",
    "What is the baggage limit?",
    "How do I reset my password?",
    "Is there a fee for transfers?",
    "My card was stolen, what now?",
    "Can I pick my seat?",
    "How long does a transfer take?",
]


def prompt(tenant: str, q: str) -> str:
    return f"{TENANTS[tenant]}\nCustomer: {q}\nAssistant:"


def make(model: Any, num_blocks: int, cache: bool, chunk: int) -> tuple[Scheduler, PagedGPT2Engine]:
    cfg = model.config
    s = Scheduler(num_blocks, chunk_size=chunk, enable_prefix_cache=cache)
    store = PyTorchKVStore(num_blocks, cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head)
    return s, PagedGPT2Engine(model, store, s)


def run(model: Any, reqs: list[tuple[str, list[int]]], *, cache: bool, max_new: int, chunk: int,
        num_blocks: int) -> dict[str, Any]:
    s, engine = make(model, num_blocks, cache, chunk)
    for rid, ids in reqs:
        s.add_request(rid, ids, max_new)
    computed = 0
    first: dict[str, float] = {}
    out: dict[str, list[int]] = {}
    model_s = sched_s = 0.0
    t0 = time.perf_counter()
    while s.has_work():
        a = time.perf_counter()
        batch = s.schedule_step(512, 8)
        b = time.perf_counter()
        sampled = engine.run(batch)
        c = time.perf_counter()
        sched_s += b - a
        model_s += c - b
        computed += sum(w.end - w.start for w in batch.work if w.end - w.start > 1 or not w.req.output)
        res = s.commit(batch, sampled)
        now = time.perf_counter() - t0
        for rid, tok in res.new_tokens.items():
            first.setdefault(rid, now)
            out.setdefault(rid, []).append(tok)
    total = time.perf_counter() - t0
    st = s.stats()
    ttft = sorted(first.values())
    return {
        "total_s": total,
        "model_s": model_s,
        "scheduler_s": sched_s,
        "prompt_tokens": st.prompt_tokens,
        "prefill_tokens_computed": computed,
        "prefix_hit_rate": st.prefix_hit_rate,
        "ttft_mean_s": statistics.mean(ttft),
        "ttft_p50_s": ttft[len(ttft) // 2],
        "output_tok_per_s": sum(len(v) for v in out.values()) / total,
        "outputs": out,
    }


def single_ttft(model: Any, tok: Any, *, repeats: int) -> dict[str, float]:
    """Time to first token for one request: cache cold vs warm (system prompt cached)."""
    sys_ids = tok.encode(prompt("travel", "warmup"))
    target = tok.encode(prompt("travel", "Can I bring my dog on the plane?"))
    cold, warm = [], []
    for _ in range(repeats):
        for is_warm, bucket in ((False, cold), (True, warm)):
            s, engine = make(model, 256, cache=True, chunk=4096)
            if is_warm:  # cache the system prompt via an earlier request
                s.add_request("w", sys_ids, 1)
                while s.has_work():
                    batch = s.schedule_step(4096, 8)
                    s.commit(batch, engine.run(batch))
            s.add_request("t", target, 1)
            t0 = time.perf_counter()
            while s.has_work():
                batch = s.schedule_step(4096, 8)
                s.commit(batch, engine.run(batch))
            bucket.append(time.perf_counter() - t0)
    return {
        "prompt_tokens": len(target),
        "cold_ms": statistics.median(cold) * 1e3,
        "warm_ms": statistics.median(warm) * 1e3,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("--max-new", type=int, default=16)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("gpt2_results.json"))
    args = p.parse_args()

    torch.manual_seed(0)
    tok = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()

    with torch.inference_mode():
        model(torch.tensor([[1, 2, 3]]))  # warm up kernels

        one = single_ttft(model, tok, repeats=args.repeats)
        print(f"Single request ({one['prompt_tokens']} prompt tokens): "
              f"cold TTFT {one['cold_ms']:.0f}ms, warm TTFT {one['warm_ms']:.0f}ms "
              f"-> {one['cold_ms'] / one['warm_ms']:.1f}x faster")

        names = list(TENANTS)
        reqs = [(f"r{i}", tok.encode(prompt(names[i % 2], QUESTIONS[i % len(QUESTIONS)])))
                for i in range(args.requests)]
        need = sum(-(-(len(ids) + args.max_new) // BLOCK_SIZE) for _, ids in reqs) + 8
        off = run(model, reqs, cache=False, max_new=args.max_new, chunk=512, num_blocks=need)
        on = run(model, reqs, cache=True, max_new=args.max_new, chunk=512, num_blocks=need)

    same = off["outputs"] == on["outputs"]
    rows = [
        ("Total time", f"{off['total_s']:.2f}s", f"{on['total_s']:.2f}s"),
        ("Prompt tokens computed", f"{off['prefill_tokens_computed']:,}", f"{on['prefill_tokens_computed']:,}"),
        ("Prefix hit rate", f"{off['prefix_hit_rate']:.0%}", f"{on['prefix_hit_rate']:.0%}"),
        ("TTFT mean", f"{off['ttft_mean_s']:.2f}s", f"{on['ttft_mean_s']:.2f}s"),
        ("Output tok/s", f"{off['output_tok_per_s']:.1f}", f"{on['output_tok_per_s']:.1f}"),
        ("Time in scheduler", f"{off['scheduler_s'] * 1e3:.1f}ms", f"{on['scheduler_s'] * 1e3:.1f}ms"),
    ]
    print(f"\n{args.requests} requests, 2 tenants, ~{len(reqs[0][1])} prompt tokens each, {args.max_new} new tokens")
    print(f"{'':26}{'cache off':>12}{'cache on':>12}")
    for name, a, b in rows:
        print(f"{name:26}{a:>12}{b:>12}")
    print(f"Speedup (total time): {off['total_s'] / on['total_s']:.2f}x")
    print(f"Identical outputs with and without cache: {same}")

    for r in (off, on):
        r.pop("outputs")
    args.out.write_text(json.dumps({
        "note": "Real GPT-2 small on CPU. One request at a time (no batched forward pass): "
                "measures prefix-cache savings only, not batching.",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "single_request": one, "cache_off": off, "cache_on": on, "identical_outputs": same,
    }, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
