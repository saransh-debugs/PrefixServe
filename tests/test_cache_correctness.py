"""Proves the prefix cache gives the same output as recomputing everything,
using a real model (GPT-2 small, CPU only).

The scheduler only hands out integer block ids. PyTorchKVStore turns those
ids into real KV tensors: a paged pool of [num_blocks, layers, heads, 16, dim].
Every step, the KV for positions [0, start) is gathered from the request's
blocks and fed to the model as past_key_values. If the cache ever handed a
request the wrong block, or a block with stale KV, the tokens would differ.

What's checked:
- Token ids match the no-cache baseline exactly.
- Logits match within 1e-5 *relative* to their size.
  Why not absolute 1e-5: GPT-2 logits reach ~300, and float32 rounding alone
  moves them ~1e-3 when the additions happen in a different order. HF's own
  KV cache differs from full recompute by that much. A wrong KV block would
  move them by whole units.

Not checked: speed. This is about correctness only.

Needs torch + transformers. On Intel Macs, torch stops at 2.2.2 (Python
<= 3.12), so this runs in a separate venv:
    python3.12 -m venv .venv-torch
    .venv-torch/bin/pip install "torch==2.2.2" "numpy<2" "transformers>=4.40,<4.50" pytest -e .
    .venv-torch/bin/python -m pytest tests/test_cache_correctness.py
In the main venv it is skipped.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from prefixserve import BLOCK_SIZE, Scheduler  # noqa: E402
from prefixserve.scheduler import Batch, WorkItem  # noqa: E402

MAX_NEW_TOKENS = 12
REL_TOL = 1e-5

# Long enough to fill several 16-token blocks with GPT-2's tokenizer.
SYSTEM = (
    "You are a careful assistant for a travel company called Blue Horizon. "
    "Answer every question in one short sentence. Be friendly and clear. "
    "Never invent prices, dates, flight numbers, or hotel names. If you do "
    "not know something, say so and suggest contacting the support team. "
)
PROMPTS = [
    SYSTEM + "Question: What is the capital of France? Answer:",
    SYSTEM + "Question: Which ocean is the largest? Answer:",
    SYSTEM + "Question: What language is spoken in Brazil? Answer:",
]


# ---------------------------------------------------------------- the bridge


class PyTorchKVStore:
    """Maps PrefixServe block ids to real KV tensors.

    One preallocated pool per K and V: [num_blocks, layers, heads, 16, head_dim].
    Block b holds the KV for 16 consecutive positions of whichever request
    owns it. That's the paged layout: a request's KV is scattered across
    whatever blocks the scheduler gave it, not one contiguous buffer.
    """

    def __init__(self, num_blocks: int, layers: int, heads: int, head_dim: int) -> None:
        shape = (num_blocks, layers, heads, BLOCK_SIZE, head_dim)
        self.k = torch.zeros(shape, dtype=torch.float32)
        self.v = torch.zeros(shape, dtype=torch.float32)
        self.layers = layers

    def write(self, block_table: list[int], start: int, layer_kv: list[tuple[Any, Any]]) -> None:
        """Stores KV for positions [start, start + n). layer_kv[l] = (k, v),
        each [1, heads, n, head_dim]."""
        n = layer_kv[0][0].shape[2]
        pos = torch.arange(start, start + n)
        blocks = torch.tensor(block_table)[pos // BLOCK_SIZE]
        offs = pos % BLOCK_SIZE
        for layer, (k, v) in enumerate(layer_kv):
            # [1, heads, n, dim] -> [n, heads, dim], scattered to (block, offset)
            self.k[blocks, layer, :, offs] = k[0].permute(1, 0, 2)
            self.v[blocks, layer, :, offs] = v[0].permute(1, 0, 2)

    def gather(self, block_table: list[int], num_tokens: int) -> list[tuple[Any, Any]]:
        """KV for positions [0, num_tokens), as per-layer [1, heads, T, dim]."""
        nb = -(-num_tokens // BLOCK_SIZE)
        idx = torch.tensor(block_table[:nb])
        out = []
        for layer in range(self.layers):
            # [nb, heads, 16, dim] -> [heads, nb*16, dim] -> trim to num_tokens
            k = self.k[idx, layer].permute(1, 0, 2, 3).reshape(self.k.shape[2], nb * BLOCK_SIZE, -1)
            v = self.v[idx, layer].permute(1, 0, 2, 3).reshape(self.v.shape[2], nb * BLOCK_SIZE, -1)
            out.append((k[None, :, :num_tokens].contiguous(), v[None, :, :num_tokens].contiguous()))
        return out


def legacy_kv(past: Any) -> list[tuple[Any, Any]]:
    """Per-layer (k, v) out of whatever HF returned. transformers 5 keeps them
    in cache.layers; 4.x had to_legacy_cache(); older still, plain tuples."""
    if hasattr(past, "layers"):
        return [(layer.keys, layer.values) for layer in past.layers]
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    return [(k, v) for k, v in past]


def to_hf_cache(kv: list[tuple[Any, Any]]) -> Any:
    """Builds a DynamicCache via update(), which exists in transformers 4 and 5
    (from_legacy_cache was removed in 5)."""
    cache = transformers.DynamicCache()
    for layer, (k, v) in enumerate(kv):
        cache.update(k, v, layer)
    return cache


class PagedGPT2Engine:
    """Runs a PrefixServe batch on GPT-2, reading and writing KV through the store."""

    def __init__(self, model: Any, store: PyTorchKVStore, scheduler: Scheduler) -> None:
        self.model = model
        self.store = store
        self.scheduler = scheduler
        self.logits: dict[str, list[Any]] = {}

    def run(self, batch: Batch) -> dict[str, int]:
        sampled: dict[str, int] = {}
        for w in batch.work:  # one request at a time keeps the test simple
            token = self._run_item(w)
            if token is not None:
                sampled[w.req.req_id] = token
        return sampled

    def _run_item(self, w: WorkItem) -> int | None:
        req = w.req
        seq = req.seq()
        # Blocks we're about to write must belong only to this request.
        # Writing into a block the cache shares would corrupt other requests.
        for b in {req.block_table[p // BLOCK_SIZE] for p in range(w.start, w.end)}:
            assert not self.scheduler.tree.is_cached(b), f"would overwrite shared block {b}"

        past = None
        if w.start > 0:
            past = to_hf_cache(self.store.gather(req.block_table, w.start))
        ids = torch.tensor([seq[w.start : w.end].tolist()])
        pos = torch.arange(w.start, w.end)[None]
        out = self.model(input_ids=ids, past_key_values=past, position_ids=pos, use_cache=True)

        # out.past_key_values covers [0, end). Store only the new part.
        new = [(k[:, :, w.start :], v[:, :, w.start :]) for k, v in legacy_kv(out.past_key_values)]
        self.store.write(req.block_table, w.start, new)

        if not w.samples:
            return None
        last = out.logits[0, -1]
        self.logits.setdefault(req.req_id, []).append(last.clone())
        return int(last.argmax())


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def gpt2() -> Iterator[tuple[Any, Any]]:
    torch.manual_seed(0)
    try:
        tok = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
    except OSError as e:  # no network and not cached
        pytest.skip(f"gpt2 weights not available: {e}")
    model = model.to("cpu").eval()
    with torch.inference_mode():
        yield tok, model
    del model, tok
    gc.collect()


def baseline(model: Any, prompt_ids: list[int]) -> tuple[list[int], list[Any]]:
    """Greedy decoding with NO cache: every step recomputes the whole sequence."""
    ids = list(prompt_ids)
    tokens, logits = [], []
    for _ in range(MAX_NEW_TOKENS):
        last = model(torch.tensor([ids]), use_cache=False).logits[0, -1]
        logits.append(last.clone())
        nxt = int(last.argmax())
        tokens.append(nxt)
        ids.append(nxt)
    return tokens, logits


def assert_matches(name: str, got: list[int], got_logits: list[Any], want: list[int], want_logits: list[Any]) -> None:
    assert got == want, f"{name}: tokens differ from no-cache baseline"
    for step, (a, b) in enumerate(zip(got_logits, want_logits, strict=True)):
        scale = float(b.abs().max())
        diff = float((a - b).abs().max())
        assert diff <= REL_TOL * scale, f"{name} step {step}: logit diff {diff:.2e} > {REL_TOL:g} x {scale:.0f}"


def make_run(model: Any, num_blocks: int, chunk_size: int) -> tuple[Scheduler, PagedGPT2Engine, PyTorchKVStore]:
    cfg = model.config
    s = Scheduler(num_blocks, chunk_size=chunk_size)
    store = PyTorchKVStore(num_blocks, cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head)
    return s, PagedGPT2Engine(model, store, s), store


def drive(s: Scheduler, engine: PagedGPT2Engine, max_prefill: int, max_seqs: int) -> dict[str, list[int]]:
    outputs: dict[str, list[int]] = {}
    for _ in range(10_000):
        if not s.has_work():
            return outputs
        batch = s.schedule_step(max_prefill, max_seqs)
        result = s.commit(batch, engine.run(batch))
        for rid, tok in result.new_tokens.items():
            outputs.setdefault(rid, []).append(tok)
    raise AssertionError("scheduler did not finish")


def cleanup(*objs: Any) -> None:
    for o in objs:
        del o
    gc.collect()


# ---------------------------------------------------------------- tests


def test_hf_generate_agrees_with_baseline(gpt2: tuple[Any, Any]) -> None:
    """Sanity check: our no-cache baseline is the same as HF generate()."""
    tok, model = gpt2
    ids = tok.encode("The quick brown fox")
    want, _ = baseline(model, ids)
    out = model.generate(
        torch.tensor([ids]), max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id
    )
    got = out[0, len(ids):].tolist()
    assert got == want[: len(got)]


def test_prefix_reuse_matches_full_recompute(gpt2: tuple[Any, Any]) -> None:
    """Request B reuses A's cached system-prompt blocks. Output must not change."""
    tok, model = gpt2
    ids = [tok.encode(p) for p in PROMPTS[:2]]
    shared = next(i for i, (a, b) in enumerate(zip(*ids)) if a != b)
    assert shared >= 2 * BLOCK_SIZE, "prompts must share at least 2 full blocks"

    s, engine, store = make_run(model, num_blocks=64, chunk_size=16)  # 16 forces chunked prefill
    s.add_request("A", ids[0], MAX_NEW_TOKENS)
    out_a = drive(s, engine, max_prefill=16, max_seqs=4)

    s.add_request("B", ids[1], MAX_NEW_TOKENS)
    req_b = s.get_request("B")
    assert req_b is not None
    out_b = drive(s, engine, max_prefill=16, max_seqs=4)
    assert req_b.cached_tokens == (shared // BLOCK_SIZE) * BLOCK_SIZE, "B must reuse A's shared blocks"

    for name, prompt, got in (("A", ids[0], out_a["A"]), ("B", ids[1], out_b["B"])):
        want, want_logits = baseline(model, prompt)
        assert_matches(name, got, engine.logits[name], want, want_logits)

    cleanup(store, engine, s)


def test_concurrent_with_preemption_matches(gpt2: tuple[Any, Any]) -> None:
    """Three requests in a pool too small for all of them: preemption,
    eviction and re-matching all happen, and every output still matches."""
    tok, model = gpt2
    ids = [tok.encode(p) for p in PROMPTS]
    need = max(-(-(len(p) + MAX_NEW_TOKENS) // BLOCK_SIZE) for p in ids)

    s, engine, store = make_run(model, num_blocks=need + 1, chunk_size=32)
    for i, p in enumerate(ids):
        s.add_request(f"R{i}", p, MAX_NEW_TOKENS)
    outputs = drive(s, engine, max_prefill=64, max_seqs=3)

    assert s.stats().preemptions > 0, "pool should be too small to avoid preemption"
    assert s.stats().cached_prompt_tokens > 0
    for i, p in enumerate(ids):
        want, want_logits = baseline(model, p)
        assert_matches(f"R{i}", outputs[f"R{i}"], engine.logits[f"R{i}"][-MAX_NEW_TOKENS:], want, want_logits)

    s.tree.evict(s.block_manager.num_blocks)
    assert s.block_manager.num_free == s.block_manager.num_blocks
    cleanup(store, engine, s)
