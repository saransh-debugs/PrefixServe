"""Exact tier: the Llama engine, driven by the real scheduler, must give the
same tokens and logits as HF LlamaForCausalLM recomputing everything.

Runs in fp32 with TorchRefAttention, because FlashInfer has no fp32 kernels.
The ref backend reads the same paged pool and the same 4 paging arrays as
FlashInfer, so a wrong block table, position or last_page_len shows up here.
(tests/gpu/test_flashinfer_vs_ref.py then checks FlashInfer against the ref
backend.)

Model:
- Default: a tiny random Llama (2 layers, GQA, llama3 rope scaling) built
  by HF and saved as safetensors. No download, runs on a laptop CPU.
- PREFIXSERVE_TEST_MODEL=/path/to/Llama-3.2-1B-Instruct runs the same
  tests on real weights (on GPU if there is one).

What's checked:
- Token ids match exactly.
- Logits within 1e-4 relative to their max.
- The engine never writes into a block the cache shares.
- Both injected bugs (relative RoPE, wrong last_page_len) are caught.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from prefixserve import BLOCK_SIZE, Scheduler  # noqa: E402
from prefixserve.llama_engine import EngineConfig, EngineError, InjectedBug, LlamaEngine  # noqa: E402
from prefixserve.scheduler import Batch  # noqa: E402

MAX_NEW_TOKENS = 12
REL_TOL = 1e-4
REAL_MODEL = os.environ.get("PREFIXSERVE_TEST_MODEL")
DEVICE = "cuda" if REAL_MODEL and torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- fixtures


def _tiny_config(tied: bool) -> Any:
    return transformers.LlamaConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,  # GQA group of 4, like 8B (32/8)
        head_dim=16,
        max_position_embeddings=4096,
        rope_theta=500000.0,
        # Small original context so all three llama3 bands are non-empty
        # for head_dim 16. With 8192 most pairs fall in one band.
        rope_scaling={
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 64,
        },
        rms_norm_eps=1e-5,
        tie_word_embeddings=tied,
        # Bigger than HF's 0.02 so logits are well separated. With near-ties,
        # a 1e-7 difference could flip argmax and fail for no real reason.
        initializer_range=0.15,
        bos_token_id=1,
        eos_token_id=2,
    )


@pytest.fixture(scope="module", params=["untied", "tied"] if not REAL_MODEL else ["real"])
def model_dir(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    if REAL_MODEL:
        yield Path(REAL_MODEL)
        return
    torch.manual_seed(0)
    hf = transformers.LlamaForCausalLM(_tiny_config(tied=request.param == "tied"))
    path = tmp_path_factory.mktemp(f"tiny-llama-{request.param}")
    hf.save_pretrained(path, safe_serialization=True)
    del hf
    yield path


@pytest.fixture(scope="module")
def hf_model(model_dir: Path) -> Iterator[Any]:
    m = transformers.LlamaForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(DEVICE).eval()
    with torch.inference_mode():
        yield m
    del m
    gc.collect()


@pytest.fixture(scope="module")
def prompts(model_dir: Path) -> list[list[int]]:
    """3 prompts sharing a 3-block 'system prompt', then different questions."""
    g = torch.Generator().manual_seed(1)
    vocab = 512 if not REAL_MODEL else 32000  # stay in the plain-token range
    system = torch.randint(3, vocab, (3 * BLOCK_SIZE + 5,), generator=g).tolist()
    return [system + torch.randint(3, vocab, (n,), generator=g).tolist() for n in (9, 21, 40)]


def baseline(hf: Any, prompt: list[int]) -> tuple[list[int], list[Any]]:
    """Greedy, no KV cache: every step recomputes the whole sequence."""
    ids = list(prompt)
    tokens, logits = [], []
    for _ in range(MAX_NEW_TOKENS):
        last = hf(torch.tensor([ids], device=DEVICE), use_cache=False).logits[0, -1].float().cpu()
        logits.append(last)
        tokens.append(int(last.argmax()))
        ids.append(tokens[-1])
    return tokens, logits


@pytest.fixture(scope="module")
def baselines(hf_model: Any, prompts: list[list[int]]) -> list[tuple[list[int], list[Any]]]:
    return [baseline(hf_model, p) for p in prompts]


# ---------------------------------------------------------------- harness


class Run:
    """Scheduler + engine on one KV pool, plus per-request logits."""

    def __init__(self, model_dir: Path, num_blocks: int, chunk: int, max_seqs: int, bug: InjectedBug | None = None) -> None:
        self.max_prefill = chunk
        self.max_seqs = max_seqs
        self.s = Scheduler(num_blocks, chunk_size=chunk)
        self.engine = LlamaEngine(EngineConfig(
            model_path=model_dir, dtype=torch.float32, device=DEVICE, attention="torch",
            max_num_seqs=max_seqs, max_prefill_tokens=chunk, max_seq_len=1024,
            num_blocks=num_blocks, inject_bug=bug,
        ))
        self.engine.capture_logits = True
        self.logits: dict[str, list[Any]] = {}

    def step(self, batch: Batch) -> dict[str, int]:
        # Blocks this step writes into must belong to one request only.
        # Writing into a cached (shared) block corrupts every other user of it.
        for w in batch.work:
            for p in range(w.start, w.end):
                b = w.req.block_table[p // BLOCK_SIZE]
                assert not self.s.tree.is_cached(b), f"{w.req.req_id} would overwrite shared block {b}"
        out = self.engine.step(batch)
        for rid, lg in self.engine.last_logits.items():
            self.logits.setdefault(rid, []).append(lg)
        self.engine.last_logits = {}
        return out

    def drive(self) -> dict[str, list[int]]:
        outputs: dict[str, list[int]] = {}
        for _ in range(10_000):
            if not self.s.has_work():
                return outputs
            batch = self.s.schedule_step(self.max_prefill, self.max_seqs)
            result = self.s.commit(batch, self.step(batch))
            for rid, tok in result.new_tokens.items():
                outputs.setdefault(rid, []).append(tok)
        raise AssertionError("scheduler did not finish")


def assert_matches(name: str, got: list[int], got_logits: list[Any], want: list[int], want_logits: list[Any]) -> None:
    for step, (a, b) in enumerate(zip(got_logits, want_logits, strict=True)):
        scale = float(b.abs().max())
        diff = float((a - b).abs().max())
        assert diff <= REL_TOL * scale, f"{name} step {step}: logit diff {diff:.2e} > {REL_TOL:g} x {scale:.2f}"
    assert got == want, f"{name}: tokens differ from full recompute"


# ---------------------------------------------------------------- tests


def test_single_request_chunked(model_dir: Path, prompts: list[list[int]], baselines: list[Any]) -> None:
    """One request, prefill in 16-token chunks, then decode."""
    r = Run(model_dir, num_blocks=32, chunk=16, max_seqs=4)
    r.s.add_request("A", prompts[0], MAX_NEW_TOKENS)
    out = r.drive()
    assert_matches("A", out["A"], r.logits["A"], *baselines[0])


def test_prefix_reuse(model_dir: Path, prompts: list[list[int]], baselines: list[Any]) -> None:
    """B reuses A's cached system-prompt blocks. B's output must not change."""
    r = Run(model_dir, num_blocks=32, chunk=16, max_seqs=4)
    r.s.add_request("A", prompts[0], MAX_NEW_TOKENS)
    r.drive()
    r.s.add_request("B", prompts[1], MAX_NEW_TOKENS)
    req_b = r.s.get_request("B")
    assert req_b is not None
    out = r.drive()
    assert req_b.cached_tokens == 3 * BLOCK_SIZE, "B must reuse the 3 shared blocks"
    assert_matches("B", out["B"], r.logits["B"], *baselines[1])


def test_mixed_batch(model_dir: Path, prompts: list[list[int]], baselines: list[Any]) -> None:
    """All three at once: chunked prefills and decodes share every step, and
    later requests hit the cache while earlier ones are still decoding."""
    r = Run(model_dir, num_blocks=64, chunk=24, max_seqs=3)
    for i, p in enumerate(prompts):
        r.s.add_request(f"R{i}", p, MAX_NEW_TOKENS)
    out = r.drive()
    for i in range(3):
        assert_matches(f"R{i}", out[f"R{i}"], r.logits[f"R{i}"], *baselines[i])


def test_preemption(model_dir: Path, prompts: list[list[int]], baselines: list[Any]) -> None:
    """Pool too small for all three: preemption, eviction, re-matching."""
    need = max(-(-(len(p) + MAX_NEW_TOKENS) // BLOCK_SIZE) for p in prompts)
    r = Run(model_dir, num_blocks=need + 1, chunk=32, max_seqs=3)
    for i, p in enumerate(prompts):
        r.s.add_request(f"R{i}", p, MAX_NEW_TOKENS)
    out = r.drive()
    assert r.s.stats().preemptions > 0, "pool should be too small to avoid preemption"
    for i in range(3):
        assert_matches(f"R{i}", out[f"R{i}"], r.logits[f"R{i}"], *baselines[i])
    r.s.tree.evict(r.s.block_manager.num_blocks)
    assert r.s.block_manager.num_free == r.s.block_manager.num_blocks


@pytest.mark.parametrize("bug", ["rope_relative", "last_page_len"])
def test_injected_bug_is_caught(bug: InjectedBug, model_dir: Path, prompts: list[list[int]], baselines: list[Any]) -> None:
    """Each bug must break the mixed-batch scenario. If it doesn't, the
    tests above are too weak to trust.

    Caught either as a mismatch, or (for last_page_len) by the engine's
    paging check before anything runs (EngineError).
    """
    r = Run(model_dir, num_blocks=64, chunk=24, max_seqs=3, bug=bug)
    for i, p in enumerate(prompts):
        r.s.add_request(f"R{i}", p, MAX_NEW_TOKENS)
    with pytest.raises((AssertionError, EngineError)):
        out = r.drive()
        for i in range(3):
            assert_matches(f"R{i}", out[f"R{i}"], r.logits[f"R{i}"], *baselines[i])
