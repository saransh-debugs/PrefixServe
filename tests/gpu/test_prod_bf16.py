"""Production tier: real Llama in bf16 through FlashInfer vs HF LlamaForCausalLM.

"First N tokens match exactly" is not a fair bar in bf16. Logits here are
~20-30, where bf16 can only step by 0.125. When HF's top two choices are
1-3 of those steps apart, either kernel can pick either one. Measured on
8B: free-run splits at token 9 (top-2 gap 0.375) and 27 (gap 0.125), with
100% teacher-forced agreement everywhere else. So:

1. Free-running greedy: we may split from HF only at a near-tie
   (HF top-1 minus top-2 < NEAR_TIE). Never on a clear decision.
2. Teacher-forced: feed both models HF's tokens. Our argmax must agree
   with HF's >= 99% of the time and logits must stay close (cosine).
   A real bug (wrong block, wrong position) drops agreement to near 0.

Both are scored only up to where HF's answer ends (its first stop token).
HF is forced to generate past that (min_new_tokens), into filler text where
the logits are nearly flat and tiny differences look huge (cosine 0.85).

All prompts run at once, so batching, prefix sharing and chunked prefill
are on the path, not just single-request decode.

Needs: GPU, flashinfer, PREFIXSERVE_PROD_MODEL=/path/to/Llama-3.1-8B-Instruct.
HF runs first, its results go to CPU, then it's freed before our engine
loads. Both at once would need 32 GB of weights.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
MODEL = os.environ.get("PREFIXSERVE_PROD_MODEL")
if not MODEL or not torch.cuda.is_available():
    pytest.skip("needs a GPU and PREFIXSERVE_PROD_MODEL", allow_module_level=True)
pytest.importorskip("flashinfer")

from prefixserve import Scheduler  # noqa: E402
from prefixserve.hf_tokenizer import HFTokenizer  # noqa: E402
from prefixserve.llama_engine import EngineConfig, LlamaEngine  # noqa: E402

GEN = 256
MIN_TOP1_AGREEMENT = 0.99
MIN_COSINE = 0.99
NEAR_TIE = 0.5  # logits; ~2-4 bf16 steps at this magnitude

SYSTEM = (
    "You are the support assistant for Blue Horizon Travel. You help with bookings, "
    "cancellations, baggage rules and seat upgrades. Be polite and brief. Never invent "
    "prices, dates or flight numbers. If you do not know, say so and offer a human agent."
)
QUESTIONS = [
    "Can I bring a bike on the plane?",
    "My flight was cancelled. What now?",
    "Write a short poem about lost luggage.",
    "Explain how refunds work, step by step, in detail.",
]


def prompts() -> list[list[int]]:
    # Through our tokenizer, same as the server. (Raw apply_chat_template
    # returns a list in transformers 4 and a BatchEncoding in 5.)
    tok = HFTokenizer(MODEL or "")
    return [tok.encode_chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]) for q in QUESTIONS]


def answer_len(want_logits: Any, stops: frozenset[int]) -> int:
    """Steps until HF's own argmax is a stop token: where its answer ends."""
    hits = [j for j, t in enumerate(want_logits.argmax(-1).tolist()) if t in stops]
    return hits[0] if hits else want_logits.shape[0]


@pytest.fixture(scope="module")
def stops() -> frozenset[int]:
    from prefixserve.model.config import LlamaConfig

    return LlamaConfig.from_dir(MODEL or "").stop_token_ids


@pytest.fixture(scope="module")
def reference() -> list[tuple[list[int], list[int], Any]]:
    """Per prompt: (prompt ids, HF greedy tokens, HF teacher-forced logits [GEN, vocab])."""
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.LlamaForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    out = []
    with torch.inference_mode():
        for p in prompts():
            ids = torch.tensor([p], device="cuda")
            gen = hf.generate(ids, max_new_tokens=GEN, min_new_tokens=GEN, do_sample=False, pad_token_id=tok.eos_token_id)
            toks = gen[0, len(p):].tolist()
            # Logits that predict each generated token, from one full pass.
            logits = hf(gen).logits[0, len(p) - 1 : -1].float().cpu()
            out.append((p, toks, logits))
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_ours(reference: list[tuple[list[int], list[int], Any]], forced: bool) -> tuple[dict[str, list[int]], dict[str, list[Any]]]:
    engine = LlamaEngine(EngineConfig(model_path=Path(MODEL or ""), max_seq_len=2048))
    engine.capture_logits = True
    s = Scheduler(engine.num_blocks, chunk_size=64, stop_token_ids=engine.stop_token_ids)
    for i, (p, _, _) in enumerate(reference):
        s.add_request(f"R{i}", p, GEN, ignore_eos=True)
    ours: dict[str, list[int]] = {}
    logits: dict[str, list[Any]] = {}
    while s.has_work():
        batch = s.schedule_step(64, 8)  # small budget: forces chunked prefill + mixed steps
        sampled = engine.step(batch)
        for rid, lg in engine.last_logits.items():
            logits.setdefault(rid, []).append(lg)
        for rid, t in sampled.items():
            ours.setdefault(rid, []).append(t)
        if forced:  # teacher forcing: continue with HF's token, not ours
            sampled = {rid: reference[int(rid[1:])][1][len(ours[rid]) - 1] for rid in sampled}
        s.commit(batch, sampled)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return ours, logits


def test_free_run_splits_only_at_near_ties(reference: list[tuple[list[int], list[int], Any]], stops: frozenset[int]) -> None:
    ours, _ = run_ours(reference, forced=False)
    for i, (_, want, want_logits) in enumerate(reference):
        n = answer_len(want_logits, stops)
        got = ours[f"R{i}"]
        split = next((j for j in range(n) if got[j] != want[j]), None)
        if split is None:
            print(f"R{i}: identical for all {n} answer tokens")
            continue
        top2 = want_logits[split].topk(2).values
        gap = float(top2[0] - top2[1])
        print(f"R{i}: split at token {split}/{n}, HF top-2 gap {gap:.3f}")
        assert gap < NEAR_TIE, f"R{i}: split at token {split} where HF was sure (gap {gap:.2f}): a bug, not rounding"


def test_teacher_forced_agreement(reference: list[tuple[list[int], list[int], Any]], stops: frozenset[int]) -> None:
    _, logits = run_ours(reference, forced=True)
    agree = total = 0
    worst_cos = 1.0
    for i, (_, _, want_logits) in enumerate(reference):
        got = torch.stack(logits[f"R{i}"])
        assert got.shape == want_logits.shape
        n = answer_len(want_logits, stops)
        got, want = got[:n], want_logits[:n]
        agree += int((got.argmax(-1) == want.argmax(-1)).sum())
        total += n
        worst_cos = min(worst_cos, float(torch.nn.functional.cosine_similarity(got, want, dim=-1).min()))
    rate = agree / total
    print(f"top-1 agreement {rate:.4f} over {total} answer steps, worst cosine {worst_cos:.5f}")
    assert total >= 200, "answers too short to mean anything"
    assert rate >= MIN_TOP1_AGREEMENT
    assert worst_cos >= MIN_COSINE
