"""Production tier: real Llama in bf16 through FlashInfer vs HF LlamaForCausalLM.

Exact token match over long generations isn't a fair bar in bf16: a
different batch mix changes the reduction order, and greedy outputs can
split after a few dozen tokens with both sides "correct". So:

1. Free-running greedy: the first 16 tokens must match HF exactly.
2. Teacher-forced: feed both models HF's 256 generated tokens. At every
   step our argmax must agree with HF's > 99% of the time, and our logits
   must stay close (cosine similarity). A real bug (wrong block, wrong
   position) drops agreement to near 0, not to 98%.

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
from prefixserve.llama_engine import EngineConfig, LlamaEngine  # noqa: E402

GEN = 256
EXACT_PREFIX = 16
MIN_TOP1_AGREEMENT = 0.99
MIN_COSINE = 0.99

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


def prompts(tok: Any) -> list[list[int]]:
    return [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            add_generation_prompt=True, tokenize=True,
        )
        for q in QUESTIONS
    ]


@pytest.fixture(scope="module")
def reference() -> list[tuple[list[int], list[int], Any]]:
    """Per prompt: (prompt ids, HF greedy tokens, HF teacher-forced logits [GEN, vocab])."""
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.LlamaForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    out = []
    with torch.inference_mode():
        for p in prompts(tok):
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


def test_first_tokens_exact(reference: list[tuple[list[int], list[int], Any]]) -> None:
    ours, _ = run_ours(reference, forced=False)
    for i, (_, want, _) in enumerate(reference):
        assert ours[f"R{i}"][:EXACT_PREFIX] == want[:EXACT_PREFIX], f"R{i}: first {EXACT_PREFIX} tokens differ"


def test_teacher_forced_agreement(reference: list[tuple[list[int], list[int], Any]]) -> None:
    ours, logits = run_ours(reference, forced=True)
    agree = total = 0
    worst_cos = 1.0
    for i, (_, _, want_logits) in enumerate(reference):
        got = torch.stack(logits[f"R{i}"])
        assert got.shape == want_logits.shape
        agree += int((got.argmax(-1) == want_logits.argmax(-1)).sum())
        total += got.shape[0]
        worst_cos = min(worst_cos, float(torch.nn.functional.cosine_similarity(got, want_logits, dim=-1).min()))
    rate = agree / total
    print(f"top-1 agreement {rate:.4f} over {total} steps, worst cosine {worst_cos:.5f}")
    assert rate >= MIN_TOP1_AGREEMENT
    assert worst_cos >= MIN_COSINE
