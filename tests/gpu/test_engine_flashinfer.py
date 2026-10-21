"""Whole engine on FlashInfer vs whole engine on the ref backend, bf16, GPU.

The kernel tier tests FlashInferBackend with hand-built metadata. This one
runs it the way the server does: metadata from the engine's pinned buffer,
the scheduler mixing chunked prefills, decodes and cache hits, and the KV
pool sized by profiling free GPU memory. No real weights needed: a tiny
random Llama with real head sizes (FlashInfer only has kernels for some).
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("flashinfer")

from prefixserve import BLOCK_SIZE, Scheduler  # noqa: E402
from prefixserve.llama_engine import EngineConfig, EngineError, LlamaEngine  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_llama_exact import _tiny_config  # noqa: E402

GEN = 32


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg = _tiny_config(tied=False)
    cfg.hidden_size, cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads = 512, 128, 4, 1
    cfg.intermediate_size = 1024
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny-llama-hd128")
    transformers.LlamaForCausalLM(cfg).save_pretrained(path)
    return path


def run(
    model_dir: Path,
    attention: str,
    dtype: Any,
    num_blocks: int | None,
    forced: dict[str, list[int]] | None = None,
    bug: Any = None,
) -> tuple[dict[str, list[int]], dict[str, Any], int]:
    """forced: teacher forcing. Continue with these tokens instead of our own,
    so several runs see identical inputs and their logits line up step by step."""
    engine = LlamaEngine(EngineConfig(
        model_path=model_dir, dtype=dtype, attention=attention,  # type: ignore[arg-type]
        max_num_seqs=8, max_prefill_tokens=48, max_seq_len=512, num_blocks=num_blocks, inject_bug=bug,
    ))
    engine.capture_logits = True
    s = Scheduler(engine.num_blocks, chunk_size=48)
    g = torch.Generator().manual_seed(3)
    system = torch.randint(3, 512, (5 * BLOCK_SIZE + 3,), generator=g).tolist()
    for i in range(6):
        q = torch.randint(3, 512, (7 + 13 * i,), generator=g).tolist()
        s.add_request(f"R{i}", system + q, GEN)
    out: dict[str, list[int]] = {}
    logits: dict[str, list[Any]] = {}
    while s.has_work():
        batch = s.schedule_step(48, 8)
        sampled = engine.step(batch)
        for rid, lg in engine.last_logits.items():
            logits.setdefault(rid, []).append(lg)
        for rid, t in sampled.items():
            out.setdefault(rid, []).append(t)
        if forced is not None:
            sampled = {rid: forced[rid][len(out[rid]) - 1] for rid in sampled}
        s.commit(batch, sampled)
    n = engine.num_blocks
    assert s.stats().cached_prompt_tokens > 0, "later requests should hit the shared prefix"
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return out, logits, n


def rel_err(got: list[Any], want: list[Any]) -> float:
    g, w = torch.stack(got), torch.stack(want)
    return float(((g - w).norm(dim=-1) / w.norm(dim=-1)).mean())


def test_flashinfer_engine_as_accurate_as_plain_bf16(model_dir: Path) -> None:
    """Why not "FlashInfer == ref within X": in bf16 both are off from the true
    (fp32) logits by a few %, and this random model amplifies that (bf16 and
    fp32 can disagree on the very first token). The meaningful question is
    whether FlashInfer adds error beyond bf16 rounding. A wiring bug (wrong
    page, wrong position) shows up as ~100% error, not a few %.
    """
    truth, truth_logits, _ = run(model_dir, "torch", torch.float32, num_blocks=256)
    _, fi_logits, profiled_blocks = run(model_dir, "flashinfer", torch.bfloat16, num_blocks=None, forced=truth)
    _, bf_logits, _ = run(model_dir, "torch", torch.bfloat16, num_blocks=256, forced=truth)
    # Profiling must leave a big pool on an otherwise empty GPU.
    assert profiled_blocks > 10_000

    for rid in truth:
        fi, bf = rel_err(fi_logits[rid], truth_logits[rid]), rel_err(bf_logits[rid], truth_logits[rid])
        print(f"{rid}: rel err vs fp32  FlashInfer {fi:.4f}  plain bf16 {bf:.4f}")
        assert fi < 0.15, f"{rid}: FlashInfer error {fi:.3f} is a bug, not rounding"
        assert fi <= 1.5 * bf, f"{rid}: FlashInfer error {fi:.4f} > 1.5x plain bf16 ({bf:.4f})"


@pytest.mark.parametrize("bug", ["rope_relative", "last_page_len"])
def test_injected_bug_shows_on_flashinfer_path(model_dir: Path, bug: str) -> None:
    """Caught either as error far above bf16 rounding, or by the engine's
    paging check (EngineError). What must never happen: FlashInfer gets the
    bad arrays and writes out of bounds (that crashed the CUDA context
    before _check_paging existed)."""
    truth, truth_logits, _ = run(model_dir, "torch", torch.float32, num_blocks=256)
    try:
        _, fi_logits, _ = run(model_dir, "flashinfer", torch.bfloat16, num_blocks=256, forced=truth, bug=bug)
    except EngineError as e:
        print(f"{bug}: refused by paging check ({e})")
        return
    worst = max(rel_err(fi_logits[rid], truth_logits[rid]) for rid in truth)
    print(f"{bug}: worst rel err {worst:.3f}")
    assert worst > 0.15
