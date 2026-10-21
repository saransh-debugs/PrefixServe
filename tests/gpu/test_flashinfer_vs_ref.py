"""Kernel tier: FlashInferBackend vs TorchRefAttention on the same paged inputs.

The exact tier proves the engine + ref backend match HF. This proves
FlashInfer matches the ref backend. Together: FlashInfer path matches HF.

Inputs cover what the scheduler produces: decodes (1 new token), prefill
chunks on top of cached KV, partial last pages, full last pages, pages
shared between requests, block tables in random order, GQA.

GPU only. Skipped without CUDA + flashinfer.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("flashinfer")

from prefixserve import BLOCK_SIZE  # noqa: E402
from prefixserve.model.attention import AttnMetadata, FlashInferBackend, TorchRefAttention  # noqa: E402

DEV = torch.device("cuda")
DTYPE = torch.bfloat16


def make_meta(seqs: list[tuple[int, list[int], int]]) -> AttnMetadata:
    """seqs: (new tokens n, block table, kv_len after this step)."""
    qo, kvp, idx, last = [0], [0], [], []
    for n, table, kv_len in seqs:
        pages = -(-kv_len // BLOCK_SIZE)
        assert len(table) >= pages and kv_len >= n
        qo.append(qo[-1] + n)
        idx += table[:pages]
        kvp.append(len(idx))
        last.append(kv_len - (pages - 1) * BLOCK_SIZE)
    host = [torch.tensor(a, dtype=torch.int32) for a in (qo, kvp, idx, last)]
    dev = [h.to(DEV) for h in host]
    return AttnMetadata(
        num_tokens=qo[-1], num_seqs=len(seqs),
        qo_indptr=dev[0], kv_indptr=dev[1], kv_indices=dev[2], kv_last_page_len=dev[3],
        qo_indptr_host=host[0], kv_indptr_host=host[1], kv_indices_host=host[2], kv_last_page_len_host=host[3],
    )


@pytest.mark.parametrize("heads,kv_heads,head_dim", [(32, 8, 128), (32, 8, 64), (8, 8, 128)])
def test_flashinfer_matches_ref(heads: int, kv_heads: int, head_dim: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(0)
    num_blocks = 256
    perm = torch.randperm(num_blocks, generator=g).tolist()
    shared = perm[:4]  # a 4-block "system prompt" two requests share
    rest = iter(perm[4:])

    def table(n_blocks: int, prefix: list[int] | None = None) -> list[int]:
        p = list(prefix or [])
        return p + [next(rest) for _ in range(n_blocks - len(p))]

    seqs = [
        (1, table(3), 33),  # decode, last page has 1 slot
        (1, table(2), 32),  # decode, last page exactly full
        (64, table(8, shared), 64 + 4 * BLOCK_SIZE),  # prefill chunk on top of shared cached prefix
        (37, table(10, shared), 37 + 4 * BLOCK_SIZE + 50),  # later chunk of a longer prompt
        (100, table(7), 100),  # fresh prefill, no cache
        (1, table(9, shared), 129),  # decode on top of shared prefix
    ]
    meta = make_meta(seqs)
    t = meta.num_tokens

    pool = torch.randn(num_blocks, 2, BLOCK_SIZE, kv_heads, head_dim, generator=g).to(DEV, DTYPE)
    q = torch.randn(t, heads, head_dim, generator=g).to(DEV, DTYPE)
    k = torch.randn(t, kv_heads, head_dim, generator=g).to(DEV, DTYPE)
    v = torch.randn(t, kv_heads, head_dim, generator=g).to(DEV, DTYPE)

    pool_ref, pool_fi = pool.clone(), pool.clone()
    ref = TorchRefAttention(heads, kv_heads, head_dim)
    ref.plan(meta)
    out_ref = ref.run(q, k, v, pool_ref)
    fi = FlashInferBackend(heads, kv_heads, head_dim, DTYPE, DEV)
    fi.plan(meta)
    out_fi = fi.run(q, k, v, pool_fi)
    torch.cuda.synchronize()

    # Appends are copies, so the pools must match bit for bit.
    assert torch.equal(pool_ref, pool_fi), "FlashInfer wrote KV to different slots than the ref backend"
    # bf16 outputs from two different reduction orders: compare loosely, but
    # far tighter than any indexing bug (those give O(1) errors).
    torch.testing.assert_close(out_fi.float(), out_ref.float(), atol=2e-2, rtol=2e-2)
