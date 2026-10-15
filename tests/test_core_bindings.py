"""Tests for the pybind11 boundary: types, errors, lifetimes."""

from __future__ import annotations

import gc
import random

import numpy as np
import pytest

from prefixserve import BLOCK_SIZE, BlockManager, CoreError, RadixTree


def tokens(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


def prefill(bm: BlockManager, tree: RadixTree, toks: list[int]) -> list[int]:
    """One request's prefill: reuse cached blocks, allocate the rest, cache them."""
    _, blocks = tree.match_prefix(toks)
    needed = -(-len(toks) // BLOCK_SIZE)
    more = bm.allocate_many(needed - len(blocks))
    assert more is not None
    blocks += more
    tree.insert(toks, blocks)
    return blocks


def test_block_size() -> None:
    assert BLOCK_SIZE == 16


def test_allocate_and_free() -> None:
    bm = BlockManager(2)
    a, b = bm.allocate(), bm.allocate()
    assert (a, b) == (0, 1)
    assert bm.allocate() is None
    bm.decref(a)
    assert bm.num_free == 1
    bm.check_invariants()


def test_double_free_raises() -> None:
    bm = BlockManager(2)
    a = bm.allocate()
    assert a is not None
    bm.decref(a)
    with pytest.raises(CoreError, match="double free"):
        bm.decref(a)
    assert isinstance(CoreError("x"), RuntimeError)


def test_bad_ids_raise() -> None:
    bm = BlockManager(2)
    with pytest.raises(CoreError):
        bm.incref(5)
    with pytest.raises(CoreError):
        bm.ref_count(-1)
    with pytest.raises(ValueError):
        BlockManager(0)


def test_allocate_many_is_all_or_nothing() -> None:
    bm = BlockManager(4)
    assert bm.allocate_many(5) is None
    assert bm.num_free == 4  # nothing taken
    got = bm.allocate_many(4)
    assert got == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        bm.allocate_many(-1)


def test_decref_many() -> None:
    bm = BlockManager(4)
    got = bm.allocate_many(3)
    assert got is not None
    bm.decref_many(got)
    assert bm.num_free == 4
    with pytest.raises(CoreError, match="index 0"):
        bm.decref_many([0])


def test_match_insert_round_trip_list_and_numpy() -> None:
    bm = BlockManager(16)
    tree = RadixTree(bm)
    toks = tokens(40)
    held = prefill(bm, tree, toks)
    assert tree.num_cached_blocks == 2

    for inp in (toks, np.asarray(toks, dtype=np.int32), np.asarray(toks, dtype=np.int64)):
        matched, blocks = tree.match_prefix(inp)
        assert matched == 32
        assert blocks == held[:2]
        bm.decref_many(blocks)

    bm.decref_many(held)
    tree.check_invariants()
    bm.check_invariants()


def test_returned_list_is_a_copy() -> None:
    bm = BlockManager(16)
    tree = RadixTree(bm)
    held = prefill(bm, tree, tokens(32))
    _, first = tree.match_prefix(tokens(32))
    first[0] = 999
    _, second = tree.match_prefix(tokens(32))  # reuses the C++ scratch buffer
    assert second == held[:2]
    assert first[0] == 999  # first list didn't change under us
    bm.decref_many(held + [held[0], held[1]] * 2)


def test_rejects_2d_input() -> None:
    tree = RadixTree(BlockManager(4))
    with pytest.raises(ValueError, match="1-D"):
        tree.match_prefix(np.zeros((2, 16), dtype=np.int32))


def test_insert_error_is_atomic() -> None:
    bm = BlockManager(8)
    tree = RadixTree(bm)
    held = prefill(bm, tree, tokens(16))
    with pytest.raises(CoreError, match="already owned"):
        tree.insert(tokens(32, 500), [held[0], held[0]])
    assert tree.num_cached_blocks == 1
    tree.check_invariants()


def test_tree_keeps_block_manager_alive() -> None:
    bm = BlockManager(8)
    tree = RadixTree(bm)
    del bm
    gc.collect()
    # Would be a use-after-free without keep_alive.
    matched, blocks = tree.match_prefix(tokens(32))
    assert (matched, blocks) == (0, [])
    tree.check_invariants()


def test_evict_negative_raises() -> None:
    with pytest.raises(ValueError):
        RadixTree(BlockManager(4)).evict(-1)


def test_random_ops_never_leak() -> None:
    rng = random.Random(7)
    bm = BlockManager(64)
    tree = RadixTree(bm)
    prefixes = [tokens(16 * (i + 1), 1000 * i) for i in range(3)]
    live: list[list[int]] = []

    for _ in range(2000):
        op = rng.random()
        if op < 0.5:
            toks = rng.choice(prefixes) + [rng.randrange(3) for _ in range(rng.randrange(40))]
            _, blocks = tree.match_prefix(toks)
            needed = -(-len(toks) // BLOCK_SIZE) - len(blocks)
            more = bm.allocate_many(needed)
            if more is None:
                tree.evict(needed)
                more = bm.allocate_many(needed)
            if more is None:
                bm.decref_many(blocks)
                continue
            blocks += more
            tree.insert(toks, blocks)
            live.append(blocks)
        elif op < 0.9 and live:
            bm.decref_many(live.pop(rng.randrange(len(live))))
        else:
            tree.evict(rng.randrange(8))

    for blocks in live:
        bm.decref_many(blocks)
    tree.evict(64)
    assert bm.num_free == 64
    tree.check_invariants()
    bm.check_invariants()
