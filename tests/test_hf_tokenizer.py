"""BPE streaming detokenizer. Uses GPT-2's byte-level BPE (same family as
Llama-3's), because it's ungated and usually already cached."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from prefixserve.hf_tokenizer import REPLACEMENT_CHAR, HFTokenizer  # noqa: E402

TEXTS = [
    "plain ascii, nothing special.",
    "Héllo wörld 👋🏽 — 日本語のテキスト. Done!",
    "🧑‍🚀👩🏾‍💻🏳️‍🌈",  # multi-codepoint emoji: several tokens per visible character
    "   leading spaces and\nnewlines\n\n",
]


@pytest.fixture(scope="module")
def tok() -> HFTokenizer:
    try:
        return HFTokenizer("gpt2")
    except OSError as e:
        pytest.skip(f"gpt2 tokenizer not available: {e}")


@pytest.mark.parametrize("text", TEXTS)
def test_stream_equals_full_decode(tok: HFTokenizer, text: str) -> None:
    ids = tok.encode_prompt(text)
    d = tok.stream_decoder()
    pieces = [d.push(i) for i in ids]
    pieces.append(d.flush())
    assert "".join(pieces) == tok.decode(ids) == text
    # Never emit a half character. A client can't take it back.
    assert not any(REPLACEMENT_CHAR in p for p in pieces)


def test_holds_back_partial_character(tok: HFTokenizer) -> None:
    ids = tok.encode_prompt("👋")
    assert len(ids) > 1, "test needs an emoji that spans several tokens"
    d = tok.stream_decoder()
    assert [d.push(i) for i in ids[:-1]] == [""] * (len(ids) - 1)
    assert d.push(ids[-1]) == "👋"


def test_window_stays_small(tok: HFTokenizer) -> None:
    """Old ids are dropped, so per-token cost doesn't grow with output length."""
    d = tok.stream_decoder()
    for i in tok.encode_prompt("word " * 500):
        d.push(i)
    assert len(d._ids) <= 4


def test_flush_emits_unfinished_bytes(tok: HFTokenizer) -> None:
    """If generation stops mid-character, flush still returns what's there."""
    ids = tok.encode_prompt("👋")[:-1]
    d = tok.stream_decoder()
    assert "".join(d.push(i) for i in ids) == ""
    assert d.flush() != ""
