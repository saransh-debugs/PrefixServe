from __future__ import annotations

from prefixserve.tokenizer import (
    BOS_ID,
    EOS_ID,
    StreamDecoder,
    apply_chat_template,
    decode,
    encode,
)


def test_round_trip_unicode() -> None:
    text = "héllo 👋 世界"
    assert decode(encode(text)) == text


def test_stream_decoder_holds_partial_characters() -> None:
    # 👋 is 4 UTF-8 bytes = 4 tokens. Decoding them one by one must not
    # print broken characters in between.
    d = StreamDecoder()
    pieces = [d.push(t) for t in encode("a👋b")]
    assert pieces == ["a", "", "", "", "👋", "b"]
    assert d.flush() == ""


def test_special_tokens_produce_no_text() -> None:
    d = StreamDecoder()
    assert d.push(EOS_ID) == ""
    assert decode([BOS_ID, *encode("x"), EOS_ID]) == "x"


def test_same_system_prompt_gives_same_prefix() -> None:
    sys_msg = {"role": "system", "content": "You are a support bot."}
    a = apply_chat_template([sys_msg, {"role": "user", "content": "hi"}])  # type: ignore[list-item]
    b = apply_chat_template([sys_msg, {"role": "user", "content": "refund please"}])  # type: ignore[list-item]
    shared = len(encode("<|system|>\nYou are a support bot.\n<|user|>\n")) + 1
    assert a[:shared] == b[:shared]
    assert a[0] == BOS_ID
