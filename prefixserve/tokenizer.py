"""Byte-level tokenizer and chat template. No dependencies.

Token ids:
    0..2           special tokens (PAD, BOS, EOS)
    3..258         one token per UTF-8 byte (byte + 3)

Real tokenizers (BPE) produce fewer tokens, but the scheduler doesn't care
what a token means, only how many there are and which prefixes repeat.
Byte-level keeps that honest and needs nothing installed.
"""

from __future__ import annotations

import codecs
from collections.abc import Iterable, Sequence
from typing import Literal, Protocol, TypedDict

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
BYTE_OFFSET = 3
VOCAB_SIZE = 256 + BYTE_OFFSET

Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


def encode(text: str) -> list[int]:
    return [b + BYTE_OFFSET for b in text.encode("utf-8")]


def decode(tokens: Iterable[int]) -> str:
    data = bytes(t - BYTE_OFFSET for t in tokens if BYTE_OFFSET <= t < VOCAB_SIZE)
    return data.decode("utf-8", errors="replace")


def apply_chat_template(messages: Sequence[ChatMessage]) -> list[int]:
    """BOS + each message + the assistant header.

    The system message comes first and is rendered the same way every time,
    so every request from the same tenant starts with the same tokens. That
    shared start is exactly what the prefix cache reuses.
    """
    parts = [f"<|{m['role']}|>\n{m['content']}\n" for m in messages]
    parts.append("<|assistant|>\n")
    return [BOS_ID, *encode("".join(parts))]


class StreamDecoder:
    """Turns tokens into text one at a time, for streaming.

    Why not decode each token on its own: one character can be several UTF-8
    bytes. Decoding them one by one would print broken characters. The
    incremental decoder holds a partial character until it's complete.
    """

    def __init__(self) -> None:
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token: int) -> str:
        if not BYTE_OFFSET <= token < VOCAB_SIZE:
            return ""  # special tokens produce no text
        return self._dec.decode(bytes([token - BYTE_OFFSET]))

    def flush(self) -> str:
        return self._dec.decode(b"", final=True)


# ------------------------------------------------- what the server needs


class Detokenizer(Protocol):
    def push(self, token: int) -> str:
        """Text that became final with this token. Often "" mid-character."""
        ...

    def flush(self) -> str: ...


class TextTokenizer(Protocol):
    def encode_prompt(self, text: str) -> list[int]:
        """Raw /v1/completions prompt -> ids, BOS included."""
        ...

    def encode_chat(self, messages: Sequence[ChatMessage]) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...

    def stream_decoder(self) -> Detokenizer: ...


class ByteTokenizer:
    """The dependency-free tokenizer above, behind the TextTokenizer interface."""

    def encode_prompt(self, text: str) -> list[int]:
        return [BOS_ID, *encode(text)]

    def encode_chat(self, messages: Sequence[ChatMessage]) -> list[int]:
        return apply_chat_template(messages)

    def decode(self, tokens: Sequence[int]) -> str:
        return decode(tokens)

    def stream_decoder(self) -> StreamDecoder:
        return StreamDecoder()
