"""HF tokenizer (Llama-3 BPE) behind the server's TextTokenizer interface.

Runs on the event loop thread. A fast (Rust) tokenizer encodes a 1k-token
prompt in well under a millisecond, which is fine there. A slow (Python)
tokenizer would not be, so we refuse to load one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from prefixserve.tokenizer import ChatMessage

REPLACEMENT_CHAR = "�"


class HFTokenizer:
    def __init__(self, path: str | Path) -> None:
        from transformers import AutoTokenizer

        self._tok: Any = AutoTokenizer.from_pretrained(str(path))
        if not self._tok.is_fast:
            raise ValueError(f"{path}: only fast tokenizers are supported (a slow one blocks the event loop)")

    def encode_prompt(self, text: str) -> list[int]:
        # add_special_tokens adds <|begin_of_text|>, same as vLLM and SGLang
        # do for /v1/completions. Without it our prompts would be one token
        # shorter than theirs and the benchmark wouldn't match.
        ids: list[int] = self._tok.encode(text, add_special_tokens=True)
        return ids

    def encode_chat(self, messages: Sequence[ChatMessage]) -> list[int]:
        """System prompt comes first in the template, so every request from
        one tenant starts with the same tokens. That's what the cache reuses."""
        if self._tok.chat_template is None:
            raise ValueError("this tokenizer has no chat template")
        out = self._tok.apply_chat_template(list(messages), add_generation_prompt=True, tokenize=True)
        ids: list[int] = out["input_ids"] if isinstance(out, dict) or hasattr(out, "keys") else out
        return ids

    def decode(self, tokens: Sequence[int]) -> str:
        text: str = self._tok.decode(list(tokens), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text

    def stream_decoder(self) -> BPEStreamDecoder:
        return BPEStreamDecoder(self)


class BPEStreamDecoder:
    """Incremental detokenization for BPE.

    Why the byte-level UTF-8 decoder doesn't carry over: a BPE token is not
    one byte. One token can end halfway through a character (an emoji split
    over 2 tokens), and decode() then returns U+FFFD for the partial bytes.
    Emitting that would print garbage the client can't take back.

    How it works: keep a small window of recent ids. Decode the window with
    and without the newest tokens; the difference is the new text. If it
    ends in U+FFFD, the character isn't finished: emit nothing, wait for
    more. Decoding only the window keeps each step O(window), not O(output).
    """

    def __init__(self, tok: HFTokenizer) -> None:
        self._tok = tok
        self._ids: list[int] = []
        self._prefix = 0  # window start; ids before it are already emitted and dropped
        self._read = 0  # text for ids[:read] has been emitted

    def push(self, token: int) -> str:
        self._ids.append(token)
        return self._emit(final=False)

    def flush(self) -> str:
        return self._emit(final=True)

    def _emit(self, final: bool) -> str:
        before = self._tok.decode(self._ids[self._prefix : self._read])
        after = self._tok.decode(self._ids[self._prefix :])
        if len(after) <= len(before) or (after.endswith(REPLACEMENT_CHAR) and not final):
            return ""
        # Keep one emitted token of context in the window: some tokenizers
        # decode a token differently depending on what precedes it.
        self._prefix, self._read = self._read, len(self._ids)
        # Drop ids that left the window, so memory stays flat on long outputs.
        if self._prefix > 0:
            del self._ids[: self._prefix]
            self._read -= self._prefix
            self._prefix = 0
        return after[len(before) :]
