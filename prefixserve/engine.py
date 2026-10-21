"""What the server needs from an engine. MockEngine and LlamaEngine both fit."""

from __future__ import annotations

from typing import Protocol

from prefixserve.scheduler import Batch


class Engine(Protocol):
    @property
    def num_blocks(self) -> int | None:
        """KV blocks the engine allocated. None = it has no real pool (mock),
        use the server config. When set, the scheduler MUST use exactly this
        many, or it hands out block ids past the end of the GPU pool."""
        ...

    @property
    def max_seq_len(self) -> int | None:
        """Longest prompt + output the engine can run. None = no limit."""
        ...

    @property
    def stop_token_ids(self) -> frozenset[int]: ...

    def step(self, batch: Batch) -> dict[str, int]:
        """Runs one batch. BLOCKING: the server calls it from a worker thread,
        never on the event loop. Returns req_id -> token for items that sample."""
        ...
