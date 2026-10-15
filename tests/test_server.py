"""Server tests against a real uvicorn server on a random port.

Real sockets on purpose: a client disconnect here is a real TCP close, not a
mock, so the abort-on-disconnect path is actually exercised.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from prefixserve.mock_engine import CostModel, MockEngine
from prefixserve.scheduler import Batch
from prefixserve.server import AdmissionError, DoneEvent, EngineLoop, ServerConfig, TokenEvent, create_app

pytestmark = pytest.mark.anyio

FAST = CostModel(base_s=0.001, per_prefill_token_s=0.0, per_decode_seq_s=0.0)
SLOW = CostModel(base_s=0.02, per_prefill_token_s=0.0, per_decode_seq_s=0.0)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def serve(cfg: ServerConfig, engine: MockEngine) -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI]]:
    app = create_app(cfg, engine)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            task.result()
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
            yield client, app
    finally:
        server.should_exit = True
        await task


def body(user: str = "hi", *, system: str | None = None, max_tokens: int = 16, stream: bool = False) -> dict[str, Any]:
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": user})
    return {"messages": msgs, "max_tokens": max_tokens, "stream": stream}


async def read_sse(resp: httpx.Response) -> list[Any]:
    events: list[Any] = []
    async for line in resp.aiter_lines():
        if line.startswith("data: "):
            data = line[len("data: "):]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


async def wait_until_idle(client: httpx.AsyncClient, timeout: float = 5.0) -> dict[str, Any]:
    async with asyncio.timeout(timeout):
        while True:
            m: dict[str, Any] = (await client.get("/metrics")).json()
            if m["num_running"] == 0 and m["num_waiting"] == 0 and m["open_streams"] == 0:
                return m
            await asyncio.sleep(0.02)


def assert_no_leak(app: FastAPI, m: dict[str, Any]) -> None:
    loop: EngineLoop = app.state.loop
    loop.scheduler.block_manager.check_invariants()
    loop.scheduler.tree.check_invariants()
    # Every block is either free or held only by the cache.
    assert m["free_blocks"] + m["cached_blocks"] == m["total_blocks"]


async def test_non_streaming_completion() -> None:
    async with serve(ServerConfig(), MockEngine(FAST)) as (client, _):
        r = await client.post("/v1/chat/completions", json=body(max_tokens=40))
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["finish_reason"] == "length"
        assert len(data["choices"][0]["message"]["content"]) == 40
        assert data["usage"]["completion_tokens"] == 40
        assert (await client.get("/health")).status_code == 200


async def test_streaming_completion() -> None:
    async with serve(ServerConfig(), MockEngine(FAST)) as (client, app):
        async with client.stream("POST", "/v1/chat/completions", json=body(max_tokens=25, stream=True)) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            events = await read_sse(r)
        assert events[-1] == "[DONE]"
        assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert events[-2]["choices"][0]["finish_reason"] == "length"
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in events[1:-2])
        assert len(text) == 25
        assert_no_leak(app, await wait_until_idle(client))


async def test_eos_finishes_with_stop() -> None:
    async with serve(ServerConfig(), MockEngine(FAST, eos_prob=1.0)) as (client, _):
        data = (await client.post("/v1/chat/completions", json=body(max_tokens=50))).json()
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["completion_tokens"] == 0


async def test_shared_system_prompt_hits_cache() -> None:
    system = "You are a support bot for ACME. " * 20
    async with serve(ServerConfig(), MockEngine(FAST)) as (client, _):
        for user in ("first question", "second question"):
            assert (await client.post("/v1/chat/completions", json=body(user, system=system))).status_code == 200
        m = (await client.get("/metrics")).json()
        assert m["prefix_hit_rate"] > 0.3


async def test_overload_returns_429() -> None:
    cfg = ServerConfig(max_waiting=1, max_decode_seqs=1)
    async with serve(cfg, MockEngine(SLOW)) as (client, app):
        rs = await asyncio.gather(
            *(client.post("/v1/chat/completions", json=body(f"q{i}", max_tokens=10)) for i in range(6))
        )
        codes = sorted(r.status_code for r in rs)
        assert 429 in codes and 200 in codes
        limited = next(r for r in rs if r.status_code == 429)
        assert limited.headers["retry-after"] == "1"
        assert limited.json()["error"]["type"] == "rate_limit_error"
        assert_no_leak(app, await wait_until_idle(client))


async def test_bad_requests() -> None:
    async with serve(ServerConfig(num_blocks=8), MockEngine(FAST)) as (client, _):
        r = await client.post("/v1/chat/completions", json=body(max_tokens=500))
        assert r.status_code == 400  # can never fit in 8 blocks
        r = await client.post("/v1/chat/completions", json={"messages": [], "max_tokens": 5})
        assert r.status_code == 422
        r = await client.post("/v1/chat/completions", json=body(max_tokens=0))
        assert r.status_code == 422


async def test_client_disconnect_frees_blocks() -> None:
    async with serve(ServerConfig(), MockEngine(SLOW)) as (client, app):
        async with client.stream("POST", "/v1/chat/completions", json=body(max_tokens=2000, stream=True)) as r:
            seen = 0
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    seen += 1
                if seen >= 4:
                    break  # hang up mid-stream
        m = await wait_until_idle(client)
        assert m["disconnect_aborts"] == 1
        assert m["generated_tokens"] < 2000
        assert_no_leak(app, m)


async def test_many_concurrent_streams() -> None:
    systems = [f"Tenant {t} system prompt. " * 15 for t in range(5)]
    async with serve(ServerConfig(num_blocks=1024), MockEngine(FAST)) as (client, app):

        async def one(i: int) -> int:
            req = body(f"user {i}", system=systems[i % 5], max_tokens=20 + i % 7, stream=True)
            async with client.stream("POST", "/v1/chat/completions", json=req) as r:
                events = await read_sse(r)
            return sum(len(e["choices"][0]["delta"].get("content", "")) for e in events[1:-2])

        lengths = await asyncio.gather(*(one(i) for i in range(50)))
        assert lengths == [20 + i % 7 for i in range(50)]
        m = await wait_until_idle(client)
        assert m["prefix_hit_rate"] > 0.5
        assert_no_leak(app, m)


# ---- EngineLoop directly (no HTTP)


async def test_slow_client_is_cut_off() -> None:
    loop = EngineLoop(ServerConfig(stream_queue_size=4), MockEngine(FAST))
    loop.start()
    handle = loop.submit(list(range(3, 40)), max_tokens=100)
    # Never read. The loop must not stall or buffer forever.
    async with asyncio.timeout(5):
        while not handle.closed:
            await asyncio.sleep(0.01)
    events = [handle.queue.get_nowait() for _ in range(handle.queue.qsize())]
    assert sum(isinstance(e, TokenEvent) for e in events) == 4
    assert isinstance(events[-1], DoneEvent) and events[-1].reason == "error"
    assert loop.metrics.slow_client_aborts == 1
    assert not loop.scheduler.has_work()
    await loop.shutdown(0)


async def test_shutdown_drains_then_refuses() -> None:
    loop = EngineLoop(ServerConfig(), MockEngine(SLOW))
    loop.start()
    handle = loop.submit(list(range(3, 40)), max_tokens=10_000)
    await loop.shutdown(drain_timeout_s=0.1)  # too short to finish
    events = [handle.queue.get_nowait() for _ in range(handle.queue.qsize())]
    assert isinstance(events[-1], DoneEvent) and "shutting down" in events[-1].message
    with pytest.raises(AdmissionError) as e:
        loop.submit([3, 4, 5], 1)
    assert e.value.status == 503


class BrokenEngine(MockEngine):
    def sample(self, batch: Batch) -> dict[str, int]:
        raise RuntimeError("GPU fell off the bus")


async def test_engine_crash_fails_requests_and_health() -> None:
    async with serve(ServerConfig(), BrokenEngine(FAST)) as (client, _):
        r = await client.post("/v1/chat/completions", json=body())
        assert r.status_code == 503
        assert "engine failure" in r.json()["error"]["message"]
        assert (await client.get("/health")).status_code == 503
        assert (await client.post("/v1/chat/completions", json=body())).status_code == 503
