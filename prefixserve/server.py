"""OpenAI-style HTTP server: POST /v1/chat/completions, with SSE streaming.

Shape:

    HTTP handlers  --submit/abort-->  EngineLoop (one asyncio task)  -->  Scheduler + MockEngine
          ^                                   |
          +------ per-request bounded queue <-+

Rules that keep one bad client from hurting everyone:
- The engine loop never awaits a client. It pushes tokens with put_nowait.
  A client that reads too slowly fills its own queue and gets cut off.
- A disconnected client's request is aborted, so its KV blocks come back.
- The waiting queue is capped. Over the cap -> 429 with Retry-After.
- Every request has a deadline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from prefixserve.mock_engine import MockEngine
from prefixserve.scheduler import AddResult, FinishReason, Scheduler, StepResult
from prefixserve.tokenizer import EOS_ID, ChatMessage, StreamDecoder, apply_chat_template

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    num_blocks: int = 4096  # 4096 x 16 = 65k tokens of KV
    max_waiting: int = 256
    # 512: same simulated throughput as 2048, ~2.7x lower p99 time between
    # tokens (bench/sim.py budget sweep). Bigger = decodes wait behind longer prefills.
    max_prefill_tokens: int = 512
    max_decode_seqs: int = 64
    chunk_size: int = 512
    stream_queue_size: int = 256  # tokens a client may fall behind before it's cut off
    request_timeout_s: float = 300.0
    drain_timeout_s: float = 30.0
    model_name: str = "prefixserve-mock"


# ---------------------------------------------------------------- engine loop


@dataclass(frozen=True, slots=True)
class TokenEvent:
    token: int


@dataclass(frozen=True, slots=True)
class DoneEvent:
    reason: Literal["stop", "length", "error"]
    message: str = ""


Event = TokenEvent | DoneEvent


@dataclass(slots=True, eq=False)
class RequestHandle:
    req_id: str
    prompt_tokens: int
    # maxsize = limit + 1: one slot is always kept for the final DoneEvent,
    # so a full queue can never swallow the "you're done" message.
    queue: asyncio.Queue[Event]
    token_limit: int
    closed: bool = False


class AdmissionError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


_ADMISSION_ERRORS: dict[AddResult, tuple[int, str]] = {
    AddResult.QUEUE_FULL: (429, "server is at capacity, retry later"),
    AddResult.TOO_LONG: (400, "prompt + max_tokens is larger than the whole KV cache"),
    AddResult.INVALID: (400, "invalid request"),
    AddResult.DUPLICATE_ID: (409, "duplicate request id"),
}


@dataclass(slots=True)
class LoopMetrics:
    started_at: float = field(default_factory=time.monotonic)
    steps: int = 0
    generated_tokens: int = 0
    slow_client_aborts: int = 0
    disconnect_aborts: int = 0
    last_step_ms: float = 0.0


class EngineLoop:
    """Owns the scheduler and engine. Runs schedule -> engine -> commit forever."""

    def __init__(self, config: ServerConfig, engine: MockEngine) -> None:
        self.config = config
        self.engine = engine
        self.scheduler = Scheduler(
            config.num_blocks,
            max_waiting=config.max_waiting,
            chunk_size=config.chunk_size,
            eos_token_id=EOS_ID,
        )
        self.metrics = LoopMetrics()
        self.draining = False
        self.failed: BaseException | None = None
        self._handles: dict[str, RequestHandle] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    # -- lifecycle

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="prefixserve-engine-loop")

    @property
    def healthy(self) -> bool:
        return self.failed is None and self._task is not None and not self._task.done()

    async def shutdown(self, drain_timeout_s: float) -> None:
        """Stop taking requests, let running ones finish, then cut off the rest."""
        self.draining = True
        deadline = time.monotonic() + drain_timeout_s
        while self.scheduler.has_work() and time.monotonic() < deadline and self.healthy:
            await asyncio.sleep(0.05)
        for rid in list(self._handles):
            self._close(rid, DoneEvent("error", "server shutting down"))
            self.scheduler.free_request(rid)
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:  # already logged in _run
                pass

    # -- called by HTTP handlers (same thread, so no locks needed)

    def submit(self, prompt: list[int], max_tokens: int) -> RequestHandle:
        if not self.healthy:
            raise AdmissionError(503, "engine is not running")
        if self.draining:
            raise AdmissionError(503, "server is shutting down")
        rid = f"chatcmpl-{uuid.uuid4().hex}"
        limit = self.config.stream_queue_size
        handle = RequestHandle(rid, len(prompt), asyncio.Queue(maxsize=limit + 1), limit)
        result = self.scheduler.add_request(rid, prompt, max_tokens)
        if result is not AddResult.ACCEPTED:
            status, msg = _ADMISSION_ERRORS[result]
            raise AdmissionError(status, msg)
        self._handles[rid] = handle
        self._wake.set()
        return handle

    def abort(self, req_id: str) -> None:
        """Client went away. Safe to call twice and after the request finished."""
        if self._handles.pop(req_id, None) is not None:
            self.metrics.disconnect_aborts += 1
            self.scheduler.free_request(req_id)

    # -- the loop

    async def _run(self) -> None:
        cfg = self.config
        try:
            while True:
                if not self.scheduler.has_work():
                    # Sleep until a request arrives. No busy polling.
                    self._wake.clear()
                    await self._wake.wait()
                    continue

                started = time.perf_counter()
                batch = self.scheduler.schedule_step(cfg.max_prefill_tokens, cfg.max_decode_seqs)
                if not batch.work:
                    # Nothing could run (e.g. every waiting request is blocked on
                    # memory). Commit the empty batch and yield, so we don't spin.
                    self.scheduler.commit(batch, {})
                    await asyncio.sleep(0.001)
                    continue

                # The "GPU" runs here. asyncio.sleep doesn't block the event
                # loop, so HTTP handlers keep accepting and streaming meanwhile.
                # With a real model this would be an await on the GPU stream or
                # a worker process, never a blocking call on this thread.
                await asyncio.sleep(self.engine.step_seconds(batch))
                result = self.scheduler.commit(batch, self.engine.sample(batch))
                self._deliver(result)

                self.metrics.steps += 1
                self.metrics.last_step_ms = (time.perf_counter() - started) * 1e3
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Don't die silently. Every open request gets an error, and
            # /health starts failing so the orchestrator restarts us.
            logger.exception("engine loop crashed")
            self.failed = exc
            for rid in list(self._handles):
                self._close(rid, DoneEvent("error", "engine failure"))
            raise

    def _deliver(self, result: StepResult) -> None:
        for rid, token in result.new_tokens.items():
            handle = self._handles.get(rid)
            if handle is None:
                continue  # aborted while the batch ran
            if handle.queue.qsize() >= handle.token_limit:
                # Client isn't reading. Cut it off instead of buffering
                # forever (memory) or waiting on it (stalls everyone).
                self.metrics.slow_client_aborts += 1
                self._close(rid, DoneEvent("error", "client too slow, stream cut off"))
                self.scheduler.free_request(rid)
                continue
            if token != EOS_ID:
                self.metrics.generated_tokens += 1
            handle.queue.put_nowait(TokenEvent(token))

        for rid, reason in result.finished.items():
            if reason is FinishReason.ABORT:
                continue  # handle already removed by abort()
            self._close(rid, DoneEvent("stop" if reason is FinishReason.STOP else "length"))

    def _close(self, rid: str, event: DoneEvent) -> None:
        handle = self._handles.pop(rid, None)
        if handle is None or handle.closed:
            return
        handle.closed = True
        handle.queue.put_nowait(event)  # the reserved slot

    def snapshot(self) -> dict[str, Any]:
        st = self.scheduler.stats()
        m = self.metrics
        uptime = time.monotonic() - m.started_at
        return {
            "healthy": self.healthy,
            "draining": self.draining,
            "uptime_s": round(uptime, 3),
            "steps": m.steps,
            "last_step_ms": round(m.last_step_ms, 3),
            "generated_tokens": m.generated_tokens,
            "generated_tokens_per_s": round(m.generated_tokens / uptime, 2) if uptime > 0 else 0.0,
            "slow_client_aborts": m.slow_client_aborts,
            "disconnect_aborts": m.disconnect_aborts,
            "open_streams": len(self._handles),
            "num_waiting": st.num_waiting,
            "num_running": st.num_running,
            "free_blocks": st.free_blocks,
            "cached_blocks": st.cached_blocks,
            "total_blocks": self.scheduler.block_manager.num_blocks,
            "prefix_hit_rate": round(st.prefix_hit_rate, 4),
            "preemptions": st.preemptions,
            "rejected": st.rejected,
        }


# ---------------------------------------------------------------- HTTP layer


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=200_000)


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(min_length=1, max_length=256)
    max_tokens: int = Field(default=128, ge=1, le=8192)
    stream: bool = False


def _error(status: int, message: str) -> JSONResponse:
    headers = {"Retry-After": "1"} if status in (429, 503) else None
    kind = "rate_limit_error" if status == 429 else "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status, headers=headers)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def create_app(config: ServerConfig | None = None, engine: MockEngine | None = None) -> FastAPI:
    cfg = config or ServerConfig()
    eng = engine or MockEngine()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loop = EngineLoop(cfg, eng)
        loop.start()
        app.state.loop = loop
        try:
            yield
        finally:
            await loop.shutdown(cfg.drain_timeout_s)

    app = FastAPI(title="PrefixServe", lifespan=lifespan)

    def get_loop(request: Request) -> EngineLoop:
        loop: EngineLoop = request.app.state.loop
        return loop

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        loop = get_loop(request)
        ok = loop.healthy and not loop.draining
        return JSONResponse({"status": "ok" if ok else "unavailable"}, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def metrics(request: Request) -> JSONResponse:
        return JSONResponse(get_loop(request).snapshot())

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(body: ChatRequest, request: Request) -> Response:
        loop = get_loop(request)
        messages: list[ChatMessage] = [{"role": m.role, "content": m.content} for m in body.messages]
        prompt = apply_chat_template(messages)
        try:
            handle = loop.submit(prompt, body.max_tokens)
        except AdmissionError as e:
            return _error(e.status, e.message)

        model = body.model or cfg.model_name
        if body.stream:
            return StreamingResponse(
                _stream(loop, handle, model, cfg.request_timeout_s),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await _complete(loop, handle, model, cfg.request_timeout_s, request)

    return app


async def _stream(loop: EngineLoop, handle: RequestHandle, model: str, timeout_s: float) -> AsyncIterator[str]:
    created = int(time.time())

    def chunk(delta: dict[str, str], finish: str | None = None) -> str:
        return _sse({
            "id": handle.req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })

    done = False
    try:
        async with asyncio.timeout(timeout_s):
            yield chunk({"role": "assistant"})
            decoder = StreamDecoder()
            while True:
                event = await handle.queue.get()
                if isinstance(event, TokenEvent):
                    text = decoder.push(event.token)
                    if text:
                        yield chunk({"content": text})
                    continue
                done = True
                tail = decoder.flush()
                if tail:
                    yield chunk({"content": tail})
                if event.reason == "error":
                    yield _sse({"error": {"message": event.message, "type": "server_error"}})
                else:
                    yield chunk({}, finish=event.reason)
                yield "data: [DONE]\n\n"
                return
    except TimeoutError:
        yield _sse({"error": {"message": "request timed out", "type": "timeout"}})
    finally:
        # Runs on normal exit, timeout, and client disconnect (Starlette
        # cancels this generator). Without it, a dropped connection would
        # keep generating and holding KV blocks until max_tokens.
        if not done:
            loop.abort(handle.req_id)


async def _complete(
    loop: EngineLoop, handle: RequestHandle, model: str, timeout_s: float, request: Request
) -> Response:
    tokens: list[int] = []
    done = False
    try:
        async with asyncio.timeout(timeout_s):
            while True:
                try:
                    event = await asyncio.wait_for(handle.queue.get(), timeout=1.0)
                except TimeoutError:
                    # Non-streaming responses never write until the end, so
                    # check for a dead client ourselves.
                    if await request.is_disconnected():
                        return Response(status_code=499)
                    continue
                if isinstance(event, TokenEvent):
                    if event.token != EOS_ID:
                        tokens.append(event.token)
                    continue
                done = True
                if event.reason == "error":
                    return _error(503, event.message)
                decoder = StreamDecoder()
                text = "".join(decoder.push(t) for t in tokens) + decoder.flush()
                return JSONResponse({
                    "id": handle.req_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": event.reason,
                    }],
                    "usage": {
                        "prompt_tokens": handle.prompt_tokens,
                        "completion_tokens": len(tokens),
                        "total_tokens": handle.prompt_tokens + len(tokens),
                    },
                })
    except TimeoutError:
        return _error(504, "request timed out")
    finally:
        if not done:
            loop.abort(handle.req_id)


def main() -> None:
    import uvicorn

    d = ServerConfig()  # read defaults from an instance: with slots=True the class attrs are descriptors
    p = argparse.ArgumentParser(description="PrefixServe mock LLM server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--num-blocks", type=int, default=d.num_blocks)
    p.add_argument("--max-waiting", type=int, default=d.max_waiting)
    p.add_argument("--max-decode-seqs", type=int, default=d.max_decode_seqs)
    p.add_argument("--max-prefill-tokens", type=int, default=d.max_prefill_tokens)
    p.add_argument("--eos-prob", type=float, default=0.0, help="chance each token is EOS")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = ServerConfig(
        num_blocks=args.num_blocks,
        max_waiting=args.max_waiting,
        max_decode_seqs=args.max_decode_seqs,
        max_prefill_tokens=args.max_prefill_tokens,
    )
    uvicorn.run(create_app(cfg, MockEngine(eos_prob=args.eos_prob)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
