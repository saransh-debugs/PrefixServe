"""OpenAI-style HTTP server: /v1/completions and /v1/chat/completions, with SSE streaming.

Shape:

    HTTP handlers  --submit/abort-->  EngineLoop (one asyncio task)  -->  Scheduler
          ^                                   |                              |
          |                                   +--> engine.step() in ONE worker thread (GPU)
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
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from prefixserve.engine import Engine
from prefixserve.mock_engine import MockEngine
from prefixserve.scheduler import AddResult, FinishReason, Scheduler, StepResult
from prefixserve.tokenizer import ByteTokenizer, ChatMessage, TextTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    num_blocks: int = 4096  # 4096 x 16 = 65k tokens of KV. Ignored if the engine owns a pool.
    max_waiting: int = 256
    # 512: same simulated throughput as 2048, ~2.7x lower p99 time between
    # tokens (bench/sim.py budget sweep). Bigger = decodes wait behind longer prefills.
    max_prefill_tokens: int = 512
    max_decode_seqs: int = 64
    chunk_size: int = 512
    enable_prefix_cache: bool = True
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
    # Where each step's time goes. scheduler_s / (scheduler_s + engine_s) is
    # the scheduler overhead number the README quotes.
    schedule_s: float = 0.0
    engine_s: float = 0.0
    commit_s: float = 0.0


class EngineLoop:
    """Owns the scheduler and engine. Runs schedule -> engine -> commit forever."""

    def __init__(self, config: ServerConfig, engine: Engine) -> None:
        self.config = config
        self.engine = engine
        # A real engine sized its KV pool from GPU memory. The scheduler must
        # hand out exactly those block ids, never more.
        num_blocks = engine.num_blocks if engine.num_blocks is not None else config.num_blocks
        self.scheduler = Scheduler(
            num_blocks,
            max_waiting=config.max_waiting,
            chunk_size=config.chunk_size,
            stop_token_ids=engine.stop_token_ids,
            enable_prefix_cache=config.enable_prefix_cache,
        )
        self.metrics = LoopMetrics()
        self.draining = False
        self.failed: BaseException | None = None
        self._handles: dict[str, RequestHandle] = {}
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        # One thread, on purpose. The engine's pinned input buffer and the
        # CUDA stream are not safe to use from two threads, and steps must
        # run in order anyway: step N+1 reads KV that step N wrote.
        #
        # Why a thread at all: engine.step() blocks until the GPU is done
        # (the .tolist() sync). Called on the event loop, it would freeze
        # every HTTP handler for the whole step: no new requests, no tokens
        # streamed, no disconnects noticed.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefixserve-engine")

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
        # A step may still be running in the worker. Wait for it off the loop,
        # so the GPU isn't torn down under a running kernel.
        await asyncio.to_thread(self._executor.shutdown, wait=True)

    # -- called by HTTP handlers (same thread, so no locks needed)

    def submit(self, prompt: list[int], max_tokens: int, *, ignore_eos: bool = False) -> RequestHandle:
        if not self.healthy:
            raise AdmissionError(503, "engine is not running")
        if self.draining:
            raise AdmissionError(503, "server is shutting down")
        limit_len = self.engine.max_seq_len
        if limit_len is not None and len(prompt) + max_tokens > limit_len:
            raise AdmissionError(400, f"prompt + max_tokens is over the {limit_len}-token limit")
        rid = f"cmpl-{uuid.uuid4().hex}"
        limit = self.config.stream_queue_size
        handle = RequestHandle(rid, len(prompt), asyncio.Queue(maxsize=limit + 1), limit)
        result = self.scheduler.add_request(rid, prompt, max_tokens, ignore_eos=ignore_eos)
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
        loop = asyncio.get_running_loop()
        m = self.metrics
        try:
            while True:
                if not self.scheduler.has_work():
                    # Sleep until a request arrives. No busy polling.
                    self._wake.clear()
                    await self._wake.wait()
                    continue

                t0 = time.perf_counter()
                batch = self.scheduler.schedule_step(cfg.max_prefill_tokens, cfg.max_decode_seqs)
                if not batch.work:
                    # Nothing could run (e.g. every waiting request is blocked on
                    # memory). Commit the empty batch and yield, so we don't spin.
                    self.scheduler.commit(batch, {})
                    await asyncio.sleep(0.001)
                    continue
                t1 = time.perf_counter()
                # HTTP handlers keep running while the worker thread waits on
                # the GPU. Aborts that arrive now are deferred to commit(), so
                # the worker never reads a block table that's being freed.
                sampled = await loop.run_in_executor(self._executor, self.engine.step, batch)
                t2 = time.perf_counter()
                result = self.scheduler.commit(batch, sampled)
                self._deliver(result)
                t3 = time.perf_counter()

                m.steps += 1
                m.schedule_s += t1 - t0
                m.engine_s += t2 - t1
                m.commit_s += t3 - t2
                m.last_step_ms = (t3 - t0) * 1e3
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
            # The stop token itself isn't output. OpenAI doesn't return it
            # and doesn't count it in completion_tokens.
            if result.finished.get(rid) is FinishReason.STOP:
                continue
            if handle.queue.qsize() >= handle.token_limit:
                # Client isn't reading. Cut it off instead of buffering
                # forever (memory) or waiting on it (stalls everyone).
                self.metrics.slow_client_aborts += 1
                self._close(rid, DoneEvent("error", "client too slow, stream cut off"))
                self.scheduler.free_request(rid)
                continue
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
        busy = m.schedule_s + m.engine_s + m.commit_s
        return {
            "healthy": self.healthy,
            "draining": self.draining,
            "uptime_s": round(uptime, 3),
            "steps": m.steps,
            "last_step_ms": round(m.last_step_ms, 3),
            "schedule_ms_total": round(m.schedule_s * 1e3, 3),
            "engine_ms_total": round(m.engine_s * 1e3, 3),
            "commit_ms_total": round(m.commit_s * 1e3, 3),
            "scheduler_overhead": round((m.schedule_s + m.commit_s) / busy, 5) if busy > 0 else 0.0,
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


class StreamOptions(BaseModel):
    include_usage: bool = False


class _GenParams(BaseModel):
    model: str | None = None
    max_tokens: int = Field(default=128, ge=1, le=32768)
    stream: bool = False
    stream_options: StreamOptions | None = None
    # Greedy only. None and 0 both mean greedy; anything else is refused
    # rather than silently run greedy, so a client never gets output it
    # didn't ask for.
    temperature: float | None = Field(default=None, ge=0)
    ignore_eos: bool = False


class ChatRequest(_GenParams):
    messages: list[Message] = Field(min_length=1, max_length=256)


class CompletionRequest(_GenParams):
    # Benchmark clients send a string. Token ids are accepted too (OpenAI allows it).
    prompt: str | list[int] = Field(min_length=1)


def _error(status: int, message: str) -> JSONResponse:
    headers = {"Retry-After": "1"} if status in (429, 503) else None
    kind = "rate_limit_error" if status == 429 else "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status, headers=headers)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _usage(handle: RequestHandle, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": handle.prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": handle.prompt_tokens + completion_tokens,
    }


@dataclass(frozen=True, slots=True)
class _Shape:
    """The two OpenAI response shapes. Everything else is shared."""

    chunk_object: str
    full_object: str
    delta: Callable[[str], dict[str, Any]]  # text -> choice fields for a stream chunk
    full: Callable[[str], dict[str, Any]]  # text -> choice fields for a full response
    first: dict[str, Any] | None  # chat sends {"role": "assistant"} first


_CHAT = _Shape(
    "chat.completion.chunk", "chat.completion",
    delta=lambda t: {"delta": {"content": t} if t else {}},
    full=lambda t: {"message": {"role": "assistant", "content": t}},
    first={"delta": {"role": "assistant"}},
)
_TEXT = _Shape(
    "text_completion", "text_completion",
    delta=lambda t: {"text": t, "logprobs": None},
    full=lambda t: {"text": t, "logprobs": None},
    first=None,
)


def create_app(
    config: ServerConfig | None = None, engine: Engine | None = None, tokenizer: TextTokenizer | None = None
) -> FastAPI:
    cfg = config or ServerConfig()
    eng: Engine = engine or MockEngine()
    tok: TextTokenizer = tokenizer or ByteTokenizer()

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

    async def generate(body: _GenParams, prompt: list[int], shape: _Shape, request: Request) -> Response:
        if body.temperature:
            return _error(400, "only greedy decoding (temperature 0) is supported")
        loop = get_loop(request)
        try:
            handle = loop.submit(prompt, body.max_tokens, ignore_eos=body.ignore_eos)
        except AdmissionError as e:
            return _error(e.status, e.message)
        model = body.model or cfg.model_name
        if body.stream:
            include_usage = body.stream_options is not None and body.stream_options.include_usage
            return StreamingResponse(
                _stream(loop, handle, tok, shape, model, cfg.request_timeout_s, include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await _complete(loop, handle, tok, shape, model, cfg.request_timeout_s, request)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(body: ChatRequest, request: Request) -> Response:
        messages: list[ChatMessage] = [{"role": m.role, "content": m.content} for m in body.messages]
        return await generate(body, tok.encode_chat(messages), _CHAT, request)

    @app.post("/v1/completions", response_model=None)
    async def completions(body: CompletionRequest, request: Request) -> Response:
        prompt = tok.encode_prompt(body.prompt) if isinstance(body.prompt, str) else body.prompt
        return await generate(body, prompt, _TEXT, request)

    return app


async def _stream(
    loop: EngineLoop,
    handle: RequestHandle,
    tok: TextTokenizer,
    shape: _Shape,
    model: str,
    timeout_s: float,
    include_usage: bool,
) -> AsyncIterator[str]:
    created = int(time.time())

    def chunk(choice: dict[str, Any], finish: str | None = None) -> str:
        return _sse({
            "id": handle.req_id,
            "object": shape.chunk_object,
            "created": created,
            "model": model,
            "choices": [{"index": 0, **choice, "finish_reason": finish}],
        })

    done = False
    completion_tokens = 0
    try:
        async with asyncio.timeout(timeout_s):
            if shape.first is not None:
                yield chunk(shape.first)
            decoder = tok.stream_decoder()
            while True:
                event = await handle.queue.get()
                if isinstance(event, TokenEvent):
                    completion_tokens += 1
                    # Only send when there's text. Benchmark clients time
                    # inter-token latency per non-empty chunk, same as for
                    # vLLM/SGLang, which also hold back partial characters.
                    text = decoder.push(event.token)
                    if text:
                        yield chunk(shape.delta(text))
                    continue
                done = True
                tail = decoder.flush()
                if tail:
                    yield chunk(shape.delta(tail))
                if event.reason == "error":
                    yield _sse({"error": {"message": event.message, "type": "server_error"}})
                else:
                    yield chunk(shape.delta(""), finish=event.reason)
                    if include_usage:
                        yield _sse({
                            "id": handle.req_id, "object": shape.chunk_object, "created": created,
                            "model": model, "choices": [], "usage": _usage(handle, completion_tokens),
                        })
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
    loop: EngineLoop,
    handle: RequestHandle,
    tok: TextTokenizer,
    shape: _Shape,
    model: str,
    timeout_s: float,
    request: Request,
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
                    tokens.append(event.token)
                    continue
                done = True
                if event.reason == "error":
                    return _error(503, event.message)
                return JSONResponse({
                    "id": handle.req_id,
                    "object": shape.full_object,
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, **shape.full(tok.decode(tokens)), "finish_reason": event.reason}],
                    "usage": _usage(handle, len(tokens)),
                })
    except TimeoutError:
        return _error(504, "request timed out")
    finally:
        if not done:
            loop.abort(handle.req_id)


def main() -> None:
    import uvicorn

    d = ServerConfig()  # read defaults from an instance: with slots=True the class attrs are descriptors
    p = argparse.ArgumentParser(description="PrefixServe LLM server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default=None, help="HF Llama folder. Omit to run the mock engine.")
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--attention", choices=["flashinfer", "torch"], default="flashinfer")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--num-blocks", type=int, default=None,
                   help="KV blocks. Default: mock 4096, real model sized from free GPU memory.")
    p.add_argument("--max-waiting", type=int, default=d.max_waiting)
    p.add_argument("--max-running-requests", type=int, default=d.max_decode_seqs)
    p.add_argument("--chunked-prefill-size", type=int, default=d.max_prefill_tokens)
    p.add_argument("--disable-prefix-cache", action="store_true")
    p.add_argument("--eos-prob", type=float, default=0.0, help="mock only: chance each token is EOS")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = ServerConfig(
        num_blocks=args.num_blocks or d.num_blocks,
        max_waiting=args.max_waiting,
        max_decode_seqs=args.max_running_requests,
        max_prefill_tokens=args.chunked_prefill_size,
        chunk_size=args.chunked_prefill_size,
        enable_prefix_cache=not args.disable_prefix_cache,
        model_name=args.served_model_name or args.model or d.model_name,
    )
    engine: Engine
    tokenizer: TextTokenizer
    if args.model is None:
        engine, tokenizer = MockEngine(eos_prob=args.eos_prob), ByteTokenizer()
    else:
        # Heavy imports only when serving a real model, so the mock server
        # and its tests don't need torch.
        from pathlib import Path

        import torch

        from prefixserve.hf_tokenizer import HFTokenizer
        from prefixserve.llama_engine import EngineConfig, LlamaEngine

        engine = LlamaEngine(EngineConfig(
            model_path=Path(args.model),
            dtype=getattr(torch, args.dtype),
            attention=args.attention,
            max_num_seqs=cfg.max_decode_seqs,
            max_prefill_tokens=cfg.max_prefill_tokens,
            max_seq_len=args.max_seq_len,
            num_blocks=args.num_blocks,
        ))
        tokenizer = HFTokenizer(args.model)
    uvicorn.run(create_app(cfg, engine, tokenizer), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
