"""OpenAI-compatible streaming server over the continuous batch scheduler."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nanoserve.engine.scheduler import ContinuousBatchScheduler, SchedulerEvent

if TYPE_CHECKING:
    from nanoserve.backends.base import Backend


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """One printable streaming segment or terminal signal."""

    token_id: int | None
    text: str
    finished: bool
    finish_reason: Literal["stop", "length"] | None = None


class StreamEngine(Protocol):
    @property
    def healthy(self) -> bool: ...

    async def stream(
        self, prompt: str, *, max_tokens: int
    ) -> AsyncIterator[StreamChunk]: ...

    async def close(self) -> None: ...


class CompletionRequest(BaseModel):
    model: str = "nanoserve"
    prompt: str
    max_tokens: int = Field(default=64, ge=1)
    stream: bool = False


class ServingEngine:
    """Bridge synchronous Metal steps to per-request async token streams."""

    def __init__(self, backend: "Backend", *, max_batch_size: int = 8) -> None:
        self.backend = backend
        self.scheduler = ContinuousBatchScheduler(
            backend, max_batch_size=max_batch_size
        )
        self._queues: dict[str, asyncio.Queue[StreamChunk | Exception]] = {}
        self._detokenizers: dict[str, Any] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._lock: asyncio.Lock | None = None
        self._wake: asyncio.Event | None = None
        self._closed = False

    async def stream(
        self, prompt: str, *, max_tokens: int
    ) -> AsyncIterator[StreamChunk]:
        """Submit one request and yield segments as scheduler steps finish."""
        if self._closed:
            raise RuntimeError("serving engine is closed")
        self._ensure_worker()
        assert self._lock is not None and self._wake is not None
        request_id = uuid4().hex
        queue: asyncio.Queue[StreamChunk | Exception] = asyncio.Queue()
        self._queues[request_id] = queue
        self._detokenizers[request_id] = self.backend.new_detokenizer()
        prompt_ids = self.backend.encode(prompt)
        async with self._lock:
            self.scheduler.submit(
                prompt_ids,
                max_tokens=max_tokens,
                request_id=request_id,
            )
            self._wake.set()

        while True:
            item = await queue.get()
            if isinstance(item, Exception):
                raise RuntimeError("serving worker failed") from item
            chunk = item
            yield chunk
            if chunk.finished:
                break

    async def close(self) -> None:
        self._closed = True
        if self._wake is not None:
            self._wake.set()
        if self._worker_task is not None:
            await self._worker_task

    @property
    def healthy(self) -> bool:
        """Whether this engine can accept and execute new requests."""
        return not self._closed

    def _ensure_worker(self) -> None:
        if self._worker_task is not None:
            return
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._worker_task = asyncio.create_task(self._run(), name="nanoserve-batcher")

    async def _run(self) -> None:
        assert self._lock is not None and self._wake is not None
        while not self._closed:
            await self._wake.wait()
            if self._closed:
                break
            while True:
                try:
                    async with self._lock:
                        if not self.scheduler.has_work:
                            self._wake.clear()
                            break
                        events = await asyncio.to_thread(self.scheduler.step)
                    await self._dispatch(events)
                    await asyncio.sleep(0)
                except Exception as exc:
                    self._closed = True
                    for queue in list(self._queues.values()):
                        await queue.put(exc)
                    self._queues.clear()
                    self._detokenizers.clear()
                    return

    async def _dispatch(self, events: list[SchedulerEvent]) -> None:
        for event in events:
            detokenizer = self._detokenizers[event.request_id]
            text = ""
            if event.token_id is not None:
                detokenizer.add_token(event.token_id)
                text = detokenizer.last_segment
            if event.finished:
                detokenizer.finalize()
                text += detokenizer.last_segment
            await self._queues[event.request_id].put(
                StreamChunk(
                    event.token_id,
                    text,
                    event.finished,
                    event.finish_reason,
                )
            )
            if event.finished:
                self.scheduler.pop_result(event.request_id)
                del self._detokenizers[event.request_id]
                del self._queues[event.request_id]


def create_app(engine: StreamEngine) -> FastAPI:
    """Create the HTTP app without hiding model state in module globals."""
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await engine.close()

    app = FastAPI(title="nanoserve", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        if not engine.healthy:
            return JSONResponse({"status": "error"}, status_code=503)
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        completion_id = f"cmpl-{uuid4().hex}"
        created = int(time.time())

        if request.stream:
            async def event_stream() -> AsyncIterator[str]:
                async for chunk in engine.stream(
                    request.prompt, max_tokens=request.max_tokens
                ):
                    payload = _completion_payload(
                        completion_id=completion_id,
                        created=created,
                        model=request.model,
                        text=chunk.text,
                        finish_reason=chunk.finish_reason,
                    )
                    yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        text_parts = []
        async for chunk in engine.stream(request.prompt, max_tokens=request.max_tokens):
            text_parts.append(chunk.text)
        return JSONResponse(
            _completion_payload(
                completion_id=completion_id,
                created=created,
                model=request.model,
                text="".join(text_parts),
                finish_reason=chunk.finish_reason,
            )
        )

    return app


def _completion_payload(
    *,
    completion_id: str,
    created: int,
    model: str,
    text: str,
    finish_reason: Literal["stop", "length"] | None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
