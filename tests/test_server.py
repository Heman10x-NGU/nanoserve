"""OpenAI-compatible response-shape tests without loading a model."""

import asyncio
import json

import httpx

from nanoserve.server import ServingEngine, StreamChunk, create_app


class FakeServingEngine:
    async def stream(self, prompt: str, *, max_tokens: int):
        assert prompt == "hello"
        assert max_tokens == 2
        yield StreamChunk(token_id=1, text=" world", finished=False)
        yield StreamChunk(
            token_id=2,
            text="!",
            finished=True,
            finish_reason="length",
        )

    async def close(self) -> None:
        return None


class FailingBackend:
    eos_token_ids: set[int] = set()

    def encode(self, prompt: str) -> list[int]:
        return [1]

    def new_detokenizer(self):
        return object()

    def prefill_batch(self, prompts: list[list[int]]):
        raise ValueError("synthetic backend failure")


def test_non_streaming_completion_has_openai_shape() -> None:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app(FakeServingEngine()))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post(
                "/v1/completions",
                json={"model": "nanoserve", "prompt": "hello", "max_tokens": 2},
            )

    response = asyncio.run(request())

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == " world!"
    assert body["choices"][0]["finish_reason"] == "length"


def test_streaming_completion_emits_token_chunks_before_done() -> None:
    async def request() -> tuple[httpx.Response, list[str]]:
        transport = httpx.ASGITransport(app=create_app(FakeServingEngine()))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            async with client.stream(
                "POST",
                "/v1/completions",
                json={
                    "model": "nanoserve",
                    "prompt": "hello",
                    "max_tokens": 2,
                    "stream": True,
                },
            ) as response:
                lines = [line async for line in response.aiter_lines() if line]
                return response, lines

    response, lines = asyncio.run(request())

    assert response.status_code == 200
    assert lines[-1] == "data: [DONE]"
    first = json.loads(lines[0].removeprefix("data: "))
    assert first["choices"][0]["text"] == " world"
    assert first["choices"][0]["finish_reason"] is None
    final = json.loads(lines[-2].removeprefix("data: "))
    assert final["choices"][0]["finish_reason"] == "length"


def test_worker_failure_reaches_waiting_stream_and_shutdown_is_clean() -> None:
    async def consume() -> None:
        engine = ServingEngine(FailingBackend())
        try:
            async for _ in engine.stream("hello", max_tokens=2):
                pass
        except RuntimeError as exc:
            assert "serving worker failed" in str(exc)
        else:
            raise AssertionError("worker failure did not reach the request")
        await engine.close()

    asyncio.run(consume())
