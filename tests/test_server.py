"""OpenAI-compatible response-shape tests without loading a model."""

import json

from fastapi.testclient import TestClient

from nanoserve.server import StreamChunk, create_app


class FakeServingEngine:
    async def stream(self, prompt: str, *, max_tokens: int):
        assert prompt == "hello"
        assert max_tokens == 2
        yield StreamChunk(token_id=1, text=" world", finished=False)
        yield StreamChunk(token_id=2, text="!", finished=True)

    async def close(self) -> None:
        return None


def test_non_streaming_completion_has_openai_shape() -> None:
    client = TestClient(create_app(FakeServingEngine()))

    response = client.post(
        "/v1/completions",
        json={"model": "nanoserve", "prompt": "hello", "max_tokens": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == " world!"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_streaming_completion_emits_token_chunks_before_done() -> None:
    client = TestClient(create_app(FakeServingEngine()))

    with client.stream(
        "POST",
        "/v1/completions",
        json={
            "model": "nanoserve",
            "prompt": "hello",
            "max_tokens": 2,
            "stream": True,
        },
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert lines[-1] == "data: [DONE]"
    first = json.loads(lines[0].removeprefix("data: "))
    assert first["choices"][0]["text"] == " world"
    assert first["choices"][0]["finish_reason"] is None

