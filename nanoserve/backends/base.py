"""The single inference-backend seam used by metrics and serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
PREFILL_BLOCK_SIZE = 64


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One evaluated output token, or final buffered text, from generation."""

    token_id: int | None
    text: str
    timestamp: float
    finished: bool = False


@dataclass(frozen=True, slots=True)
class ForwardOutput:
    """Logits plus the cache mutated by a direct model forward pass."""

    logits: Any
    cache: Sequence[Any]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """A completed generation and the timestamps needed for metrics."""

    text: str
    token_ids: tuple[int, ...]
    token_timestamps: tuple[float, ...]
    started_at: float
    prompt_tokens: int


@runtime_checkable
class Backend(Protocol):
    """Single interface required by generation, batching, and serving."""

    model_id: str
    eos_token_ids: set[int]

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL) -> "Backend": ...

    def forward_logits(
        self, token_ids: Sequence[int], cache: Sequence[Any] | None = None
    ) -> ForwardOutput: ...

    def encode(
        self, prompt: str | Sequence[int], *, add_special_tokens: bool = True
    ) -> list[int]: ...

    def new_detokenizer(self) -> Any: ...

    def prefill_batch(self, prompts: list[list[int]]) -> tuple[list[int], Any]: ...

    def decode_batch(
        self, token_ids: list[int], cache: Any
    ) -> tuple[list[int], Any]: ...

    def extend_batch_cache(self, active: Any, admitted: Any) -> Any: ...

    def filter_batch_cache(self, cache: Any, indices: list[int]) -> Any: ...

    def generate(
        self,
        prompt: str | Sequence[int],
        cache: Sequence[Any] | None = None,
        stream: bool = False,
        max_tokens: int = 64,
    ) -> GenerationResult | Iterator[TokenEvent]: ...
