"""The single inference-backend seam used by metrics and serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One evaluated output token from a streaming generation."""

    token_id: int
    text: str
    timestamp: float


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
    """Minimal interface shared by current and future device adapters."""

    model_id: str

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL) -> "Backend": ...

    def forward_logits(
        self, token_ids: Sequence[int], cache: Sequence[Any] | None = None
    ) -> ForwardOutput: ...

    def generate(
        self,
        prompt: str | Sequence[int],
        cache: Sequence[Any] | None = None,
        stream: bool = False,
        max_tokens: int = 64,
    ) -> GenerationResult | Iterator[TokenEvent]: ...

