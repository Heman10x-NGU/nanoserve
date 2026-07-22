"""MLX adapter with a deliberately hand-written autoregressive decode loop.

The loop follows the model/cache contract documented by mlx-lm, but does not
call ``mlx_lm.generate``, ``stream_generate``, or ``generate_step``. See
``docs/reading_notes.md`` for the pinned source studied before implementation.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from time import perf_counter
from typing import Any

import mlx.core as mx
from mlx_lm import load as load_mlx_model
from mlx_lm.models.cache import make_prompt_cache

from nanoserve.backends.base import (
    DEFAULT_MODEL,
    ForwardOutput,
    GenerationResult,
    TokenEvent,
)


class MLXBackend:
    """Run one MLX language model through nanoserve's backend interface."""

    def __init__(self, model: Any, tokenizer: Any, model_id: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL) -> "MLXBackend":
        """Load and evaluate an MLX model and its tokenizer."""
        model, tokenizer = load_mlx_model(model_id, lazy=False)
        return cls(model=model, tokenizer=tokenizer, model_id=model_id)

    def new_cache(self) -> list[Any]:
        """Create an empty per-layer prompt cache for this model."""
        return make_prompt_cache(self.model)

    def encode(
        self, prompt: str | Sequence[int], *, add_special_tokens: bool = True
    ) -> list[int]:
        """Convert text or an existing token sequence into owned token IDs."""
        if isinstance(prompt, str):
            return list(
                self.tokenizer.encode(
                    prompt,
                    add_special_tokens=add_special_tokens,
                )
            )
        return [int(token_id) for token_id in prompt]

    def forward_logits(
        self, token_ids: Sequence[int], cache: Sequence[Any] | None = None
    ) -> ForwardOutput:
        """Run a direct model forward pass and force MLX evaluation."""
        if not token_ids:
            raise ValueError("token_ids must contain at least one token")
        prompt_cache = list(cache) if cache is not None else self.new_cache()
        inputs = mx.array([list(token_ids)])
        logits = self.model(inputs, cache=prompt_cache)
        mx.eval(logits, [entry.state for entry in prompt_cache])
        return ForwardOutput(logits=logits, cache=prompt_cache)

    def generate(
        self,
        prompt: str | Sequence[int],
        cache: Sequence[Any] | None = None,
        stream: bool = False,
        max_tokens: int = 64,
    ) -> GenerationResult | Iterator[TokenEvent]:
        """Generate greedily, optionally returning evaluated tokens as a stream."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least one")
        started_at = perf_counter()
        prompt_ids = self.encode(prompt, add_special_tokens=cache is None)
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")

        events = self._decode(
            prompt_ids=prompt_ids,
            cache=list(cache) if cache is not None else self.new_cache(),
            max_tokens=max_tokens,
        )
        if stream:
            return events

        collected = list(events)
        token_ids = tuple(event.token_id for event in collected)
        return GenerationResult(
            text=self.tokenizer.decode(list(token_ids), skip_special_tokens=True),
            token_ids=token_ids,
            token_timestamps=tuple(event.timestamp for event in collected),
            started_at=started_at,
            prompt_tokens=len(prompt_ids),
        )

    def _decode(
        self,
        *,
        prompt_ids: Sequence[int],
        cache: Sequence[Any],
        max_tokens: int,
    ) -> Iterator[TokenEvent]:
        """Prefill once, then append one sampled token per model forward pass."""
        current_ids = list(prompt_ids)
        detokenizer = self.tokenizer.detokenizer
        eos_token_ids = set(self.tokenizer.eos_token_ids)

        for _ in range(max_tokens):
            inputs = mx.array([current_ids])
            logits = self.model(inputs, cache=cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)

            # MLX is lazy. This synchronization point defines when the token is
            # genuinely available and therefore the TTFT/TPOT timestamp.
            mx.eval(next_token, [entry.state for entry in cache])
            token_id = int(next_token.item())
            timestamp = perf_counter()
            if token_id in eos_token_ids:
                break

            detokenizer.add_token(token_id)
            yield TokenEvent(
                token_id=token_id,
                text=detokenizer.last_segment,
                timestamp=timestamp,
            )
            current_ids = [token_id]

