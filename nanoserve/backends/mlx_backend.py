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
import mlx_lm
from mlx_lm import load as load_mlx_model
from mlx_lm.models.cache import BatchKVCache, KVCache, make_prompt_cache
from mlx.utils import tree_map

from nanoserve.backends.base import (
    DEFAULT_MODEL,
    ForwardOutput,
    GenerationResult,
    PREFILL_BLOCK_SIZE,
    TokenEvent,
)


class MLXBackend:
    """Run one MLX language model through nanoserve's backend interface."""

    def __init__(self, model: Any, tokenizer: Any, model_id: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.eos_token_ids = set(tokenizer.eos_token_ids)
        self.batch_forward_count = 0
        self.batch_token_slots = 0

    @property
    def cache_namespace(self) -> str:
        """Identity inputs that must match before a prompt cache is reused."""
        tokenizer_id = getattr(self.tokenizer, "name_or_path", self.model_id)
        return f"{self.model_id}:{tokenizer_id}:mlx-lm-{mlx_lm.__version__}:v1"

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL) -> "MLXBackend":
        """Load and evaluate an MLX model and its tokenizer."""
        model, tokenizer = load_mlx_model(model_id, lazy=False)
        return cls(model=model, tokenizer=tokenizer, model_id=model_id)

    def new_cache(self) -> list[Any]:
        """Create an empty per-layer prompt cache for this model."""
        return make_prompt_cache(self.model)

    def clone_cache(self, cache: Sequence[Any]) -> list[Any]:
        """Copy cache objects and MLX arrays so future appends cannot alias."""
        clones = []
        for entry in cache:
            state = tree_map(
                lambda value: mx.array(value) if isinstance(value, mx.array) else value,
                entry.state,
            )
            clones.append(type(entry).from_state(state, entry.meta_state))
        mx.eval([entry.state for entry in clones])
        return clones

    def prefill_batch(
        self, prompts: list[list[int]]
    ) -> tuple[list[int], list[BatchKVCache]]:
        """Prefill a right-aligned prompt batch in one model forward pass."""
        if not prompts or any(not prompt for prompt in prompts):
            raise ValueError("prompts must contain non-empty token sequences")
        if any(type(cache) is not KVCache for cache in self.new_cache()):
            raise TypeError("continuous batching v1 requires standard KVCache layers")

        max_length = max(len(prompt) for prompt in prompts)
        left_padding = [max_length - len(prompt) for prompt in prompts]
        pad_token_id = int(self.tokenizer.pad_token_id or 0)
        padded = [
            [pad_token_id] * padding + prompt
            for prompt, padding in zip(prompts, left_padding)
        ]
        cache = [BatchKVCache(left_padding) for _ in self.model.layers]
        logits = self.model(mx.array(padded), cache=cache)
        next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_tokens, [entry.state for entry in cache])
        self.batch_forward_count += 1
        self.batch_token_slots += len(prompts)
        return [int(token) for token in next_tokens.tolist()], cache

    def decode_batch(
        self, token_ids: list[int], cache: list[BatchKVCache]
    ) -> tuple[list[int], list[BatchKVCache]]:
        """Advance every active request through one shared model forward."""
        if not token_ids:
            raise ValueError("token_ids must contain at least one active request")
        logits = self.model(mx.array([[token] for token in token_ids]), cache=cache)
        next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_tokens, [entry.state for entry in cache])
        self.batch_forward_count += 1
        self.batch_token_slots += len(token_ids)
        return [int(token) for token in next_tokens.tolist()], cache

    def extend_batch_cache(
        self,
        active: list[BatchKVCache],
        admitted: list[BatchKVCache],
    ) -> list[BatchKVCache]:
        """Append newly prefetched request state to the active cache batch."""
        for active_layer, admitted_layer in zip(active, admitted):
            active_layer.extend(admitted_layer)
        mx.eval([entry.state for entry in active])
        return active

    def filter_batch_cache(
        self, cache: list[BatchKVCache], indices: list[int]
    ) -> list[BatchKVCache]:
        """Drop finished rows while preserving the surviving batch order."""
        for entry in cache:
            entry.filter(indices)
        mx.eval([entry.state for entry in cache])
        return cache

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
        logits = self._prefill(token_ids, prompt_cache)
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
        detokenizer = self.tokenizer.detokenizer
        eos_token_ids = set(self.tokenizer.eos_token_ids)
        logits = self._prefill(prompt_ids, cache)
        previous_token: int | None = None

        for _ in range(max_tokens):
            if previous_token is not None:
                logits = self.model(mx.array([[previous_token]]), cache=cache)
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
            previous_token = token_id

    def _prefill(self, token_ids: Sequence[int], cache: Sequence[Any]) -> Any:
        """Process prompt blocks in a stable order shared by cold and warm runs.

        MLX kernels can produce small floating-point differences when the same
        prefix is grouped into different sequence lengths. Fixed-size blocks,
        together with block-aligned cache publication, make both paths execute
        the same reductions and preserve greedy token identity.
        """
        logits = None
        for start in range(0, len(token_ids), PREFILL_BLOCK_SIZE):
            block = token_ids[start : start + PREFILL_BLOCK_SIZE]
            logits = self.model(mx.array([list(block)]), cache=cache)
        assert logits is not None
        return logits
