"""A small continuous-batching scheduler inspired by Orca and vLLM.

Waiting requests are admitted whenever the active batch has capacity. Every
decode call sends all active requests through one backend forward pass. The
backend owns tensor/cache mechanics; this module owns queue and lifecycle state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol, Sequence
from uuid import uuid4


class BatchBackend(Protocol):
    """Tensor operations required by the scheduler."""

    eos_token_ids: set[int]

    def prefill_batch(
        self, prompts: list[list[int]]
    ) -> tuple[list[int], Any]: ...

    def decode_batch(
        self, token_ids: list[int], cache: Any
    ) -> tuple[list[int], Any]: ...

    def extend_batch_cache(self, active: Any, admitted: Any) -> Any: ...

    def filter_batch_cache(self, cache: Any, indices: list[int]) -> Any: ...


@dataclass(frozen=True, slots=True)
class SchedulerEvent:
    """One token or terminal signal emitted by a scheduler step."""

    request_id: str
    token_id: int | None
    timestamp: float
    finished: bool


@dataclass(frozen=True, slots=True)
class ScheduledResult:
    """Completed token/timing record for one request."""

    request_id: str
    token_ids: tuple[int, ...]
    token_timestamps: tuple[float, ...]
    submitted_at: float
    completed_at: float


@dataclass(slots=True)
class _RequestState:
    request_id: str
    prompt_ids: list[int]
    max_tokens: int
    submitted_at: float
    token_ids: list[int] = field(default_factory=list)
    token_timestamps: list[float] = field(default_factory=list)


class ContinuousBatchScheduler:
    """Admit queued work continuously and decode the active batch together."""

    def __init__(self, backend: BatchBackend, *, max_batch_size: int = 8) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least one")
        self.backend = backend
        self.max_batch_size = max_batch_size
        self._waiting: deque[_RequestState] = deque()
        self._active: list[_RequestState] = []
        self._batch_cache: Any = None
        self._results: dict[str, ScheduledResult] = {}
        self._request_ids: set[str] = set()

    @property
    def pending_count(self) -> int:
        return len(self._waiting)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def has_work(self) -> bool:
        return bool(self._waiting or self._active)

    def submit(
        self,
        prompt_ids: Sequence[int],
        *,
        max_tokens: int,
        request_id: str | None = None,
        submitted_at: float | None = None,
    ) -> str:
        """Queue a request and return its stable ID."""
        tokens = [int(token_id) for token_id in prompt_ids]
        if not tokens:
            raise ValueError("prompt_ids must contain at least one token")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least one")
        request_id = request_id or uuid4().hex
        if request_id in self._request_ids:
            raise ValueError(f"duplicate request_id: {request_id}")
        self._request_ids.add(request_id)
        self._waiting.append(
            _RequestState(
                request_id=request_id,
                prompt_ids=tokens,
                max_tokens=max_tokens,
                submitted_at=(perf_counter() if submitted_at is None else submitted_at),
            )
        )
        return request_id

    def step(self, *, now: float | None = None) -> list[SchedulerEvent]:
        """Admit available requests, then advance the full active batch once."""
        if not self.has_work:
            return []
        events: list[SchedulerEvent] = []
        timestamp = (lambda: perf_counter()) if now is None else (lambda: now)

        capacity = self.max_batch_size - len(self._active)
        admitted = [self._waiting.popleft() for _ in range(min(capacity, len(self._waiting)))]
        if admitted:
            first_tokens, admitted_cache = self.backend.prefill_batch(
                [request.prompt_ids for request in admitted]
            )
            admitted, admitted_cache, prefill_events = self._record_batch(
                admitted,
                first_tokens,
                admitted_cache,
                timestamp(),
            )
            events.extend(prefill_events)
            if admitted:
                if self._batch_cache is None:
                    self._batch_cache = admitted_cache
                else:
                    self._batch_cache = self.backend.extend_batch_cache(
                        self._batch_cache, admitted_cache
                    )
                self._active.extend(admitted)

        if self._active:
            next_tokens, self._batch_cache = self.backend.decode_batch(
                [request.token_ids[-1] for request in self._active],
                self._batch_cache,
            )
            self._active, self._batch_cache, decode_events = self._record_batch(
                self._active,
                next_tokens,
                self._batch_cache,
                timestamp(),
            )
            events.extend(decode_events)

        return events

    def pop_result(self, request_id: str) -> ScheduledResult | None:
        """Remove and return a completed result, if available."""
        result = self._results.pop(request_id, None)
        if result is not None:
            self._request_ids.discard(request_id)
        return result

    def _record_batch(
        self,
        requests: list[_RequestState],
        token_ids: Sequence[int],
        cache: Any,
        timestamp: float,
    ) -> tuple[list[_RequestState], Any, list[SchedulerEvent]]:
        if len(requests) != len(token_ids):
            raise RuntimeError("backend returned the wrong number of batch tokens")

        survivors: list[_RequestState] = []
        survivor_indices: list[int] = []
        events: list[SchedulerEvent] = []
        for index, (request, token_id) in enumerate(zip(requests, token_ids)):
            token_id = int(token_id)
            if token_id in self.backend.eos_token_ids:
                finished = True
                public_token: int | None = None
            else:
                request.token_ids.append(token_id)
                request.token_timestamps.append(timestamp)
                finished = len(request.token_ids) >= request.max_tokens
                public_token = token_id

            events.append(
                SchedulerEvent(
                    request_id=request.request_id,
                    token_id=public_token,
                    timestamp=timestamp,
                    finished=finished,
                )
            )
            if finished:
                self._results[request.request_id] = ScheduledResult(
                    request_id=request.request_id,
                    token_ids=tuple(request.token_ids),
                    token_timestamps=tuple(request.token_timestamps),
                    submitted_at=request.submitted_at,
                    completed_at=timestamp,
                )
            else:
                survivor_indices.append(index)
                survivors.append(request)

        if survivors:
            cache = self.backend.filter_batch_cache(cache, survivor_indices)
        else:
            cache = None
        return survivors, cache, events

