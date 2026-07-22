"""Deterministic scheduler tests through its public submission/step seam."""

import pytest

from nanoserve.engine.scheduler import ContinuousBatchScheduler


class FakeBatchBackend:
    eos_token_ids: set[int] = set()

    def __init__(self) -> None:
        self.prefill_batches: list[list[list[int]]] = []
        self.decode_batches: list[list[int]] = []

    def prefill_batch(self, prompts: list[list[int]]):
        self.prefill_batches.append(prompts)
        return [prompt[-1] + 1 for prompt in prompts], [list(p) for p in prompts]

    def decode_batch(self, token_ids: list[int], cache):
        self.decode_batches.append(list(token_ids))
        return [token_id + 1 for token_id in token_ids], cache

    def extend_batch_cache(self, active, admitted):
        return active + admitted

    def filter_batch_cache(self, cache, indices: list[int]):
        return [cache[index] for index in indices]


def test_scheduler_batches_waiting_requests_up_to_capacity() -> None:
    backend = FakeBatchBackend()
    scheduler = ContinuousBatchScheduler(backend, max_batch_size=2)
    scheduler.submit([10], max_tokens=2, request_id="a")
    scheduler.submit([20], max_tokens=2, request_id="b")
    scheduler.submit([30], max_tokens=2, request_id="c")

    events = scheduler.step(now=1.0)

    assert backend.prefill_batches == [[[10], [20]]]
    assert backend.decode_batches == [[11, 21]]
    assert [(event.request_id, event.token_id) for event in events] == [
        ("a", 11),
        ("b", 21),
        ("a", 12),
        ("b", 22),
    ]
    assert scheduler.pending_count == 1
    assert scheduler.active_count == 0


def test_scheduler_admits_new_work_while_an_older_request_is_active() -> None:
    backend = FakeBatchBackend()
    scheduler = ContinuousBatchScheduler(backend, max_batch_size=2)
    scheduler.submit([1], max_tokens=3, request_id="older")
    scheduler.step(now=1.0)
    scheduler.submit([100], max_tokens=3, request_id="new")

    events = scheduler.step(now=2.0)

    assert backend.prefill_batches[-1] == [[100]]
    assert backend.decode_batches[-1] == [3, 101]
    assert [(event.request_id, event.token_id) for event in events] == [
        ("new", 101),
        ("older", 4),
        ("new", 102),
    ]
    assert scheduler.active_count == 1


def test_scheduler_records_completed_result_and_token_timestamps() -> None:
    backend = FakeBatchBackend()
    scheduler = ContinuousBatchScheduler(backend, max_batch_size=1)
    scheduler.submit([7], max_tokens=2, request_id="request", submitted_at=0.5)

    scheduler.step(now=1.25)
    result = scheduler.pop_result("request")

    assert result is not None
    assert result.token_ids == (8, 9)
    assert result.token_timestamps == (1.25, 1.25)
    assert result.submitted_at == 0.5
    assert result.completed_at == 1.25


def test_eos_finishes_without_exposing_the_stop_token() -> None:
    backend = FakeBatchBackend()
    backend.eos_token_ids = {8}
    scheduler = ContinuousBatchScheduler(backend, max_batch_size=1)
    scheduler.submit([7], max_tokens=5, request_id="request")

    events = scheduler.step(now=1.0)
    result = scheduler.pop_result("request")

    assert [(event.token_id, event.finished) for event in events] == [(None, True)]
    assert result is not None
    assert result.token_ids == ()
    assert backend.decode_batches == []


@pytest.mark.integration
def test_mlx_scheduler_batches_active_requests_in_one_forward() -> None:
    from nanoserve.backends.base import DEFAULT_MODEL
    from nanoserve.backends.mlx_backend import MLXBackend

    backend = MLXBackend.load(DEFAULT_MODEL)
    scheduler = ContinuousBatchScheduler(backend, max_batch_size=2)
    scheduler.submit(backend.encode("The capital of France is"), max_tokens=4, request_id="a")
    scheduler.submit(backend.encode("The capital of Japan is"), max_tokens=4, request_id="b")

    while scheduler.has_work:
        scheduler.step()

    first = scheduler.pop_result("a")
    second = scheduler.pop_result("b")
    assert first is not None and len(first.token_ids) == 4
    assert second is not None and len(second.token_ids) == 4
    assert backend.batch_forward_count == 4
    assert backend.batch_token_slots == 8
