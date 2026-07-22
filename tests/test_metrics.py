"""Behavioral tests for deterministic serving metrics."""

import pytest

from nanoserve.metrics import (
    cache_hit_rate,
    percentile_summary,
    request_metrics,
)


def test_request_metrics_uses_first_token_for_ttft_and_intervals_for_tpot() -> None:
    metrics = request_metrics(
        started_at=10.0,
        token_timestamps=[10.2, 10.5, 10.9],
    )

    assert metrics.ttft_seconds == pytest.approx(0.2)
    assert metrics.tpot_seconds == pytest.approx(0.35)
    assert metrics.tokens_per_second == pytest.approx(3 / 0.9)
    assert metrics.output_tokens == 3


def test_single_token_request_has_ttft_but_no_tpot() -> None:
    metrics = request_metrics(started_at=3.0, token_timestamps=[3.25])

    assert metrics.ttft_seconds == pytest.approx(0.25)
    assert metrics.tpot_seconds is None
    assert metrics.tokens_per_second == pytest.approx(4.0)


def test_request_metrics_rejects_missing_or_non_monotonic_timestamps() -> None:
    with pytest.raises(ValueError, match="at least one"):
        request_metrics(started_at=1.0, token_timestamps=[])

    with pytest.raises(ValueError, match="monotonic"):
        request_metrics(started_at=1.0, token_timestamps=[1.3, 1.2])


def test_percentile_summary_uses_linear_percentiles() -> None:
    summary = percentile_summary([0.1, 0.2, 0.3, 0.4])

    assert summary.p50 == pytest.approx(0.25)
    assert summary.p95 == pytest.approx(0.385)
    assert summary.p99 == pytest.approx(0.397)


def test_percentile_summary_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percentile_summary([])


def test_cache_hit_rate_reports_hits_over_lookups() -> None:
    assert cache_hit_rate([True, False, True]) == pytest.approx(2 / 3)
    assert cache_hit_rate([]) == 0.0
