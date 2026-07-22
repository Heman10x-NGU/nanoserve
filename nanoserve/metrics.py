"""Pure latency and cache metrics for inference benchmarks."""

from dataclasses import dataclass
from math import fsum
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    """Latency metrics derived from one request's token timestamps."""

    ttft_seconds: float
    tpot_seconds: float | None
    tokens_per_second: float
    output_tokens: int


@dataclass(frozen=True, slots=True)
class PercentileSummary:
    """The latency percentiles reported by nanoserve."""

    p50: float
    p95: float
    p99: float


def request_metrics(
    *, started_at: float, token_timestamps: Sequence[float]
) -> RequestMetrics:
    """Calculate TTFT, mean TPOT, and end-to-end output throughput.

    TTFT spans request start through the first evaluated output token. TPOT is
    the mean interval between evaluated output tokens and is undefined for a
    one-token response. Throughput includes TTFT in its denominator.
    """
    if not token_timestamps:
        raise ValueError("token_timestamps must contain at least one token")

    previous = started_at
    for timestamp in token_timestamps:
        if timestamp < previous:
            raise ValueError("token_timestamps must be monotonic")
        previous = timestamp

    ttft = token_timestamps[0] - started_at
    elapsed = token_timestamps[-1] - started_at
    if elapsed <= 0:
        raise ValueError("the final token timestamp must follow started_at")

    intervals = [
        current - prior
        for prior, current in zip(token_timestamps, token_timestamps[1:])
    ]
    tpot = fsum(intervals) / len(intervals) if intervals else None

    return RequestMetrics(
        ttft_seconds=ttft,
        tpot_seconds=tpot,
        tokens_per_second=len(token_timestamps) / elapsed,
        output_tokens=len(token_timestamps),
    )


def percentile_summary(samples: Sequence[float]) -> PercentileSummary:
    """Return linearly interpolated p50, p95, and p99 values."""
    if not samples:
        raise ValueError("samples must contain at least one value")
    p50, p95, p99 = np.percentile(np.asarray(samples, dtype=float), [50, 95, 99])
    return PercentileSummary(float(p50), float(p95), float(p99))


def cache_hit_rate(hits: Sequence[bool]) -> float:
    """Return cache hits divided by lookups, or zero when no lookup occurred."""
    return sum(hits) / len(hits) if hits else 0.0
