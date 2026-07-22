"""Reproducible benchmark reporting for nanoserve."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nanoserve.metrics import PercentileSummary, percentile_summary


def system_info(model_id: str) -> dict[str, str]:
    """Return portable host metadata without shelling out or collecting secrets."""
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }


def write_benchmark_report(
    *,
    rows: Sequence[dict[str, Any]],
    model_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Write raw JSON, system metadata, and a TTFT/TPOT PNG."""
    if not rows:
        raise ValueError("rows must contain at least one benchmark result")
    output_dir.mkdir(parents=True, exist_ok=True)

    ttft = [float(row["ttft_seconds"]) for row in rows]
    tpot = [
        float(row["tpot_seconds"])
        for row in rows
        if row["tpot_seconds"] is not None
    ]
    ttft_percentiles = percentile_summary(ttft)
    tpot_percentiles = percentile_summary(tpot) if tpot else None
    report = {
        "model": model_id,
        "system": system_info(model_id),
        "runs": len(rows),
        "ttft_seconds": asdict(ttft_percentiles),
        "tpot_seconds": asdict(tpot_percentiles) if tpot_percentiles else None,
        "requests": list(rows),
    }

    (output_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "sysinfo.json").write_text(
        json.dumps(system_info(model_id), indent=2) + "\n", encoding="utf-8"
    )
    _write_latency_chart(rows, ttft_percentiles, tpot_percentiles, output_dir)
    return report


def write_cache_report(
    *,
    rows: Sequence[dict[str, Any]],
    model_id: str,
    prefix_tokens: int,
    cache_hit_rate: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Write cold-vs-warm TTFT evidence and its comparison chart."""
    if not rows:
        raise ValueError("rows must contain at least one cache benchmark result")
    output_dir.mkdir(parents=True, exist_ok=True)
    cold = percentile_summary([float(row["cold_ttft_seconds"]) for row in rows])
    warm = percentile_summary([float(row["warm_ttft_seconds"]) for row in rows])
    drop = (cold.p50 - warm.p50) / cold.p50 if cold.p50 else 0.0
    report = {
        "model": model_id,
        "system": system_info(model_id),
        "runs": len(rows),
        "prefix_tokens": prefix_tokens,
        "cache_hit_rate": cache_hit_rate,
        "token_identical": all(bool(row["token_identical"]) for row in rows),
        "cold_ttft_seconds": asdict(cold),
        "warm_ttft_seconds": asdict(warm),
        "p50_ttft_drop_fraction": drop,
        "requests": list(rows),
    }
    (output_dir / "cache_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_cache_chart(rows, output_dir)
    return report


def _write_latency_chart(
    rows: Sequence[dict[str, Any]],
    ttft: PercentileSummary,
    tpot: PercentileSummary | None,
    output_dir: Path,
) -> None:
    runs = list(range(1, len(rows) + 1))
    ttft_ms = [float(row["ttft_seconds"]) * 1000 for row in rows]
    tpot_ms = [
        float(row["tpot_seconds"]) * 1000
        if row["tpot_seconds"] is not None
        else float("nan")
        for row in rows
    ]

    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(runs, ttft_ms, marker="o", label="TTFT")
    axis.plot(runs, tpot_ms, marker="s", label="mean TPOT")
    axis.axhline(ttft.p95 * 1000, linestyle="--", alpha=0.6, label="TTFT p95")
    if tpot is not None:
        axis.axhline(
            tpot.p95 * 1000,
            linestyle=":",
            alpha=0.6,
            label="TPOT p95",
        )
    axis.set(
        title="nanoserve latency by request",
        xlabel="request",
        ylabel="milliseconds",
        xticks=runs,
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "benchmark.png", dpi=160)
    plt.close(figure)


def _write_cache_chart(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    runs = list(range(1, len(rows) + 1))
    cold_ms = [float(row["cold_ttft_seconds"]) * 1000 for row in rows]
    warm_ms = [float(row["warm_ttft_seconds"]) * 1000 for row in rows]
    width = 0.36

    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar([run - width / 2 for run in runs], cold_ms, width, label="cold")
    axis.bar([run + width / 2 for run in runs], warm_ms, width, label="warm")
    axis.set(
        title="Prefix reuse: cold vs warm time to first token",
        xlabel="paired run",
        ylabel="TTFT (milliseconds)",
        xticks=runs,
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "cache_benchmark.png", dpi=160)
    plt.close(figure)
