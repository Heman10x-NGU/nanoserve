"""Reproducible benchmark reporting for nanoserve."""

from __future__ import annotations

import json
import platform
import subprocess
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
    """Return portable host metadata without collecting user or secret data."""
    chip = platform.processor() or "unknown"
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            chip = completed.stdout.strip()
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": chip,
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


def write_batch_report(
    *,
    rows: Sequence[dict[str, Any]],
    model_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Write concurrency latency percentiles and a p95 scaling chart."""
    if not rows:
        raise ValueError("rows must contain at least one batch benchmark result")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for concurrency in sorted({int(row["concurrency"]) for row in rows}):
        group = [row for row in rows if int(row["concurrency"]) == concurrency]
        ttft = percentile_summary([float(row["ttft_seconds"]) for row in group])
        latency = percentile_summary(
            [float(row["latency_seconds"]) for row in group]
        )
        summaries[str(concurrency)] = {
            "requests": len(group),
            "ttft_seconds": asdict(ttft),
            "latency_seconds": asdict(latency),
        }
    report = {
        "model": model_id,
        "system": system_info(model_id),
        "concurrency": summaries,
        "requests": list(rows),
    }
    (output_dir / "batch_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_batch_chart(summaries, output_dir)
    return report


def write_baseline_report(
    *,
    rows: Sequence[dict[str, Any]],
    model_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Write an end-to-end throughput comparison with mlx_lm.generate."""
    if not rows:
        raise ValueError("rows must contain at least one baseline result")
    output_dir.mkdir(parents=True, exist_ok=True)
    implementations: dict[str, Any] = {}
    for implementation in ("nanoserve", "mlx_lm.generate"):
        group = [row for row in rows if row["implementation"] == implementation]
        if not group:
            raise ValueError(f"missing baseline rows for {implementation}")
        latency = percentile_summary(
            [float(row["latency_seconds"]) for row in group]
        )
        throughput = percentile_summary(
            [float(row["tokens_per_second"]) for row in group]
        )
        implementations[implementation] = {
            "runs": len(group),
            "latency_seconds": asdict(latency),
            "tokens_per_second": asdict(throughput),
        }
    report = {
        "model": model_id,
        "system": system_info(model_id),
        "method": (
            "Same loaded model, tokenizer, fixed prompts, greedy decoding, and "
            "requested token limit. Pair order alternates. End-to-end wall time "
            "includes prefill and decode; mlx_lm.generate output text is re-tokenized "
            "because that public API returns text rather than token IDs."
        ),
        "implementations": implementations,
        "requests": list(rows),
    }
    (output_dir / "baseline_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_baseline_chart(implementations, output_dir)
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


def _write_batch_chart(summaries: dict[str, Any], output_dir: Path) -> None:
    concurrency = sorted(int(value) for value in summaries)
    ttft_p95 = [
        summaries[str(value)]["ttft_seconds"]["p95"] * 1000
        for value in concurrency
    ]
    latency_p95 = [
        summaries[str(value)]["latency_seconds"]["p95"] * 1000
        for value in concurrency
    ]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(concurrency, ttft_p95, marker="o", label="TTFT p95")
    axis.plot(concurrency, latency_p95, marker="s", label="end-to-end p95")
    axis.set(
        title="Continuous batching latency under concurrency",
        xlabel="concurrent requests",
        ylabel="milliseconds",
        xticks=concurrency,
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "batch_benchmark.png", dpi=160)
    plt.close(figure)


def _write_baseline_chart(
    implementations: dict[str, Any], output_dir: Path
) -> None:
    labels = ["nanoserve", "mlx_lm.generate"]
    throughput = [
        implementations[label]["tokens_per_second"]["p50"] for label in labels
    ]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.bar(labels, throughput, color=["#2563eb", "#64748b"])
    axis.bar_label(bars, fmt="%.1f tok/s", padding=3)
    axis.set(
        title="End-to-end generation throughput",
        ylabel="output tokens per second",
    )
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "baseline_benchmark.png", dpi=160)
    plt.close(figure)
