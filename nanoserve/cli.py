"""Command-line interface for nanoserve."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nanoserve.backends.base import DEFAULT_MODEL, GenerationResult
from nanoserve.metrics import request_metrics

app = typer.Typer(
    no_args_is_help=True,
    help="A minimal MLX inference and serving engine.",
)


@app.command()
def bench(
    runs: int = typer.Option(10, min=1, help="Benchmark repetitions."),
    max_tokens: int = typer.Option(32, min=1, help="Output-token limit per run."),
    model: str = typer.Option(DEFAULT_MODEL, help="MLX model ID or local path."),
    output_dir: Path = typer.Option(Path("results"), help="Artifact directory."),
) -> None:
    """Benchmark the hand-written decode loop."""
    # Import lazily so CLI help and pure-metric tests do not initialize Metal.
    from nanoserve.backends.mlx_backend import MLXBackend
    from nanoserve.report import write_benchmark_report

    prompt_path = Path(__file__).parents[1] / "prompts" / "bench.json"
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    backend = MLXBackend.load(model)
    rows = []

    for run_index in range(runs):
        prompt = prompts[run_index % len(prompts)]
        result = backend.generate(prompt, max_tokens=max_tokens)
        if not isinstance(result, GenerationResult):
            raise RuntimeError("non-streaming generation returned a token iterator")
        metrics = request_metrics(
            started_at=result.started_at,
            token_timestamps=result.token_timestamps,
        )
        rows.append(
            {
                "run": run_index + 1,
                "prompt_id": run_index % len(prompts),
                "prompt_tokens": result.prompt_tokens,
                "output_tokens": metrics.output_tokens,
                "ttft_seconds": metrics.ttft_seconds,
                "tpot_seconds": metrics.tpot_seconds,
                "tokens_per_second": metrics.tokens_per_second,
            }
        )

    report = write_benchmark_report(
        rows=rows,
        model_id=backend.model_id,
        output_dir=output_dir,
    )
    _print_benchmark_table(report)


@app.command("cache")
def cache_command(
    runs: int = typer.Option(5, min=1, help="Paired cold/warm repetitions."),
    max_tokens: int = typer.Option(16, min=1, help="Output-token limit."),
    model: str = typer.Option(DEFAULT_MODEL, help="MLX model ID or local path."),
    output_dir: Path = typer.Option(Path("results"), help="Artifact directory."),
) -> None:
    """Compare cold and prefix-cache-warm time to first token."""
    from nanoserve.backends.base import PREFILL_BLOCK_SIZE
    from nanoserve.backends.mlx_backend import MLXBackend
    from nanoserve.engine.kv_cache import PrefixCache
    from nanoserve.report import write_cache_report

    prompt_path = Path(__file__).parents[1] / "prompts" / "cache.json"
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    prefix_text = " ".join(prompt["prefix_segments"] * int(prompt["repeat"]))
    backend = MLXBackend.load(model)
    full_ids = backend.encode(f"{prefix_text} {prompt['suffix']}")
    prefix_length = ((len(full_ids) - 16) // PREFILL_BLOCK_SIZE) * PREFILL_BLOCK_SIZE
    if prefix_length < PREFILL_BLOCK_SIZE:
        raise RuntimeError("cache prompt is too short for one reusable block")

    # Keep exactly one loaded model process throughout the benchmark.
    prefix_ids = full_ids[:prefix_length]
    prefix_state = backend.forward_logits(prefix_ids).cache
    prefix_cache = PrefixCache(
        namespace=backend.cache_namespace,
        block_size=PREFILL_BLOCK_SIZE,
        clone=backend.clone_cache,
    )
    prefix_cache.put(prefix_ids, prefix_state)

    # Compile both paths before collecting paired measurements.
    backend.generate(full_ids, max_tokens=2)
    warmup_match = prefix_cache.longest_prefix(full_ids)
    assert warmup_match is not None
    backend.generate(
        full_ids[warmup_match.prefix_length :],
        cache=warmup_match.cache,
        max_tokens=2,
    )

    rows = []
    for run_index in range(runs):
        cold = backend.generate(full_ids, max_tokens=max_tokens)
        match = prefix_cache.longest_prefix(full_ids)
        if match is None:
            raise RuntimeError("published cache entry unexpectedly missed")
        warm = backend.generate(
            full_ids[match.prefix_length :],
            cache=match.cache,
            max_tokens=max_tokens,
        )
        if not isinstance(cold, GenerationResult) or not isinstance(
            warm, GenerationResult
        ):
            raise RuntimeError("cache benchmark requires completed generations")
        identical = cold.token_ids == warm.token_ids
        if not identical:
            raise RuntimeError(
                f"token identity gate failed on paired run {run_index + 1}"
            )
        cold_metrics = request_metrics(
            started_at=cold.started_at,
            token_timestamps=cold.token_timestamps,
        )
        warm_metrics = request_metrics(
            started_at=warm.started_at,
            token_timestamps=warm.token_timestamps,
        )
        rows.append(
            {
                "run": run_index + 1,
                "cold_ttft_seconds": cold_metrics.ttft_seconds,
                "warm_ttft_seconds": warm_metrics.ttft_seconds,
                "token_identical": identical,
            }
        )

    report = write_cache_report(
        rows=rows,
        model_id=backend.model_id,
        prefix_tokens=prefix_length,
        cache_hit_rate=prefix_cache.hit_rate,
        output_dir=output_dir,
    )
    _print_cache_table(report)


@app.command()
def serve() -> None:
    """Run the OpenAI-compatible streaming server."""
    typer.echo("Phase 3 server is not implemented yet.", err=True)
    raise typer.Exit(code=2)


def _print_benchmark_table(report: dict[str, object]) -> None:
    table = Table(title=f"nanoserve benchmark ({report['runs']} requests)")
    table.add_column("metric")
    table.add_column("p50", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("p99", justify="right")
    for label, key in (("TTFT", "ttft_seconds"), ("TPOT", "tpot_seconds")):
        values = report[key]
        if not isinstance(values, dict):
            continue
        table.add_row(
            label,
            f"{float(values['p50']) * 1000:.2f} ms",
            f"{float(values['p95']) * 1000:.2f} ms",
            f"{float(values['p99']) * 1000:.2f} ms",
        )
    Console().print(table)


def _print_cache_table(report: dict[str, object]) -> None:
    cold = report["cold_ttft_seconds"]
    warm = report["warm_ttft_seconds"]
    if not isinstance(cold, dict) or not isinstance(warm, dict):
        raise TypeError("cache report percentile rows are missing")
    table = Table(title=f"prefix reuse ({report['prefix_tokens']} cached tokens)")
    table.add_column("path")
    table.add_column("p50 TTFT", justify="right")
    table.add_column("p95 TTFT", justify="right")
    table.add_row(
        "cold",
        f"{float(cold['p50']) * 1000:.2f} ms",
        f"{float(cold['p95']) * 1000:.2f} ms",
    )
    table.add_row(
        "warm",
        f"{float(warm['p50']) * 1000:.2f} ms",
        f"{float(warm['p95']) * 1000:.2f} ms",
    )
    Console().print(table)
    Console().print(
        f"p50 TTFT drop: {float(report['p50_ttft_drop_fraction']) * 100:.1f}% | "
        f"token-identical: {report['token_identical']}"
    )
