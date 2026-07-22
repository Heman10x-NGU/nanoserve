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


@app.command("batch-bench")
def batch_bench(
    runs: int = typer.Option(3, min=1, help="Batches per concurrency level."),
    max_tokens: int = typer.Option(16, min=2, help="Output-token limit."),
    model: str = typer.Option(DEFAULT_MODEL, help="MLX model ID or local path."),
    output_dir: Path = typer.Option(Path("results"), help="Artifact directory."),
) -> None:
    """Measure p50/p95/p99 latency at concurrency 1, 2, 4, and 8."""
    from time import perf_counter

    from nanoserve.backends.mlx_backend import MLXBackend
    from nanoserve.engine.scheduler import ContinuousBatchScheduler
    from nanoserve.report import write_batch_report

    prompt_path = Path(__file__).parents[1] / "prompts" / "bench.json"
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    backend = MLXBackend.load(model)
    encoded_prompts = [backend.encode(prompt) for prompt in prompts]
    rows = []

    # One untimed batch warms the default model kernels.
    warmup = ContinuousBatchScheduler(backend, max_batch_size=1)
    warmup.submit(encoded_prompts[0], max_tokens=2, request_id="warmup")
    while warmup.has_work:
        warmup.step()
    warmup.pop_result("warmup")

    for concurrency in (1, 2, 4, 8):
        for run_index in range(runs):
            scheduler = ContinuousBatchScheduler(
                backend, max_batch_size=concurrency
            )
            request_ids = []
            submitted_at = perf_counter()
            for request_index in range(concurrency):
                request_id = f"c{concurrency}-r{run_index}-q{request_index}"
                scheduler.submit(
                    encoded_prompts[request_index % len(encoded_prompts)],
                    max_tokens=max_tokens,
                    request_id=request_id,
                    submitted_at=submitted_at,
                )
                request_ids.append(request_id)
            while scheduler.has_work:
                scheduler.step()
            for request_id in request_ids:
                result = scheduler.pop_result(request_id)
                if result is None:
                    raise RuntimeError(f"missing completed result: {request_id}")
                metrics = request_metrics(
                    started_at=result.submitted_at,
                    token_timestamps=result.token_timestamps,
                )
                rows.append(
                    {
                        "request_id": request_id,
                        "concurrency": concurrency,
                        "run": run_index + 1,
                        "ttft_seconds": metrics.ttft_seconds,
                        "tpot_seconds": metrics.tpot_seconds,
                        "latency_seconds": result.completed_at - result.submitted_at,
                        "output_tokens": metrics.output_tokens,
                    }
                )

    report = write_batch_report(
        rows=rows,
        model_id=backend.model_id,
        output_dir=output_dir,
    )
    _print_batch_table(report)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, min=1, max=65535, help="Bind port."),
    max_batch_size: int = typer.Option(8, min=1, help="Maximum active requests."),
    model: str = typer.Option(DEFAULT_MODEL, help="MLX model ID or local path."),
) -> None:
    """Run the OpenAI-compatible streaming server."""
    import uvicorn

    from nanoserve.backends.mlx_backend import MLXBackend
    from nanoserve.server import ServingEngine, create_app

    backend = MLXBackend.load(model)
    engine = ServingEngine(backend, max_batch_size=max_batch_size)
    uvicorn.run(create_app(engine), host=host, port=port)


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


def _print_batch_table(report: dict[str, object]) -> None:
    summaries = report["concurrency"]
    if not isinstance(summaries, dict):
        raise TypeError("batch report summaries are missing")
    table = Table(title="continuous batching latency")
    table.add_column("concurrency", justify="right")
    table.add_column("TTFT p50", justify="right")
    table.add_column("TTFT p95", justify="right")
    table.add_column("TTFT p99", justify="right")
    table.add_column("latency p95", justify="right")
    for concurrency in (1, 2, 4, 8):
        summary = summaries[str(concurrency)]
        ttft = summary["ttft_seconds"]
        latency = summary["latency_seconds"]
        table.add_row(
            str(concurrency),
            f"{float(ttft['p50']) * 1000:.2f} ms",
            f"{float(ttft['p95']) * 1000:.2f} ms",
            f"{float(ttft['p99']) * 1000:.2f} ms",
            f"{float(latency['p95']) * 1000:.2f} ms",
        )
    Console().print(table)
