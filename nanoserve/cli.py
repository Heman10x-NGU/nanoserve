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


@app.command()
def cache() -> None:
    """Compare cold and prefix-cache-warm time to first token."""
    typer.echo("Phase 2 cache benchmark is not implemented yet.", err=True)
    raise typer.Exit(code=2)


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
