"""Command-line interface for nanoserve."""

import typer

app = typer.Typer(
    no_args_is_help=True,
    help="A minimal MLX inference and serving engine.",
)


@app.command()
def bench(runs: int = typer.Option(10, min=1, help="Benchmark repetitions.")) -> None:
    """Benchmark the hand-written decode loop."""
    raise typer.Exit("Phase 1 benchmark is not implemented yet.")


@app.command()
def cache() -> None:
    """Compare cold and prefix-cache-warm time to first token."""
    raise typer.Exit("Phase 2 cache benchmark is not implemented yet.")


@app.command()
def serve() -> None:
    """Run the OpenAI-compatible streaming server."""
    raise typer.Exit("Phase 3 server is not implemented yet.")

