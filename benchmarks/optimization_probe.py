"""Isolate the MLX optimization candidates evaluated in the performance report."""

from __future__ import annotations

import argparse
import json
import statistics
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import mlx.core as mx

from nanoserve.backends.mlx_backend import MLXBackend

ROOT = Path(__file__).resolve().parents[1]


def _milliseconds(values: list[float]) -> list[float]:
    return [round(value * 1000, 6) for value in values]


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 6)


def _prefill_probe(
    backend: MLXBackend, runs: int
) -> dict[str, dict[str, Any]]:
    cache_prompt = json.loads(
        (ROOT / "prompts/cache.json").read_text(encoding="utf-8")
    )
    text = " ".join(
        cache_prompt["prefix_segments"] * int(cache_prompt["repeat"])
    )
    token_ids = backend.encode(f"{text} {cache_prompt['suffix']}")
    baseline_tokens: list[int] | None = None
    rows = {}
    for block_size in (64, 128, 256, 512, 1024, 2048):
        samples = []
        outputs = []
        for run in range(runs + 1):
            cache = backend.new_cache()
            started = perf_counter()
            logits = None
            for start in range(0, len(token_ids), block_size):
                block = token_ids[start : start + block_size]
                logits = backend.model(mx.array([block]), cache=cache)
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            elapsed = perf_counter() - started
            if run:
                samples.append(elapsed)
                outputs.append(int(token.item()))
        if baseline_tokens is None:
            baseline_tokens = outputs
        sample_ms = _milliseconds(samples)
        rows[str(block_size)] = {
            "samples_ms": sample_ms,
            "p50_ms": _median(sample_ms),
            "token_identical_to_64": outputs == baseline_tokens,
            "prompt_tokens": len(token_ids),
        }
    return rows


def _decode_variant(
    backend: MLXBackend,
    prompt_ids: list[int],
    *,
    mode: str,
    max_tokens: int,
) -> tuple[list[int], float, float]:
    cache = backend.new_cache()
    started = perf_counter()
    logits = backend._prefill(prompt_ids, cache)
    token_ids = []
    timestamps = []

    if mode in {"serial_all", "serial_token"}:
        previous_token = None
        for _ in range(max_tokens):
            if previous_token is not None:
                logits = backend.model(
                    mx.array([[previous_token]]),
                    cache=cache,
                )
            token = mx.argmax(logits[:, -1, :], axis=-1)
            if mode == "serial_all":
                mx.eval(token, [entry.state for entry in cache])
            else:
                mx.eval(token)
            previous_token = int(token.item())
            token_ids.append(previous_token)
            timestamps.append(perf_counter())
    elif mode == "double_buffer":
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.async_eval(token)
        for index in range(max_tokens):
            if index + 1 < max_tokens:
                next_logits = backend.model(token[None], cache=cache)
                next_token = mx.argmax(next_logits[:, -1, :], axis=-1)
                mx.async_eval(next_token)
            token_ids.append(int(token.item()))
            timestamps.append(perf_counter())
            if index + 1 < max_tokens:
                token = next_token
    else:
        raise ValueError(f"unknown decode mode: {mode}")

    return token_ids, timestamps[0] - started, timestamps[-1] - timestamps[0]


def _decode_probe(
    backend: MLXBackend, runs: int
) -> dict[str, dict[str, Any]]:
    prompts = json.loads(
        (ROOT / "prompts/bench.json").read_text(encoding="utf-8")
    )
    prompt_ids = backend.encode(prompts[0])
    baseline_tokens: list[list[int]] | None = None
    rows = {}
    for mode in ("serial_all", "serial_token", "double_buffer"):
        ttft_samples = []
        decode_samples = []
        outputs = []
        for run in range(runs + 1):
            tokens, ttft, elapsed = _decode_variant(
                backend,
                prompt_ids,
                mode=mode,
                max_tokens=64,
            )
            if run:
                ttft_samples.append(ttft)
                decode_samples.append(elapsed)
                outputs.append(tokens)
        if baseline_tokens is None:
            baseline_tokens = outputs
        ttft_ms = _milliseconds(ttft_samples)
        decode_ms = _milliseconds(decode_samples)
        rows[mode] = {
            "ttft_samples_ms": ttft_ms,
            "decode_63_intervals_samples_ms": decode_ms,
            "ttft_p50_ms": _median(ttft_ms),
            "decode_63_intervals_p50_ms": _median(decode_ms),
            "tokens_per_second_p50": round(
                63 / statistics.median(decode_samples),
                6,
            ),
            "token_identical_to_serial_all": outputs == baseline_tokens,
        }
    return rows


def _fork_cache(cache: list[Any]) -> list[Any]:
    return [
        type(entry).from_state(entry.state, entry.meta_state)
        for entry in cache
    ]


def _cache_fork_probe(
    backend: MLXBackend, runs: int
) -> dict[str, Any]:
    cache_prompt = json.loads(
        (ROOT / "prompts/cache.json").read_text(encoding="utf-8")
    )
    text = " ".join(
        cache_prompt["prefix_segments"] * int(cache_prompt["repeat"])
    )
    full_ids = backend.encode(f"{text} {cache_prompt['suffix']}")
    prefix_ids = full_ids[:576]
    suffix_ids = full_ids[576:]
    source = backend.forward_logits(prefix_ids).cache
    source_offsets = [entry.offset for entry in source]

    deep_samples = []
    fork_samples = []
    for _ in range(runs):
        started = perf_counter()
        backend.clone_cache(source)
        deep_samples.append(perf_counter() - started)
        started = perf_counter()
        _fork_cache(source)
        fork_samples.append(perf_counter() - started)

    fork_a = _fork_cache(source)
    fork_b = _fork_cache(source)
    cold = backend.generate(full_ids, max_tokens=12)
    warm_a = backend.generate(suffix_ids, cache=fork_a, max_tokens=12)
    warm_b = backend.generate(suffix_ids, cache=fork_b, max_tokens=12)
    deep_ms = _milliseconds(deep_samples)
    fork_ms = _milliseconds(fork_samples)
    return {
        "deep_clone_samples_ms": deep_ms,
        "copy_on_grow_fork_samples_ms": fork_ms,
        "deep_clone_p50_ms": _median(deep_ms),
        "copy_on_grow_fork_p50_ms": _median(fork_ms),
        "source_offsets_unchanged": source_offsets
        == [entry.offset for entry in source],
        "two_forks_token_identical": (
            cold.token_ids == warm_a.token_ids == warm_b.token_ids
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least one")

    backend = MLXBackend.load()
    results = {
        "runs": args.runs,
        "warmup_runs_per_variant": 1,
        "model": backend.model_id,
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
        "prefill": _prefill_probe(backend, args.runs),
        "decode": _decode_probe(backend, args.runs),
        "cache_fork": _cache_fork_probe(backend, args.runs),
    }
    rendered = json.dumps(results, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
