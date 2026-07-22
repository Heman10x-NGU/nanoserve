# nanoserve | minimal MLX inference engine

Built a from-scratch, readable LLM serving path for Apple Silicon with a
hand-written autoregressive decode loop, exact block-hashed KV prefix reuse,
continuous batching, and an OpenAI-compatible streaming FastAPI endpoint.

**Measured proof:** on an M4 with Qwen2.5-0.5B-Instruct-4bit, a reusable
576-token prefix reduced median TTFT from 344.44 ms to 117.48 ms (65.9%) across
five paired runs. Cold and warm greedy outputs were token-identical. Concurrency
1/2/4/8 reports include per-request raw data and p50/p95/p99 latency.

**Engineering depth:** MLX lazy-evaluation-aware timing, per-layer KV cache
cloning and invalidation, dynamic batch row extension/filtering, async token
routing, deterministic metric tests, and real-model integration gates.

**Honest boundary:** mlx-lm's optimized reference generated 41.74 median tok/s
versus nanoserve's 31.41 tok/s in the paired local baseline. The project sells
mechanistic understanding and reproducible evidence, not a production-speed
claim.

Stack: Python 3.11, MLX, mlx-lm, FastAPI, Typer, pytest, NumPy, Matplotlib.
