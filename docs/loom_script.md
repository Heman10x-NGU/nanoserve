# 90-second demo script

## 0:00-0:12 | Result first

Show the cache chart at the top of the README.

> Nanoserve is a minimal inference engine I built directly on MLX. On this M4,
> reusing a verified 576-token prefix cut median time to first token from 344 to
> 117 milliseconds, with token-identical output across five paired runs.

## 0:12-0:32 | Mechanism

Open `nanoserve/backends/mlx_backend.py` at `_decode` and `_prefill`.

> This is my decode loop. It calls the model directly, greedily samples one
> token, forces MLX evaluation so the timing is real, records the timestamp, and
> feeds only that token back with the mutable KV cache. I do not call the mlx-lm
> generation helpers here.

## 0:32-0:49 | Correct prefix reuse

Open `nanoserve/engine/kv_cache.py`, then the integration test.

> Prefix entries are chained hashes of token blocks plus the model and tokenizer
> namespace. Arrays are cloned at the store boundary. The load-bearing test
> compares cold and restored-cache greedy tokens exactly, not just decoded text.

## 0:49-1:06 | Continuous batching

Open `nanoserve/engine/scheduler.py`.

> Waiting requests join whenever a slot opens. Every active row advances through
> one batched model forward, and finished rows are filtered from the batch cache.
> The committed concurrency report includes raw rows and p50, p95, and p99.

## 1:06-1:21 | Stream a real response

Run `nanoserve serve`, then the README `curl -N` command in a second terminal.

> FastAPI receives an OpenAI-compatible completion request. Each evaluated token
> becomes an SSE chunk immediately, then the stream ends with `[DONE]`.

## 1:21-1:30 | Honest boundary

Show the baseline table.

> mlx-lm's reference generator is faster: 41.74 versus 31.41 median tokens per
> second. This project demonstrates why serving is fast and where my simple
> design pays overhead; it does not claim production parity.
