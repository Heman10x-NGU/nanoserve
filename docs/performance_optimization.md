# Decode performance optimization

This report records the optimization work performed on 2026-07-25. It includes
the original baseline, isolated candidate measurements, accepted implementation,
failed experiment, and final benchmark. Results are local measurements on one
Apple M4 and are not claims about other machines or models.

## Reproduction environment

- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- Python: 3.11.15
- MLX: 0.32.0
- MLX-LM: 0.31.3
- Before commit: `89eccd0`
- Optimized runtime commit: `7f790d7`
- Sampling: greedy

The standard latency benchmark used 10 requests and 32 output tokens. Cache
results used five paired cold/warm runs with a 576-token reusable prefix. The
MLX-LM comparison used five pairs in one loaded-model process and alternated
measurement order. Raw percentile summaries and request rows are under
`results/published/`.

## Before and after

| Metric | Before | After | Change |
|---|---:|---:|---:|
| TTFT p50 | 70.29 ms | 65.25 ms | 7.16% lower |
| TPOT p50 | 4.72 ms | 3.73 ms | 20.90% lower |
| end-to-end latency p50 | 212.66 ms | 186.42 ms | 12.34% lower |
| nanoserve throughput p50 | 150.47 tok/s | 171.65 tok/s | 14.08% higher |
| cache cold TTFT p50 | 242.71 ms | 245.98 ms | 1.35% higher |
| cache warm TTFT p50 | 78.12 ms | 80.77 ms | 3.38% higher |

In the final paired reference run, nanoserve measured 171.65 tok/s and
`mlx_lm.generate` measured 165.55 tok/s. This is a result for this particular
run, not evidence that nanoserve is universally faster than MLX-LM.

## What changed

The old loop completed one model forward, synchronously evaluated the sampled
token and every cache array, converted the token on the CPU, yielded it, and
only then constructed the next step.

The optimized loop asynchronously schedules the current token. Before yielding
it, the loop constructs and schedules the next one-token forward. Calling
`item()` remains the synchronization boundary for the current token, so TTFT
and TPOT timestamps still represent materialized token availability. CPU
detokenization and response handling can overlap the next GPU step.

Batch token paths now materialize their output tokens without separately
requesting evaluation of every cache array. Evaluating the token already
evaluates the model dependency graph that produced it.

This follows the double-buffer pattern used by
[MLX-LM 0.31.3 `generate_step`](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py).
MLX documents both the fixed cost of excessive graph evaluation and the fact
that scalar `item()` access evaluates an array in its
[lazy-evaluation guide](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

## Candidate evaluation

### `mx.async_eval` and double buffering: accepted

An isolated 64-token decode probe improved from 214.76 to 269.23 tok/s, a
25.37% increase. The full benchmark retained a smaller but still material
14.08% throughput improvement and 20.90% lower TPOT.

### Stop evaluating all cache arrays: accepted as part of the pipeline

Changing only the evaluated outputs improved the isolated loop by 0.54%.
Therefore this was not a large independent optimization, but it is the correct
synchronization shape for double buffering.

### Larger prefill blocks: tested and rejected

For a 639-token isolated prefill, 512-token blocks reduced median time from
192.40 to 178.64 ms, a 7.15% improvement. Blocks of 1024 and 2048 were slower
than 64 on this workload, so the proposed 2–2.5x improvement did not reproduce.

The complete cache benchmark exposed the decisive tradeoff. Moving the
canonical block from 64 to 512 improved cold TTFT from 243.49 to 231.21 ms
(5.04%), but reduced the reusable prefix from 576 to 512 tokens and regressed
warm TTFT from 80.04 to 103.20 ms (28.93%). The candidate was reverted.

### Cache checkout/return: rejected

MLX's standard `KVCache` writes new keys and values into mutable backing arrays.
Giving the stored object to a caller would make concurrent cache hits alias
mutable request state. A safe copy-on-grow fork was also probed, but reduced
median clone time only from 0.512 to 0.064 ms—an absolute 0.448 ms saving. Cache
cloning was not the observed 78 ms warm-TTFT bottleneck.

PagedAttention systems solve sharing with block-level ownership and
copy-on-write rather than exclusive checkout; see the
[PagedAttention paper](https://arxiv.org/abs/2309.06180).

### Compile the whole decode step: deferred

MLX compilation can fuse graph operations, but compiled functions are intended
to be pure. Captured mutable state must be declared as input and output, while
shape changes can cause recompilation. Nanoserve's cache objects mutate on each
token, so wrapping the existing method in `mx.compile()` would not be a safe
five-line optimization. See the
[MLX compilation guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html).

## Correctness and limitations

The optimized implementation passed 19 pure tests and five real-model
integration tests. Those gates require exact equality with a serial greedy
reference, cold/warm prefix token identity, streaming/completed text identity,
and correct EOS completion.

Before and after commands were run sequentially on the same machine, but they
were not interleaved at the individual-request level. Thermal state and normal
system noise can therefore affect the exact percentages. The MLX-LM comparison
does alternate pair order. Re-run the commands below for current local results:

```bash
MPLCONFIGDIR=results/.mplconfig nanoserve bench --runs 10
MPLCONFIGDIR=results/.mplconfig nanoserve cache --runs 5
MPLCONFIGDIR=results/.mplconfig nanoserve baseline --runs 5
pytest -o addopts='' -q -m integration
```
