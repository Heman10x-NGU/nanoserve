# Architecture

Nanoserve keeps policy and tensor mechanics separate. `Backend` is the narrow
single-request seam consumed by metrics and future adapters. The scheduler uses
a small batch extension of the MLX adapter because batch-cache shape changes
are device-specific.

```text
POST /v1/completions
        |
        v
ServingEngine -> waiting queue -> ContinuousBatchScheduler
                                      | prefill admitted prompts together
                                      | decode every active row together
                                      v
                                  MLXBackend
                                      |
                       Qwen2.5 model + BatchKVCache
```

## Hand-written generation

`MLXBackend.generate` tokenizes once, creates one cache object per transformer
layer, and prefills the prompt in stable 64-token blocks. It selects the final
position's argmax, forces MLX evaluation, records the timestamp, and passes only
that sampled token into the next model call. The loop never calls
`mlx_lm.generate`, `stream_generate`, or `generate_step`.

The evaluation barrier matters because MLX is lazy. A timestamp taken before
`mx.eval` measures Python dispatch, not token availability.

## Prefix reuse

`PrefixCache` stores complete per-layer cache snapshots at full block
boundaries. Its key is a SHA-256 chain:

```text
H0 = SHA256(format_version || model_and_tokenizer_namespace)
Hi = SHA256(Hi-1 || block_length || token_ids_in_block_i)
```

Lookup finds the longest exact cached prefix, clones its arrays, and forwards
only the suffix. Cold and warm prefills use identical block grouping. This
avoids small floating-point changes from different reduction shapes that can
eventually flip a greedy choice. The integration test is the authority: cold
and warm output tokens must be identical.

This is a bounded LRU of whole snapshots, not a paged allocator. It spends more
memory to keep the correctness argument readable.

## Continuous batching

The scheduler owns waiting, active, and completed request state. Each `step()`:

1. admits waiting requests up to `max_batch_size`;
2. prefills admitted prompts in one batched forward;
3. extends the active `BatchKVCache` rows;
4. advances every active request with one shared decode forward; and
5. filters finished rows from the cache.

The active set may change between steps, which is the defining property of
continuous batching. The server runs synchronous Metal steps on a worker thread
and routes evaluated token events into per-request async queues. FastAPI exposes
those chunks as OpenAI-style server-sent events.

## Extension boundary

A future CUDA adapter can implement `Backend` without changing metrics or HTTP
payloads. Production batching would benefit from making the batch operations a
formal second protocol and adding cancellation, token budgets, backpressure,
paged allocation, and failure cleanup. Those are explicit future work, not v1
claims.
