# Phase 0 reading notes

These notes pin the sources read before implementing the cache. They are not a
claim that nanoserve reimplements any source wholesale; the goal is to retain
the small set of design invariants that make reuse correct.

## Sources read

- [nano-vllm at `bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8): `sequence.py`, `block_manager.py`, `scheduler.py`, `llm_engine.py`, `model_runner.py`, and the attention path.
- [vLLM at `a287eb1`](https://github.com/vllm-project/vllm/tree/a287eb163fb6f8f007a4a78411fb54c8dde64cc7/vllm/v1/core): `block_pool.py`, `kv_cache_manager.py`, full-attention prefix lookup, hash construction, and the scheduler's running/waiting paths.
- [mlx-lm at `cf10f96`](https://github.com/ml-explore/mlx-lm/tree/cf10f962b7a20e63a6df43dbf0faf06070153d40): `generate_step`, `generate`, `models/cache.py`, and Qwen2's attention/cache path.
- [still-mini-kv-compactor at `f1dbdae`](https://github.com/Heman10x-NGU/still-mini-kv-compactor/tree/f1dbdaec1a06c7411774baf0afba667a15f2779e): cache canonicalization, byte accounting, and direct-versus-replayed-logit tests.

## How does a prefix hash map to reusable KV?

1. Tokenization comes first. Cache identity is over token IDs, never raw text.
2. Tokens are divided into fixed-size blocks. Each full block receives a
   chained hash of `(parent_hash, current_block_token_ids, identity_keys)`.
   Including the parent makes a block identify the entire prefix ending there,
   not merely a repeated token fragment at an unrelated position.
3. The cache store maps that hash to a cache entry containing one per-layer
   key/value tensor pair plus its covered token count. The tensors have the MLX
   attention layout `[batch, kv_heads, sequence, head_dim]`.
4. Lookup walks block hashes from the start and stops at the first miss. Chained
   hashes mean a later block cannot be a valid hit after an earlier miss.
5. On a hit, generation restores a fresh copy of every layer cache and sets its
   offset to the reused prefix length. Only suffix tokens are passed to the
   model. RoPE uses the restored offset, and attention reads cached prefix K/V
   plus newly appended suffix K/V.
6. If the full prompt is cached, at least its last token must be recomputed to
   produce next-token logits. A cache entry represents model state, not logits.
7. Reuse is acceptable only when cold full-context greedy generation and warm
   prefix-replay greedy generation produce identical token IDs. The test checks
   the behavior through the backend interface, not internal dictionary state.

For v1, nanoserve stores complete per-layer cache snapshots only at full 64-token
prefix-block boundaries rather than implementing vLLM's paged allocator. Cold
and warm prefills use the same 64-token grouping because changing MLX reduction
shapes can introduce small floating-point differences that later flip a greedy
choice. This is enough to prove the prefix-reuse mechanism on one process while
keeping the module small. vLLM's block pool, reference counts, duplicate hashes, LRU
eviction, partial blocks, hybrid cache groups, and distributed cache transfer
remain cited future design, not implied functionality.

## What invalidates a block?

A cached prefix is invalid if any input capable of changing its K/V tensors is
different:

- any token ID or token order in the covered prefix;
- the model identity, exact weight revision, or active adapter/LoRA;
- tokenizer identity or revision, chat template, or special-token policy when
  text is the caller input;
- attention-affecting model configuration, including RoPE/scaling behavior;
- cache tensor layout, dtype, backend/cache format version, or layer count;
- multimodal features or prompt embeddings represented by the prefix;
- an explicit cache salt or namespace change;
- a weight reload/fine-tune while the process is alive;
- eviction or capacity reuse of the physical entry; or
- mutation of a stored MLX cache after publication.

Nanoserve's v1 key therefore includes a stable model namespace, tokenizer
namespace, cache format version, and the chained token-block hash. Stored MLX
arrays are copied on put and again on get so a decode cannot append into the
shared entry. `clear()` is required after model reload. Unsupported adapters,
multimodal inputs, and prompt embeddings are outside v1 rather than silently
sharing an unsafe namespace.

## Scheduler source lessons and v1 boundary

- Production schedulers treat scheduling as a token-budget problem. Running requests consume one
  decode token per step; waiting requests consume their remaining prefill
  tokens, bounded by request-count and token budgets.
- Admit running work first to avoid inflating inter-token latency, then use
  remaining capacity for prefills.
- A request owns mutable decode state while running. Finished requests release
  it; cancellation and failure must also release it.
- Continuous batching means the active batch may change between forward passes.
  It does not mean concatenating unrelated histories into one logical cache.
- When allocation cannot fit, vLLM preempts and recomputes. Nanoserve v1 has a
  bounded active batch but an unbounded waiting deque; it has no token budget,
  backpressure, or capacity rejection. Those remain required production work.

## MLX decode-loop lessons retained

- Build a per-layer prompt cache, prefill all prompt tokens, take logits at the
  final position, greedily sample, then repeatedly call the model with only the
  sampled token and the same mutable cache.
- Force evaluation at the measurement boundary. MLX is lazy; timing Python
  dispatch without `mx.eval` is not model latency.
- Measure TTFT from immediately before tokenization/prefill through evaluation
  of the first sampled token. Subsequent timestamps support TPOT.
- Stream only evaluated token IDs/text pieces. The public implementation calls
  the model directly and never calls `mlx_lm.generate` or `generate_step`.
