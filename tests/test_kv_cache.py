"""Behavioral and real-model gates for prefix-cache reuse."""

from copy import deepcopy

import pytest

from nanoserve.engine.kv_cache import PrefixCache


def test_longest_prefix_returns_a_fresh_copy_of_the_longest_match() -> None:
    cache = PrefixCache(
        namespace="model-a:tokenizer-a:v1",
        block_size=4,
        clone=deepcopy,
    )
    short_state = [{"offset": 4}]
    long_state = [{"offset": 8}]
    cache.put([1, 2, 3, 4], short_state)
    cache.put([1, 2, 3, 4, 5, 6, 7, 8], long_state)

    match = cache.longest_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9])

    assert match is not None
    assert match.prefix_length == 8
    assert match.cache == long_state
    match.cache[0]["offset"] = 99
    assert cache.longest_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9]).cache == long_state


def test_hash_chain_rejects_changed_tokens_and_namespace() -> None:
    cache = PrefixCache(
        namespace="model-a:tokenizer-a:v1",
        block_size=4,
        clone=deepcopy,
    )
    cache.put([10, 11, 12, 13], [{"offset": 4}])

    assert cache.longest_prefix([10, 11, 12, 99, 20]) is None

    other_namespace = PrefixCache(
        namespace="model-b:tokenizer-a:v1",
        block_size=4,
        clone=deepcopy,
    )
    assert other_namespace.longest_prefix([10, 11, 12, 13, 20]) is None


def test_cache_rejects_a_partial_block_boundary() -> None:
    cache = PrefixCache(namespace="test", block_size=4, clone=deepcopy)

    with pytest.raises(ValueError, match="full block"):
        cache.put([1, 2, 3], ["state"])


def test_capacity_evicts_the_least_recently_used_entry() -> None:
    cache = PrefixCache(
        namespace="test",
        block_size=2,
        max_entries=2,
        clone=deepcopy,
    )
    cache.put([1, 2], ["first"])
    cache.put([3, 4], ["second"])
    assert cache.longest_prefix([1, 2, 9]) is not None  # first is now newest

    cache.put([5, 6], ["third"])

    assert cache.longest_prefix([3, 4, 9]) is None
    assert cache.longest_prefix([1, 2, 9]) is not None


def test_hit_rate_counts_successful_and_failed_lookups() -> None:
    cache = PrefixCache(namespace="test", block_size=2, clone=deepcopy)
    cache.put([1, 2], ["state"])

    cache.longest_prefix([1, 2, 3])
    cache.longest_prefix([9, 9, 9])

    assert cache.hit_rate == pytest.approx(0.5)


@pytest.mark.integration
def test_prefix_reuse_token_identical() -> None:
    """Load-bearing gate: warm greedy output must equal cold output exactly."""
    from nanoserve.backends.base import DEFAULT_MODEL, GenerationResult
    from nanoserve.backends.mlx_backend import MLXBackend

    backend = MLXBackend.load(DEFAULT_MODEL)
    context = (
        "Inference engines reuse attention keys and values from a stable prefix. "
        "The cache is valid only when every identity input matches. "
    )
    full_ids = backend.encode(context * 12 + "Question: what must warm decode preserve?")
    split_at = ((len(full_ids) - 10) // 64) * 64
    assert split_at >= 64
    prefix_ids, suffix_ids = full_ids[:split_at], full_ids[split_at:]

    prefix_state = backend.forward_logits(prefix_ids).cache
    cache = PrefixCache(
        namespace=backend.cache_namespace,
        block_size=64,
        clone=backend.clone_cache,
    )
    cache.put(prefix_ids, prefix_state)
    match = cache.longest_prefix(full_ids)
    assert match is not None

    cold = backend.generate(full_ids, max_tokens=12)
    warm = backend.generate(
        full_ids[match.prefix_length :],
        cache=match.cache,
        max_tokens=12,
    )
    assert isinstance(cold, GenerationResult)
    assert isinstance(warm, GenerationResult)
    assert warm.token_ids == cold.token_ids
