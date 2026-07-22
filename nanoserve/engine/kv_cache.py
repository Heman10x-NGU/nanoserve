"""Small, correctness-first prefix-cache store inspired by vLLM's hash chain.

This is deliberately not a paged allocator. Each entry owns one complete MLX
prompt-cache snapshot at a token-prefix boundary. The chained token-block hash
and collision check decide identity; an LRU controls the bounded store.
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

CACHE_FORMAT_VERSION = "nanoserve-prefix-v1"


@dataclass(frozen=True, slots=True)
class CacheMatch:
    """An owned cache snapshot for the longest reusable token prefix."""

    prefix_length: int
    cache: list[Any]


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    token_ids: tuple[int, ...]
    cache: list[Any]


class PrefixCache:
    """Bounded exact-prefix cache with chained token-block hashes."""

    def __init__(
        self,
        *,
        namespace: str,
        clone: Callable[[Sequence[Any]], list[Any]],
        block_size: int = 16,
        max_entries: int = 32,
    ) -> None:
        if not namespace:
            raise ValueError("namespace must not be empty")
        if block_size < 1:
            raise ValueError("block_size must be at least one")
        if max_entries < 1:
            raise ValueError("max_entries must be at least one")
        self.namespace = namespace
        self.block_size = block_size
        self.max_entries = max_entries
        self._clone = clone
        self._entries: OrderedDict[bytes, _CacheEntry] = OrderedDict()
        self._lookups = 0
        self._hits = 0

    @property
    def hit_rate(self) -> float:
        """Successful longest-prefix lookups divided by all lookups."""
        return self._hits / self._lookups if self._lookups else 0.0

    def put(self, token_ids: Sequence[int], cache: Sequence[Any]) -> None:
        """Publish an immutable snapshot for exactly ``token_ids``."""
        tokens = _owned_tokens(token_ids)
        if not tokens:
            raise ValueError("token_ids must contain at least one token")
        if len(tokens) % self.block_size:
            raise ValueError("cache prefixes must end at a full block boundary")
        key = self._prefix_hash(tokens)
        self._entries[key] = _CacheEntry(tokens, self._clone(cache))
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def longest_prefix(self, token_ids: Sequence[int]) -> CacheMatch | None:
        """Return a fresh copy of the longest cached prefix of ``token_ids``."""
        tokens = _owned_tokens(token_ids)
        self._lookups += 1
        candidate_lengths = sorted(
            {len(entry.token_ids) for entry in self._entries.values()}, reverse=True
        )
        for length in candidate_lengths:
            if length > len(tokens):
                continue
            prefix = tokens[:length]
            key = self._prefix_hash(prefix)
            entry = self._entries.get(key)
            # The token comparison is the collision guard. A digest match alone
            # never authorizes tensor reuse.
            if entry is None or entry.token_ids != prefix:
                continue
            self._entries.move_to_end(key)
            self._hits += 1
            return CacheMatch(length, self._clone(entry.cache))
        return None

    def clear(self) -> None:
        """Invalidate every cache entry, for example after a weight reload."""
        self._entries.clear()

    def block_hashes(self, token_ids: Sequence[int]) -> tuple[str, ...]:
        """Expose the chained hashes for explanation and deterministic tests."""
        tokens = _owned_tokens(token_ids)
        parent = hashlib.sha256(
            f"{CACHE_FORMAT_VERSION}\0{self.namespace}".encode("utf-8")
        ).digest()
        hashes: list[str] = []
        for start in range(0, len(tokens), self.block_size):
            block = tokens[start : start + self.block_size]
            encoded = struct.pack(">I", len(block)) + b"".join(
                struct.pack(">I", token_id) for token_id in block
            )
            parent = hashlib.sha256(parent + encoded).digest()
            hashes.append(parent.hex())
        return tuple(hashes)

    def _prefix_hash(self, token_ids: Sequence[int]) -> bytes:
        hashes = self.block_hashes(token_ids)
        if not hashes:
            raise ValueError("token_ids must contain at least one token")
        return bytes.fromhex(hashes[-1])


def _owned_tokens(token_ids: Sequence[int]) -> tuple[int, ...]:
    tokens = tuple(int(token_id) for token_id in token_ids)
    if any(token_id < 0 or token_id > 0xFFFFFFFF for token_id in tokens):
        raise ValueError("token IDs must be unsigned 32-bit integers")
    return tokens
