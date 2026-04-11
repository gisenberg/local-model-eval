"""
Tests for TTLCache.

Each test uses ``unittest.mock.patch`` to replace ``time.monotonic`` with a
deterministic counter, allowing us to control expiry without ``time.sleep``.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from ttcache import TTLCache  # assuming the implementation is saved in ttcache.py


def _monotonic_generator(start: float = 0.0, step: float = 1.0):
    """Yield increasing monotonic times."""
    t = start
    while True:
        yield t
        t += step


@pytest.fixture
def mock_time():
    """Patch time.monotonic with a controllable generator."""
    gen = _monotonic_generator()
    with patch("time.monotonic", side_effect=lambda: next(gen)):
        yield gen


def test_basic_get_put(mock_time):
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2


def test_capacity_eviction_lru(mock_time):
    cache = TTLCache(capacity=3, default_ttl=10)
    cache.put("a", 1)   # LRU
    cache.put("b", 2)
    cache.put("c", 3)   # MRU

    # cache is full; accessing 'a' makes it MRU
    assert cache.get("a") == 1   # now order: b, c, a (a MRU)

    cache.put("d", 4)   # should evict LRU = 'b'
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4


def test_ttl_expiry(mock_time):
    cache = TTLCache(capacity=2, default_ttl=2)
    cache.put("x", "hello")   # expires at t=2
    assert cache.get("x") == "hello"

    # advance time to just before expiry
    next(mock_time)  # t=1
    assert cache.get("x") == "hello"

    # advance time to expiry
    next(mock_time)  # t=2
    assert cache.get("x") is None   # expired and removed


def test_custom_per_key_ttl(mock_time):
    cache = TTLCache(capacity=2, default_ttl=5)
    cache.put("short", 1, ttl=1)   # expires at t=1
    cache.put("long", 2, ttl=10)   # expires at t=10

    # t=0 -> both present
    assert cache.get("short") == 1
    assert cache.get("long") == 2

    # t=1 -> short expired
    next(mock_time)
    assert cache.get("short") is None
    assert cache.get("long") == 2

    # t=2 -> long still valid
    next(mock_time)
    assert cache.get("long") == 2


def test_delete(mock_time):
    cache = TTLCache(capacity=2, default_ttl=10)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("a") is False   # already gone
    assert cache.get("b") == 2
    assert cache.delete("b") is True
    assert cache.size() == 0


def test_size_with_mixed_expired_and_valid(mock_time):
    cache = TTLCache(capacity=3, default_ttl=5)
    cache.put("a", 1)   # expires at t=5
    cache.put("b", 2)   # expires at t=5
    cache.put("c", 3)   # expires at t=5

    # all valid
    assert cache.size() == 3

    # advance time so that 'a' and 'b' expire, 'c' still valid
    next(mock_time)  # t=1
    next(mock_time)  # t=2
    next(mock_time)  # t=3
    next(mock_time)  # t=4
    assert cache.size() == 3   # lazy cleanup hasn't run yet

    # accessing an expired item triggers its removal
    assert cache.get("a") is None
    # now size() will clean up the expired 'b' as well
    assert cache.size() == 2   # 'b' and 'c' are still in map; 'b' expired but not accessed yet
    # accessing 'b' removes it
    assert cache.get("b") is None
    assert cache.size() == 1   # only 'c' remains
