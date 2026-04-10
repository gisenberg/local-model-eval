import pytest
from unittest.mock import patch
from ttl_cache import TTLCache


def test_basic_get_put():
    with patch('ttl_cache.time.monotonic', side_effect=[0, 0, 10]):
        cache = TTLCache(capacity=3, default_ttl=100)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.get("b") is None


def test_capacity_eviction_lru_order():
    with patch('ttl_cache.time.monotonic', side_effect=[0, 0, 0, 0, 0, 0, 0, 0]):
        cache = TTLCache(capacity=2, default_ttl=100)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # evicts "a" (LRU)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

        # Now "b" is LRU (since "c" was just added)
        cache.put("d", 4)  # evicts "b"
        assert cache.get("b") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4


def test_ttl_expiry():
    with patch('ttl_cache.time.monotonic', side_effect=[0, 0, 5, 10]):
        cache = TTLCache(capacity=3, default_ttl=5)
        cache.put("a", 1)
        assert cache.get("a") == 1  # at t=5, still valid
        assert cache.get("a") is None  # at t=10, expired (5+5=10)


def test_custom_per_key_ttl():
    with patch('ttl_cache.time.monotonic', side_effect=[0, 0, 0, 3, 6]):
        cache = TTLCache(capacity=3, default_ttl=10)
        cache.put("a", 1, ttl=2)  # expires at t=2
        cache.put("b", 2, ttl=5)  # expires at t=5
        assert cache.get("a") == 1  # t=3 → expired? No: 3 > 2 → expired!
        assert cache.get("a") is None  # t=3: a expired (2 < 3)
        assert cache.get("b") == 2  # t=3: b valid (3 < 5)
        assert cache.get("b") is None  # t=6: b expired (6 > 5)


def test_delete():
    with patch('ttl_cache.time.monotonic', side_effect=[0, 0, 0, 0]):
        cache = TTLCache(capacity=3, default_ttl=100)
        cache.put("a", 1)
        assert cache.delete("a") is True
        assert cache.get("a") is None
        assert cache.delete("a") is False  # already gone


def test_size_mixed_expired_valid():
    with patch('ttl_cache.time.monotonic', side_effect=[0, 0, 0, 0, 5, 5]):
        cache = TTLCache(capacity=5, default_ttl=3)
        cache.put("a", 1)  # expires at 3
        cache.put("b", 2)  # expires at 3
        cache.put("c", 3)  # expires at 3
        assert cache.size() == 3  # all valid at t=0

        # At t=5, all expired
        assert cache.size() == 0  # lazy cleanup on size()

        # Add new item
        cache.put("d", 4)  # expires at 8
        assert cache.size() == 1
