import pytest
from typing import Any
from unittest.mock import patch
from time import monotonic

from ttl_cache import TTLCache, CacheNode

@pytest.fixture
def mock_time():
    """Fixture to mock time.monotonic() for deterministic tests."""
    with patch('time.monotonic') as mock:
        yield mock

def test_basic_get_put(mock_time):
    """Test basic get and put operations."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") is None

def test_capacity_eviction(mock_time):
    """Test LRU eviction when cache is at capacity."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # Should evict "a" as it's LRU

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_ttl_expiry(mock_time):
    """Test that items expire after their TTL."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("a", 1, ttl=5.0)
    cache.put("b", 2, ttl=15.0)

    mock_time.return_value = 105.0  # 5 seconds after put
    assert cache.get("a") is None  # Expired
    assert cache.get("b") == 2  # Still valid

def test_custom_per_key_ttl(mock_time):
    """Test that custom TTL overrides default TTL."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("a", 1, ttl=5.0)
    cache.put("b", 2)  # Uses default TTL

    mock_time.return_value = 105.0
    assert cache.get("a") is None  # Expired after 5 seconds
    assert cache.get("b") == 2  # Still valid (default TTL is 10 seconds)

def test_delete(mock_time):
    """Test delete operation."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=2, default_ttl=10.0)

    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.delete("a") is True
    assert cache.get("a") is None
    assert cache.delete("c") is False  # Non-existent key

def test_size_with_mixed_expired_valid(mock_time):
    """Test size() returns correct count with mixed expired/valid items."""
    mock_time.return_value = 100.0
    cache = TTLCache(capacity=3, default_ttl=10.0)

    cache.put("a", 1, ttl=5.0)
    cache.put("b", 2, ttl=15.0)
    cache.put("c", 3, ttl=5.0)

    mock_time.return_value = 105.0  # 5 seconds after puts
    assert cache.size() == 1  # Only "b" is valid
    assert cache.get("a") is None  # Expired
    assert cache.get("b") == 2  # Valid
    assert cache.get("c") is None  # Expired

    # Add a new item that will expire immediately
    cache.put("d", 4, ttl=0.0)
    assert cache.size() == 1  # "d" is expired

    # Add an item that will expire in the future
    cache.put("e", 5, ttl=10.0)
    mock_time.return_value = 110.0  # 10 seconds after put
    assert cache.size() == 0  # All items expired
