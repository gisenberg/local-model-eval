import pytest
from unittest.mock import patch
from typing import Any
from your_module import TTLCache  # Replace 'your_module' with the actual filename

# Helper to create a cache instance
def create_cache(capacity: int = 3, default_ttl: float = 10.0):
    return TTLCache(capacity, default_ttl)

def test_basic_get_put():
    """Test basic insertion and retrieval."""
    cache = create_cache(capacity=2, default_ttl=10.0)
    
    with patch('your_module.time.monotonic', return_value=0.0):
        cache.put("a", 1)
        cache.put("b", 2)
        
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") is None

def test_capacity_eviction_lru_order():
    """Test that LRU items are evicted when capacity is reached."""
    cache = create_cache(capacity=2, default_ttl=100.0)
    
    with patch('your_module.time.monotonic', return_value=0.0):
        cache.put("a", 1)
        cache.put("b", 2)
        
        # Access 'a' to make it MRU
        cache.get("a")
        
        # Add 'c', should evict 'b' (LRU)
        cache.put("c", 3)
        
        assert cache.get("a") == 1
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3

def test_ttl_expiry():
    """Test that items expire after default TTL."""
    cache = create_cache(capacity=5, default_ttl=5.0)
    
    with patch('your_module.time.monotonic', return_value=0.0):
        cache.put("x", 100)
        assert cache.get("x") == 100
        
    # Simulate time passing (6 seconds later)
    with patch('your_module.time.monotonic', return_value=6.0):
        assert cache.get("x") is None  # Expired
        assert cache.size() == 0

def test_custom_per_key_ttl():
    """Test that custom TTL overrides default TTL."""
    cache = create_cache(capacity=5, default_ttl=10.0)
    
    with patch('your_module.time.monotonic', return_value=0.0):
        cache.put("short", 1, ttl=2.0)
        cache.put("long", 2, ttl=20.0)
        
    # At t=5: 'short' should be expired, 'long' should be valid
    with patch('your_module.time.monotonic', return_value=5.0):
        assert cache.get("short") is None
        assert cache.get("long") == 2

def test_delete():
    """Test manual deletion of keys."""
    cache = create_cache(capacity=3, default_ttl=10.0)
    
    with patch('your_module.time.monotonic', return_value=0.0):
        cache.put("key1", "val1")
        cache.put("key2", "val2")
        
        assert cache.delete("key1") is True
        assert cache.delete("key1") is False  # Already deleted
        assert cache.get("key1") is None
        assert cache.get("key2") == "val2"

def test_size_with_mixed_expired_valid():
    """Test size() returns count of valid items and cleans up expired ones."""
    cache = create_cache(capacity=5, default_ttl=5.0)
    
    with patch('your_module.time.monotonic', return_value=0.0):
        cache.put("valid1", 1)
        cache.put("valid2", 2)
        cache.put("exp1", 3, ttl=2.0)
        cache.put("exp2", 4, ttl=3.0)
        
    # At t=4: exp1 and exp2 are expired, valid1 and valid2 are valid
    with patch('your_module.time.monotonic', return_value=4.0):
        # size() should clean up expired items and return 2
        assert cache.size() == 2
        
        # Verify expired items are gone
        assert cache.get("exp1") is None
        assert cache.get("exp2") is None
        
        # Verify valid items remain
        assert cache.get("valid1") == 1
        assert cache.get("valid2") == 2
